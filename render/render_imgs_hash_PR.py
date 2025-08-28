import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ["KITTI360_DATASET"] = "/sdf1/kitti360"
# os.environ["XGRIDS_DATASET"] = "/xgrids/roma_playground" #"/sdf1/xgrids/reconstruct"
os.environ['QT_QPA_PLATFORM']="offscreen"
import torch
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from wisp.core import Rays
import sys
sys.path.append("./")
from models.rays import *
import open3d as o3d
import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from wisp.ops.differential import autodiff_gradient, finitediff_gradient
import time
import torch.nn.functional as F
import kaolin.render.spc as spc_render
# from labels_ours import labels, id2label
from Fig_paper.labels_unify import labels, id2label
import imageio
from tqdm import tqdm
# from utils.scannet_data_tools.kitti360Viewer3DRaw import init_camera
from lietorch import SE3
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from models.decoder import SemanticSDFDecoder
import math
import kaolin.ops.spc as spc_ops
from wisp.models.grids import *
from instance.models.query_model import QueryModel

h_SI = True #False #
DATASET = "scannet" #"kitti360"
# FRAME = range(6330,6530,1) # range(250,1851,1)#range(250,1800,1)
CUBE_NUM = 1 #1
frame_num = 320 #355 #300 #343 #0, 250, 400, 600 1095
step = 1 #5
START = 0
TEST = [1230,1550] #[395,750] # [430, 860] #[220, 343] #[0.343]
# TEST = [250,1851] 
FRAME = range(TEST[0],TEST[1])
BKGD = False # True # 
PRO_PATH = '/mnt/nas_new/yx/pami/exp/scannet/0000_02' #'/sdf1/yx/pami/draw_fig/replica_new'
MODEL = PRO_PATH + '/v0/grid/optimed_grid_0_19.pth' #'/mnt/data/yx/exp_result/panoptic_recon++/semantic/v0/grid/optimed_grid_0_11.pth'
result_path = '/mnt/nas_new/yx/pami/exp/scannet/0000_02/v0' #'/mnt/data/yx/exp_result/panoptic_recon++/semantic/v0'
# MODEL = '/sde1/yx/dataset/replica/result/apartment_2/rgb_depth_new/v0/grid/optimed_grid_0_19.pth'
# result_path = '/sde1/yx/dataset/replica/result/apartment_2/rgb_depth_new/v0'
# MODEL = '/sde1/yx/result/panoptic_recon++/scannetpp/scen_1ada7a0617/panoptic_test/v0/grid/optimed_grid_0_19.pth'
# result_path = '/sde1/yx/result/panoptic_recon++/scannetpp/scen_1ada7a0617/panoptic_test/v0'
DEBUG = False
SENCE = ['camera'] #['camera', 'lidar']

BATCH = 80000 #8000 #30000 #241600 # #40000
cos_anneal_ratio = 1
def gen_pose(t, radius, v, pose_old):
        z = radius * torch.cos(torch.pi/3*t) 
        y = radius * torch.sin(torch.pi/3*t)
        x = v * t
        position = torch.stack([x,y,z],-1).to(pose_old.device) + pose_old[:3][None,:]
        quat = pose_old[3:][None,:].expand(position.shape[0],-1)
        poses = torch.hstack((position, quat))
        return poses
def gen_rays(height, width, intrinsic, pose, TrCam0ToVelo):
    grid = create_meshgrid(height, width, normalized_coordinates=False, device=pose.device)[0]
    i, j = grid.unbind(-1)
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    rays_d_cam = \
        torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], dim=-1) # (H, W, 3)
    
    rays_o = TrCam0ToVelo[:3,3].expand(height,-1)
    rays_d = rays_d_cam @ TrCam0ToVelo[:3, :3].T
    dir_norm = torch.norm(rays_d, dim=-1, keepdim=True)
    rays_d = rays_d / dir_norm
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

def loadCameraToPose(filename):
    # open file
    Tr = {}
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lineData = list(line.strip().split())
            data = np.array(lineData[1:]).reshape(3,4).astype(np.float)
            data = np.concatenate((data,lastrow), axis=0)
            Tr[lineData[0][:-1]] = data
    return Tr

class Render():
        def  __init__(self, cube_idx, yaml_path, MODEL_PATH, mode='val'):
               
                self.MODEL_PATH = MODEL_PATH
                with open(yaml_path, "r") as f:
                        self.cfg = yaml.safe_load(f)
                self.cam_id = 0 if mode=='val' else 1
                self.s = 500
                self.trunc_d = 0.3
                self.width, self.height = self.cfg['img']['width'], self.cfg['img']['height']
                self.sem_model = self.cfg['sem_model']
                self.normal_type = 'trace' # 'render'
                self.weight_mode = 'mid' #'grad' #
                if self.cfg['path']['dataset_type'] == 'kitti360':
                        self.camera0, self.TrCam0ToVelo = init_camera(seq=int(self.cfg['path']['seq']), cam_id=0)
                        self.camera1, self.TrCam1ToVelo = init_camera(seq=int(self.cfg['path']['seq']), cam_id=1)
                        self.TrCamToPose = loadCameraToPose(os.path.join(os.path.join(os.environ['KITTI360_DATASET'], 'calibration'),'calib_cam_to_pose.txt'))
                        self.TrCam1ToCam0 = torch.from_numpy(np.linalg.inv(self.TrCamToPose['image_00']) @ self.TrCamToPose['image_01'] @ np.linalg.inv(self.camera1.R_rect))
                self.num_lods = 2
                self.pretrain = self.cfg['pretrain']
                #=========rgb========
                self.rgb_flag = False #True #
                self.appearance_embedding = self.cfg['decoder']['rgb']['embedding_a_dim'] > 0 if self.rgb_flag else False
                self.embedding_a_dim = self.cfg['decoder']['rgb']['embedding_a_dim'] if self.rgb_flag else 0
                #=========dinov2========
                self.feat_flag = False  #True #
                self.feat_dim = 64 #384 #64
                #=========semantic========
                self.semantic_flag = True #False #
                self.class_num = 10 #44 #
                #=========instance========
                self.instance_flag = True #False #False #
                if self.instance_flag:
                        self.instance_num = 1000 #self.cfg['instance']['num_queries']
                        self.Thing_class = self.cfg['Thing_class']
                        np.random.seed(42)
                        bg_id_mapping = [0]+self.cfg['Stuff_class']
                        self.color_id = np.random.random( [self.instance_num+len(bg_id_mapping), 3] )*255.
                        for i in range(len(self.cfg['Stuff_class'])):
                                semID = bg_id_mapping[i]
                                stuff_color = np.vstack(id2label[semID].color) 
                                self.color_id[i] = stuff_color.squeeze()
                        # self.color_id[0] = np.zeros([1,3])
                self.panoptic_flag = True #False #False #
                if self.panoptic_flag:
                        self.stuff_class = self.cfg['Stuff_class']
                self.pca = None
                self.load_model(cube_idx)
                self.cfg['path']['proj_dir'] = PRO_PATH
                self.mode = mode
                self.save_path = result_path if self.mode == 'val' else PRO_PATH+'/NVS'
                os.makedirs(self.save_path, exist_ok=True)
                self.pc_debug = []

        def load_model(self, idx):
                # path = os.path.join(self.cfg['path']['proj_dir'], f'grid/optimed_grid_{idx}.pth')
                # path = MODEL + f"/optimed_grid_{idx}.pth"
                path = self.MODEL_PATH
                data = torch.load(path)
                self.grid = data['grid']
                # self.rgb_feat_grid = data['rgb_feat_grid']
                self.decoder = data['decoder']
                self.octree = data['octree']
                if self.appearance_embedding:
                        self.embedding_a = data['embedding_a']
                self.scale = data['scale']#.cpu().numpy()
                self.origin = data['origin']#.cpu().numpy()
                self.device = self.scale.device
                self.s = data['s_val']
                self.s = torch.exp(self.s * 10.0).clip(1e-6, 1e6)
                if BKGD:
                        from models.fields import NeRF
                        self.nerf_outside = NeRF(**self.cfg['outside']['nerf']).to(self.device)
                        self.nerf_outside.load_state_dict(data['nerf_outside'])
                if self.instance_flag:
                        from instance.models.mask3d import Mask3D
                        # self.instance_corners = data['instance_corners']
                        self.instance_model = Mask3D(**self.cfg['instance'], query_position=data['query_position'], device=self.device)
                        self.instance_model.to(self.device)
                        self.instance_model.load_state_dict(data['instance_model']) #, strict=False
                        # self.instance_model.update_network(data['query_mask'], data['query_update_map'])
                        self.instance_model.gaussians = QueryModel()
                        self.instance_model.gaussians.load_ply(os.path.join(PRO_PATH, f'v0/grid/optimed_query_19.ply'))
                        self.THING = self.cfg['Thing_class']

        def query_pts_sem(self, pts):
                # pc_semantic = torch.zeros(pts.shape[0],1).to(self.device)
                pts = (pts-self.origin)/self.scale
                pts = (pts+1.0)/2
                if self.sem_model=='mlp':
                        radiance_feats = self.grid['feature'](pts.reshape(-1, 3),  self.grid['feature'].num_lods-1)
                        feats = self.decoder['feature'](radiance_feats)
                        sem = self.decoder['semantic'](feats)
                elif self.sem_model=='grid':
                        feats = self.grid['semantic'](pts.reshape(-1, 3),  self.grid['semantic'].num_lods-1)
                        sem = self.decoder['semantic'](feats)
                elif self.sem_model=='geo':
                        feats = self.grid['sdf'](pts.reshape(-1, 3))
                        _,sem = self.decoder['sdf'].forward(feats)
                sem = sem.reshape(-1, self.class_num)
                sem = torch.argmax(torch.nn.functional.softmax(sem, dim=-1), -1)
                # pc_semantic[ridx_hit.long(), :] = sem[:,None]
                return sem
        
        def query_pts(self, pts, channel=[]):
                # pc_semantic = torch.zeros(pts.shape[0],1).to(self.device)
                pts = (pts-self.origin)/self.scale
                pts = (pts+1.0)/2
                sem_batch = 50000
                semantic, instance, panoptic = torch.zeros(pts.shape[0]), torch.zeros(pts.shape[0]), torch.zeros(pts.shape[0])
                color = torch.zeros([pts.shape[0],3])
                pidxs = torch.zeros(pts.shape[0])
                for i in tqdm(range(0, pts.shape[0], sem_batch), desc='get semantic labels'):
                        next_ind = min(i+sem_batch, pts.shape[0])
                        p_g = pts[i:next_ind]

                        if 'rgb' in channel:
                                # feats = grid['sdf'](p_g.reshape(-1, 3))
                                radiance_feats = self.grid['rgb'](p_g.reshape(-1, 3))
                                # rgb_g = decoder['rgb'](radiance_feats)
                                grad = autodiff_gradient(p_g.reshape(-1, 3), sdf) / self.scale
                                normal = grad / grad.norm(2, 1)[:, None]
                                embedding = data['embedding_a'][0] if 'embedding_a' in data.keys() else None
                                rgb_g = self.decoder['rgb'](radiance_feats=radiance_feats, view_dirs=-grad/grad.norm(2, 1)[:, None], appearance_embedding=embedding, grads=normal)
                                # rgbs = self.decoder['rgb'](radiance_feats=radiance_feats, view_dirs=view_dirs.reshape(-1,3),appearance_embedding=embedding, grads=normals)
                                rgb = torch.sigmoid(rgb_g)
                                color[i:next_ind] = rgb.cpu()
                        if 'sem' in channel:
                                # #                                 feats = self.grid['sdf'](p_g.reshape(-1, 3))
                                _, sdf_last_layer = self.decoder['sdf'](feats,return_last=True)

                                semantic_feat = self.grid['semantic'](p_g.reshape(-1, 3))
                                sem_g_ = self.decoder['semantic'].forward(semantic_feat)
                                sem = torch.argmax(torch.nn.functional.softmax(sem_g_.squeeze(), dim=-1), -1)
                                semantic[i:next_ind] = sem.cpu()
                        if 'ins' in channel:
                                instance_corners = [(p_g*2-1).reshape(-1, 3)]
                                ins_feat = [self.grid['instance'](p_g.reshape(-1, 3))]
                                instance_heat_map, thing_semantic_heat_map, query_position, query_loss = self.instance_model(
                                                                                                        ins_feat, 
                                                                                                        instance_corners,
                                                                                                        self.scale
                                                                                                        ) 
                                ins_g_ = instance_heat_map
                                ins = torch.argmax(torch.nn.functional.softmax(ins_g_.squeeze(), dim=-1), -1)
                                instance[i:next_ind] = ins.cpu()

                                stuff_mask = torch.ones(sem_g_.shape[-1]).bool().to(self.device)
                                stuff_mask[self.THING] = False
                                sem_g_norm = F.softmax(sem_g_)
                                unlabel_mask = torch.logical_and(sem_g_norm[:,stuff_mask].sum(-1)>0.2, sem_g_norm[:,stuff_mask].sum(-1)<0.8)
                                sem_g_norm[unlabel_mask] = 0
                                sem_g_norm[unlabel_mask,0] = 1
                                panoptic_instance_prob = (1 - sem_g_norm[:,stuff_mask].sum(-1))[:,None] * ins_g_/ins_g_.sum(-1)[:,None] # * out['instance']
                                pan_g_ = torch.hstack((sem_g_norm[:,stuff_mask], panoptic_instance_prob))
                                # pan_g_ = torch.hstack((sem_g_[:,stuff_mask], ins_g_))
                                pan = torch.argmax(torch.nn.functional.softmax(pan_g_.squeeze(), dim=-1), -1)
                                panoptic[i:next_ind] = pan.cpu()  

                                stuff_label = self.cfg['Stuff_class']
                                thing_label = torch.argmax(thing_semantic_heat_map,-1)
                                thing_map = [0]+ self.cfg['Thing_class']
                                pan2sem_mapping = { **{pan_id+1: sem_id for pan_id, sem_id in enumerate(stuff_label)}, \
                                                **{pan_id+1+len(stuff_label): thing_map[sem_id] for pan_id, sem_id in enumerate(thing_label)} }
                                sem = torch.zeros_like(pan)
                                for key, val in pan2sem_mapping.items():
                                        sem[pan==key] = val
                                semantic[i:next_ind] = sem.cpu()

                return {'rgb':color, 'sem':semantic, 'ins':instance, 'pan':panoptic}
        

        def read_pose(self):
                self.vec_es = {}
                if self.cfg['path']['dataset_type'] == 'kitti360':
                        from utils.scannet_data_tools.init_data import init_poses_from_gt, init_all_poses, get_novel_poses
                        if self.mode == 'NVS':
                                self.cfg['path']['proj_dir'] = os.path.join(self.cfg['path']['proj_dir'], "NVS3")
                                self.vec_es, self.vec_cam = get_novel_poses(self.device)
                        else:
                                self.vec_es, self.vec_cam, _, _, _, _, _  = init_all_poses(self.cfg['path']['dataset_dir'], int(self.cfg['path']['seq']), FRAME, self.device, cam_id=self.cam_id) 
                                if self.mode=='test':
                                        self.cfg['path']['proj_dir'] = os.path.join(self.cfg['path']['proj_dir'], "NVS")
                        
                elif self.cfg['path']['dataset_type'] == 'xgrids':
                        if self.mode=='val':
                                from utils.scannet_data_tools.init_data_xgrids import init_all_poses
                                pose_path = os.path.join(self.cfg['path']['proj_dir'], "pose/pose_optim.json") #_init
                                self.vec_es, self.vec_cam, _, _, _, _, _  = init_all_poses(self.cfg, pose_path, FRAME, self.device, colmap=self.cfg['colmap']) 
                        elif self.mode=='test':
                                from utils.scannet_data_tools.init_data_xgrids import get_novel_poses
                                self.vec_cam, self.width, self.height = get_novel_poses(self.device)
                                self.vec_es = self.vec_cam
                                self.cfg['path']['proj_dir'] = os.path.join(self.cfg['path']['proj_dir'], "NVS")

                elif self.cfg['path']['dataset_type'] == 'scannet':
                        from utils.scannet_data_tools.init_scannet_pose import init_all_poses, init_all_poses_custom
                        if self.mode=='val':
                                self.vec_es, self.vec_cam, _ = init_all_poses(self.cfg['path']['dataset_dir'], FRAME, self.device) 
                        else:
                                # self.vec_es, self.vec_cam = init_all_poses_custom("/sdf1/yx/ijrr/demo/traj_data_all_2.npy", self.device) 
                                self.vec_es, self.vec_cam = init_all_poses_custom("/sdf1/yx/demo/traj_scannet_5748_fixed.npy", self.device) 

        def get_render_rays(self, render_start, render_end, step):
                if self.cfg['path']['dataset_type'] == 'kitti360':
                        from kitti_dataset import KITTIDataset
                        test_list = list(range(render_start, render_end, step)) #+20, 10
                        dataset_test = KITTIDataset(self.cfg, test_list, self.device, self.vec_es, self.vec_es, self.vec_cam, self.mode, optim_flag=self.cfg["loss_term"],cam_id=self.cam_id)
                        # self.TrCam0ToVelo, self.fx, self.fy, self.cx, self.cy = dataset_test.TrCam0ToVelo, dataset_test.fx, dataset_test.fy, dataset_test.cx, dataset_test.cy
                        dataloader_test = DataLoader(dataset = dataset_test, batch_size=1)
                elif self.cfg['path']['dataset_type'] == 'xgrids':
                        from xgrids_dataset import KITTIDataset
                        test_list = list(range(render_start, render_end, step)) #+20, 10
                        dataset_test = KITTIDataset(self.cfg, test_list, test_list, self.device, self.vec_es, self.vec_es, self.vec_cam, self.mode, optim_flag=self.cfg["loss_term"])
                        self.fx, self.fy, self.cx, self.cy = dataset_test.fx, dataset_test.fy, dataset_test.cx, dataset_test.cy
                        dataloader_test = DataLoader(dataset = dataset_test, batch_size=1)
                elif self.cfg['path']['dataset_type'] == 'scannet':
                        from scannet_dataset import ScannetDataset
                        test_list = list(range(render_start, render_end, step))
                        # test_list = [3,142]
                        dataset_test = ScannetDataset(self.cfg, test_list, self.device, self.vec_es, self.mode, optim_flag=self.cfg["loss_term"]) 
                        dataloader_test = DataLoader(dataset = dataset_test, batch_size=1)
                else:
                        from maicity_dataset import MaiCityDataset
                        test_list = list(range(render_start, render_end, step))
                        dataloader_test = MaiCityDataset(self.cfg, test_list, self.device, self.vec_es, self.vec_es, "test", self.cfg['path']['dataset_type'], None, normal_flag=False,semantic_flag=False) 
                return dataloader_test
        
        
        def pca_feature(self, features):
                features = features.reshape(-1, self.feat_dim)
                # features_ob = features_ob.reshape(-1, self.feat_dim)
                # data = np.vstack([features, features_ob])
                features_pca = PCA(n_components=3).fit_transform(features)
                features_pca = minmax_scale(features_pca)
                return features_pca.reshape(self.height,self.width,3) #scaled_features[0:features.shape[0]], scaled_features[features.shape[0]:]
        
        def render_sdf_video(self,  start_frame, time_long):
                h = self.height
                w = self.width
                normal_images, rgb_images = [], []
                camera, TrCam0ToVelo = init_camera(seq=0, cam_id=0)
                intrinsic = camera.K
                pose_old = SE3(self.vec_es[f"{start_frame}"]).vec() #matrix() #.to(self.device) 
                r = 0.5
                v = 0.2
                time = range(0,time_long,1)
                pose = gen_pose(torch.IntTensor(time), r, v, pose_old)
                images_test = []
                for ii in tqdm(range(time_long)):
                        _, ray_directions_cam = gen_rays(h, w, intrinsic, pose[ii,:], TrCam0ToVelo.to(pose.device))
                        render_pose_c2w = SE3(pose[ii]).matrix() #.to(self.device) 
                        rays_all_num2 = ray_directions_cam.shape[0]
                        rgb_list = []
                        normal_list = []
                        depth_list = []
                        for i in range(0, rays_all_num2, BATCH):
                                i_next = i+BATCH if (i+BATCH <rays_all_num2) else rays_all_num2
                                depth_chunk, normal_chunk, rgb_chunk = self.render_core(ray_directions_cam[i:i_next,:], render_pose_c2w)
                                
                                normal_list.append(255*normal_chunk.cpu().numpy())
                                # semantic_list.append(semantic_chunk.cpu().numpy())
                                rgb_list.append(255 * rgb_chunk.cpu().numpy())
                                depth_list.append(depth_chunk[:,None].cpu().numpy())

                        normal_img = np.vstack(normal_list)
                        rgb = normal_img.copy()
                        # rgb[:,0] = normal_img[:,2]
                        # rgb[:,2] = normal_img[:,0]
                        normal_images.append(rgb.astype(np.uint8).reshape(-1, self.width, 3)) 
                        self.draw_rgb_render(normal_img, ii, save_dir='normal_novel_img')
                        if self.rgb_flag:
                                rgb_img = np.vstack(rgb_list).squeeze()
                                rgb = rgb_img.copy()
                                # rgb[:,0] = rgb_img[:,2]
                                # rgb[:,2] = rgb_img[:,0]
                                rgb_images.append(rgb.astype(np.uint8).reshape(-1, self.width, 3)) 
                                self.draw_rgb_render(rgb, ii, save_dir='rgb_novel_img')
                return normal_images, rgb_images
        
        @torch.no_grad()
        def render(self, dataloader, save_path=None):
                # self.save_path = save_path #self.cfg['path']['proj_dir']
                # print("************************Render Test Ray!************************")
                normal_images, rgb_images = [], []
                pc_list, color_list = [], []
                result = {}
                with tqdm(total=len(dataloader)) as t:
                        t.set_description('render images: ')
                        for ii, data in enumerate(dataloader):
                                if len(data)==0:
                                        continue       
                                render_pose_c2w = data['pose_cam'].squeeze() #[0]

                                depth_list = []
                                uncertaionty_list = []
                                normal_list = []
                                semantic_list, instance_list, panoptic_list = [], [], []
                                rgb_list, feat_list = [], []
                                depth2_list = []
                                
                                # #======================camera=======================
                                if 'camera' in SENCE:
                                        ray_directions_cam = data['direction_img'].squeeze()
                                        ray_origin_cam = data['origin'].squeeze()  
                                        # ray_origin, ray_directions = ray_origin_cam, ray_directions_cam
                                        rays_all_num2 = ray_directions_cam.size(0)    

                                        # self.grid.iter_n = 0
                                        for i in range(0, rays_all_num2, BATCH):
                                                # if i==210000:
                                                #         l=1
                                                i_next = i+BATCH if (i+BATCH <rays_all_num2) else rays_all_num2
                                                rays_d_cam_batch = ray_directions_cam[i:i_next].clone()
                                                res = self.render_core_test(ray_origin_cam, ray_directions_cam[i:i_next,:], render_pose_c2w, frame_id=ii)
                                                normal_list.append(255*res['normal'].cpu().numpy())
                                                semantic_list.append(res['semantic'].cpu().numpy()) if self.semantic_flag else semantic_list.append(np.zeros(ray_directions_cam[i:i_next,:].shape[0]))
                                                instance_list.append(res['instance'].cpu().numpy()) if self.instance_flag else instance_list.append(np.zeros(ray_directions_cam[i:i_next,:].shape[0]))
                                                panoptic_list.append(res['panoptic'].cpu().numpy()) if self.panoptic_flag else panoptic_list.append(np.zeros(ray_directions_cam[i:i_next,:].shape[0]))
                                                rgb_list.append(255 * res['rgb'].cpu().numpy()) if self.rgb_flag else rgb_list.append(np.zeros([ray_directions_cam[i:i_next,:].shape[0],3]))
                                                feat_list.append(res['feature'].cpu().numpy()) if self.feat_flag else feat_list.append(np.zeros([ray_directions_cam[i:i_next,:].shape[0],self.feat_dim]))
                                                depth_list.append((res['depth']*rays_d_cam_batch[:,2:]).cpu().numpy())
                                                # uncertaionty_list.append(res['uncertainty'][:,None].cpu().numpy())
                                                if DEBUG:
                                                        depth2_list.append(res['depth2'][:,None].cpu().numpy())
                                            
                                        normal_img = np.vstack(normal_list)
                                        rgb = normal_img.copy()
                                        # rgb[:,0] = normal_img[:,2]
                                        # rgb[:,2] = normal_img[:,0]
                                        normal_images.append(rgb.astype(np.uint8).reshape(-1, self.width, 3)) 
                                        self.draw_depth_render(np.vstack(depth_list), data['idx'], folder_name=f'depth_render_img') #color
                                        self.draw_depth_render(np.vstack(depth_list), data['idx'], folder_name=f'depth_render_value', mode='value') #value
                                        # self.draw_depth_render(np.vstack(uncertaionty_list), data['idx'], folder_name="uncer_ernder_img") #-250
                                        self.draw_rgb_render(normal_img, data['idx'], save_dir=f'normal_render_img')
                                        # # # self.draw_lidar_img(normal_img, data['idx'].item(), data['points'].squeeze())
                                        if self.rgb_flag:
                                                rgb_img = np.vstack(rgb_list).squeeze()
                                                rgb_images.append(rgb_img.astype(np.uint8).reshape(-1, self.width, 3)) 
                                                self.draw_rgb_render(rgb_img, data['idx'], save_dir=f'rgb_render_img')
                                                result['rgb'] = rgb_img
                                        # if self.feat_flag:
                                        #         feat_img = np.vstack(feat_list).squeeze()
                                        #         feat_rgb = self.pca_feature(feat_img) * 255
                                        #         self.draw_rgb_render(feat_rgb, data['idx'], save_dir='feature_img')
                                        if self.panoptic_flag:
                                                panoptic_img_ = np.vstack(panoptic_list).squeeze()
                                                panoptic_img = self.color_id[panoptic_img_]
                                                self.draw_semantic_render(panoptic_img, data['idx'], save_dir='panoptic_render_img')
                                                result['panoptic'] = panoptic_img_
                                        if self.semantic_flag:
                                                semantic_img = np.vstack(semantic_list).squeeze()
                                                if self.instance_flag:
                                                        thing_mask = np.isin(semantic_img, self.Thing_class)
                                                self.draw_semantic_render(semantic_img, data['idx'], save_dir='semantic_render_label')
                                                semantic_img = np.vstack([id2label[semID].color for semID in semantic_img.tolist()]) 
                                                self.draw_semantic_render(semantic_img, data['idx'], save_dir='semantic_render_img')
                                                result['semantic'] = semantic_img
                                        if self.instance_flag:
                                                # semantic_img = cv2.imread(f"/home/yx/myProject/PaopticRecon++/data/scene0087_02/semantic_image/{data['idx'].item()}.png", -1)
                                                # thing_mask = np.isin(semantic_img, self.Thing_class)
                                                # instance_img = thing_mask.reshape(-1) * np.vstack(instance_list).squeeze()
                                                instance_img = np.vstack(instance_list).squeeze()
                                                self.draw_semantic_render(instance_img, data['idx'], save_dir='instance_render_label')
                                                instance_img = self.color_id[instance_img] #.reshape(-1,self.width,3)
                                                self.draw_semantic_render(instance_img, data['idx'], save_dir='instance_render_img')
                                                result['instance'] = instance_img

                                
                                # #======================lidar=======================
                                if 'lidar' in SENCE:
                                        depth_list = []
                                        semantic_list = []
                                        # rgb_list = []
                                        if self.mode == 'test':
                                                #LIDAR
                                                render_pose_l2w = data['pose_lidar'].squeeze() 
                                                ray_origin_lidar = data['origin_lidar'].squeeze()
                                                ray_directions_lidar = data['direction_lidar'].squeeze()
                                                rays_all_num2 = ray_directions_lidar.size(0)  
                                                for i in range(0, rays_all_num2, BATCH):
                                                        i_next = i+BATCH if (i+BATCH <rays_all_num2) else rays_all_num2
                                                        res = self.render_core_test(ray_origin_lidar, ray_directions_lidar[i:i_next,:], render_pose_l2w, frame_id=ii, depth_method='median')
                                                        semantic_list.append(res['semantic'].cpu().numpy()) if self.semantic_flag else semantic_list.append(np.zeros(ray_directions_lidar[i:i_next,:].shape[0])[:,None])
                                                        # rgb_list.append(255 * res['rgb'].cpu().numpy()) if self.rgb_flag else rgb_list.append(np.zeros([ray_directions_lidar[i:i_next,:].shape[0],3]))
                                                        depth_list.append(res['depth'][:,None].cpu().numpy())
                                                scan_range = np.vstack(depth_list)
                                                scan_pc = ray_origin_lidar[None,:].expand(ray_directions_lidar.shape[0],3).cpu().numpy() \
                                                        + scan_range.squeeze()[:,None] * ray_directions_lidar.cpu().numpy()
                                                scan_pc = scan_pc*self.scale.cpu().numpy() #+self.origin.cpu().numpy()
                                                scan_semantic = np.vstack(semantic_list).squeeze()
                                                color = np.vstack([id2label[semID].color for semID in scan_semantic.tolist()])

                                                # pcd = o3d.geometry.PointCloud()
                                                # pcd.points = o3d.utility.Vector3dVector(scan_pc)
                                                # pcd.colors = o3d.utility.Vector3dVector(color/255.)
                                                # savepath = os.path.join(self.cfg['path']['proj_dir'], "lidar_render", "%010d.ply"%ii)
                                                # o3d.io.write_point_cloud(savepath, pcd)

                                                # scan_pc = ((torch.FloatTensor(scan_pc).to(self.device)@render_pose_l2w[:3, :3].T).squeeze()+render_pose_l2w[:3, 3]).cpu().numpy()
                                                # pc_list.append(scan_pc)
                                                # color_list.append(color)

                                                data = np.hstack([scan_pc, scan_semantic[:,None]])
                                                savepath = os.path.join("/mnt/northcn4/Code_yx/dataset/roma_mx/trajectory_02/data_3d_semantics_novel", "%010d.bin"%ii)
                                                data.astype(np.float32).tofile(savepath)
                                                # binfile = open(savepath, 'wb')                                                 # binfile.write(binary_data)
                                                # binfile.close()                                        
                                
                                t.update(1)
                        # pc_list = np.vstack(pc_list)
                        # color_list = np.vstack(color_list)
                        # pcd = o3d.geometry.PointCloud()
                        # pcd.points = o3d.utility.Vector3dVector(pc_list)
                        # pcd.colors = o3d.utility.Vector3dVector(color_list/255.)
                        # savepath = os.path.join("lidar_test.ply")
                        # o3d.io.write_point_cloud(savepath, pcd)
                        # yx = 1
                        
                return result
                      
        def get_feature_ob(self, idx):
               feat_ob = torch.load(os.path.join(self.feat_path, f'{idx}.pth')).reshape(47,176,self.feat_dim)
               feat_ob = torch.from_numpy(np.transpose(feat_ob.cpu().numpy(),(2,0,1)))
               feat_ob = torch.nn.functional.interpolate(feat_ob.unsqueeze(0), scale_factor=8, mode="nearest")[0].cpu().numpy()
               feat_ob = np.transpose(feat_ob, (1,2,0)).reshape(-1,self.feat_dim)
               return feat_ob
                                 
        def render_core(self, rays_o, rays_d, c2w, frame_id=None):
                rays_o, rays_d = get_rays(rays_o, rays_d, c2w)
                rays_go = get_rays_sfm(rays_o[None,:,:], self.scale, self.origin)[0]
                rays_g = Rays(origins=rays_go, dirs=rays_d)
                rgb_buffer = torch.zeros(*rays_o.shape[:-1], 3, device=rays_o.device)
                result = {}
                # semantic_buffer = torch.zeros(*rays_o.shape[:-1], 3, device=rays_o.device)
                # normal_buffer = torch.zeros(rays_g.shape[0], 3, device=self.device)
                with torch.no_grad():
                        ridx, deltas, depth_samples, samples, boundary = self.raymarch_2(rays_g)
                        # boundary = boundary.bool()
                        deltas = deltas*self.scale
                        # depth_samples = depth_samples*self.scale
                        if ridx.shape[0]>0:
                                ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
                                # hit_ray_d = rays_g.dirs.index_select(0, ridx)
                                # sdf_value = self.sdf(coords=samples, lod_idx=self.grid.num_lods - 1)
                                # sdf_value, semantics = self.get_output(coords=samples, lod_idx=self.grid.num_lods - 1)
                                hit_ray_d = rays_g.dirs.index_select(0, ridx)
                                # sdf_value, rgbs = self.get_output(coords=samples, view_dirs=hit_ray_d, lod_idx=self.grid.num_lods - 1, return_rgb=self.rgb_flag)
                                channels = ['sdf']
                                if self.feat_flag:
                                        channels.append('feature')
                                if self.rgb_flag:
                                        channels.append('rgb')
                                if self.semantic_flag:
                                        channels.append('semantic')
                                if self.panoptic_flag:
                                        channels.append('panoptic')
                                embeddings = self.embedding_a((torch.ones([1],dtype=int)*frame_id).to(self.device)).expand(samples.shape[0], -1) if self.appearance_embedding else None
                                grads = autodiff_gradient(samples, self.sdf, self.origin, self.scale) /self.scale
                                # grads = finitediff_gradient(samples, self.sdf, eps=0.005/self.scale)/self.scale
                                out = self.get_output(samples, channels=channels, view_dirs=hit_ray_d, embedding=embeddings, normals=grads.reshape(-1,3))
                                # sdf_value, feats, semantics = out['sdf'], out['feature'], out['semantic']
                                weights = self.get_weight_2(out['sdf'], deltas, self.s, cos_anneal_ratio, boundary)
                                # weights = self.get_weight_2(rays_g.dirs[ridx.long()], samples.shape, out['sdf'], grads, deltas, self.s, cos_anneal_ratio, boundary)
                                semantic = torch.zeros(rays_g.shape[0], 1, device=self.device).long()
                                if self.semantic_flag:
                                        ray_semantic = spc_render.sum_reduce(out['semantic'].squeeze() * weights.reshape(-1,1), boundary)
                                        ray_semantic = torch.argmax(torch.nn.functional.softmax(ray_semantic, dim=-1), -1)
                                        semantic[ridx_hit.long(), :] = ray_semantic[:,None]
                                        result['semantic'] = semantic
                                rgb = torch.zeros(rays_g.shape[0], 3, device=self.device)
                                feat = torch.zeros(rays_g.shape[0], self.feat_dim, device=self.device)
                                if self.normal_type=='render':
                                        ray_normal = spc_render.sum_reduce(grads.squeeze() * weights.reshape(-1,1), boundary)
                                if self.rgb_flag:
                                        ray_rgb = spc_render.sum_reduce(out['rgb'].squeeze() * weights.reshape(-1,1), boundary)
                                        rgb[ridx_hit.long(), :] = ray_rgb
                                if self.feat_flag:
                                        ray_feat = spc_render.sum_reduce(out['feature'].squeeze() * weights.reshape(-1,1), boundary)
                                        feat[ridx_hit.long(), :] = ray_feat
                                        result['feature'] = feat
                                ray_depth = spc_render.sum_reduce(depth_samples.reshape(-1,1) * weights.reshape(-1,1), boundary)
                                depth = torch.zeros(rays_g.shape[0], 1, device=self.device)
                                depth[ridx_hit.long(), :] = ray_depth
                                result['depth'] = depth
                                result['rgb'] = rgb
                                surface_points = rays_go + rays_d * depth
                                # _, ray_semantic = self.get_output(surface_points[ridx_hit.long(), :])
                                # ray_semantic = torch.argmax(torch.nn.functional.softmax(ray_semantic.squeeze(), dim=-1), -1)
                                # semantic[ridx_hit.long(), :] = ray_semantic[:,None]

                                # try:
                                #         grad = autodiff_gradient(surface_points[ridx_hit.long(), :], self.sdf, self.origin, self.scale) /self.scale
                                # except:
                                #         # print("sphere_tracing_grad_error")
                                #         grad = torch.zeros_like(surface_points[ridx_hit.long(), :])
                                if self.normal_type=='trace':       
                                        if ridx_hit.shape[0]>1000:
                                                grad = autodiff_gradient(surface_points[ridx_hit.long(), :], self.sdf, self.origin, self.scale) /self.scale
                                        #        grad = finitediff_gradient(surface_points[ridx_hit.long(), :], self.sdf, eps=0.01/self.scale)/self.scale
                                        else:
                                                grad = torch.zeros_like(surface_points[ridx_hit.long(), :])
                                elif self.normal_type=='render':
                                        grad = ray_normal
                                normal_buffer = F.normalize(grad, p=2, dim=-1, eps=1e-5)
                                rgb_buffer[ridx_hit.long(), :3] = (normal_buffer + 1.0) / 2.0
                                result['normal'] = rgb_buffer
                        else:
                                depth = torch.zeros(rays_g.shape[0], 1, device=self.device)
                                semantic = torch.zeros(rays_g.shape[0], 1, device=self.device).long()
                                rgb = torch.zeros(rays_g.shape[0], 3, device=self.device)
                                feat = torch.zeros(rays_g.shape[0], self.feat_dim, device=self.device)
                                rgb_buffer = torch.zeros(*rays_o.shape[:-1], 3, device=rays_o.device)
                                result = {'depth': depth, 'semantic': semantic, 'rgb': rgb, 'feature': feat, 'normal': rgb_buffer}
                                # normal_buffer = torch.zeros(rays_g.shape[0], 3, device=self.device)
                                
                return result

      
        def render_core_test(self, rays_o, rays_d, c2w, frame_id=None, depth_method='expected'):
                rays_cam_o, rays_cam_d = rays_o, rays_d
                rays_o, rays_d = get_rays(rays_o, rays_d, c2w)
                rays_go = get_rays_sfm(rays_o[None,:,:], self.scale, self.origin)[0]
                rays_g = Rays(origins=rays_go, dirs=rays_d)
                rgb_buffer = torch.zeros(*rays_o.shape[:-1], 3, device=rays_o.device)
                result = {}

                with torch.no_grad():
                        # ridx, samples, depth_samples, deltas, boundary, valid_mask, depths_outside = self.raymarch_ray(rays_g, rays_cam_o, rays_cam_d) 
                        samples, depth_samples, deltas, valid_mask, depths_outside = self.raymarch_ray(rays_g, rays_cam_o, rays_cam_d) 
                        boundary = None
                        
                        # background
                        if depths_outside is not None:
                                ret_outside = self.render_core_outside(rays_g[~valid_mask].origins, rays_g[~valid_mask].dirs, depths_outside)

                                color_outside = ret_outside['color']

                        deltas = deltas*self.scale
                        deltas = deltas.clip(0, self.trunc_d)
                        
                        channels = ['sdf']
                        if self.feat_flag:
                                channels.append('feature')
                        if self.rgb_flag:
                                channels.append('rgb')
                        if self.semantic_flag:
                                channels.append('semantic')
                        if self.instance_flag:
                                channels.append('instance')
                        if self.panoptic_flag:
                                channels.append('panoptic')
                        embeddings = None #self.embedding_a((torch.ones([1],dtype=int)*frame_id).to(self.device)).expand(samples.shape[0], -1) if self.appearance_embedding else None
                        grads = autodiff_gradient(samples, self.sdf, self.origin, self.scale) /self.scale
                        # grads = finitediff_gradient(samples, self.sdf, eps=0.005/self.scale)/self.scale
                        # out = self.get_output(samples, channels=channels, view_dirs=hit_ray_d, embedding=embeddings, normals=grads.reshape(-1,3))
                        out = self.get_output(samples, channels=channels, embedding=embeddings, normals=grads.reshape(-1,3))
                        # sdf_value, feats, semantics = out['sdf'], out['feature'], out['semantic']
                        if self.weight_mode == 'mid':
                                weights, alpha = self.get_weight_2(out['sdf'], deltas, self.s, cos_anneal_ratio, boundary)
                                # uncertainty = alpha*(1-alpha)
                                # ray_uncertainty = spc_render.sum_reduce(torch.ones_like(depth_samples.reshape(-1,1)) * uncertainty.reshape(-1,1), boundary)
                                # samplr_num = torch.zeros_like(ray_uncertainty)
                                # samplr_num[:-1,:] = torch.nonzero(boundary)[1:] - torch.nonzero(boundary)[:-1]
                                # samplr_num[-1,:] = boundary.shape[0] - torch.nonzero(boundary)[-1]
                                # ray_uncertainty = ray_uncertainty / samplr_num
                                # uncer = torch.zeros(rays_g.shape[0], 1, device=self.device)
                                # buffer = torch.zeros_like(uncer[valid_mask],device=uncer.device)
                                # buffer[ridx_hit.long(),:] = ray_uncertainty
                                # uncer[valid_mask] = buffer
                                # result['uncertainty'] = uncer
                                if DEBUG:
                                        weights2 = self.get_weight_train(rays_g.dirs[valid_mask][ridx.long()], samples.shape, out['sdf'], grads, deltas, self.s, cos_anneal_ratio, boundary)
                        elif self.weight_mode == 'grad':
                                weights = self.get_weight_train(hit_ray_d, samples.shape, out['sdf'], grads, deltas, self.s, cos_anneal_ratio, boundary)
                        semantic = torch.zeros(rays_g.shape[0], 1, device=self.device).long()
                        if self.semantic_flag:
                                ray_semantic = torch.sum(weights[...,None].detach() * torch.nn.functional.softmax(out['semantic'].reshape(weights.shape[0], weights.shape[1], -1), dim=-1), dim=-2)
                                ray_semantic = torch.argmax(ray_semantic, -1)
                                semantic[valid_mask] = ray_semantic[:,None]
                                result['semantic'] = semantic
                        #         # # 2d softmax
                        #         # ray_semantic = spc_render.sum_reduce(out['semantic'].squeeze() * weights.reshape(-1,1), boundary)
                        #         # ray_semantic = torch.argmax(torch.nn.functional.softmax(ray_semantic, dim=-1), -1)
                        #         # 3d softmax
                        #         ray_semantic = spc_render.sum_reduce(torch.nn.functional.softmax(out['semantic'].squeeze(), dim=-1) * weights.reshape(-1,1), boundary)
                        #         ray_semantic = torch.argmax(ray_semantic, -1)

                        #         buffer = torch.zeros_like(semantic[valid_mask],device=semantic.device)
                        #         buffer[ridx_hit.long(),:] = ray_semantic[:,None]
                        #         semantic[valid_mask] = buffer
                        #         result['semantic'] = semantic
                        instance = torch.zeros(rays_g.shape[0], 1, device=self.device).long()
                        # instance_masks = torch.zeros(rays_g.shape[0], self.instance_num, device=self.device)
                        if self.instance_flag:
                                ray_instance = torch.sum(weights[...,None].detach() * torch.nn.functional.softmax(out['instance'].reshape(weights.shape[0], weights.shape[1], -1), dim=-1), dim=-2)
                                ray_instance = torch.argmax(ray_instance, -1)
                                instance[valid_mask] = ray_instance[:,None] + 1
                                thing_mask = torch.isin(semantic.squeeze(-1), torch.tensor(self.cfg['Thing_class'], device=semantic.device))
                                instance[~thing_mask] = 0
                                result['instance'] = instance
                        #         # 3d softmax
                        #         ray_instance = spc_render.sum_reduce(out['instance'].squeeze() * weights.reshape(-1,1), boundary)
                                
                        #         ray_instance = torch.argmax(ray_instance, -1)
                        #         buffer = torch.zeros_like(instance[valid_mask],device=instance.device)
                        #         buffer[ridx_hit.long(),:] = ray_instance[:,None]
                        #         instance[valid_mask] = buffer
                        #         result['instance'] = instance

                                # semantic filter
                                stuff_label = self.cfg['Stuff_class']
                                if h_SI:
                                        thing_label = torch.argmax(self.thing_semantic_heat_map,-1)
                                        thing_map = [0]+ self.cfg['Thing_class']
                                        ins2sem_mapping = {  #**{pan_id+1: sem_id for pan_id, sem_id in enumerate(stuff_label)}, \
                                                        **{pan_id+1: thing_map[sem_id] for pan_id, sem_id in enumerate(thing_label)} }
                                        for key, val in ins2sem_mapping.items():
                                                mask = torch.logical_and(thing_mask.reshape(-1,1), instance==key)
                                                semantic[mask] = val
                                        result['semantic'] = semantic

                        panoptic = torch.zeros(rays_g.shape[0], 1, device=self.device).long()
                        if self.panoptic_flag:
                                # ray_panoptic = spc_render.sum_reduce(torch.nn.functional.softmax(out['panoptic'].squeeze(), dim=-1) * weights.reshape(-1,1), boundary)
                                ray_panoptic = torch.sum(weights[...,None] * torch.nn.functional.softmax(out['panoptic'].reshape(weights.shape[0], weights.shape[1], -1), dim=-1), dim=-2)
                                ray_panoptic = torch.argmax(ray_panoptic, -1)

                                # buffer = torch.zeros_like(panoptic[valid_mask],device=panoptic.device)
                                # buffer[ridx_hit.long(),:] = ray_panoptic[:,None]
                                # panoptic[valid_mask] = buffer
                                panoptic[valid_mask] = ray_panoptic[:,None]
                                result['panoptic'] = panoptic

                                # ray_semantic = spc_render.sum_reduce(torch.nn.functional.softmax(out['semantic'].squeeze(), dim=-1) * weights.reshape(-1,1), boundary)
                                ray_semantic = torch.sum(weights[...,None].detach() * torch.nn.functional.softmax(out['semantic'].reshape(weights.shape[0], weights.shape[1], -1), dim=-1), dim=-2)
                                ray_semantic = torch.argmax(ray_semantic, -1)
                                if not h_SI:
                                        semantic = torch.zeros_like(panoptic)
                                        semantic[valid_mask] = ray_semantic[:,None]
                                        result['semantic'] = semantic

                                # semantic
                                stuff_label = self.cfg['Stuff_class']
                                if h_SI:
                                        thing_label = torch.argmax(self.thing_semantic_heat_map,-1)
                                        thing_map = [0]+ self.cfg['Thing_class']
                                        pan2sem_mapping = { **{pan_id+1: sem_id for pan_id, sem_id in enumerate(stuff_label)}, \
                                                        **{pan_id+1+len(stuff_label): thing_map[sem_id] for pan_id, sem_id in enumerate(thing_label)} }
                                        semantic = torch.zeros_like(panoptic)
                                        for key, val in pan2sem_mapping.items():
                                                semantic[panoptic==key] = val
                                        result['semantic'] = semantic
                                # instance
                                pan2ins_mapping = {pan_id+1: 0 for pan_id in range(len(stuff_label))}
                                instance = panoptic.clone()
                                for key, val in pan2ins_mapping.items():
                                        instance[panoptic==key] = val
                                result['instance'] = instance

                        rgb = torch.zeros(rays_g.shape[0], 3, device=self.device)
                        feat = torch.zeros(rays_g.shape[0], self.feat_dim, device=self.device)
                        if self.normal_type=='render':
                                ray_normal = spc_render.sum_reduce(grads.squeeze() * weights.reshape(-1,1), boundary)
                        if self.rgb_flag:
                                ray_rgb = spc_render.sum_reduce(out['rgb'].squeeze() * weights.reshape(-1,1), boundary)
                                buffer = torch.zeros_like(rgb[valid_mask],device=rgb.device)
                                buffer[ridx_hit.long(),:] = ray_rgb
                                rgb[valid_mask] = buffer
                                # outside
                                if depths_outside is not None:
                                        rgb[~valid_mask] = color_outside
                        if self.feat_flag:
                                ray_feat = spc_render.sum_reduce(out['feature'].squeeze() * weights.reshape(-1,1), boundary)
                                buffer = torch.zeros_like(feat[valid_mask],device=feat.device)
                                buffer[ridx_hit.long(),:] = ray_feat
                                feat[valid_mask] = buffer
                                result['feature'] = feat
                        if depth_method == 'expected':
                                # ray_depth = spc_render.sum_reduce(depth_samples.reshape(-1,1) * weights.reshape(-1,1), boundary)
                                ray_depth = torch.sum(weights[...,None] * depth_samples.reshape(weights.shape[0],weights.shape[1],1), dim=-2)
                        elif depth_method == 'median': # too slow
                                cumulative_weights = spc_render.cumsum(weights.reshape(-1,1), boundary)
                                true_index = torch.nonzero(boundary).squeeze().tolist()
                                median_depth = []
                                for ray_idx in range(boundary.sum()):
                                        end_idx = true_index[ray_idx+1] if ray_idx+1<len(true_index) else cumulative_weights.shape[0]
                                        weight_ray = cumulative_weights[true_index[ray_idx]:end_idx].reshape(-1)[None,:]
                                        median_idx = torch.searchsorted(weight_ray, torch.FloatTensor([0.5])[:,None].to(weight_ray.device), side="left")
                                        median_idx = torch.clamp(median_idx, 0, weight_ray.shape[-1]-1)
                                        depth = depth_samples[true_index[ray_idx]:end_idx][median_idx]
                                        median_depth.append(depth)
                                ray_depth = torch.stack(median_depth).squeeze()[:,None]
                        depth = torch.zeros(rays_g.shape[0], 1, device=self.device)
                        # buffer = torch.zeros_like(depth[valid_mask],device=depth.device)
                        # buffer[ridx_hit.long(),:] = ray_depth
                        if DEBUG:
                                weight_sum = spc_render.sum_reduce(torch.ones_like(depth_samples) * weights.reshape(-1,1), boundary)

                                ray_depth2 = spc_render.sum_reduce(depth_samples.reshape(-1,1) * weights2.reshape(-1,1), boundary)
                                weight2_sum = spc_render.sum_reduce(torch.ones_like(depth_samples) * weights2.reshape(-1,1), boundary)
                                depth2 = torch.zeros(rays_g.shape[0], 1, device=self.device)
                                buffer2 = torch.zeros_like(depth[valid_mask],device=depth.device) 
                                buffer2[ridx_hit.long(),:] = ray_depth2
                                depth2[valid_mask] = buffer2
                                result['depth2'] = depth2
                        # depth[valid_mask] = buffer
                        depth[valid_mask] = ray_depth
                        result['depth'] = depth
                        result['rgb'] = rgb
                        surface_points = rays_go + rays_d * depth
                        # _, ray_semantic = self.get_output(surface_points[ridx_hit.long(), :])
                        # ray_semantic = torch.argmax(torch.nn.functional.softmax(ray_semantic.squeeze(), dim=-1), -1)
                        # semantic[ridx_hit.long(), :] = ray_semantic[:,None]

                        # try:
                        #         grad = autodiff_gradient(surface_points[ridx_hit.long(), :], self.sdf, self.origin, self.scale) /self.scale
                        # except:
                        #         # print("sphere_tracing_grad_error")
                        #         grad = torch.zeros_like(surface_points[ridx_hit.long(), :])
                        if self.normal_type=='trace':       
                                # if ridx_hit.shape[0]>1000:
                                #         grad = autodiff_gradient(surface_points[valid_mask][ridx_hit.long(), :], self.sdf, self.origin, self.scale) /self.scale
                                # #        grad = finitediff_gradient(surface_points[ridx_hit.long(), :], self.sdf, eps=0.01/self.scale)/self.scale
                                # else:
                                #         grad = torch.zeros_like(surface_points[valid_mask][ridx_hit.long(), :])
                                grad = autodiff_gradient(surface_points[valid_mask], self.sdf, self.origin, self.scale) /self.scale
                        elif self.normal_type=='render':
                                grad = ray_normal

                        grad_cam = (c2w[:3,:3].T @ grad.T).T
                        
                        normal = torch.zeros(rays_g.shape[0], 3, device=self.device)
                        normal_buffer = F.normalize(grad_cam, p=2, dim=-1, eps=1e-5)
                        normal[valid_mask] = (normal_buffer + 1.0) / 2.0
                        result['normal'] = normal
                             
                        # else:
                        #         depth = torch.zeros(rays_g.shape[0], 1, device=self.device)
                        #         semantic = instance = panoptic = torch.zeros(rays_g.shape[0], 1, device=self.device).long()
                        #         rgb = torch.zeros(rays_g.shape[0], 3, device=self.device)
                        #         feat = torch.zeros(rays_g.shape[0], self.feat_dim, device=self.device)
                        #         rgb_buffer = torch.zeros(*rays_o.shape[:-1], 3, device=rays_o.device)
                        #         result = {'depth': depth, 'semantic': semantic, 'instance': instance, 'panoptic': panoptic, 'rgb': rgb, 'feature': feat, 'normal': rgb_buffer}
                        #         # normal_buffer = torch.zeros(rays_g.shape[0], 3, device=self.device)
                                
                return result
        
        def sdf(self, coords, pidx=None, lod_idx=None):
            shape = coords.shape
            
            if shape[0] == 0:
                return dict(sdf=torch.zeros_like(coords)[...,0:1])
            
            if len(shape) == 2:
                coords = coords[:, None]
            num_samples = coords.shape[1]
            coords = (coords+1.0)/2
            # TODO(ttakikawa): this should return [batch, ns, f] but it returns [batch, f]
            feats = self.grid['sdf'](coords.reshape(-1, 3))
                #     # mask = feats.sum(-1)==0
                #     sdfs = self.decoder['sdf'].forward(feats)[:,0:1]
                
                #     sdfs = sdfs.reshape(-1,num_samples,1)
                #     if len(shape) == 2:
                #         sdfs = sdfs[:,0]
                #     return sdfs
            if self.decoder['sdf'].__class__==SemanticSDFDecoder:
                sdfs = self.decoder['sdf'].forward_sdf(feats) 
            else:
                sdfs = self.decoder['sdf'].forward(feats) 
            if type(sdfs)==tuple:
                sdf = sdfs[0].reshape(-1,num_samples,1) 
                sem = sdfs[1]
                if len(shape) == 2:
                    sdf = sdf[:,0]
                return sdf, sem
            else:
                sdfs = sdfs.reshape(-1,num_samples,1)
        
                if torch.isnan(sdfs).sum()>0:
                    yx = 1

                if len(shape) == 2:
                    sdfs = sdfs[:,0]
                return sdfs

        
        def get_output(self, coords, channels=['sdf'], view_dirs=None, embedding=None, normals=None):
            shape = coords.shape
            out = {}
            if shape[0] == 0:
                return dict(sdf=torch.zeros_like(coords)[...,0:1])
            
            if len(shape) == 2:
                coords = coords[:, None]
            if view_dirs is not None and len(view_dirs.shape) == 2:
                view_dirs = view_dirs[:, None].expand(view_dirs.shape[0], coords.shape[1], view_dirs.shape[1])
            num_samples = coords.shape[1]
            coords = (coords+1.0)/2
            sdf_feats = self.grid['sdf'](coords.reshape(-1, 3)) #.reshape(-1, self.grid['sdf'].encoding_config['n_features_per_level'], self.grid['sdf'].encoding_config['n_levels']).sum(-1)
            net_output = self.decoder['sdf'].forward(sdf_feats)
            if 'rgb' in channels:
                if self.cfg['model_type'] == 'shared':
                        radiance_feats = net_output[:,1:]
                elif self.cfg['model_type'] == 'separate':         
                        radiance_feats = self.grid['rgb'](coords.reshape(-1, 3))
                if 'semantic' in channels and self.pretrain:
                        rgbs, rgb_last_layer = self.decoder['rgb'].forward(radiance_feats=radiance_feats, view_dirs=view_dirs.reshape(-1,3),appearance_embedding=embedding, grads=normals, return_last=True)
                else:
                        rgbs = self.decoder['rgb'](radiance_feats=radiance_feats, view_dirs=view_dirs.reshape(-1,3),appearance_embedding=embedding, grads=normals)
                # rgbs = self.decoder['rgb'](radiance_feats=radiance_feats, view_dirs=view_dirs.reshape(-1,3), appearance_embedding=embedding, grads=normals)
                # rgbs = self.decoder['rgb'](radiance_feats)
                rgbs = rgbs.reshape(-1,num_samples,3)
                rgbs = torch.sigmoid(rgbs)
                out['rgb'] = rgbs

                # # outside

                # out['rgb_outside'] = 
                
            if 'sdf' in channels:
                # # sdf_feats = self.grid['sdf'](coords.reshape(-1, 3))
                # # net_output = self.decoder['sdf'].forward(sdf_feats)
                # sdfs = net_output[:,0:1]
                # sdfs = sdfs.reshape(-1,num_samples,1)
                # if len(shape) == 2:
                #     sdfs = sdfs[:,0]
                # out['sdf'] = sdfs
                sdf_feats = self.grid['sdf'](coords.reshape(-1, 3))
                # if self.decoder['sdf'].__class__ == SemanticSDFDecoder:
                #         sdfs,semantics = self.decoder['sdf'].forward(sdf_feats)
                #         semantics = semantics.reshape(-1,num_samples,self.class_num)
                #         out['semantic'] = semantics
                # else:
                #         if 'semantic' in channels and self.pretrain:                 #                 sdfs, sdf_last_layer = self.decoder['sdf'].forward(sdf_feats,return_last=True)
                #                 # input = torch.hstack((sdf_last_layer, rgb_last_layer))
                #                 # # input = sdf_last_layer
                #                 # input = torch.hstack((input, coords.reshape(-1, 3)))
                #                 semantic_feat = self.grid['semantic'](coords.reshape(-1, 3))
                #                 # input = torch.hstack((semantic_feat, sdf_last_layer))
                #                 # # input = torch.hstack((input, rgb_last_layer))
                #                 # input = torch.hstack((input, coords.reshape(-1, 3)))
                #                 semantics = self.decoder['semantic'].forward(semantic_feat)
                #                 out['semantic'] = semantics
                #         else:
                #                 sdfs = self.decoder['sdf'].forward(sdf_feats)
                sdfs = self.decoder['sdf'].forward(sdf_feats)
                sdfs = sdfs.reshape(-1,num_samples,1)
                if torch.isnan(sdfs).sum()>0:
                    yx = 1
                if len(shape) == 2:
                    sdfs = sdfs[:,0]
                out['sdf'] = sdfs
            # TODO(ttakikawa): this should return [batch, ns, f] but it returns [batch, f]
            if 'feature' in channels:
                radiance_feats = self.grid['feature'](coords.reshape(-1, 3))
                feats = self.decoder['feature'](radiance_feats)
                if 'semantic' in channels:
                    semantics = self.decoder['semantic'](feats)
                    semantics = semantics.reshape(-1,num_samples,self.class_num)
                    out['semantic'] = semantics
                feats = feats.reshape(-1,num_samples,self.feat_dim)
                out['feature'] = feats
            if 'semantic' in channels:
                semantic_feat = self.grid['semantic'](coords.reshape(-1, 3))
                semantics = self.decoder['semantic'].forward(semantic_feat)
                out['semantic'] = semantics
            if 'instance' in channels:
                # ## octree_pc
                # self.instance_corners = [(coords*2-1).reshape(-1, 3)]
                # self.instance_feat = [self.instance_interpolate((coords*2-1).reshape(-1, 3)[:,None], self.grid['instance'].features[0], self.grid['instance'].active_lods[-1], self.grid['instance'])]
                # instance_heat_map, self.thing_semantic_heat_map, _, _ = self.instance_model(self.instance_feat, self.instance_corners, self.scale)
                ## hash_pc
                self.instance_corners = [(coords*2-1).reshape(-1, 3)]
                self.ins_feat = [self.grid['instance'](coords.reshape(-1, 3))]
                instance_heat_map, self.thing_semantic_heat_map, self.query_position, self.query_loss = self.instance_model(
                                                                                                    self.ins_feat, 
                                                                                                    self.instance_corners,
                                                                                                    self.scale
                                                                                                ) 
                out['instance'] = instance_heat_map

            if 'panoptic' in channels:
                stuff_mask = torch.ones(out['semantic'].shape[-1]).bool().to(self.device)
                stuff_mask[self.Thing_class] = False
                sem_g_norm = F.softmax(out['semantic'])
                panoptic_instance_prob = (1 - sem_g_norm[:,stuff_mask].sum(-1))[:,None] * (out['instance']/out['instance'].sum(-1)[:,None]) # * out['instance']
                out['panoptic'] = torch.hstack((sem_g_norm[:,stuff_mask], panoptic_instance_prob))
                # out['panoptic'] = torch.hstack((out['semantic'][:,stuff_mask], out['instance']))
            return out

        def instance_interpolate(self, coords, feats, lod, instance_model):
                query_results = instance_model.blas.query(coords[:,0], lod, with_parents=False)
                pidx = query_results.pidx
                fs = spc_ops.unbatched_interpolate_trilinear(coords, pidx.int(), instance_model.blas.points, instance_model.trinkets.int(),
                                                                feats.half(), lod).float()
                return fs.reshape(coords.shape[0], feats.shape[-1])
                
        def raymarch_2(self, rays):
                # result = self.grid.raymarch(rays, raymarch_type='voxel', num_samples=2, level=self.grid.active_lods[-2])
                result = self.octree.raymarch(rays, raymarch_type='voxel', num_samples=5, level=self.octree.active_lods[-1])
                ridx, deltas, depth_samples, samples, boundary = \
                result.ridx, result.deltas, result.depth_samples, result.samples, result.boundary
                return ridx, deltas, depth_samples, samples, boundary
        
        def render_core_outside(self, rays_o, rays_d, z_vals, background_rgb=None):
                """
                Render background
                """
                batch_size, n_samples = z_vals.shape

                # Section length
                dists = z_vals[..., 1:] - z_vals[..., :-1]
                # dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
                dists = torch.cat([dists, dists[:,:1]], -1)
                mid_z_vals = z_vals + dists * 0.5

                # Section midpoints
                pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

                dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
                pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

                dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

                pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
                dirs = dirs.reshape(-1, 3)

                density, sampled_color = self.nerf_outside(pts, dirs)
                sampled_color = torch.sigmoid(sampled_color)
                alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
                alpha = alpha.reshape(batch_size, n_samples)
                weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=self.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
                sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
                color = (weights[:, :, None] * sampled_color).sum(dim=1)
                if background_rgb is not None:
                        color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

                return {
                'color': color,
                'sampled_color': sampled_color,
                'alpha': alpha,
                'weights': weights,
                }
                
        @ torch.no_grad()
        def grid_sample(self, rays, sample_lod, num_samples_list):
                grid_z_val = []
                for level in sample_lod:
                        result = self.octree.raytrace(rays, level, with_exit=True) #level
                        ridx, pidx, depth_in_out = result.ridx, result.pidx, result.depth

                        rays_pid = torch.ones_like(rays.origins[:, :1]) * -1
                        rays_near = torch.zeros_like(rays.origins[:, :1])
                        rays_far = torch.zeros_like(rays.origins[:, :1])
                        near_index, near_count = torch.unique_consecutive(ridx, return_counts=True)
                        near_index = near_index.to(torch.int64)
                        near_inv = torch.roll(torch.cumsum(near_count, dim=0), shifts=1)
                        near_inv[0] = 0
                        far_index, far_count = torch.unique_consecutive(torch.flip(ridx, [0]), return_counts=True)
                        far_index = far_index.to(torch.int64)
                        far_inv = torch.roll(torch.cumsum(far_count, dim=0), shifts=1)
                        far_inv[0] = 0
                        far_inv = ((ridx.size()[0] - 1) - far_inv).long()
                        rays_pid[near_index] = pidx[near_inv].reshape(-1, 1).float()
                        rays_near[near_index] = depth_in_out[near_inv, :1]+0.1/self.scale
                        # rays_near = 2.75*torch.ones_like(rays.origins[:, :1])/self.scale
                        rays_far[far_index] =depth_in_out[far_inv, 1:]

                        num_samples = num_samples_list[sample_lod.index(level)]
                        z_val_new = torch.linspace(0, 1.0, num_samples, device=rays.origins.device)[None].expand(rays.origins.shape[0], num_samples)

                        # Normalize between near and far plane
                        z_val_new = (rays_far - rays_near)*z_val_new + rays_near
                        grid_z_val.append(z_val_new+(0.1*torch.rand(rays.origins.shape[0], num_samples, device=rays.origins.device)-0.05)/self.scale / num_samples)
                        if sample_lod.index(level)==0:
                        #     z_vals = z_val_new
                                rays_near_est = rays_near
                grid_z_val = torch.cat(grid_z_val, dim=-1)

                return grid_z_val, rays_near_est
        
        def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s, step):
                device = sdf.device
                batch_size, n_samples = z_vals.shape
                prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
                prev_z_vals, next_z_vals = z_vals[:, :-1].clone(), z_vals[:, 1:].clone()
                prev_z_vals *= self.scale
                next_z_vals *= self.scale
                mid_sdf = (prev_sdf + next_sdf) * 0.5
                cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
                prev_cos_val = torch.cat(
                [torch.zeros([batch_size, 1], device=device), cos_val[:, :-1]], dim=-1
                )
                cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
                cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
                cos_val = cos_val.clip(-1e3, 0.0) #* inside_sphere

                dist = next_z_vals - prev_z_vals
                prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
                next_esti_sdf = mid_sdf + cos_val * dist * 0.5
                prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
                next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
                alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                # transient alpha
                # alpha = alpha_s + alpha_t
                weights = (
                alpha
                * torch.cumprod(
                        torch.cat(
                        [torch.ones([batch_size, 1], device=device), 1.0 - alpha + 1e-7], -1
                        ),
                        -1,
                )[:, :-1]
                )
                z_samples = self.sample_pdf(z_vals, weights, n_importance, det=True).detach()
                return z_samples

        def sample_pdf(self, bins, weights, N_importance, det=False):
                # This implementation is from NeRF
                # Get pdf
                eps = 1e-5
                N_rays, N_samples_ = weights.shape
                weights = weights + eps  # prevent nans
                pdf = weights / torch.sum(weights, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
                # Take uniform samples
                if det:
                        u = torch.linspace(0.1, 0.9, steps=N_importance, device=weights.device)
                        # u = torch.linspace(0.0, 1.0, steps=N_importance, device=weights.device)
                        u = u.expand(N_rays, N_importance)
                else:
                        u = torch.rand(N_rays, N_importance, device=weights.device)

                # Invert CDF
                u = u.contiguous()
                inds = torch.searchsorted(cdf, u, side='right')
                below = torch.clamp_min(inds-1, 0)
                above = torch.clamp_max(inds, N_samples_)
                inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
                cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
                bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

                denom = cdf_g[...,1]-cdf_g[...,0]
                denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                                # anyway, therefore any value for it is fine (set to 1 here)
                samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])

                return samples

        def raymarch_ray2(self, rays):
                depth_cam = self.get_img_depth(rays).squeeze()
                valid_mask = depth_cam>0
                valid_depth = depth_cam[valid_mask]
                valid_rays = rays[valid_mask]
                if valid_mask.sum()>0:
                        # surface guided sampling
                        with torch.no_grad():
                                num_samples_lidar = [12,0,0]#[10] # 4 [4,4,6] #[2 2 8] [2,2,2,3] #[4,8,6,4,2]
                                num_samples_sur = [7,15] #5
                                # num_samples_sur = [22,0]
                                n_important = 4 #20 #8
                                up_sample_steps = 2 #4
                                if (valid_mask).sum() != 0:
                                        rays_far = valid_depth + 0.1 #0.2
                                        rays_near = valid_depth - 0.1 #0.2
                                        rays_near_est = rays_near[:,None]/self.scale
                                        z_vals = torch.linspace(0, 1.0, num_samples_sur[0], device=valid_rays.origins.device)[None] + \
                                                (torch.zeros(valid_rays.origins.shape[0], num_samples_sur[0], device=valid_rays.origins.device) / num_samples_sur[0])
                                        z_vals *= (rays_far - rays_near)[:,None]/self.scale
                                        z_vals += rays_near[:,None]/self.scale

                                        #new
                                        n_samples = num_samples_sur[0]
                                        pts = (valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * z_vals[..., :, None])  # N_rays, N_samples, 3
                                        sdf = self.sdf(pts).squeeze()
                                        for i in range(int(num_samples_sur[1]/5)):
                                                new_z_vals = self.up_sample(valid_rays.origins, valid_rays.dirs, z_vals, sdf, 5, 64 * 2 ** i, i)
                                                z_vals = torch.cat([z_vals, new_z_vals+0.05*torch.rand_like(new_z_vals)/self.scale], dim=-1)
                                                z_vals, index = torch.sort(z_vals, dim=-1)
                                                n_samples = n_samples + 5 

                                                pts = valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * new_z_vals[..., :, None]
                                                new_sdf = self.sdf(pts).squeeze()
                                                sdf = torch.cat([sdf, new_sdf], dim=-1)
                                                xx = (
                                                        torch.arange(z_vals.shape[0])[:, None]
                                                        .expand(-1, n_samples)
                                                        .reshape(-1)
                                                )
                                                index = index.reshape(-1)
                                                sdf = sdf[(xx, index)].reshape(-1, n_samples)

                        # lidar_n_samples = num_samples_sur.sum()
                        lidar_sample_lod = self.octree.active_lods[:1]

                        z_val_new, rays_near_est = self.grid_sample(valid_rays, lidar_sample_lod, num_samples_lidar)

                        z_vals = torch.cat([z_vals, z_val_new], dim=-1)
                        z_vals, index = torch.sort(z_vals, dim=-1)
                        n_samples = n_samples + sum(num_samples_lidar)

                        pts = (valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * z_vals[..., :, None])  # N_rays, N_samples, 3
                        sdf = self.sdf(pts).squeeze()
                        for i in range(up_sample_steps):
                                new_z_vals = self.up_sample(valid_rays.origins, valid_rays.dirs, z_vals, sdf, n_important // up_sample_steps,
                                64 * 2 ** i, i)
                                z_vals = torch.cat([z_vals, new_z_vals+0.05*torch.rand_like(new_z_vals)/self.scale], dim=-1)
                                z_vals, index = torch.sort(z_vals, dim=-1)
                                n_samples = n_samples + n_important // up_sample_steps
                                pts = valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * new_z_vals[..., :, None]
                                new_sdf = self.sdf(pts).squeeze()
                                sdf = torch.cat([sdf, new_sdf], dim=-1)
                                xx = (
                                torch.arange(z_vals.shape[0])[:, None]
                                .expand(-1, n_samples)
                                .reshape(-1)
                                )
                                index = index.reshape(-1)
                                sdf = sdf[(xx, index)].reshape(-1, n_samples)

                        # deltas = z_vals.diff(dim=-1,prepend=torch.zeros(valid_rays.origins.shape[0], 1, device=z_vals.device)+rays_near_est)
                        # samples = torch.addcmul(valid_rays.origins[:, None], valid_rays.dirs[:, None], z_vals[..., None])
                        # query_results = self.octree.blas.query(samples.reshape(-1, 3), self.octree.active_lods[0], with_parents=True)
                        # mask = (query_results.pidx[:,self.octree.active_lods[0]]<0).reshape(-1, n_samples)
                        # z_vals[mask] = (torch.rand_like(z_vals)*0.04+valid_depth[:,None].expand(valid_depth.shape[0],n_samples) - 0.02)[mask]/self.scale
                        # z_vals, index = torch.sort(z_vals, dim=-1)
                        deltas = z_vals.diff(dim=-1,prepend=torch.zeros(valid_rays.origins.shape[0], 1, device=z_vals.device)+rays_near_est)
                        samples = torch.addcmul(valid_rays.origins[:, None], valid_rays.dirs[:, None], z_vals[..., None])
                else:
                        samples = torch.zeros([0,0,3], device=rays.origins.device)
                        z_vals = torch.zeros([0,0], device=rays.origins.device)
                        deltas = torch.zeros([0,0], device=rays.origins.device)
                return samples, z_vals, deltas, valid_mask 

        def raymarch_ray(self, rays, rays_cam_o, rays_cam_d):
                # mask camera rays and lidar rays
                depth_cam,samples = self.get_img_depth(rays, num_samples=5, level=self.octree.active_lods[0])
                if depth_cam is not None:
                        depth_cam = depth_cam.squeeze()
                        # # pc_cam = rays_cam_o + depth_cam[:,None] * rays_cam_d
                        # pc = rays.origins + rays.dirs * depth_cam[:,None]/self.scale
                        # pc = pc *self.scale + self.origin
                        # self.pc_debug.append(pc)

                        # depth_cam = self.get_img_depth_im(rays).squeeze()
                        valid_mask = depth_cam>0
                        # valid_mask = torch.logical_and(depth_cam>0.5, pc_cam[:,1]>=-2)  #>-2
                        # valid_mask = torch.ones_like(rays.origins[:,0], device=rays.origins.device, dtype=bool)
                        valid_depth = depth_cam[valid_mask]
                        valid_rays = rays[valid_mask]
                        outside_rays = rays[~valid_mask]
                        # set near and far
                        rays_far = (valid_depth + 0.1)/self.scale #0.2
                        rays_near = (valid_depth - 0.1)/self.scale #0.2
                else:
                        valid_mask = torch.zeros([rays.shape[0],1]).to(self.device)

             
                if valid_mask.sum()>0:
                        with torch.no_grad():
                                num_samples_lidar = [10,0,0]
                                n_important = 10 #20 #8
                                up_sample_steps = 1 #4
                                num_samples_sur = [10] #5 
                                if sum(num_samples_sur) > 0:
                                        rays_near_est = rays_near[:,None]#/self.scale
                                        z_vals = torch.linspace(0, 1.0, num_samples_sur[0], device=valid_rays.origins.device)[None] + \
                                                (torch.zeros(valid_rays.origins.shape[0], num_samples_sur[0], device=valid_rays.origins.device) / num_samples_sur[0])
                                        z_vals *= (rays_far - rays_near)[:,None]#/self.scale
                                        z_vals += rays_near[:,None]#/self.scale
                                        n_samples = num_samples_sur[0]

                                # rays_far = 10
                                # rays_near = 0.5
                                # num = 15
                                # rays_near_est = rays_near/self.scale
                                # z_vals_new = torch.linspace(0, 1.0, num, device=valid_rays.origins.device)[None] + \
                                #         (torch.zeros(valid_rays.origins.shape[0], num, device=valid_rays.origins.device) / num)
                                # z_vals_new *= (rays_far - rays_near)/self.scale
                                # z_vals_new += rays_near/self.scale                                  # z_vals = torch.cat([z_vals, z_vals_new], dim=-1)
                                # z_vals, index = torch.sort(z_vals, dim=-1)
                                # n_samples = n_samples + num
                                                
                                pts = (valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * z_vals[..., :, None])  # N_rays, N_samples, 3
                                sdf = self.sdf(pts).squeeze()
                                for i in range(up_sample_steps):
                                        new_z_vals = self.up_sample(valid_rays.origins, valid_rays.dirs, z_vals, sdf, n_important // up_sample_steps,
                                                64 * 2 ** i, i)
                                        # z_vals = torch.cat([z_vals, new_z_vals+0.05*torch.rand_like(new_z_vals)/self.scale], dim=-1)
                                        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
                                        z_vals, index = torch.sort(z_vals, dim=-1)
                                        n_samples = n_samples + n_important // up_sample_steps
                                        pts = valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * new_z_vals[..., :, None]
                                        new_sdf = self.sdf(pts).squeeze()
                                        sdf = torch.cat([sdf, new_sdf], dim=-1)
                                        xx = (
                                                torch.arange(z_vals.shape[0])[:, None]
                                                .expand(-1, n_samples)
                                                .reshape(-1)
                                        )
                                        index = index.reshape(-1)
                                        sdf = sdf[(xx, index)].reshape(-1, n_samples) 

                                # deltas = z_vals.diff(dim=-1,prepend=torch.zeros(valid_rays.origins.shape[0], 1, device=z_vals.device)+rays_near_est)
                                deltas = z_vals.diff(dim=-1,prepend=z_vals[:,0:1])
                                samples = torch.addcmul(valid_rays.origins[:, None], valid_rays.dirs[:, None], z_vals[..., None])
                                # query_results = self.octree.blas.query(samples.reshape(-1, 3), self.octree.active_lods[0])
                                # pidx = query_results.pidx.reshape(-1, n_samples)
                                # mask = pidx>-1
                                # z_vals = z_vals[mask][:,None]
                                # num_hit_samples = z_vals.shape[0]
                                # deltas = deltas[mask].reshape(num_hit_samples, 1)
                                # samples = samples[mask]
                                # ridx = torch.arange(0, pidx.shape[0], device=pidx.device)
                                # ridx = ridx[..., None].repeat(1, n_samples)[mask]
                                # boundary = spc_render.mark_pack_boundaries(ridx.int())
                else:
                        samples = torch.zeros([0,3], device=rays.origins.device)
                        z_vals = torch.zeros([0,1], device=rays.origins.device)
                        deltas = torch.zeros([0,1], device=rays.origins.device)
                        boundary = torch.zeros([0], device=rays.origins.device)
                        ridx = torch.zeros([0], device=rays.origins.device)

                ## outside octree -- background
                z_vals_outside = None
                if BKGD:
                        if outside_rays.shape[0] > 0:
                                with torch.no_grad():
                                        self.n_outside = 10
                                        batch_size = outside_rays.shape[0]
                                        far = 0.5
                                        
                                        if self.n_outside > 0:
                                                z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside, device=self.device)
                                                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                                                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                                                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                                                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]], device=self.device)
                                                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

                                                z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / n_samples

                # return ridx, samples, z_vals, deltas, boundary, valid_mask, z_vals_outside #, samples, z_vals, deltas #samples_d, z_vals_d, deltas_d
                return samples, z_vals, deltas, valid_mask, z_vals_outside
                
        def raymarch_ray_uniform(self, rays):
                depth_cam = self.get_img_depth(rays).squeeze()
                valid_mask = depth_cam>0
                valid_depth = depth_cam[valid_mask]
                valid_rays = rays[valid_mask]
                if valid_mask.sum()>0:
                        with torch.no_grad():
                                num_samples_lidar = [50,0,0]#[10] # 4 [4,4,6] #[2 2 8] [2,2,2,3] #[4,8,6,4,2]
                                num_samples_sur = [7,15] #5
                                # num_samples_sur = [22,0]
                                n_important = 25 #20 #8
                                up_sample_steps = 5 #4
                                # if (valid_mask).sum() != 0:
                                #         rays_far = valid_depth + 0.1 #0.2
                                #         rays_near = valid_depth - 0.1 #0.2
                                #         z_vals = torch.linspace(0, 1.0, num_samples_sur[0], device=valid_rays.origins.device)[None] + \
                                #                 (torch.zeros(valid_rays.origins.shape[0], num_samples_sur[0], device=valid_rays.origins.device) / num_samples_sur[0])
                                #         z_vals *= (rays_far - rays_near)[:,None]/self.scale
                                #         z_vals += rays_near[:,None]/self.scale                                  #         #new
                                #         n_samples = num_samples_sur[0]

                                # lidar_n_samples = num_samples_sur.sum()
                                rays_far = valid_depth + 2.0 #0.2
                                rays_near = torch.zeros_like(valid_depth) + 2.0 #0.2
                                z_vals = torch.linspace(0, 1.0, num_samples_lidar[0], device=valid_rays.origins.device)[None] + \
                                                (torch.zeros(valid_rays.origins.shape[0], num_samples_lidar[0], device=valid_rays.origins.device) / num_samples_lidar[0])
                                z_vals *= (rays_far - rays_near)[:,None]/self.scale
                                z_vals += rays_near[:,None]/self.scale
                                rays_near_est = rays_near[:,None]/self.scale
                                # z_vals = torch.cat([z_vals, z_val_new], dim=-1)
                                # z_vals, index = torch.sort(z_vals, dim=-1)
                                # n_samples = n_samples + sum(num_samples_lidar)
                                n_samples = sum(num_samples_lidar)

                                pts = (valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * z_vals[..., :, None])  # N_rays, N_samples, 3
                                sdf = self.sdf(pts).squeeze()
                                for i in range(up_sample_steps):
                                        new_z_vals = self.up_sample(valid_rays.origins, valid_rays.dirs, z_vals, sdf, n_important // up_sample_steps,
                                        64 * 2 ** i, i)
                                        # z_vals = torch.cat([z_vals, new_z_vals+0.05*torch.rand_like(new_z_vals)/self.scale], dim=-1)
                                        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
                                        z_vals, index = torch.sort(z_vals, dim=-1)
                                        n_samples = n_samples + n_important // up_sample_steps
                                        pts = valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * new_z_vals[..., :, None]
                                        new_sdf = self.sdf(pts).squeeze()
                                        sdf = torch.cat([sdf, new_sdf], dim=-1)
                                        xx = (
                                        torch.arange(z_vals.shape[0])[:, None]
                                        .expand(-1, n_samples)
                                        .reshape(-1)
                                        )
                                        index = index.reshape(-1)
                                        sdf = sdf[(xx, index)].reshape(-1, n_samples)

                        samples = torch.addcmul(valid_rays.origins[:, None], valid_rays.dirs[:, None], z_vals[..., None])
                        deltas = z_vals.diff(dim=-1,prepend=torch.zeros(valid_rays.origins.shape[0], 1, device=z_vals.device)+rays_near_est)
                else:
                        samples = torch.zeros([0,0,3], device=rays.origins.device)
                        z_vals = torch.zeros([0,0], device=rays.origins.device)
                        deltas = torch.zeros([0,0], device=rays.origins.device)
                return samples, z_vals, deltas, valid_mask #, samples, z_vals, deltas #samples_d, z_vals_d, deltas_d

        @ torch.no_grad()
        def get_img_depth(self, rays_g, num_samples=6, level=8):
                # result = self.octree.raymarch(rays_g, raymarch_type='voxel', num_samples=num_samples, level=level)
                # # result = self.octree.raymarch(rays_g, raymarch_type='ray', num_samples=15, level=level)
                # depth = torch.zeros(rays_g.shape[0], 1, device=self.device)
                # ridx, deltas, depth_samples, samples, boundary = result.ridx, result.deltas, result.depth_samples, result.samples, result.boundary
                # depth_samples = depth_samples*self.scale
                # deltas = deltas*self.scale
                # if ridx.shape[0]>0:
                #         ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
                #         sdfs = self.sdf(samples)
                #         sdfs = sdfs.reshape(-1, num_samples, 1)
                #         weights,_ = self.get_weight_2(sdfs, deltas, 500, 1, boundary)
                #         ray_depth = spc_render.sum_reduce(depth_samples.reshape(-1,1) * weights.reshape(-1,1), boundary)
                #         depth[ridx_hit.long(), :] = ray_depth
                #         # depth[ridx_hit.long(),:][depth_samples[boundary==1]>depth[ridx_hit.long(), :]]=0
                # return depth,samples    

                # scannet++
                num_samples = 10
                result = self.octree.raytrace(rays_g, 8, with_exit=True)
                ridx, pidx, depth = result.ridx,  result.pidx,  result.depth
                if ridx.shape[0]==0:
                        return None, None
                ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]

                section = depth[:,1] - depth[:,0]
                mask = section < 0.05/self.scale
                depth[mask,0] -= 0.025/self.scale
                depth[mask,1] += 0.025/self.scale
                depth_samples = wisp_spc_ops.sample_from_depth_intervals(depth, num_samples)[...,None]
                deltas = depth_samples[...,0].diff(dim=-1, prepend=depth[...,0:1]).reshape(-1, 1)*self.scale

                samples = torch.addcmul(rays_g.origins.index_select(0, ridx)[:,None], 
                                rays_g.dirs.index_select(0, ridx)[:,None], depth_samples)
        
                boundary = wisp_spc_ops.expand_pack_boundary(spc_render.mark_pack_boundaries(ridx.int()), num_samples)
                sdfs = self.sdf(samples)
                sdfs = sdfs.reshape(-1, num_samples, 1)
                weights,_ = self.get_weight_2(sdfs, deltas, 500, 1, boundary)
                ray_depth = spc_render.sum_reduce(depth_samples.reshape(-1,1) * weights.reshape(-1,1), boundary)
                depth = torch.zeros(rays_g.shape[0], 1, device=self.device)
                depth[ridx_hit.long(), :] = ray_depth*self.scale
                return depth,samples.reshape(-1,3)


        def get_img_depth_im(self, rays_g):
                depth = torch.zeros(rays_g.shape[0], 1, device=self.device)
                with torch.no_grad():
                        num_samples = [80,0]
                        n_important = 20 #8
                        up_sample_steps = 2 #4

                        depth_sample_lod = self.octree.active_lods[:1]
                        z_vals, rays_near_est = self.grid_sample(rays_g, depth_sample_lod, num_samples)
                        # z_vals = torch.cat([z_vals, z_val_new], dim=-1)
                        z_vals, index = torch.sort(z_vals, dim=-1)
                        n_samples = sum(num_samples)
                        # z_vals, rays_near_est = self.grid_sample(valid_rays, lidar_sample_lod, num_samples_lidar)
                        # n_samples = sum(num_samples_lidar)

                        pts = (rays_g.origins[:, None, :] + rays_g.dirs[:, None, :] * z_vals[..., :, None])  # N_rays, N_samples, 3
                        sdf = self.sdf(pts).squeeze()
                        for i in range(up_sample_steps):
                                new_z_vals = self.up_sample(rays_g.origins, rays_g.dirs, z_vals, sdf, n_important // up_sample_steps,
                                64 * 2 ** i, i)
                                # z_vals = torch.cat([z_vals, new_z_vals+0.05*torch.rand_like(new_z_vals)/self.scale], dim=-1)
                                z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
                                z_vals, index = torch.sort(z_vals, dim=-1)
                                n_samples = n_samples + n_important // up_sample_steps
                                pts = rays_g.origins[:, None, :] + rays_g.dirs[:, None, :] * new_z_vals[..., :, None]
                                new_sdf = self.sdf(pts).squeeze()
                                sdf = torch.cat([sdf, new_sdf], dim=-1)
                                xx = (
                                torch.arange(z_vals.shape[0])[:, None]
                                .expand(-1, n_samples)
                                .reshape(-1)
                                )
                                index = index.reshape(-1)
                                sdf = sdf[(xx, index)].reshape(-1, n_samples)

                # deltas = z_vals.diff(dim=-1,prepend=torch.zeros(valid_rays.origins.shape[0], 1, device=z_vals.device)+rays_near_est)
                deltas = z_vals.diff(dim=-1,prepend=z_vals[:,0:1])
                samples = torch.addcmul(rays_g.origins[:, None], rays_g.dirs[:, None], z_vals[..., None])
                query_results = self.octree.blas.query(samples.reshape(-1, 3), self.octree.active_lods[0])
                pidx = query_results.pidx.reshape(-1, n_samples)
                mask = pidx>-1
                # ray_hit_mask = mask.sum(-1)>0
                z_vals = z_vals*self.scale
                deltas = deltas*self.scale
                z_vals = z_vals[mask][:,None]
                num_hit_samples = z_vals.shape[0]
                deltas = deltas[mask].reshape(num_hit_samples, 1)
                samples = samples[mask]
                ridx = torch.arange(0, pidx.shape[0], device=pidx.device)
                ridx = ridx[..., None].repeat(1, n_samples)[mask]
                boundary = spc_render.mark_pack_boundaries(ridx)

                sdfs = self.sdf(samples)
                weights = self.get_weight_2(sdfs, deltas, 500, 1, boundary)
                ray_depth = spc_render.sum_reduce(z_vals * weights, boundary)
                depth[ridx.unique(), :] = ray_depth

                return depth
        
        def get_weight(self, sdfs, deltas, inv_s, cos_anneal_ratio):
                if len(sdfs.shape)==3:
                        sdfs = sdfs.squeeze(2)
                prev_sdf, next_sdf = sdfs[:, :-1], sdfs[:, 1:]
                mid_sdf = (prev_sdf + next_sdf) * 0.5
                mid_sdf = torch.hstack((mid_sdf,sdfs[:, -1:]))
                deltas_sdf = sdfs.diff(dim=-1, prepend=sdfs[:,0:1]).reshape(-1, 1)
                cos_val = deltas_sdf / (deltas.reshape(-1,1) + 1e-5)

                cos_val = cos_val.clip(-1e3, 0.0)
        
                weight = neus_weights(mid_sdf, deltas.reshape(-1,mid_sdf.size(1)), inv_s, cos_val.reshape(-1,mid_sdf.size(1)))
                return weight

        def get_weight_2(self, sdfs, deltas, inv_s, cos_anneal_ratio, boundaries):
                if len(sdfs.shape)==3:
                        sdfs = sdfs.squeeze(2)
                        prev_sdf, next_sdf = sdfs[:, :-1], sdfs[:, 1:]
                        mid_sdf = (prev_sdf + next_sdf) * 0.5
                        deltas_sdf = sdfs.diff(dim=-1, prepend=sdfs[:,0:1]).reshape(-1, 1)
                        mid_sdf = torch.hstack((mid_sdf,sdfs[:,-1:]))
                else:
                        mid_sdf = (sdfs[:-1]+sdfs[1:])*0.5
                        mid_sdf = torch.vstack((mid_sdf,sdfs[-1:]))
                        deltas_sdf = spc_render.diff(sdfs, boundaries)
                
                cos_val = deltas_sdf / (deltas.reshape(-1,1) + 1e-5)
                cos_val = cos_val.clip(-1e3, 0.0)
        
                weight, alpha = neus_weights_2(mid_sdf, deltas.reshape(-1,mid_sdf.size(1)), inv_s, cos_val.reshape(-1,mid_sdf.size(1)), boundaries, True)
                return weight, alpha
        
        def get_weight_train(self, dirs, shape, sdfs, grads, deltas, inv_s, cos_anneal_ratio, boundaries):
                if len(sdfs.shape)==3:
                        sdfs = sdfs.squeeze(2)
                prev_sdf, next_sdf = sdfs[:, :-1], sdfs[:, 1:]
                mid_sdf = (prev_sdf + next_sdf) * 0.5
                mid_sdf = torch.hstack((mid_sdf,sdfs[:, -1:]))
                # dirs = dirs[:, None, :].expand(shape).reshape(-1, 3)
                n = F.normalize(grads, dim=-1) 
                true_cos = (dirs * n.reshape(-1, 3)).sum(-1, keepdim=True)
                # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
                # the cos value "not dead" at the beginning training iterations, for better convergence.
                iter_cos = -(
                F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
                + F.relu(-true_cos) * cos_anneal_ratio
                )  # always non-positive
                # iter_cos = -F.relu(-true_cos) 
                weight = neus_weights_2(mid_sdf, deltas.reshape(-1,mid_sdf.size(1)), inv_s, true_cos.reshape(-1,mid_sdf.size(1)), boundaries, True)
                #---------
                # estimated_next_sdf = mid_sdf.reshape(-1, 1) + iter_cos * deltas.reshape(-1, 1) * 0.5
                # estimated_prev_sdf = mid_sdf.reshape(-1, 1) - iter_cos * deltas.reshape(-1, 1) * 0.5
                # prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                # next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
                # p = prev_cdf - next_cdf
                # c = prev_cdf
                # alphas = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0) #.reshape(shape[0], shape[1])
                # if boundaries == None:
                #         transmittance = torch.cumprod(
                #                 torch.cat([torch.ones([alphas.size()[0], 1], device=alphas.device),1.0 - alphas + 1e-7,], -1,), -1)[:, :-1]
                #         weight = alphas* transmittance
                # else:
                #         transmittance = torch.exp(spc_render.cumsum(torch.log(1. - alphas.contiguous() + 1e-7).reshape(-1,1).contiguous(), boundaries.contiguous(), exclusive=True))
                #         weight = alphas.reshape(-1,1) * transmittance
                # # weight = neus_weights_2(sdfs, deltas.reshape(-1,sdfs.size(1)), inv_s, iter_cos.reshape(-1,sdfs.size(1)), boundaries, True)
                return weight
            
        def draw_depth_render(self, depth, i, folder_name='depth_render_img', mode='color'):
                if MODE == 'test':
                        folder_name = 'depth_render_img_nvs'
                savepath = os.path.join(self.save_path , f"{folder_name}/%010d.png"%i)
                folder = os.path.join(self.save_path , folder_name)                
                os.makedirs(folder, exist_ok=True)
                if mode=='value':
                        depth = depth*self.scale.cpu().numpy()
                        depth = depth.squeeze()*100
                        cv2.imwrite(savepath, depth.reshape(-1,self.width,1).astype(np.uint16))
                elif mode=='color':
                        # #----------------color-----------------
                        depth = depth*256/(depth.max()-depth.min())
                        # depth = np.vstack((np.zeros([176*self.width,1]),depth.squeeze(-1))).squeeze()
                        depth = depth.squeeze()
                        # cv2.imshow('depth_render', depth.reshape(-1,self.width,1))                        
                        im_color = cv2.applyColorMap(cv2.convertScaleAbs(depth.reshape(-1,self.width,1), alpha=1), cv2.COLORMAP_JET)
                        cv2.imwrite(savepath, im_color)

        def draw_rgb_render(self, rgb, i, save_dir = None):
                if MODE == 'test':
                        save_dir = save_dir+'_nvs'
                # rgb_normal = np.vstack((np.zeros([176*self.width,3]),rgb)).squeeze()
                rgb_normal = rgb.squeeze().astype(np.uint8)
                # cv2.imshow('sdf_render', rgb_normal.reshape(-1,self.width,3))
                savepath = os.path.join(self.save_path , save_dir, "%010d.png"%i)
                folder = os.path.join(self.save_path , save_dir)            
                os.makedirs(folder, exist_ok=True)
                rgb_normal = rgb_normal.reshape(-1,self.width,3)
                cv2.imwrite(savepath, rgb_normal[:,:,::-1])
        def draw_lidar_img(self, rgb_list, i, lidar_data, save_dir=None):
                lidar_points = lidar_data.cpu()
                lidar_points = lidar_points @ self.TrCam0ToVelo[:3, :3] - self.TrCam0ToVelo[:3, :3].T @ self.TrCam0ToVelo[:3,3][:,None].squeeze()
                u = (self.fx * lidar_points[:,0] / lidar_points[:,2] + self.cx).long()
                v = (self.fy * lidar_points[:,1] / lidar_points[:,2] + self.cy).long()
                rgb = torch.zeros_like(lidar_points)
                # mask = torch.logical_and(idx>0, idx<self.w*self.h)
                z_mask = lidar_points[:,2]>0
                u_mask = torch.logical_and(u>0, u<self.width)
                v_mask = torch.logical_and(v>0, v<376)
                mask = torch.logical_and(u_mask,v_mask)
                mask = torch.logical_and(mask, z_mask)
                # img = np.zeros([376,self.width,3])
                img = rgb_list.reshape(-1,self.width,3).copy()

                lidar_points = lidar_points[mask]
                colors = np.zeros([lidar_points.shape[0],3])
                height_max = lidar_points[:, 2].max()
                height_min = lidar_points[:, 2].min()
                delta_c = abs(height_max - height_min) / (255 * 2)
                for j in range(lidar_points.shape[0]):
                    color_n = (lidar_points[j, 2] - height_min) / delta_c
                    if color_n <= 255:
                        colors[j, :] = [0, 1 - color_n / 255, 1]
                    else:
                        colors[j, :] = [(color_n - 255) / 255, 0, 1]
                # img[v[mask], u[mask]]=self.rgb_list[frame][v[mask],u[mask]].float().cpu().numpy()
                # img[v[mask], u[mask]]=colors*255
                img[v[mask], u[mask]]=[0,255,0]
                rgb_img = img.copy()
                rgb_img[:,:,0] = img[:,:,2]
                rgb_img[:,:,2] = img[:,:,0]
                cv2.imwrite(self.save_path + f'/lidar_normal/{i}.png', rgb_img)

        def draw_semantic_render(self, rgb_semantic, i, save_dir=None):
                if MODE == 'test':
                        save_dir = save_dir+'_nvs'
                # rgb_normal = np.vstack((np.zeros([176*self.width,3]),rgb)).squeeze()
                # cv2.imshow('sdf_render', rgb_normal.reshape(-1,self.width,3))
                savepath = os.path.join(self.save_path , save_dir+"/%010d.png"%i)
                folder = os.path.join(self.save_path , save_dir)              
                os.makedirs(folder, exist_ok=True)
                rgb_semantic = rgb_semantic.reshape(self.height,self.width,-1)
                cv2.imwrite(savepath, rgb_semantic[:,:,::-1])

def neus_weights(sdf, dists, inv_s, cos_val, z_vals=None):    
    estimated_next_sdf = sdf + cos_val * dists * 0.5
    estimated_prev_sdf = sdf - cos_val * dists * 0.5
    
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
    weights = alpha * torch.cumprod(torch.cat([torch.ones([sdf.shape[0], 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    
    return weights #F.normalize(weights, dim=-1)

def neus_weights_2(sdf, dists, inv_s, cos_val, boundaries, exclusive, z_vals=None):    
        estimated_next_sdf = sdf + cos_val * dists * 0.5
        estimated_prev_sdf = sdf - cos_val * dists * 0.5
        
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alphas = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        if boundaries == None:
                transmittance = torch.cumprod(
                torch.cat([torch.ones([alphas.size()[0], 1], device=alphas.device),1.0 - alphas + 1e-7,], -1,), -1)[:, :-1]
                weights = alphas* transmittance
        else:
                transmittance = torch.exp(spc_render.cumsum(torch.log(1. - alphas.contiguous() + 1e-7).reshape(-1,1).contiguous(), boundaries.contiguous(), exclusive=exclusive))
                weights = alphas.reshape(-1,1) * transmittance
        # transmittance = torch.exp(spc_render.cumsum(torch.log(1. - alpha.contiguous() + 1e-7).reshape(-1,1).contiguous(), boundaries.contiguous(), exclusive=exclusive))
        # weights = alpha.reshape(-1,1) * transmittance  #torch.cumprod(torch.cat([torch.ones([sdf.shape[0], 1], device=alpha.device), 1. - alpha ], -1), -1)[:, :-1]
    
        return weights, alphas #F.normalize(weights, dim=-1)

def generate_video(rgb_vid_path, images_test):
    imageio.mimwrite(rgb_vid_path, images_test, fps=2, quality=8)
    print(f"Test RGB video saved in : {rgb_vid_path}")

def load_pointscloud(seq, frame_idx_list, PTS_MODE, render):
    if PTS_MODE == "LiDAR_ALL":
        pcd = o3d.io.read_point_cloud("feature_result_kitti/00/semantic_points_gt.ply")
        lidar_pts = torch.FloatTensor(pcd.points)

    elif PTS_MODE == "LiDAR_CAM":
        from utils.scannet_data_tools.kitti360Viewer3DRaw import Kitti360Viewer3DRaw
        velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq)
        lidar_pts = torch.FloatTensor([])
        pc_radius = 30
        min_z, max_z = -3, 10
        for i in tqdm(frame_idx_list, desc='preparing lidar pointcloud'):
                # LiDAR
                data = velo.loadVelodyneData(i)
                data = velo.curlVelodyneData(i, data)
                lidar_points = torch.FloatTensor(data[:, :3])
                # crop
                depth = np.linalg.norm(lidar_points[:,:3],axis=1)
                mask = depth>2.75
                point_ = o3d.geometry.PointCloud()
                point_.points = o3d.utility.Vector3dVector(lidar_points[mask, 0:3])
                bbx_min = np.array([-pc_radius, -pc_radius, min_z])
                bbx_max = np.array([pc_radius, pc_radius, max_z])
                bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)
                point_ = point_.crop(bbx)
                lidar_points = torch.FloatTensor(np.array(point_.points))
                # camera
                camera, TrCam0ToVelo = init_camera(seq=seq, cam_id=0)
                TrCam0ToVelo = (TrCam0ToVelo @ np.linalg.inv(camera.R_rect)).float()
                fx, fy, cx, cy = camera.K[0, 0], camera.K[1, 1], camera.K[0, 2], camera.K[1, 2]
                # projection
                lidar_points_cam = lidar_points @ TrCam0ToVelo[:3, :3] - TrCam0ToVelo[:3, :3].T @ TrCam0ToVelo[:3,3][:,None].squeeze()
                u = (fx * lidar_points_cam[:,0] / lidar_points_cam[:,2] + cx).long()
                v = (fy * lidar_points_cam[:,1] / lidar_points_cam[:,2] + cy).long()
                z_mask = lidar_points_cam[:,2]>0
                u_mask = torch.logical_and(u>0, u<self.width)
                v_mask = torch.logical_and(v>0, v<376)
                mask = torch.logical_and(u_mask,v_mask)
                mask = torch.logical_and(mask, z_mask)
                # lidar2world
                pose = SE3(render.vec_es[f"{i}"]).matrix().cpu()
                lidar_in_cam_world = (pose[:3, :3] @ lidar_points[mask].T).T + pose[:3,3]  #.cuda()
                if OUTPUT == 'per':
                        # output per frame pointcloud
                        sem = render.query_pts_sem(lidar_in_cam_world.to(render.device))
                        v_colors = np.vstack([id2label[semID].color for semID in sem.tolist()])
                        o3d_pts = o3d.geometry.PointCloud()
                        o3d_pts.points = o3d.utility.Vector3dVector(lidar_points[mask].numpy())
                        o3d_pts.colors = o3d.utility.Vector3dVector(v_colors/255.0)
                        # o3d_pts = o3d_pts.voxel_down_sample(0.1)
                        o3d.io.write_point_cloud('/home/yx/code_model/Round1_afterNF_3d/%010d.ply' % i, o3d_pts)
                elif OUTPUT == 'all':
                        # output all poincloud
                        lidar_pts = torch.cat((lidar_pts, lidar_in_cam_world), dim=0) if lidar_pts.shape[0]>0 else lidar_in_cam_world
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(lidar_pts.numpy())

    return lidar_pts




# CODE = "query_pts_semantic" # 
CODE = "render_img"
PTS_MODE = "LiDAR_CAM" #"LiDAR_ALL" #"CAM_DEPTH"
OUTPUT = 'all' #'per' #'all'
MODE = 'val' #'NVS'#'test' #

yaml_path = "config/ijrr/scannet0000_02/render_scannet_hash_++_all.yaml"
# yaml_path = "config/ijrr/scannet0087_02/render_scannet_hash_++_all.yaml"
# yaml_path = "config/ijrr/scannet0420_01/render_scannet_hash_++_all.yaml"
# yaml_path = "config/ijrr/scannetpp_5748ce6f01/render_scannet_hash_++_all.yaml"
# yaml_path = "config/ijrr/scannetpp_1ada7a0617/render_scannet_hash_++_all.yaml"
# yaml_path = "config/ijrr/scannetpp_f6659a3107/render_scannet_hash_++_all.yaml"
# yaml_path = "config/ijrr/replica_apartment_2/render/_scannet_hash_++_all.yaml"

# yaml_path = "/sdb2/xyx/PanopticRecon++/config/ijrr/replica_apartment_2/render_scannet_hash_++_all.yaml"

if __name__ == "__main__":
        if CODE == "query_pts_semantic":
                PTS, SEM = np.array([]), np.array([])
                start_frame, end_frame = 0, 0
                for Vol_idx in range(START,START+CUBE_NUM): # volumes num
                        render = Render(Vol_idx)
                        if end_frame==0:
                                start_frame = Vol_idx*frame_num+250 + int(frame_num/2) if Vol_idx>0 else 250
                                end_frame = start_frame+frame_num if Vol_idx>0 else start_frame+frame_num+int(frame_num/2)
                        else:
                                start_frame = end_frame
                                end_frame = min(start_frame+frame_num, (Vol_idx+1)*frame_num+250+math.ceil(frame_num/2)) 
                        pts = load_pointscloud(00, range(start_frame,end_frame,step), PTS_MODE, render)
                        # pts = load_pointscloud(00, range(254,257,step), PTS_MODE, render)
                        if OUTPUT == 'all':
                                sem = render.query_pts_sem(pts.to(render.device))
                                v_colors = np.vstack([id2label[semID].color for semID in sem.tolist()])
                                
                                PTS = np.concatenate((PTS, pts), axis=0) if PTS.shape[0]>0 else pts.numpy()
                                SEM = np.concatenate((SEM, v_colors), axis=0) if SEM.shape[0]>0 else v_colors
                
                o3d_pts = o3d.geometry.PointCloud()
                o3d_pts.points = o3d.utility.Vector3dVector(PTS)
                o3d_pts.colors = o3d.utility.Vector3dVector(SEM/255.0)
                # o3d_pts = o3d_pts.voxel_down_sample(0.1)
                o3d.io.write_point_cloud('/home/yx/code_model/semantic_points_round1.ply', o3d_pts)

        elif CODE == "render_img":
                render = Render(START, yaml_path, MODEL, mode=MODE)
                render.read_pose()
                frame_start = TEST[0] + START*frame_num//2
                frame_end = frame_start + frame_num# + frame_num//2
                dataloader_test = render.get_render_rays(frame_start,frame_end,step)
                result = render.render(dataloader_test)
                # normal_imgs, semantic_imgs = render.render_sdf_video(255, 20)
                # for idx in range(START+1,START+CUBE_NUM):
                #         print(f'rendering images from cube_{idx}:')
                #         render.load_model(idx)
                #         frame_start = frame_end
                #         if idx == START+CUBE_NUM-1:
                #                 frame_end +=  frame_num# + frame_num//2
                #         else:
                #                 frame_end +=  frame_num
                #         dataloader_test = render.get_render_rays(frame_start,frame_end,step)
                #         result = render.render(dataloader_test)
                #         # result = render.render_sdf_video(260, 20)
                #         normal_imgs.extend(result[0])
                #         semantic_imgs.extend(result[1])
                # normal_video_path = os.path.join(render.cfg['path']['proj_dir'], "normal_render_img/normal.mov")
                # # normal_video_path = os.path.join(render.cfg['path']['proj_dir'], "normal_novel_img/normal.mov")
                # # generate_video(normal_video_path, normal_imgs)
                # if render.rgb_flag:
                #         rgb_video_path = os.path.join(render.cfg['path']['proj_dir'], "rgb_render_img/rgb.mov")
                #         # rgb_video_path = os.path.join(render.cfg['path']['proj_dir'], "rgb_novel_img/rgb.mov")
                #         generate_video(rgb_video_path, semantic_imgs)
                # if render.semantic_flag:
                #         semantic_video_path = os.path.join(render.cfg['path']['proj_dir'], "semantic_render_img/semantic.mov")
                #         generate_video(semantic_video_path, semantic_imgs)
