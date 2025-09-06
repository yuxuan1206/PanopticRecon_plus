'''
Author: yuxuan1206 610939662@qq.com
Date: 2022-11-08 19:03:51
LastEditors: yuxuan1206 610939662@qq.com
LastEditTime: 2022-12-05 14:10:05
FilePath: /occuSLAM3D_indoor_KITTI/models/SDFGrid.py
Description: 

Copyright (c) 2022 by yuxuan1206 610939662@qq.com, All Rights Reserved. 
'''
from .BaseGrid import *

import torch
from wisp.models.grids import *
import kaolin.ops.spc as spc_ops
from lietorch import SE3, LieGroupParameter
from models.decoder import SDFDecoder, SemanticSDFDecoder, NeRFSDFDecoder, PositionDecoder, RadianceDecoder, SemanticRGBDecoder, PEMAP, instanceDecoder
from wisp.ops.geometric import find_depth_bound
from wisp.ops.differential import autodiff_gradient
from wisp.core import Rays
import open3d as o3d
import kaolin.render.spc as spc_render
from itertools import chain

import cv2
from scipy.spatial.transform import Rotation as R_scipy
import scipy
# try:
import tinycudann as tcnn
import matplotlib.pyplot as plt
from instance.models.matcher import batch_sigmoid_ce_loss, batch_dice_loss
from data.label import get_labels #id2label
id2label = get_labels("0420_01")

# from torch.utils.data import DataLoader
# import torchvision 
from tqdm import tqdm
from instance.models.mask3d import Mask3D
from instance.models.query_model import QueryModel



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def axis_angle_to_matrix(data):
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle, axis_angle_to_quaternion
    batch_dims = data.shape[:-1]
    theta = torch.norm(data, dim=-1, keepdim=True)
    omega = data / theta
    omega1 = omega[...,0:1]
    omega2 = omega[...,1:2]
    omega3 = omega[...,2:3]
    zeros = torch.zeros_like(omega1)

    K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[...,None,:],
                      torch.concat([omega3, zeros, -omega1], dim=-1)[...,None,:],
                      torch.concat([-omega2, omega1, zeros], dim=-1)[...,None,:]], dim=-2)
    I = torch.eye(3, device=data.device).expand(*batch_dims,3,3)

    return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)

def matrix_to_axis_angle(rot):
    """
    :param rot: [N, 3, 3]
    :return:
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(rot))

def pose6d_to_matrix(batch_poses):
    c2w = torch.eye(4, device=batch_poses.device).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
    c2w[:,:3,:3] = axis_angle_to_matrix(batch_poses[:,3:])
    c2w[:,:3,3] = batch_poses[:,0:3]
    return c2w

def matrix_to_pose6d(batch_matrices):
    return torch.cat([batch_matrices[:,:3,3],
                    matrix_to_axis_angle(batch_matrices[:,:3,:3])], dim=-1)

class SDFGrid(BaseGrid):
    #-------------------------------init----------------------------------
    def __init__(self, device, config, data_list, vec_es, vec_cam, mode):
        super(SDFGrid, self).__init__(device, config)
        self.device = device
        self.data_list = data_list
        self.vec_es = vec_es
        self.vec_cam = vec_cam
        self.get_module_from_config(config)
        if config['path']['dataset_type'] == 'kitti360':
            self.camera, self.TrCam0ToVelo = init_camera(seq=config['path']['seq'], cam_id=0)
            self.TrCam0ToVelo = (self.TrCam0ToVelo @ np.linalg.inv(self.camera.R_rect)).float()
        else:
            self.TrCam0ToVelo = np.eye(4)

        self.mode = mode
        self.iter_n = 0
        self.epoch = 0
        # self.init_grid()
        self.ray_mode = config['ray_mode']
        self.origin = None
        self.scale = None
        self.optim_mode = 0
        self.appearance_embedding = False
        if not config['pretrain']:
            self.init_decoder(config)
        self.s = torch.nn.Parameter(torch.FloatTensor([0.4]).to(self.device))
        self.sample_points = []
        self.pretrain = config['pretrain']
        self.BKGD = config['BKGD']
        self.cfg = config
        # self.load_grid("/mnt/northcn4/ywx1261068/Round1_vit/optimed_grid_0.pth")
    
    def get_module_from_config(self, config):
        self.grid_cfg = config['grid']
        self.decoder_cfg = config['decoder']
        self.model_type = config['model_type']
        self.feature_std = 0.01 #0.01
        self.feature_bias = 0.
        self.cfg = config
        self.voxel_sizes = config["params"]["voxel_sizes"]
        self.normal_flag = config["loss_term"]["normal"]
        self.depth_completion = config["loss_term"]["depth_completion"]
        self.semantic_flag = config["loss_term"]["semantic2d"] or config["loss_term"]["semantic3d"] 
        self.instance_flag = config["loss_term"]["instance"]
        self.panoptic_flag = config["loss_term"]["panoptic"]
       
        if self.semantic_flag:
            self.sem_class_num, self.positionencoding = config['decoder']['semantic']['output_ch'], config['decoder']['semantic']['concat_qp']
        self.rgb_flag = config["loss_term"]["rgb"]
        self.gaussian = None
        if self.rgb_flag and self.model_type == '3dgs':
            from Gaussian.train_3dgs import Pipeline_3dgs
            self.gaussian_train_mode = config['3dgs_mode']
            self.gaussian = Pipeline_3dgs()  
            ## load rays
            if config['path']['dataset_type'] == 'scannet':
                from utils.scannet_data_tools.scannet_dataset import ScannetDataset
                self.render_factor = 2
                self.dataset_test = ScannetDataset(config, self.data_list, self.device, self.vec_es, 'val', factor=self.render_factor, optim_flag=config["loss_term"]) 
        self.feature_flag = config["loss_term"]["feature"]
        if self.feature_flag:
            self.feat_dim = config['decoder']['feature']['output_ch'] 
        if self.instance_flag or self.panoptic_flag:
            self.prun_epoch = 1 #1 #2
            self.Thing_class = config['Thing_class']
            self.Stuff_class = config['Stuff_class']
            all_class = config['Stuff_class'] + config['Thing_class']
            self.label2label_mapping = {0:-1}
            for i in all_class:
                self.label2label_mapping[i] = self.Thing_class.index(i)+1 if i in self.Thing_class else 0
            self.cost_mask, self.cost_dice = 1, 1
            self.color_map = np.random.randint(0,255,size=(1000,3), dtype=np.uint8)
            self.token_train_iter = 0
            self.instance_threthod = 0.8 #0.9 #0.5
            self.instance_corners_all = []
            # self.color_map[0,:] = 0
        self.PEMAP = config['PEmap']
        self.interpolation_type = 'linear'
        self.multiscale_type = 'sum'
        self.ray_mode = config['ray_mode']
        self.net = config['model']['network_structure']
    
    def init_grid(self):
        """Initialize the grid object.
        """
        grid_class = OctreeGrid
        self.grid = {}
        for key, cfg in self.grid_cfg.items():
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": cfg['num_lods'],
                "n_features_per_level": cfg['feature_dim'],
                "log2_hashmap_size": cfg['log2_hashmap_size'],
                "base_resolution": cfg['base_lod'],
                "per_level_scale": cfg['per_level_scale'],
                "interpolation": self.interpolation_type,
            }
            self.grid[key] = self.tcnn_encoding = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)

    def freeze_geometry(self):
        print('----------freeze the geometry----------')
        for child in chain(self.decoder['sdf'].children(),self.grid['sdf'].children()): #.modules()
                for param in child.parameters():
                        param.requires_grad = False
        for param in self.grid['sdf'].parameters():
            param.requires_grad = False
        self.s.requires_grad = False #True

    def unfreeze_geometry(self, channels=['sdf']):
        print('----------unfreeze the decoder----------')
        for channel in channels:
            if channel in self.decoder.keys():
                for child in self.decoder[channel].children():
                        for param in child.parameters():
                                param.requires_grad = True
            if channel in self.grid.keys():
                for child in self.grid[channel].children():
                        for param in child.parameters():
                                param.requires_grad = True
                for param in self.grid[channel].parameters():
                    param.requires_grad = True 
        self.s.requires_grad = True 

    def freeze_param(self, channels=['sdf']):
        print('----------freeze the model----------')
        for channel in channels:
            if channel in self.decoder.keys():
                for child in self.decoder[channel].children():
                        for param in child.parameters():
                                param.requires_grad = False
            if channel in self.grid.keys():
                for child in self.grid[channel].children():
                        for param in child.parameters():
                                param.requires_grad = False
                for param in self.grid[channel].parameters():
                        param.requires_grad = False
        self.s.requires_grad = False #True #
    
    def freeze_query_param(self):
        print('----------freeze the query----------')
        for child in self.instance_model.children():
            for param in child.parameters():
                param.requires_grad = False
        for param in self.instance_model.class_embed_head.parameters():
            param.requires_grad = True
        # self.super_point.requires_grad = False

    def init_from_pointcloud(self, pc, vox_down_m=0.0005, down_flag=False, dilate=0):
        self.grid = {}
        for key, cfg in self.grid_cfg.items():
            self.chose_grid(key, cfg)
        self.octree = OctreeGrid.from_pointcloud(pointcloud=pc,feature_dim=1,
                                                base_lod=8, num_lods=2,
                                                interpolation_type=self.interpolation_type, multiscale_type=self.multiscale_type,dilate=dilate)
        # self.channels = list(self.grid_cfg.keys())
        # print('channels: ', self.channels)

    def update_octree(self, vox_down_m=0.0005, down_flag=False, dilate=0):
        print("----------update octree-------------")
        traj = [self.vec_es]
        traj1, traj2 = traj, traj
        traj1[:,1] -= 3.5
        traj2[:,1] += 3.5
        
        pc_add_world = 1
        self.octree = OctreeGrid.from_pointcloud(pointcloud=pc,feature_dim=1,
                                                base_lod=8, num_lods=2,
                                                interpolation_type=self.interpolation_type, multiscale_type=self.multiscale_type,dilate=dilate)

    def chose_grid(self, key, cfg):
        if key != 'rgb' or (key == 'rgb' and self.model_type == 'separate'):
            growth_factor = np.exp((np.log(cfg['max_resolution']) - np.log(cfg['base_resolution'])) / (cfg['n_levels'] - 1))
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": cfg['n_levels'],
                "n_features_per_level": cfg['n_features_per_level'],
                "log2_hashmap_size": cfg['log2_hashmap_size'],
                "base_resolution": cfg['base_resolution'],
                "per_level_scale": growth_factor,
                "interpolation": 'Smoothstep',
            }
            self.grid[key] = self.tcnn_encoding = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config, dtype=torch.float32)

    @ torch.no_grad()
    def plot_traj(self):
        from matplotlib import pyplot as plt
        # fig = plt.figure(figsize=(20, 20))
        fig, ax = plt.subplots()
        vec_cam_init = SE3.exp(self.R_cam_init.log()).vec().cpu().numpy()
        vec_cam = SE3.exp(self.R_cam.log()).vec().cpu().numpy()
        ax.plot(vec_cam_init[:,0], vec_cam_init[:,1], c='g')
        ax.plot(vec_cam[:,0], vec_cam[:,1], c='b')
        rot_cam_init = (R_scipy.from_quat(vec_cam_init[:,3:]).as_matrix() @ np.array([0,0,1])[None,:,None]).squeeze()
        rot_cam = (R_scipy.from_quat(vec_cam[:,3:]).as_matrix() @ np.array([0,0,1])[None,:,None]).squeeze() 
        ax.quiver(vec_cam_init[:,0], vec_cam_init[:,1], rot_cam_init[:,0], rot_cam_init[:,1], scale_units ='xy', scale = 0.4, color='g')
        ax.quiver(vec_cam[:,0], vec_cam[:,1], rot_cam[:,0], rot_cam[:,1], scale_units ='xy', scale = 0.4, color='b')
        # ax.quiver(vec_cam[:,0], vec_cam[:,1], vec_cam[:,2], rot_cam[:,0], rot_cam[:,1], rot_cam[:,2], length=0.1, normalize=True, color='y')
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # plt.savefig(os.path.join("semantic_result_kitti/00/pose_img/%d_%d.png" % (step,i)))
        plt.savefig("./pose.png" )
        plt.close('all')
    
    def save_grid(self, path, epoch):
        state = {'grid': self.grid, 'decoder': self.decoder, \
            "world_dims": self.world_dims, "origin": self.origin, "scale": self.scale, 's_val':self.s, 'octree':self.octree}
        if self.appearance_embedding:
            state['embedding_a'] = self.embedding_a
        if self.BKGD:
            state['nerf_outside'] = self.nerf_outside.state_dict()
        if self.PEMAP:
            state['PE_model'] = self.PE_model.state_dict()
        if self.instance_flag:
            state['instance_model'] = self.instance_model.state_dict()
            # state['instance_corners'] = self.instance_corners
            state['query_position'] = self.query_position
            self.instance_model.gaussians.save_ply(os.path.join(self.cfg['path']['proj_dir'], f'grid/optimed_query_{epoch}.ply'))
            # state['radius'] = self.radius
            # state['query_mask'] = self.instance_model.query_mask
            # state['query_update_map'] = self.instance_model.query_update_map

        torch.save(state, path)
        print("state saved in : ", path)
    
    def read_last_grid(self, path):
        data = torch.load(path)
        # self.nef.grid = data['grid'] #update octree
        self.world_dims = data['world_dims'] 
        self.scale = data['scale'] 
        # self.decoder = data['decoder'] 
        # self.nef.origin = data['origin'] 

    def load_grid(self, path):
        state = torch.load(path)
        self.grid = state['grid']
        self.decoder = state['decoder']
        # self.decoder['semantic'] = PositionDecoder(**self.decoder_cfg['semantic'], input_feat_dim=self.feat_dim, output_ch=self.sem_class_num).to(self.device)
        self.world_dims = state['world_dims']
        self.origin = state['origin']
        self.scale = state['scale']
        self.s = state['s_val']
        self.octree = state['octree']
        if self.appearance_embedding:
            self.embedding_a = state['embedding_a']
        if self.BKGD and 'nerf_outside' in state.keys():
            from models.fields import NeRF
            self.nerf_outside = NeRF(**self.cfg['outside']['nerf']).to(self.device)
            self.nerf_outside.load_state_dict(state['nerf_outside'])
        if self.PEMAP and 'PE_model' in state.keys():
            self.PE_model = PEMAP(self.cfg['decoder']['feature']['output_ch']).to(self.device)
            self.PE_model.load_state_dict(state['PE_model'])
        if self.instance_flag:
            # self.instance_corners = data['instance_corners']
            self.instance_model = Mask3D(**self.cfg['instance'], query_position=state['query_position'], device=self.device)
            self.instance_model.to(self.device)
            self.instance_model.load_state_dict(state['instance_model'])
            # self.instance_model.update_network(state['query_mask'], state['query_update_map'])
            self.instance_model.gaussians = QueryModel()
            self.instance_model.gaussians.load_ply(os.path.join(self.cfg['path']['proj_dir'], f'grid/optimed_query_{path.split(".")[0].split("_")[-1]}.ply'))
            self.query_position = state['query_position']
            num = self.instance_model.gaussians._xyz.shape[0]
            self.query_assignment_record = torch.zeros(num).to(self.device)
            self.query_assignment_record_add = torch.zeros(num).to(self.device)
            self.query_assignment_record_add_history = torch.zeros(num).to(self.device)
            self.query_update_map = {i: i for i in range(num)}
        print("load model from: : ", path)
        print(self.decoder)
        self.set_channels()

    def init_decoder(self, config):
        self.decoder = {}
        for key, cfg in self.decoder_cfg.items():
            self.chose_decoder(key, cfg, config)
        print(self.decoder)
        self.set_channels()

    def chose_decoder(self, key, cfg, config):
        if key=='semantic':
            if self.net=='geo_sem':
                self.decoder['sdf'] = SemanticSDFDecoder(config['decoder']['sdf'], config['decoder']['semantic'], 
                                                    sdf_feat_dim=self.grid_cfg['sdf']['n_levels']*self.grid_cfg['sdf']['n_features_per_level']).to(self.device)
            elif self.net=='rgb_sem':
                self.decoder['rgb'] = SemanticRGBDecoder(config['decoder']['rgb'], config['decoder']['semantic'], 
                                                    rgb_feat_dim=self.grid_cfg['rgb']['n_levels']*self.grid_cfg['rgb']['n_features_per_level'],
                                                    sem_num=self.sem_class_num).to(self.device)
            elif self.net=='geo_rgb_sem':
                self.decoder['semantic'] = PositionDecoder(**cfg, input_feat_dim=self.grid_cfg['semantic']['n_levels']*self.grid_cfg['semantic']['n_features_per_level']
                                                                                    ).to(self.device) #+ config['decoder']['rgb']['W']   + config['decoder']['sdf']['W'] + 3
            # self.decoder[key] = PositionDecoder(**cfg, input_feat_dim=self.feat_dim, output_ch=self.sem_class_num).to(self.device)
        elif key == 'rgb':
            if self.model_type == 'shared':
                self.decoder[key] = RadianceDecoder(**cfg, input_feat_dim=self.decoder_cfg['sdf']['output_ch']-1).to(self.device)
            elif self.model_type == 'separate':
                # if 'semantic' in self.decoder_cfg.keys():
                #     self.decoder[key] = 
                # else:
                self.decoder[key] = RadianceDecoder(**cfg, input_feat_dim=self.grid_cfg[key]['n_levels']*self.grid_cfg[key]['n_features_per_level']).to(self.device)

            self.appearance_embedding = (cfg['embedding_a_dim']>0)
            self.embedding_a_dim = cfg['embedding_a_dim']

            from models.fields import NeRF
            self.nerf_outside = NeRF(**config['outside']['nerf']).to(self.device)

        elif key == 'feature': # input+depth
            self.decoder[key] = PositionDecoder(**cfg, input_feat_dim=self.grid_cfg['semantic']['n_levels']*self.grid_cfg['semantic']['n_features_per_level']).to(self.device)
            if self.PEMAP:
                feature_embedding_dim = config['decoder']['feature']['output_ch']
                # positional embedding (PE) decomposition
                # globally-shared low-resolution learnable PE map
                # self.learnable_pe_map = torch.nn.Parameter(
                #     0.05 * torch.randn(1, feature_embedding_dim // 2, 80, 120),
                #     requires_grad=True,
                # ).to(self.device)
                # # a PE head to decode PE features
                # self.pe_head = torch.nn.Sequential(
                #     torch.nn.Linear(feature_embedding_dim // 2, feature_embedding_dim),
                # ).to(self.device)
                self.PE_model = PEMAP(feature_embedding_dim).to(self.device)
        # elif key == 'instance':
        #     self.decoder[key] = instanceDecoder(**cfg, input_feat_dim=3).to(self.device)
        else:
            self.decoder[key] = PositionDecoder(**cfg, input_feat_dim=self.grid_cfg[key]['n_levels']*self.grid_cfg[key]['n_features_per_level']).to(self.device)
        
    def set_channels(self):
        self.channels = list(self.decoder_cfg.keys())
        if self.rgb_flag and 'rgb' not in self.channels:
            self.channels.append('rgb')
        if self.instance_flag and 'instance' not in self.channels:
            self.channels.append('instance')
        if self.panoptic_flag:
            if 'instance' not in self.channels:
                self.channels.append('instance') 
            if 'panoptic' not in self.channels:
                self.channels.append('panoptic') 
        print('channels: ', self.channels)
        
    def init_pose(self, base_dir, data_list, vec_gt, vec_es_odom, key_idx):
        pose_path = os.path.join(base_dir, "pose")
        vecs, cam_vecs = [], []
        vecs_gt = []
        self.pose = []
        self.data_list = data_list
        with torch.no_grad():
            for i in data_list:
                # pose = torch.FloatTensor(np.loadtxt(os.path.join(pose_path, f"{i}.txt"))).to(self.device)
                vec = vec_gt[f"{i}"]

                # # vec_noise = self.vec_es[f"{i}"]
                self.vec_es[f"{i}"] = vec_es_odom[f"{i}"]
                vecs.append(self.vec_es[f"{i}"])
                vec_cam_init = self.vec_cam[f"{i}"]
                # vec_cam_init[0:3] += torch.randn(3).to(self.device)*0.1
                cam_vecs.append(vec_cam_init)
            
                vecs_gt.append(vec)
                self.pose.append(SE3(self.vec_es[f"{i}"]).matrix()) # noise
        
        R = torch.stack([x for x in vecs])
        R = SE3(R.cuda())
        R_cam = torch.stack([x for x in cam_vecs])
        R_cam_init = SE3(R_cam.clone().cuda())
        R_cam = SE3(R_cam.cuda())
        R_cam_gt = R.matrix().detach().cpu() @ self.TrCam0ToVelo[None,:,:]
        vec_cam_gt = torch.cat([R_cam_gt[:, 0:3, 3], torch.FloatTensor(R_scipy.from_matrix(R_cam_gt[:, 0:3, 0:3].numpy()).as_quat())], dim=1)
        R_cam_gt = SE3(vec_cam_gt.cuda())
        self.first_R = R[0]

        # self.R = LieGroupParameter(R[1:])
        if self.mode==1:
            self.R = LieGroupParameter(R[-1:])
            self.T_fix = LieGroupParameter(R[:-1])
        else:
            self.span = 1 if self.mode==2 else 10 #8
            self.R = LieGroupParameter(R[self.span:])
            self.R_cam = LieGroupParameter(R_cam)
            # self.R_cam = nn.Parameter(matrix_to_pose6d(R_cam.matrix()))
            self.R_cam_init = LieGroupParameter(R_cam_init)
            self.R_cam_gt = LieGroupParameter(R_cam_gt)
            self.T_fix = LieGroupParameter(R[:self.span])

        # if len(vecs) > 1:
        #     self.print_pose_error(vecs, vecs_gt, vec_loam[data_list[-2]-250:data_list[-1]+1-250,:3], key_idx, traj_gt, traj_loam)
        

    #---------------------------------optimer--------------------------------
    def set_optimer(self, occ_lr=0.005, pose_lr = 0.01, s_lr = 0.1, rgb_lr=0.02, mlp_lr=0.02): 
        self.optim_pose = (pose_lr>0)
        self.optim_list = []
        for key, cfg in self.decoder.items():
            decoder_parm =count_parameters(self.decoder[key])
            decoder_memory = decoder_parm/(1024*1024) * 4
            print(key+" decoder memory cost:",  decoder_parm , str(decoder_memory)+" M")  
            if key=='sdf' or key=='feature' or key=='instance' or  key=='semantic': 
                self.optim_list.append({'name':'decoder_'+key, 'params': self.decoder[key].parameters(), 'lr':0.0002}) #0.001
            # elif key=='semantic': 
            #     self.optim_list.append({'name':'decoder_'+key, 'params': self.decoder[key].parameters(), 'lr':0.001}) #0.001
            else:
                self.optim_list.append({'name':'decoder_'+key, 'params': self.decoder[key].parameters(), 'lr':mlp_lr})
        for key, cfg in self.grid_cfg.items():
            encoder_parm =count_parameters(self.grid[key])
            encoder_memory = encoder_parm/(1024*1024) * 4
            print(key+" encoder memory cost:",  encoder_parm , str(encoder_memory)+" M")
            if key=='feature':
                self.optim_list.append({'name':'encoder_'+key, 'params': self.grid[key].parameters(), 'lr':0.003}) #
            elif key=='instance' or key=='semantic':
                self.optim_list.append({'name':'encoder_'+key, 'params': self.grid[key].parameters(), 'lr':0.01})
            #     continue
            else:
                self.optim_list.append({'name':'encoder_'+key, 'params': self.grid[key].parameters(), 'lr':occ_lr})
        # if self.optim_pose:
        #     self.optim_list.append({'name':'camera_pose', 'params': [self.R_cam], 'lr': pose_lr})
        self.optim_list.append({'name':'s_value', 'params':[self.s], 'eps':1e-3, 'lr':s_lr})
        if self.appearance_embedding:
            self.embedding_a = torch.nn.Embedding(len(self.R_cam), self.embedding_a_dim).to(self.device)
            self.optim_list.append({'name':'a_embedding', 'params': self.embedding_a.parameters(), 'lr':0.5})
        if self.PEMAP:
            self.optim_list.append({'name':'PE_model', 'params': self.PE_model.parameters(), 'lr':mlp_lr})
        if 'instance' in self.grid.keys():
            self.optim_list.append({'name':'instance_transformer', 'params': self.instance_model.parameters(), 'lr': 0.001}) #0.01
            # self.optim_list.append({'name':'instance_transformer', 'params': self.instance_model.mask_features_head.parameters(), 'lr': 0.01}) #0.001
            # self.optim_list.append({'name':'instance_transformer', 'params': self.instance_model.class_embed_head.parameters(), 'lr': 0.001}) #0.001
            # self.optim_list.append({'name':'instance_transformer', 'params': self.instance_model.decoder_norm.parameters(), 'lr': 0.001})
            if self.cfg['train']['mode2']['query_radius_optim']:
                # self.optim_list.append({'name':'quety_radius', 'params': self.radius, 'lr': 0.005})#0.1
                self.optim_list.append({'name':'quety_xyz', 'params': [self.instance_model.gaussians._xyz], 'lr': 0.005})#0.002(good)
                self.optim_list.append({'name':'quety_radius', 'params': [self.instance_model.gaussians._scaling], 'lr': 0.01})#0.01 0.002
                self.optim_list.append({'name':'quety_rotation', 'params': [self.instance_model.gaussians._rotation], 'lr': 0.002})#0.02 0.01 0.002
                self.optim_list.append({'name':'quety_feature', 'params': [self.instance_model.gaussians._features_token], 'lr': 0.005}) #0.01
                self.optim_list.append({'name':'quety_label_feature', 'params': [self.instance_model.gaussians._label_features_token], 'lr': 0.01})
        self.optim_occ = torch.optim.Adam(self.optim_list)
        # new optimer
        if self.optim_pose:
            self.pose_optimizer = torch.optim.Adam([{'name':'camera_pose', 'params': [self.R_cam], 'lr': pose_lr, 'eps':1e-3}])
            
    def update_optimizater_prune(self, query_mask):
        new_query = {}
        if 'instance' in self.grid.keys():
            for group in self.optim_occ.param_groups:
                if group["name"] in ["quety_xyz", "quety_radius", "quety_rotation", "quety_feature","quety_label_feature"]:
                    stored_state = self.optim_occ.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = stored_state["exp_avg"][query_mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][query_mask]

                    del self.optim_occ.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(group["params"][0][query_mask].requires_grad_(True))
                    self.optim_occ.state[group['params'][0]] = stored_state

                    new_query[group["name"]] = group["params"][0]
        return new_query
    
    def update_optimizater_add(self, idx, xyz, scaling, rotation, feature):
        tensors_dict = {"quety_xyz":xyz.reshape(-1,3), "quety_radius":scaling.reshape(-1,3), "quety_rotation":rotation.reshape(-1,4), "quety_feature":feature.reshape(1,-1), "quety_label_feature":123}
        new_query = {}
        if 'instance' in self.grid.keys():
            for group in self.optim_occ.param_groups:
                if group["name"] in ["quety_xyz", "quety_radius", "quety_rotation", "quety_feature", "quety_label_feature"]:
                    extension_tensor = tensors_dict[group["name"]]
                    stored_state = self.optim_occ.state.get(group['params'][0], None)
                    if stored_state is not None:

                        stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], stored_state["exp_avg"][idx:idx+1,:]), dim=0)
                        stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], stored_state["exp_avg_sq"][idx:idx+1,:]), dim=0)

                        del self.optim_occ.state[group['params'][0]]
                        group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                        self.optim_occ.state[group['params'][0]] = stored_state

                        new_query[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                        new_query[group["name"]] = group["params"][0]

        return new_query

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
            # u = torch.linspace(0.0 + 0.5 / N_importance, 1.0 - 0.5 / N_importance, steps=N_importance, device=weights.device)
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

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s, step):
        """
        Up sampling give a fixed inv_s
        """
        device = sdf.device
        batch_size, n_samples = z_vals.shape
        # pts = (
        #     rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        # )  # n_rays, n_samples, 3
        # radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        # inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        # sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1].clone(), z_vals[:, 1:].clone()
        prev_z_vals *= self.scale
        next_z_vals *= self.scale
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
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
    
    @ torch.no_grad()
    def grid_sample(self, rays, sample_lod, num_samples_list):
        grid_z_val = []
        rays_near_est = 0
        for level in sample_lod:
            result = self.octree.raytrace(rays, level, with_exit=True) #level
            ridx, pidx, depth_in_out = result.ridx, result.pidx, result.depth
            # near, far = rays.dist_min[ridx.tolist()], rays.dist_max[ridx.tolist()]
            # mask = depth_in_out[:,0]*self.scale<far+0.1
            # ridx, depth_in_out = ridx[mask], depth_in_out[mask]

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
            # rays_far = torch.min(rays_far, rays.dist_max[:,None]/self.scale)#0.2

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
    
    def raymarch(self, rays, depth_gt, level=None, num_samples=4, imp_num_sample_block=6):
        # mask camera rays and lidar rays
        ray_mask = (depth_gt>0)
        rays_cam = rays[~ray_mask]
        valid_mask = ray_mask.clone()
        valid_depth = depth_gt.clone()
        if self.iter_n>self.geometry_iter and (~ray_mask).sum() != 0:
            depth_cam = self.get_img_depth(rays_cam).squeeze()
            cam_hit_mask = depth_cam>0
            valid_mask[~ray_mask] = cam_hit_mask
            valid_depth[~ray_mask] = depth_cam
        valid_rays = rays[valid_mask]
        valid_depth = valid_depth[valid_mask]

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
                    # z_vals = torch.cat([z_vals, new_z_vals+0.05*torch.rand_like(new_z_vals)/self.scale], dim=-1)
                    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
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
            # coords = valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * z_val_new[..., :, None]
            # query_results = self.octree.blas.query(coords.reshape(-1, 3), self.octree.active_lods[0], with_parents=True)
            # mask = query_results.pidx[:,self.octree.active_lods[0]]<0
            # z_val_new[mask.reshape(-1,12)]=(torch.rand_like(z_val_new)*0.2+valid_depth[:,None].expand(valid_depth.shape[0],12) - 0.1)[mask.reshape(-1,12)]/self.scale

            z_vals = torch.cat([z_vals, z_val_new], dim=-1)
            z_vals, index = torch.sort(z_vals, dim=-1)
            n_samples = n_samples + sum(num_samples_lidar)

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
        samples = torch.addcmul(valid_rays.origins[:, None], valid_rays.dirs[:, None], z_vals[..., None])
        query_results = self.octree.blas.query(samples.reshape(-1, 3), self.octree.active_lods[0], with_parents=True)
        mask = (query_results.pidx[:,self.octree.active_lods[0]]<0).reshape(-1, n_samples)
        z_vals[mask] = (torch.rand_like(z_vals)*0.04+valid_depth[:,None].expand(valid_depth.shape[0],n_samples) - 0.02)[mask]/self.scale
        z_vals, index = torch.sort(z_vals, dim=-1)
        deltas = z_vals.diff(dim=-1,prepend=torch.zeros(valid_rays.origins.shape[0], 1, device=z_vals.device)+rays_near_est)
        samples = torch.addcmul(valid_rays.origins[:, None], valid_rays.dirs[:, None], z_vals[..., None])
        return samples, z_vals, deltas, valid_mask #, samples, z_vals, deltas #samples_d, z_vals_d, deltas_d
    
    def raymarch_ray_uniform(self, rays, depth_gt, level=None, num_samples=4, imp_num_sample_block=6):
        # mask camera rays and lidar rays
        ray_mask = (depth_gt>0)
        rays_cam = rays[~ray_mask]
        valid_mask = ray_mask.clone()
        valid_depth = depth_gt.clone()
        if self.iter_n>self.geometry_iter and (~ray_mask).sum() != 0:
            depth_cam = self.get_img_depth(rays_cam).squeeze()
            cam_hit_mask = depth_cam>0
            valid_mask[~ray_mask] = cam_hit_mask
            valid_depth[~ray_mask] = depth_cam
        valid_rays = rays[valid_mask]
        valid_depth = valid_depth[valid_mask]

        # surface guided sampling
        with torch.no_grad():
            num_samples_lidar = [50,0,0]#[10] # 4 [4,4,6] #[2 2 8] [2,2,2,3] #[4,8,6,4,2]
            num_samples_sur = [7,15] #5
            # num_samples_sur = [22,0]
            n_important = 25 #20 #8
            up_sample_steps = 5 #4
            if (valid_mask).sum() != 0:
                rays_far = valid_depth + 0.1 #0.2
                rays_near = valid_depth - 0.1 #0.2
                z_vals = torch.linspace(0, 1.0, num_samples_sur[0], device=valid_rays.origins.device)[None] + \
                        (torch.zeros(valid_rays.origins.shape[0], num_samples_sur[0], device=valid_rays.origins.device) / num_samples_sur[0])
                z_vals *= (rays_far - rays_near)[:,None]/self.scale
                z_vals += rays_near[:,None]/self.scale

                #new
                n_samples = num_samples_sur[0]
                # pts = (valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * z_vals[..., :, None])  # N_rays, N_samples, 3
                # sdf = self.sdf(pts).squeeze()
                # for i in range(int(num_samples_sur[1]/5)):
                #     new_z_vals = self.up_sample(valid_rays.origins, valid_rays.dirs, z_vals, sdf, 5, 64 * 2 ** i, i)
                #     # z_vals = torch.cat([z_vals, new_z_vals+0.05*torch.rand_like(new_z_vals)/self.scale], dim=-1)
                #     z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
                #     z_vals, index = torch.sort(z_vals, dim=-1)
                #     n_samples = n_samples + 5 

                #     pts = valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * new_z_vals[..., :, None]
                #     new_sdf = self.sdf(pts).squeeze()
                #     sdf = torch.cat([sdf, new_sdf], dim=-1)
                #     xx = (
                #         torch.arange(z_vals.shape[0])[:, None]
                #         .expand(-1, n_samples)
                #         .reshape(-1)
                #     )
                #     index = index.reshape(-1)
                #     sdf = sdf[(xx, index)].reshape(-1, n_samples)

            # lidar_n_samples = num_samples_sur.sum()
            rays_far = valid_depth + 2.0 #0.2
            rays_near = torch.zeros_like(valid_depth) + 2.0 #0.2
            z_val_new = torch.linspace(0, 1.0, num_samples_lidar[0], device=valid_rays.origins.device)[None] + \
                        (torch.zeros(valid_rays.origins.shape[0], num_samples_lidar[0], device=valid_rays.origins.device) / num_samples_lidar[0])
            z_val_new *= (rays_far - rays_near)[:,None]/self.scale
            z_val_new += rays_near[:,None]/self.scale
            rays_near_est = rays_near[:,None]/self.scale
            z_vals = torch.cat([z_vals, z_val_new], dim=-1)
            z_vals, index = torch.sort(z_vals, dim=-1)
            n_samples = n_samples + sum(num_samples_lidar)

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
        return samples, z_vals, deltas, valid_mask #, samples, z_vals, deltas #samples_d, z_vals_d, deltas_d
    
    def raymarch_ray(self, rays, depth_gt, observation, level=None, num_samples=4, imp_num_sample_block=6):
        # mask camera rays and lidar rays
        if self.depth_completion:
            raytrace_results = self.octree.blas.raytrace(rays, level, with_exit=True)
            ridx = raytrace_results.ridx.long()
            ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
            valid_mask = torch.zeros_like(rays.origins[:, 0], dtype=bool, device=rays.origins.device)
            valid_mask[ridx_hit] = True
            valid_rays = rays[valid_mask]
            valid_depth = depth_gt[valid_mask]
        else:
            ray_mask = (depth_gt>0)
            rays_cam = rays[~ray_mask]
            valid_mask = ray_mask.clone()
            valid_depth = depth_gt.clone()
            car_mask = None
            if self.pretrain or (self.iter_n>self.geometry_iter and (~ray_mask).sum() != 0):
               
                depth_cam = self.get_img_depth(rays_cam, num_samples=5, level=self.octree.active_lods[-1]).squeeze()
                cam_hit_mask = depth_cam>0
                valid_mask[~ray_mask] = cam_hit_mask
                valid_depth[~ray_mask] = depth_cam #TODO

            valid_rays = rays[valid_mask]
            valid_depth = valid_depth[valid_mask]

        # surface guided sampling
        # with torch.no_grad():
        num_samples_lidar = [10,0,0] #25
        num_samples_sur = [10] #[20] #[5,10]
        # num_samples_sur = [22,0]
        n_important = 10 #20 #8
        up_sample_steps = 1 #4
        if (valid_mask).sum() != 0:
            with torch.no_grad():
                ## =========================layor================================
                rays_far = valid_depth + 0.1 #5 #0.2
                rays_near = valid_depth - 0.1 #5 #0.2
                rays_near_est = rays_near[:,None]/self.scale
                z_vals = torch.linspace(0, 1.0, num_samples_sur[0], device=valid_rays.origins.device)[None] + \
                        (torch.zeros(valid_rays.origins.shape[0], num_samples_sur[0], device=valid_rays.origins.device) / num_samples_sur[0])
                z_vals *= (rays_far - rays_near)[:,None]/self.scale
                z_vals += rays_near[:,None]/self.scale
                n_samples = num_samples_sur[0]
                # #new
                # pts = (valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * z_vals[..., :, None])  # N_rays, N_samples, 3
                # sdf = self.sdf(pts).squeeze()
                # SAM_NUM = 10 #5
                # for i in range(int(num_samples_sur[1]/SAM_NUM)):
                #     new_z_vals = self.up_sample(valid_rays.origins, valid_rays.dirs, z_vals, sdf, SAM_NUM, 64 * 2 ** i, i)
                #     # z_vals = torch.cat([z_vals, new_z_vals+0.05*torch.rand_like(new_z_vals)/self.scale], dim=-1)
                #     z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
                #     z_vals, index = torch.sort(z_vals, dim=-1)
                #     n_samples = n_samples + SAM_NUM 

                #     pts = valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * new_z_vals[..., :, None]
                #     new_sdf = self.sdf(pts).squeeze()
                #     sdf = torch.cat([sdf, new_sdf], dim=-1)
                #     xx = (
                #         torch.arange(z_vals.shape[0])[:, None]
                #         .expand(-1, n_samples)
                #         .reshape(-1)
                #     )
                #     index = index.reshape(-1)
                #     sdf = sdf[(xx, index)].reshape(-1, n_samples)

                ## =========================octree================================
                # lidar_sample_lod = self.octree.active_lods[:1]
                # z_val_new, rays_near_est = self.grid_sample(valid_rays, lidar_sample_lod, num_samples_lidar)
                # z_vals = torch.cat([z_vals, z_val_new], dim=-1)
                # z_vals, index = torch.sort(z_vals, dim=-1)
                # n_samples = n_samples + sum(num_samples_lidar)
                # # z_vals, rays_near_est = self.grid_sample(valid_rays, lidar_sample_lod, num_samples_lidar)
                # # n_samples = sum(num_samples_lidar)         

                rays_far = 10
                rays_near = 0.5
                num = 15
                rays_near_est = rays_near/self.scale
                z_vals_new = torch.linspace(0, 1.0, num, device=valid_rays.origins.device)[None] + \
                        (torch.zeros(valid_rays.origins.shape[0], num, device=valid_rays.origins.device) / num)
                z_vals_new *= (rays_far - rays_near)/self.scale
                z_vals_new += rays_near/self.scale
                z_vals = torch.cat([z_vals, z_vals_new], dim=-1)
                z_vals, index = torch.sort(z_vals, dim=-1)
                n_samples = n_samples + num

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

                # pts = (valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * z_vals_new[..., :, None])  # N_rays, N_samples, 3
                # sdf = self.sdf(pts).squeeze()
                # for i in range(up_sample_steps):
                #     new_z_vals = self.up_sample(valid_rays.origins, valid_rays.dirs, z_vals_new, sdf, n_important // up_sample_steps,
                #         64 * 2 ** i, i)
                #     # z_vals = torch.cat([z_vals, new_z_vals+0.05*torch.rand_like(new_z_vals)/self.scale], dim=-1)
                #     z_vals_new = torch.cat([z_vals_new, new_z_vals], dim=-1)
                #     z_vals_new, index = torch.sort(z_vals_new, dim=-1)
                #     num = num + n_important // up_sample_steps
                #     pts = valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * new_z_vals[..., :, None]
                #     new_sdf = self.sdf(pts).squeeze()
                #     sdf = torch.cat([sdf, new_sdf], dim=-1)
                #     xx = (
                #         torch.arange(z_vals_new.shape[0])[:, None]
                #         .expand(-1, num)
                #         .reshape(-1)
                #     )
                #     index = index.reshape(-1)
                #     sdf = sdf[(xx, index)].reshape(-1, num)    

                # z_vals = torch.cat([z_vals, z_vals_new], dim=-1)
                # z_vals, index = torch.sort(z_vals, dim=-1)
                # n_samples = n_samples + num                
                    

            # deltas = z_vals.diff(dim=-1,prepend=torch.zeros(valid_rays.origins.shape[0], 1, device=z_vals.device)+rays_near_est)
            deltas = z_vals.diff(dim=-1,prepend=z_vals[:,0:1])
            samples = torch.addcmul(valid_rays.origins[:, None], valid_rays.dirs[:, None], z_vals[..., None])
            # with torch.no_grad():
            #     query_results = self.octree.blas.query(samples.reshape(-1, 3), self.octree.active_lods[0])
            #     pidx = query_results.pidx.reshape(-1, n_samples)
            #     #            #     # print(torch.sum(pidx>0,dim=-1).min())
            #     # too_small_ray_mask = torch.sum(pidx>0,dim=-1)<5
            #     # too_small_ray_idx = torch.nonzero(valid_mask)[too_small_ray_mask]
            #     # # valid_mask[too_small_ray_idx] = False
            #     # pidx[too_small_ray_mask] = -1
            #     mask = pidx>-1
            # z_vals = z_vals[mask][:,None]
            # num_hit_samples = z_vals.shape[0]
            # deltas = deltas[mask].reshape(num_hit_samples, 1)
            # samples = samples[mask]
            # ridx = torch.arange(0, pidx.shape[0], device=pidx.device)
            # ridx = ridx[..., None].repeat(1, n_samples)[mask]
            # boundary = spc_render.mark_pack_boundaries(ridx)
            # return ridx, samples, z_vals, deltas, boundary, valid_mask #, samples, z_vals, deltas #samples_d, z_vals_d, deltas_d
            return samples, z_vals, deltas, valid_mask
        else:
            return [], [], [], [] #, [], []
    
        
    def raymarch2(self, rays, depth_gt, level=None, num_samples=4, imp_num_sample_block=6):
        
        with torch.no_grad():
            # result = self.grid.raymarch(rays, raymarch_type='ray', num_samples=12, level=7)
            # ridx, samples, depth_samples, deltas, boundary = \
            # result.ridx, result.samples, result.depth_samples, result.deltas, result.boundary

            result = self.octree.raytrace(rays, level, with_exit=True) #level
            ridx, pidx, depth = result.ridx, result.pidx, result.depth

            near, far = rays.dist_min[ridx.tolist()], rays.dist_max[ridx.tolist()]
            mask = depth[:,0]*self.scale<far+0.3
            ridx, depth = ridx[mask], depth[mask]

            depth_samples = wisp_spc_ops.sample_from_depth_intervals(depth, num_samples)[...,None]
            deltas = depth_samples[...,0].diff(dim=-1, prepend=depth[...,0:1]).reshape(-1, 1)
            # deltas = depth_samples[...,0].diff(dim=-1, prepend=depth[...,0:1])

        samples = torch.addcmul(rays.origins.index_select(0, ridx)[:,None], 
                                rays.dirs.index_select(0, ridx)[:,None], depth_samples)
        
        boundary = wisp_spc_ops.expand_pack_boundary(spc_render.mark_pack_boundaries(ridx.int()), num_samples)
        # boundary_ray = spc_render.mark_first_hit(ridx_imp.int())
        return ridx, samples, depth_samples, deltas, boundary
    
    @ torch.no_grad()
    def get_img_depth(self, rays_g, level, num_samples):
        result = self.octree.raymarch(rays_g, raymarch_type='voxel', num_samples=num_samples, level=level)
        depth = torch.zeros(rays_g.shape[0], 1, device=self.device)
        ridx, deltas, depth_samples, samples, boundary = result.ridx, result.deltas, result.depth_samples, result.samples, result.boundary
        depth_samples = depth_samples*self.scale
        deltas = deltas*self.scale
        if ridx.shape[0]>0:
            ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
            sdfs = self.sdf(samples).reshape(-1,num_samples,1)
            weights = self.get_weight_2(sdfs, deltas, 500, 1, boundary) #voxel_weight2
    
            ray_depth = spc_render.sum_reduce(depth_samples.reshape(-1,1) * weights.reshape(-1,1), boundary)
            depth[ridx_hit.long(), :] = ray_depth
        return depth

    def get_img_depth_im(self, rays_g, level, num_samples):
        depth = torch.zeros(rays_g.shape[0], 1, device=self.device)
        with torch.no_grad():
            num_samples = [30,0]
            n_important = 30 #20 #8
            up_sample_steps = 3 #4

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
        z_vals = z_vals*self.scale
        deltas = deltas*self.scale
        samples = torch.addcmul(rays_g.origins[:, None], rays_g.dirs[:, None], z_vals[..., None])
        query_results = self.octree.blas.query(samples.reshape(-1, 3), self.octree.active_lods[0])
        pidx = query_results.pidx.reshape(-1, n_samples)
        mask = pidx>-1
        # ray_hit_mask = mask.sum(-1)>0
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
        depth[ridx, :] = ray_depth

        return depth
    
    def get_weight_2(self, sdfs, deltas, inv_s, cos_anneal_ratio, boundaries):
        if len(sdfs.shape)==3:
            sdfs = sdfs.squeeze(2)
            prev_sdf, next_sdf = sdfs[:, :-1], sdfs[:, 1:]
            mid_sdf = (prev_sdf + next_sdf) * 0.5
            mid_sdf = torch.hstack((mid_sdf,sdfs[:, -1:]))
            deltas_sdf = sdfs.diff(dim=-1, prepend=sdfs[:,0:1]).reshape(-1, 1)
        else:
            mid_sdf = (sdfs[:-1]+sdfs[1:])*0.5
            mid_sdf = torch.vstack((mid_sdf,sdfs[-1:]))
            deltas_sdf = spc_render.diff(sdfs, boundaries)
            
        cos_val = deltas_sdf / (deltas.reshape(-1,1) + 1e-5)
        cos_val = cos_val.clip(-1e3, 0.0)

        weight = neus_weights2(mid_sdf, deltas.reshape(-1,mid_sdf.size(1)), inv_s, cos_val.reshape(-1,mid_sdf.size(1)), boundaries, True)
        return weight

        # if len(sdfs.shape)==3:
        #     sdfs = sdfs.squeeze(2)
        #     prev_sdf, next_sdf = sdfs[:, :-1], sdfs[:, 1:]
        #     mid_sdf = (prev_sdf + next_sdf) * 0.5
        #     deltas_sdf = sdfs.diff(dim=-1, prepend=sdfs[:,0:1]).reshape(-1, 1)
        #     mid_sdf = torch.hstack((mid_sdf,sdfs[:,-1:]))
        # else:
        #     mid_sdf = (sdfs[:-1]+sdfs[1:])*0.5
        #     mid_sdf = torch.vstack((mid_sdf,sdfs[-1:]))
        #     deltas_sdf = spc_render.diff(sdfs, boundaries)
        
        # cos_val = deltas_sdf / (deltas.reshape(-1,1) + 1e-5)
        # cos_val = cos_val.clip(-1e3, 0.0)

        # weight, alpha = neus_weights_2(mid_sdf, deltas.reshape(-1,mid_sdf.size(1)), inv_s, cos_val.reshape(-1,mid_sdf.size(1)), boundaries, True)
        # return weight #, alpha
    

    def Render(self, rays_g, observation, train_data_list, iter_n, epoch_n, cos_anneal_ratio, flag=1, output='loss'): #output=["pre","loss"]
        num_steps = 4
        loss_semantic = 0
        loss_rgb = 0
        self.inv_s = torch.exp(self.s * 10.0).clip(1e-6, 1e6)
        level = self.octree.active_lods[0] #max
        if 'depth' in observation.keys():
            depth_gt = observation['depth'] 
        else:
            depth_gt = -1*torch.ones(rays_g.shape[-1]).to(self.device)
        # mask padding samples
        mask= [depth_gt != 0]
        depth_gt = depth_gt[mask]
        rays_g = rays_g[mask]
        image_mask = depth_gt < 0

        for key in observation.keys():
            if self.model_type == '3dgs' and key=='rgb':
                continue
            observation[key] = observation[key][mask]
        # self.uv_ = self.uv[mask]
        
        # ridx, samples, depths, deltas, boundary, valid_mask = self.raymarch_ray(rays_g, depth_gt, observation, level=level, num_samples=num_steps)
        samples, depths, deltas, valid_mask = self.raymarch_ray(rays_g, depth_gt, observation, level=level, num_samples=num_steps)
        boundary = None
        
        # if len(ridx)==0: # no inter ray
        #     if output=='pre':
        #         pre_out = {'samvit':torch.zeros(rays_g.shape[0], self.feat_dim, device=self.device)}
        #         pre_out["dino_pe"] = self.PE_model(observation["vu"].reshape(1, 1, -1, 2) * 2 - 1) if self.PEMAP else torch.zeros_like(pre_out['samvit']) 
        #         valid_mask = torch.zeros([rays_g.shape[0],1]).to(self.device)
        #         return pre_out, valid_mask
        #     else:
        #         return None
        
        if self.depth_completion:
            ray_mask = (observation['confidence'] == 1)[valid_mask]
        else:
            ray_mask = (depth_gt > 0)[valid_mask]
        if output=='loss':
            rays_g = rays_g[valid_mask]
            depth_gt = depth_gt[valid_mask]
            for key in observation.keys():
                if self.model_type == '3dgs' and key=='rgb':
                    continue
                observation[key] = observation[key][valid_mask]
            # self.uv_ = self.uv_[valid_mask]
        N = rays_g.shape[0]
        
        if output=='loss':
            if self.rgb_flag and self.model_type!='3dgs':
                rgb_gt = observation['rgb']
            # if self.feature_flag:
            #     feature_ob = observation['feature']
            if self.semantic_flag:
                semantic_gt = observation['semantic']
            if self.instance_flag:
                depth_gt_cam = observation['depth']
                instance_gt = observation['instance']
            if self.panoptic_flag:
                semantic_gt = observation['semantic']
                instance_gt = observation['instance']
                panoptic_gt = observation['panoptic'].squeeze()
            if self.normal_flag:
                normal_ob = observation['normal']
            if self.depth_completion:
                conf_ob = observation['confidence']

        frame_id = observation['frame_id']
            
        # # Get the indices of the ray tensor which correspond to hits
        # ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
        # # Compute the color and density for each ray and their samplesd
        # hit_ray_d = rays_g.dirs.index_select(0, ridx)
        # samples = samples.reshape(-1,1,3)
        grads = self.gradient(samples)
        if torch.isnan(grads).sum()>0:
            yx=1

        embeddings = None
        
        # delete none query
        if self.instance_flag and instance_gt.max()>0:
            self.token_train_iter += 1

            # # add
            # if self.token_train_iter%100==0:
            #     add_mask = self.query_assignment_record_add > 10
            #     # print(self.query_assignment_record_add.unique())
            #     if add_mask.sum()>0:
            #         for idx in torch.where(add_mask>0)[0]:
            #             if self.query_assignment_record_add_history[self.instance_model.gaussians._id[idx].int()]==0:
            #                 xyz = self.instance_model.gaussians._xyz[idx].clone()
            #                 scaling = self.instance_model.gaussians._scaling[idx].clone()
            #                 feature = self.instance_model.gaussians._features_token[idx].clone()
            #                 rot = self.instance_model.gaussians._rotation[idx].clone() #torch.FloatTensor([1,0,0,0]).to(self.device)
            #                 after_qurey = self.update_optimizater_add(idx, xyz, scaling, rot, feature)
            #                 self.instance_model.gaussians.add(after_qurey)
            #                 print(f"==================for {idx} add {self.instance_model.gaussians._xyz.shape[0]}=======================")
            #                 self.query_assignment_record_add_history[self.instance_model.gaussians._id[idx].int()] = 1
            #                 self.query_assignment_record_add_history = torch.hstack((self.query_assignment_record_add_history, torch.zeros_like(self.query_assignment_record[0])))
            #                 self.query_assignment_record = torch.hstack((self.query_assignment_record, torch.zeros_like(self.query_assignment_record[0])))

            #     self.query_assignment_record_add = torch.zeros(self.instance_model.gaussians._xyz.shape[0]).to(self.device)

            # prun_flag = False
            if self.token_train_iter%300==0 and self.cfg['train']['mode2']['delete_query'] and self.epoch>self.prun_epoch: #1
                query_mask = self.query_assignment_record>0 #>10 #50 10
                print(f"===============after assign {query_mask.sum()}========================")
                print(f"query_mask is {query_mask}")
                # if self.token_train_iter>=400:
                with torch.no_grad():
                    # all_pc
                    pcd = o3d.io.read_point_cloud(os.path.join(self.cfg['path']['proj_dir'], '../rgb/input.ply'))
                    pcd = pcd.voxel_down_sample(0.03)
                    coords = torch.FloatTensor(np.array(pcd.points)).to(self.device)
                    self.instance_corners_all = [(coords - self.origin) / self.scale]
                    ins_feat = self.grid['instance'](((self.instance_corners_all[0]+1.0)/2).reshape(-1, 3)) #self.grid['instance']
                    self.corner_id_all = []
                    for b in range(0,coords.shape[0],100000):
                        corner_id_all, self.thing_semantic_heat_map_all, _, _ = self.instance_model(
                                                                    [ins_feat[b:b+100000]], 
                                                                    [self.instance_corners_all[0][b:b+100000]],
                                                                    self.scale
                                                                ) 
                        self.corner_id_all.append(corner_id_all)
                    self.corner_id_all = torch.vstack(self.corner_id_all)

                    if self.epoch>0:
                        # # std
                        # for i in range(self.corner_id_all.shape[-1]):                            
                        #     binary_mask = self.corner_id_all[:,i].clone()>0.7 #self.instance_threthod
                        #     # quet_points = self.instance_corners_all[0][binary_mask,:].clone()
                        #     # if quet_points.shape[0] > 0:
                        #     #     rand_idx = torch.randint(0, quet_points.shape[0], size=[10000])
                        #     #     quet_points_dis = torch.norm(quet_points[rand_idx,None,:] - quet_points[None,rand_idx,:], dim=2)
                        #     #     std_deviation = torch.std(quet_points_dis)

                        #     # if binary_mask.sum() < 10 or std_deviation > 0.3:
                        #     if binary_mask.sum() < 10: #== 0:
                        #         # delete_q = self.query_update_map[i]
                        #         # query_mask[delete_q] = False
                        #         # query_now_mask[i] = False
                        #         query_mask[i] = False
                        #         print(f"==================std delet {i}=======================")
                        # # print(f"query_mask is {query_mask}")

                        # nms
                        vertice_query_mask = self.corner_id_all.transpose(1,0).detach().cpu().numpy() > self.instance_threthod
                        intersection = np.logical_and(vertice_query_mask[:,np.newaxis,:], vertice_query_mask[np.newaxis,:,:])
                        score1 = np.sum(intersection, axis=-1) / (vertice_query_mask.sum(axis=-1)[:,None] + 1e-5)
                        score2 = np.sum(intersection, axis=-1) / (vertice_query_mask.sum(axis=-1)[None,:] + 1e-5)
                        score = np.maximum(score1, score2)
                        score[np.tril_indices(score.shape[0], k=-1)] = 0
                        np.fill_diagonal(score, 0)
                        idx = np.where(score > 0.7) #0.7
                        for i in range(idx[0].shape[0]):
                            # make sure both query are exist
                            # if query_mask[self.query_update_map[idx[0][i]]] and  query_mask[self.query_update_map[idx[1][i]]]:
                            if query_mask[idx[0][i]] and  query_mask[idx[1][i]]:
                                q_match = [idx[0][i], idx[1][i]]
                                # delete_q = self.query_update_map[q_match[self.corner_id_all.transpose(1,0).detach().cpu().numpy()[q_match,:][:,intersection[q_match[0],q_match[1]]].sum(-1).argmin()]]
                                q_idx_now = q_match[vertice_query_mask[q_match,:].sum(-1).argmin()]
                                # delete_q = self.query_update_map[q_idx_now]
                                # query_mask[delete_q] = False
                                # query_now_mask[q_idx_now] = False
                                query_mask[q_idx_now] = False
                                print(f"==================nms delet {q_idx_now}=======================")
                    print(f"query number: {query_mask.sum()} \nquery_mask is {query_mask}")
                    self.query_assignment_record_add_history[self.instance_model.gaussians._id[query_mask].int().tolist()] = 1

                # with torch.no_grad():      
                #     # split 
                #     r_min = self.instance_model.gaussians.get_scaling.min(dim=-1).values #*self.scale
                #     query_update_map_inverse = {v:k for k,v in self.query_update_map.items()}
                #     for query_idx in range(query_now_mask.shape[0]):
                #         if query_now_mask[query_idx]:
                #             pc = self.instance_corners_all[0][vertice_query_mask[query_idx,:]]
                #             if pc.shape[0] > 50:
                #                 clustering = DBSCAN(eps=r_min[query_idx].item(), min_samples=50).fit(pc.cpu().numpy()) 
                #                 clusters = np.unique(clustering.labels_)
                #                 if (clusters>=0).sum() > 1:
                #                     yx = 1
                #                     for cluster in clusters[clusters>0]:
                #                         # activate a discard query
                #                         add_query_raw_idx = torch.nonzero(query_mask==False)[-1]
                #                         query_mask[add_query_raw_idx] = True
                #                         query_raw_idx = self.query_update_map[query_idx]
                #                         self.instance_model.query_feat.weight[add_query_raw_idx] = self.instance_model.query_feat.weight[query_raw_idx].clone()
                                        
                #                         pc1, pc2 = pc[clustering.labels_==cluster-1], pc[clustering.labels_==cluster]
                #                         center1, center2 = pc1.mean(dim=0), pc2.mean(dim=0)
                #                         radius1, radius2 = (pc1.max(dim=0)[0]-pc1.min(dim=0)[0])/2, (pc2.max(dim=0)[0]-pc2.min(dim=0)[0])/2
                #                         # radius1, radius2 = torch.ones([3]).to(self.device)/self.scale, torch.ones([3]).to(self.device)/self.scale
                #                         # add_query_now_idx = query_update_map_inverse[add_query_raw_idx.item()]
                #                         add_query_now_idx = ((torch.IntTensor(list(self.query_update_map.values())).to(self.device) - add_query_raw_idx)<0).sum()
                #                         self.instance_model.gaussians.split(query_idx, add_query_now_idx, center1, center2, radius1, radius2) 
                #                         query_now_mask = torch.hstack(( torch.hstack((query_now_mask[:add_query_now_idx], torch.ones([1]).bool().to(self.device) )),  query_now_mask[add_query_now_idx:]))
                #                         # self.query_update_map[self.instance_model.gaussians._xyz.shape[0]+1] = raw_query_idx
                #                         print(f"==================split for {query_raw_idx} add {add_query_raw_idx.item()}=======================")
                #     print(f"query number: {query_mask.sum()} \nquery_mask is {query_mask}")

                after_qurey = self.update_optimizater_prune(query_mask)
                self.instance_model.gaussians.prune(query_mask, after_qurey)
                self.query_assignment_record_add = self.query_assignment_record_add[query_mask]

                pcd = o3d.geometry.PointCloud()
                pc = self.instance_model.gaussians.get_xyz.detach()*self.scale + self.origin
                pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
                pcd.colors = o3d.utility.Vector3dVector(self.color_map[range(pc.shape[0])]/255.)
                o3d.io.write_point_cloud( os.path.join(self.cfg['path']['proj_dir'], f'ply/test_query.ply'), pcd)
                 
                # add_mask = self.query_assignment_record_add > 10
                # print(self.query_assignment_record_add.unique())
                # if add_mask.sum()>0:
                #     for idx in torch.where(add_mask>0)[0]:
                #         if self.query_assignment_record_add_history[self.instance_model.gaussians._id[idx].int()]==0:
                #             xyz = self.instance_model.gaussians._xyz[idx].clone()
                #             scaling = self.instance_model.gaussians._scaling[idx].clone()
                #             feature = self.instance_model.gaussians._features_token[idx].clone()
                #             rot = self.instance_model.gaussians._rotation[idx].clone() #torch.FloatTensor([1,0,0,0]).to(self.device)
                #             after_qurey = self.update_optimizater_add(idx, xyz, scaling, rot, feature)
                #             self.instance_model.gaussians.add(after_qurey)
                #             print(f"==================for {idx} add {self.instance_model.gaussians._xyz.shape[0]}=======================")
                #             self.query_assignment_record_add_history[self.instance_model.gaussians._id[idx].int()] = 1
                #             self.query_assignment_record_add_history = torch.hstack((self.query_assignment_record_add_history, torch.zeros_like(self.query_assignment_record[0])))
                    
                # self.query_update_map = {new_idx: old_idx for new_idx, old_idx in enumerate(np.where(query_mask.cpu().numpy())[0])}
                # self.query_assignment_record[...] = 0 #
                self.query_assignment_record = torch.zeros(self.instance_model.gaussians._xyz.shape[0]).to(self.device)
                self.query_assignment_record_add = torch.zeros(self.instance_model.gaussians._xyz.shape[0]).to(self.device)
                # self.instance_model.update_network(query_mask, self.query_update_map)
                # self.update_optimizater()

        # out = self.get_output(samples, channels=self.channels, view_dirs=hit_ray_d, embedding=embeddings, normals=(grads/self.scale).reshape(-1,3))
        image_mask_valid = image_mask[valid_mask]
        out = self.get_output(samples, channels=self.channels, embedding=embeddings, normals=(grads/self.scale).reshape(-1,3),image_mask=image_mask_valid)
        
        depths = depths.clone()*self.scale
        deltas = deltas.clone()*self.scale

        pre_out = {}
        if flag:
            # weights = self.get_weight_2(out['sdf'], deltas, inv_s, cos_anneal_ratio, boundary) 
            # weights, alphas = self.get_weight(rays_g.dirs[ridx.long()], samples.shape,out['sdf'], grads, deltas, None, boundary, self.inv_s, cos_anneal_ratio, exclusive=True)
            weights, alphas = self.get_weight(rays_g.dirs, samples.shape,out['sdf'], grads, deltas, None, boundary, self.inv_s, cos_anneal_ratio, exclusive=True)
            # depth img
            # ray_depth = spc_render.sum_reduce(depths.reshape(-1,1) * weights.reshape(-1,1), boundary)
            ray_depth = torch.sum(weights[...,None] * depths[...,None], dim=-2)
            # =================SAM ViT=================
            if self.feature_flag and output=='pre':
                #depth
                pre_out['depth'] = torch.zeros(N, 1, device=ray_depth.device)
                depth_img_valid = pre_out['depth'][valid_mask]
                depth_img_valid[ridx_hit.long(), :] = ray_depth
                pre_out['depth'][valid_mask] = depth_img_valid
                # feat_mask = ~ray_mask
                ray_feat = spc_render.sum_reduce(out['feature'].reshape(-1,self.feat_dim) * weights.reshape(-1,1), boundary)
                pre_out['samvit'] = torch.zeros(N, self.feat_dim, device=ray_feat.device)
                vit_img_valid = pre_out['samvit'][valid_mask]
                vit_img_valid[ridx_hit.long(), :] = ray_feat
                pre_out['samvit'][valid_mask] = vit_img_valid
                if self.PEMAP:
                    pre_out["dino_pe"] = self.PE_model(observation["vu"].reshape(1, 1, -1, 2) * 2 - 1)
                else:
                    pre_out['dino_pe'] = torch.zeros_like(pre_out['samvit'])
               
                return pre_out, valid_mask.reshape(-1,1)
            # =================reconstruction=================
            else:
                #depth
                # pre_out['depth'] = torch.zeros(N, 1, device=ray_depth.device)
                # pre_out['depth'][ridx_hit.long(), :] = ray_depth 
                pre_out['depth'] = ray_depth
                # normal
                if (self.normal_flag and self.iter_n>self.geometry_iter) or (self.gaussian is not None and self.gaussian.opt.normal_loss):
                    normals = F.normalize(grads/self.scale, p=2, dim=-1, eps=1e-5)
                    # ray_normal = spc_render.sum_reduce(normals.reshape(-1,3) * weights.reshape(-1,1), boundary)
                    ray_normal = torch.sum(weights[...,None] * normals, dim=-2)
                    # pre_out['normal'] = torch.zeros(N, 3, device=ray_normal.device)
                    # pre_out['normal'][ridx_hit.long(), :] = ray_normal
                    pre_out['normal'] = ray_normal
                # rgb
                if self.rgb_flag and (~ray_mask).sum()!=0 and self.model_type != '3dgs':
                    ray_rgb = spc_render.sum_reduce(out['rgb'].reshape(-1,3) * weights.reshape(-1,1), boundary)
                    pre_out['rgb'] = torch.zeros(N, 3, device=ray_rgb.device)
                    pre_out['rgb'][ridx_hit.long(), :] = ray_rgb
                if self.rgb_flag and self.model_type == '3dgs' and self.cfg['3dgs_mode'] != 'None':
                    uv = torch.cat([self.uv[i] for i in range(len(train_data_list))])
                    for i, idx in enumerate(train_data_list):
                        idx_3dgs, idx_nerf, need_nerf = self.gaussian.select_frame(idx)
                        if self.gaussian_train_mode == 'only_rgb':
                            self.gaussian.train_step(idx_3dgs)
                        else:
                            # if self.iter_n%100==0 or need_nerf:
                                # normal, depth = self.render_nerf(idx_nerf) 
                            depth = torch.zeros([self.cfg['img']['height'] * self.cfg['img']['width']]).to(self.device)
                            normal = torch.zeros([self.cfg['img']['height'] * self.cfg['img']['width'], 3]).to(self.device)
                            frame_mask = frame_id[image_mask_valid] == idx
                            depth[observation['uv'][image_mask_valid][frame_mask].long()] = pre_out['depth'][image_mask_valid][frame_mask].squeeze().detach()
                            depth = depth.reshape(-1, self.cfg['img']['width'])
                            if self.gaussian_train_mode == 'rgb_depth_normal':
                                R_w2c = SE3(self.vec_es[f'{idx_nerf}']).matrix()[:3,:3].T
                                normal_world = pre_out['normal'][image_mask_valid][frame_mask].squeeze().detach()
                                normal_cam = (R_w2c @ normal_world.T).T
                                normal[observation['uv'][image_mask_valid][frame_mask].long(),:] = normal_cam #.flip(dims=[-1])
                                normal = normal.reshape(-1, self.cfg['img']['width'],3)
                                self.gaussian.train_step(idx_3dgs, depth=depth, normal=normal.permute(2,0,1))  #
                            elif self.gaussian_train_mode == 'rgb_depth':
                                self.gaussian.train_step(idx_3dgs, depth=depth)  #
                # semantic
                if self.semantic_flag and semantic_gt.max()>0:
                    semantic_gt, instance_gt, depth_gt_cam = semantic_gt[image_mask_valid], instance_gt[image_mask_valid], depth_gt_cam[image_mask_valid]
                    if self.panoptic_flag:
                        panoptic_gt = panoptic_gt[image_mask_valid]

                    # self.semantic_mask = torch.logical_and(semantic_gt>0, ~ray_mask)  
                    self.semantic_mask = semantic_gt>0
                    # ## 2d softmax
                    # ray_semantic = spc_render.sum_reduce(out['semantic'].squeeze() * weights.reshape(-1,1), boundary)
                    # semantic = torch.nn.functional.softmax(ray_semantic, dim=-1)

                    ## 3d softmax
                    # semantic = spc_render.sum_reduce(torch.nn.functional.softmax(out['semantic'].squeeze(), dim=-1) * weights.detach().reshape(-1,1), boundary)
                    semantic = torch.sum(weights[image_mask_valid,:,None].detach() * torch.nn.functional.softmax(out['semantic'].reshape(image_mask_valid.sum(), weights.shape[1],-1), dim=-1), dim=-2)
                    
                    # class_frequencies = torch.bincount(semantic_gt[self.semantic_mask], minlength=self.sem_class_num)/semantic_gt[self.semantic_mask].shape[0]
                    # class_weights = (1.0 / class_frequencies).detach()
                    # class_weights[class_weights>1e5] = 0
                    # CrossEntropyLoss = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=0) #
                    CrossEntropyLoss = torch.nn.CrossEntropyLoss(ignore_index=0)
                    # pre_out['sem'] = torch.zeros(N, semantic.shape[-1], device=semantic.device)
                    # pre_out['sem'][ridx_hit.long(), :] = semantic
                    pre_out['sem'] = semantic
                # instance
                if self.instance_flag and instance_gt.max()>0:
                    # instance = spc_render.sum_reduce(out['instance'].reshape(weights.shape[0],-1) * weights.detach().reshape(-1,1), boundary)
                    instance = torch.sum(weights[image_mask_valid,:,None].detach() * out['instance'].reshape(image_mask_valid.sum(), weights.shape[1],-1), dim=-2)
                    ## 3d softmax
                    # instance = spc_render.sum_reduce(torch.nn.functional.softmax(out['instance'].squeeze(), dim=-1) * weights.detach().reshape(-1,1), boundary)
                    # class_frequencies = torch.bincount(instance_gt[~ray_mask].squeeze(), minlength=self.cfg['decoder']['instance']['output_ch'])/instance_gt[~ray_mask].shape[0]
                    # class_weights = (1.0 / class_frequencies).detach()
                    # class_weights[class_weights>1e5] = 0
                    # CrossEntropyLoss_ins = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=0) #
                   
                    # pre_out['ins'] = torch.zeros(N, instance.shape[-1], device=instance.device)
                    # pre_out['ins'][ridx_hit.long(), :] = instance
                    pre_out['ins'] = instance

                    #                    self.assignment_all = []
                    loss_instance_clustering_dice, loss_instance_clustering_bce, distance_loss, loss_semantic_thing = 0, 0, 0, 0 
                    if epoch_n>0 or True:
                        visual_flag = True #False #
                        frame_id = frame_id[image_mask[valid_mask]] if frame_id.shape[0]>1 else frame_id
                        self.assignment_loss = 0
                        CrossEntropyLoss_thing = torch.nn.CrossEntropyLoss(ignore_index=0) #ignore_index=len(self.Thing_class)
                        # targets_thing_labels = []
                        for img_idx in frame_id.unique().tolist():
                            img_mask = frame_id==img_idx if frame_id.shape[0]>1 else torch.ones(instance.shape[0], dtype=torch.bool)
                            # instance_features = pre_out['ins'][~ray_mask][img_mask] 
                            # instance_gt_img = instance_gt[~ray_mask][img_mask]
                            # panoptic_gt_ray = panoptic_gt[~ray_mask]
                            # panoptic_gt_img = panoptic_gt_ray[img_mask]
                            instance_features = pre_out['ins'][img_mask] 
                            instance_gt_img = instance_gt[img_mask]
                            depth_gt_img = depth_gt_cam[img_mask]
                            
                            if instance_gt_img.shape[0]==0 or instance_gt_img.max() == 0:
                                loss_instance_clustering_ = 0.
                                dice_loss, bce_loss = 0., 0.
                            else:
                                self.frame_id = img_idx
                                self.loss_thing_mask = instance_gt_img>0
                                loss_mask = torch.logical_and(instance_gt_img>=0, -depth_gt_img[:,None]>0.3)

                                targets_thing_label = semantic_gt[img_mask].cpu().apply_(lambda x: self.label2label_mapping[x]) #[~ray_mask]
                                targets_thing_label[instance_gt_img.squeeze()==0] = 0
                                # targets_thing_labels.append(targets_thing_label)
                                pc_g = rays_g.origins[image_mask_valid][img_mask] + rays_g.dirs[image_mask_valid][img_mask] * (-depth_gt_img[:,None])/self.scale
                                dice_loss, bce_loss, virtual_gt_labels = self.calculate_instance_clustering_loss(instance_gt_img[loss_mask], instance_features[loss_mask.squeeze()], pc_g[loss_mask.squeeze()], targets_thing_label=targets_thing_label[loss_mask.squeeze()])
                                # loss_instance_clustering_ = 
                                # visual_flag = False
                                # virtual_gt_labels[virtual_gt_labels<0] = 0 - len(self.cfg['Stuff_class']) - 1
                                loss_mask = instance_gt_img[-depth_gt_img[:,None]>0.3] >0
                                # panoptic_gt_img[instance_gt_img.squeeze()>0] = (virtual_gt_labels + len(self.cfg['Stuff_class']) + 1)[instance_gt_img.squeeze()>0]
                                if self.panoptic_flag:
                                    panoptic_gt_img = panoptic_gt[img_mask]
                                    if panoptic_gt.max()>=instance_features.shape[-1]:
                                        yx = 1
                                    panoptic_gt_img[-depth_gt_img<=0.3] = 0
                                    temp = panoptic_gt_img[-depth_gt_img>0.3]
                                    temp[loss_mask.squeeze()] = (virtual_gt_labels + len(self.cfg['Stuff_class']) + 1)[loss_mask.squeeze()]
                                    if virtual_gt_labels.max()>=instance_features.shape[-1]:
                                        yx = 1
                                    panoptic_gt_img[-depth_gt_img>0.3] = temp
                                    panoptic_gt[img_mask] = panoptic_gt_img

                                target_object = torch.full(self.thing_semantic_heat_map.shape[:1], 0, dtype=torch.int64, device=self.device)   
                                for i in self.query2targetClass.keys(): # new_label_pred 2 new_semantic_gt
                                    target_object[i] = self.query2targetClass[i]
                                loss_semantic_thing += CrossEntropyLoss_thing(self.thing_semantic_heat_map.softmax(-1), target_object)
                                    
                            loss_instance_clustering_dice += dice_loss
                            loss_instance_clustering_bce += bce_loss
                            # if panoptic_gt[img_mask].max()-len(self.cfg['Stuff_class'])-1 >= instance_features.shape[-1]:
                            #     yx = 1

                        with torch.no_grad():
                            if visual_flag and (self.token_train_iter%300 == 0):
                                # # all_pc
                                if len(self.instance_corners_all) == 0:
                                    # all_pc
                                    pcd = o3d.io.read_point_cloud(os.path.join(self.cfg['path']['proj_dir'], '../rgb/input.ply'))
                                    pcd = pcd.voxel_down_sample(0.03)
                                    coords = torch.FloatTensor(np.array(pcd.points)).to(self.device)
                                    self.instance_corners_all = [(coords - self.origin) / self.scale]
                                    ins_feat = [self.grid['instance'](((self.instance_corners_all[0]+1.0)/2).reshape(-1, 3))] #self.grid['instance']
                                    self.corner_id_all, self.thing_semantic_heat_map_all, _, _ = self.instance_model(
                                                                                ins_feat, 
                                                                                self.instance_corners_all,
                                                                                self.scale
                                                                            ) 
                                    
                                pcd = o3d.geometry.PointCloud()
                                for i in range(len(self.instance_corners_all)):
                                    pcd.points = o3d.utility.Vector3dVector(self.instance_corners_all[i].cpu().numpy())
                                    # color_prob = self.corner_id.detach().cpu().sigmoid().numpy()
                                    color_prob = self.corner_id_all.detach().cpu().numpy()
                                    corner_label = torch.argmax(self.corner_id_all @ self.thing_semantic_heat_map_all, -1).detach().cpu()
                                    thing_class2semantic_label = ([0]+self.Thing_class)
                                    corner_label.apply_(lambda x: thing_class2semantic_label[x]).numpy() 
                                    v_colors = np.vstack([id2label[semID].color for semID in corner_label.tolist()])
                                    for j in range(self.corner_id_all.shape[-1]):
                                        pcd.colors = o3d.utility.Vector3dVector(color_prob[:,j][:,None].repeat(3,-1) * self.color_map[j]/255.) #
                                        o3d.io.write_point_cloud( os.path.join(self.cfg['path']['proj_dir'], f'ply/test_octree_wo_assign_{i}_{j}_epoch{epoch_n}.ply'), pcd)
                                        # mask = color_prob[:,j] > self.instance_threthod #0.8
                                        # # pcd.colors = o3d.utility.Vector3dVector(mask[:,None].repeat(3,-1) * octree_color/255.)
                                        # # pcd.colors = o3d.utility.Vector3dVector(mask[:,None].repeat(3,-1) * self.color_map[j]/255. )
                                        # pcd.colors = o3d.utility.Vector3dVector(mask[:,None].repeat(3,-1) * v_colors/255.) 
                                        # o3d.io.write_point_cloud( os.path.join(self.cfg['path']['proj_dir'], f'ply/mask_octree_wo_assign_{i}_{j}.ply'), pcd)
                            
                                pcd.points = o3d.utility.Vector3dVector(self.query_position)
                                pcd.colors = o3d.utility.Vector3dVector(self.color_map[range(self.query_position.shape[0])]/255.)
                                o3d.io.write_point_cloud( os.path.join(self.cfg['path']['proj_dir'], f'ply/test_query.ply'), pcd)
                                
                # panoptic
                if self.panoptic_flag and self.cfg['train']['mode2']['panoptic_term'] and panoptic_gt.max()>0 and self.epoch>3: #self.prun_epoch: #self.token_train_iter>300:
                    # self.panoptic_mask = torch.logical_and(panoptic_gt>0, ~ray_mask) 
                    self.panoptic_mask = panoptic_gt>0
                    # panoptic = spc_render.sum_reduce(torch.nn.functional.softmax(out['panoptic'].squeeze(), dim=-1) * weights.detach().reshape(-1,1), boundary)
                    panoptic = torch.sum(weights[image_mask_valid,:,None].detach() * torch.nn.functional.softmax(out['panoptic'].reshape(image_mask_valid.sum(), weights.shape[1],-1), dim=-1), dim=-2)
                    # # CrossEntropyLoss_panoptic = torch.nn.CrossEntropyLoss(ignore_index=0)
                    # pre_out['pan'] = torch.zeros(N, panoptic.shape[-1], device=panoptic.device)
                    # pre_out['pan'][ridx_hit.long(), :] = panoptic
                    pre_out['pan'] = panoptic

                # loss = self.cal_loss(pre_out, out, ridx, samples, depths, boundary, grads)
                loss = {}
                loss['feature'], loss['rgb'], loss['semantic'], loss['instance'], loss['normal'], loss['near'], loss['far'], loss['eikonal'] = 0, 0, 0, 0, 0, 0, 0, 0
                loss['eikonal'] = self.get_eikonal_loss(out['sdf'], grads, samples, weights, rays=rays_g)
                # loss['eikonal'] = self.get_eikonal_loss2(out['sdf'], grads, samples, weights)
                # loss['smooth'] = self.get_smoothing_loss2(rays_g.origins[ridx_hit.long(), :], rays_g.dirs[ridx_hit.long(), :], pre_out['depth'][ridx_hit.long(), :]/self.scale)
                loss['smooth'] = self.get_smoothing_loss2(rays_g.origins, rays_g.dirs, pre_out['depth']/self.scale)

                # loss['depth'] = torch.log(torch.abs(depth.squeeze()[ray_mask] - depth_gt[ray_mask])+1).mean() #L1
                depth_mask = torch.logical_and(depth_gt>0.5, depth_gt<10)
                loss['depth'] = torch.abs(pre_out['depth'].squeeze()[depth_mask] - depth_gt[depth_mask]).mean() #L1
                # # loss['depth'] = ((1/depth_gt[ray_mask]) * torch.abs(pre_out['depth'].squeeze()[ray_mask] - depth_gt[ray_mask])).mean() 
                # # if epoch_n < 3:
                # # loss['near'], loss['far'] = self.get_sdf_loss2(depths, depth_gt, ridx, boundary, out['sdf'].squeeze(-1), samples)
                loss['near'], loss['far'] = self.get_sdf_loss(depths[depth_mask], depth_gt[depth_mask,None], out['sdf'][depth_mask].squeeze(-1))
                # loss['near'], _ = self.get_sdf_loss(depths[depth_mask], depth_gt[depth_mask,None], out['sdf'][depth_mask].squeeze(-1))
                if self.depth_completion:
                    conf_mask = torch.logical_and(conf_ob>0.98, ~ray_mask)
                    loss['img_depth'] = (torch.abs(depth_gt[conf_mask] - depth.squeeze()[conf_mask])).mean()
                if self.normal_flag and self.iter_n>self.geometry_iter:
                    render_mask = normal_ob[ray_mask].norm(2,1)>0
                    loss['normal'] = self.get_normal_loss(pre_out['normal'][ray_mask], normal_ob[ray_mask], render_mask=render_mask)
                if self.rgb_flag and (~ray_mask).sum()!=0 and self.model_type != '3dgs':
                    rgb_mask = torch.logical_and(~ray_mask, rgb_gt.norm(2,1)>0)
                    loss['rgb'] = torch.nn.functional.mse_loss(pre_out['rgb'][rgb_mask], rgb_gt[rgb_mask], reduction='none').sum(1).mean()
                if self.semantic_flag and semantic_gt.max()>0:
                    # label2label_mapping = {0:0, 1:1, 5:2, 6:3, 7:4, 8:5, 9:6}
                    # semantic_target = semantic_gt[self.semantic_mask].cpu().apply_(lambda x: label2label_mapping[x]).to(self.device)
                    semantic_target = semantic_gt[self.semantic_mask]
                    loss['semantic'] = CrossEntropyLoss(pre_out['sem'][self.semantic_mask], semantic_target.to(torch.long))
                    # if iter_n%100 == 0:
                    #     print(pre_out['sem'][self.semantic_mask].max(-1).indices.unique())
                    #     print(semantic_gt[self.semantic_mask].unique())
                if self.instance_flag and instance_gt.max()>0:
                    # loss['instance'] = CrossEntropyLoss_ins(pre_out['ins'][~ray_mask], instance_gt[~ray_mask].squeeze().to(torch.long)) + loss_instance_clustering
                    # if self.token_train_iter>300:
                    #     loss['instance'] = loss_instance_clustering_bce + loss_semantic_thing
                    # else:
                    loss['instance'] = loss_instance_clustering_dice + loss_instance_clustering_bce + loss_semantic_thing #+ entropy_loss + 0.1*self.assignment_loss #self.query_loss
                    # loss['entropy'] = entropy_loss
                    loss['query'] = self.query_loss
                    # loss['assignment'] = self.assignment_loss
                    # if iter_n%100 == 0:
                    #     print(pre_out['ins'].max(-1).indices.unique()) #[~ray_mask]
                    #     print(instance_gt.unique()) #[~ray_mask]
                    #     # print(pre_out['thing_sem'][~ray_mask].max(-1).indices.unique())
                if self.panoptic_flag and panoptic_gt.max()>0 and 'pan' in pre_out.keys():
                    panoptic_target = panoptic_gt[self.panoptic_mask]
                    # # panoptic_target[panoptic_target>6]=0
                    # class_frequencies = torch.bincount(panoptic_target, minlength=pre_out['pan'].shape[-1])/panoptic_target.shape[0]
                    # class_weights = (1.0 / class_frequencies).detach()
                    # class_weights[class_weights>1e5] = 0
                    CrossEntropyLoss_panoptic = torch.nn.CrossEntropyLoss(ignore_index=0) #weight=class_weights, 
                    # loss['instance'] += 10 * CrossEntropyLoss_panoptic(pre_out['pan'][self.panoptic_mask], panoptic_target.to(torch.long))
                    if pre_out['pan'][self.panoptic_mask].shape[-1] <= panoptic_target.max():
                        print(pre_out['pan'][self.panoptic_mask].shape[-1])
                        print(panoptic_target.max())
                    loss['instance'] = loss_instance_clustering_bce + loss_semantic_thing + 10*CrossEntropyLoss_panoptic(pre_out['pan'][self.panoptic_mask], panoptic_target.to(torch.long))
                    loss['semantic'] = 0
                    # if iter_n%100 == 0:
                    #     print("--------")
                    #     print(pre_out['pan'][self.panoptic_mask].max(-1).indices.unique())
                    #     print(panoptic_gt[self.panoptic_mask].unique())

                return loss          

    ###############################mask 3d################################
    @torch.no_grad()
    def calculate_instance_ce_dice_loss(self, targets, outputs, targets_thing_label, pc_g):
        new_labels = -1 * torch.ones_like(targets) 
        # with autocast(enabled=False):
        out_mask = outputs.float()
        tgt_binary_mask = []
        labels = targets.unique().cpu().tolist()
        for i in range(targets.unique().shape[0]):
            if i==0:
                continue
            binary_mask = targets==labels[i]
            tgt_binary_mask.append(binary_mask.float())
        tgt_binary_mask = torch.stack(tgt_binary_mask,-1)
        # # tgt_binary_mask = (torch.arange(targets.unique().shape[0]).to(self.device) == (targets[..., None]-1)).float()
        # # Compute the focal loss between masks
        cost_mask = batch_sigmoid_ce_loss(
            torch.clip(out_mask.transpose(1,0),0,1), tgt_binary_mask.transpose(1,0)
        )

        # Compute the dice loss between masks
        cost_dice = batch_dice_loss(
            out_mask.transpose(1,0), tgt_binary_mask.transpose(1,0)
        )

        # Compute the semantic loss between masks
        if labels[0]==0:
            labels = labels[1:]
        if targets_thing_label is not None:
            targets_thing_label_ = [targets_thing_label[torch.where(targets==ins)[0][0]].item() for ins in labels]
            cost_class = - self.thing_semantic_heat_map.softmax(-1)[:, targets_thing_label_]
        else:
            cost_class = 0

        # Final cost matrix
        rand_query_id = torch.randperm(cost_dice.shape[0])
        C = cost_dice+cost_mask+0.5*(cost_class-cost_class.min()) #[rand_query_id,:] #5*cost_dice + cost_class#2 * cost_dice + (cost_class + 1.) #
        # C = C.reshape(self.instance_model.num_queries,-1).detach().cpu().numpy()
        # print(f"se: {cost_class.max()}, dice: {cost_dice.max()}")


        assignment = scipy.optimize.linear_sum_assignment(C.detach().cpu().numpy().transpose(1,0))
        # assignment = [assignment[0], rand_query_id[assignment[1]].numpy()] 

        # if self.epoch <= 1:
        #     check = (C[assignment[1], assignment[0]] == C[:,assignment[0]].min(0).values).cpu().numpy()
        #     if check.min() == 0:
        #         idxs = np.where(check==0)[0]
        #         for idx in idxs:
        #             # if C[:,assignment[0][idx]].min(0).values < 0.9:
        #             self.query_assignment_record_add[C[:,assignment[0][idx]].min(0).indices] += 1
        #             assignment[1][idx] = -1
        #             # # add token
        #             # pc_mask = pc_g[targets==(assignment[0][idx]+1).item()]
        #             # xyz = pc_mask.mean(0)
        #             # scaling = torch.max(torch.cdist(pc_mask, pc_mask)) * torch.ones([3]).to(self.device)
        #             # feature = self.instance_model.gaussians._features_token[C[:,assignment[0][idx]].min(0).indices]
        #             # self.instance_model.gaussians.add(xyz, scaling, feature)
        #             # # update assignment
        #             # # assignment[1][idx] = self.instance_model.gaussians._xyz.shape[0]-1

        #             # self.query_assignment_record = torch.hstack((self.query_assignment_record, torch.zeros_like(self.query_assignment_record[0])))
        #             # print(f"query num: {self.instance_model.gaussians._xyz.shape[0]}")
                    
        if self.epoch >= 1:
            check = (cost_dice[assignment[1], assignment[0]] == cost_dice[:,assignment[0]].min(0).values).cpu().numpy()
            if check.min() == 0:
                idxs = np.where(check==0)[0]
                for idx in idxs:
                    if cost_dice[:,assignment[0][idx]].min(0).values < 0.9:
                        self.query_assignment_record_add[cost_dice[:,assignment[0][idx]].min(0).indices] += 1
                    # else:
                    #     assignment[1][idx] = -1
        
        assignment[1][cost_dice.cpu().numpy()[assignment[1], assignment[0]]> 0.9] = -1
           
        for aidx, lidx in enumerate(assignment[0]):
            new_labels[targets == labels[lidx]] = assignment[1][aidx] # self.query_update_map[]
            # self.query_assignment_record[self.query_update_map[assignment[1][aidx]]] += 1
            self.query_assignment_record[assignment[1][aidx]] += 1

        # cross = torch.nn.CrossEntropyLoss(reduction='none')
        # C_torch = 1-cost_dice.transpose(1,0)
        # self.assignment_loss += cross(C_torch/C_torch.sum(dim=-1)[:,None], torch.from_numpy(assignment[1]).to(self.device).to(torch.long)).mean()

        targetID2query = {i:j for i, j in zip(labels, assignment[1])} # {gt: new_label(query)}

        # targetID2targetClass = {i.item():j.item() for i, j in zip(targets, targets_thing_label)}
        self.query2targetClass = {targetID2query[i.item()]:j.item() for i, j in zip(targets, targets_thing_label) if i.item() in targetID2query.keys() }
        # self.query2SemanticLabel = {targetID2query[i.item()]:j.item() for i, j in zip(targets, targets_thing_label) if i.item() in targetID2query.keys() }
                    
        return new_labels

    #############################pamoptic lifting##############################
    def calculate_instance_clustering_loss(self, labels_gt, instance_features, pc_g, targets_thing_label=None, confidences=1):

        virtual_gt_labels = self.calculate_instance_ce_dice_loss(labels_gt, instance_features, targets_thing_label, pc_g)
        # virtual_gt_labels = labels_gt.clone() - 1
        # if torch.any(virtual_gt_labels != predicted_labels):  # should never reinforce correct labels
            # class_frequencies = torch.bincount(virtual_gt_labels, minlength=self.instance_num)/virtual_gt_labels.shape[0]
            # class_weights = (1.0 / class_frequencies).detach()
            # class_weights[class_weights>1e4] = 0
            # self.loss_instances_cluster = torch.nn.CrossEntropyLoss(reduction='none') #weight=class_weights,  , ignore_index=0

        tgt_binary_mask = (torch.arange(instance_features.shape[-1]).to(self.device) == virtual_gt_labels[..., None]).float()
        virtual_gt_labels_list = virtual_gt_labels.unique().tolist()
        if virtual_gt_labels_list[0]==-1:
            virtual_gt_labels_list = virtual_gt_labels_list[1:]
        inputs = instance_features[:,virtual_gt_labels_list]
        targets = tgt_binary_mask[:,virtual_gt_labels_list].float()
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none").mean(0).sum(0)/len(virtual_gt_labels_list)
        # numerator = 2 * (inputs.sigmoid() * targets).sum(0)
        # denominator = inputs.sigmoid().sum(0) + targets.sum(0)
        numerator = 2 * (inputs * targets).sum(0)
        denominator = inputs.sum(0) + targets.sum(0)
        dice_loss = (1 - (numerator + 1e-6) / (denominator + 1e-6)).sum(0)/len(virtual_gt_labels_list)
        # print(f"ce: {ce_loss}, dice: {dice_loss}")

        return dice_loss, ce_loss, virtual_gt_labels # +10*dice_loss
        # return 0, virtual_gt_labels

    @torch.no_grad()
    def create_virtual_gt_with_linear_assignment(self, labels_gt, predicted_scores):
        labels = sorted(torch.unique(labels_gt).cpu().tolist())[:predicted_scores.shape[-1]]
        # labels = [e for e in labels if e not in {0}]
        new_labels = torch.zeros_like(labels_gt) 
        if len(labels)>0:
            # predicted_probabilities = predicted_scores[:,1:] #torch.softmax(predicted_scores, dim=-1)
            predicted_probabilities = predicted_scores
            cost_matrix = np.zeros([len(labels), predicted_probabilities.shape[-1]])
            for lidx, label in enumerate(labels):
                cost_matrix[lidx, :] = -(predicted_probabilities[labels_gt == label, :].sum(dim=0) / ((labels_gt == label).sum() + 1e-4)).cpu().numpy()
            assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
            # print(labels)
            # print(assignment[1])
            self.assignment_all.append(assignment[1])
            for aidx, lidx in enumerate(assignment[0]):
                new_labels[labels_gt == labels[lidx]] = assignment[1][aidx] #+ 1   
        return new_labels
        #############################pamoptic lifting##############################


    def gradient(self, samples):
        # eps = (0.5*0.6**self.epoch)/self.scale 
        # grad = finitediff_gradient(samples, self.sdf, eps=0.005/self.scale) #
        # samples_w = samples * self.scale + self.origin
        grad = autodiff_gradient(samples, self.sdf, self.origin, self.scale) # /self.scale
        return grad

    #-------------------------------------loss-------------------------------------
    def get_eikonal_loss(self, sdf_val, grads, pts, weights, rays=None): #world_bound_mask   
        eikonal_loss = torch.tensor([0]).float()
        eikonal_weights = sdf_val.reshape(-1).detach().abs() + 1e-2
        # eikonal_weights = sdf_val.reshape(-1).detach()
        eik_mask = eikonal_weights >=0 #>0.2
        # eikonal_loss = torch.square(grads.norm(dim=-1).reshape(-1) - 1.).reshape(-1).sum() / (grads.reshape(-1,3).shape[0])
        if eik_mask.sum()>0:
            eikonal_loss = (torch.square(grads.norm(dim=-1).reshape(-1)[eik_mask]/self.scale - 1.).reshape(-1) * eikonal_weights[eik_mask]).sum() / eikonal_weights[eik_mask].sum()
            # eikonal_loss += (weights.reshape(-1)[~eik_mask] * eikonal_weights[~eik_mask]).sum()/eikonal_weights[~eik_mask].sum()
        return eikonal_loss
    
    def get_eikonal_loss2(self, z_vals, target_d, grads, weights):
        truncation = 0.2
        eik_mask = ((target_d - z_vals).abs() <= truncation).reshape(-1)
        ray_w_num = torch.count_nonzero(target_d)
        eikonal_loss = torch.square(grads.norm(dim=-1).reshape(-1)[eik_mask]/self.scale-1.).mean()
        eikonal_loss = eikonal_loss + 2*weights.reshape(-1)[~eik_mask].mean()
        return eikonal_loss


    def get_smoothing_loss2(self, rays_o, rays_d, rendered_depth):
        truncation = 0.15 #0.16 0.2
        smoothness_std = 0.001/self.scale #0.04
        normal_regularisation_loss = torch.zeros([]).to(self.device)

        sample = 2 #4
        corners = spc_ops.unbatched_get_level_points(self.octree.blas.points, self.octree.blas.pyramid, self.octree.active_lods[-1])
        # coords = wisp_spc_ops.sample_spc(corners, self.active_lods[0], sample)
        corners = (corners[...,None,:3]+torch.rand(corners.shape[0], sample, 3, device=self.device)).reshape(-1,3)
        coords = (corners/2**self.octree.active_lods[-1])*2-1
        world = coords # + torch.rand_like(coords) #((coords + torch.rand_like(coords)) * voxel_sizes[1] + volume_origin).unsqueeze(0)
        # world = 2*(torch.rand(Nx*Nx*Nx,3)-0.5).to(self.device)
        surf = rays_o + rays_d * rendered_depth
        # points_o3d = o3d.geometry.PointCloud()
        # points_o3d.points = o3d.utility.Vector3dVector(surf.cpu().detach().numpy())
        # o3d.io.write_point_cloud(os.path.join("01_loam_map_init", "ply/surf.ply"), points_o3d)
        # surf_mask = torch.logical_and(((surf + 1)>=0).sum(1)==3, ((surf - 1)<=0).sum(1)==3).reshape(-1)
        pidx = self.octree.blas.query(surf.reshape(-1, 3), self.octree.active_lods[-1], with_parents=True).pidx[...,-1]
        surf_mask = pidx>-1
        if surf_mask.any().item():
            surf = surf[surf_mask,:] #.unsqueeze(0)

            # weight = torch.cat([torch.ones(world.shape[:-1], device=world.device) * 0.1, torch.ones(surf.shape[:-1], device=surf.device)], dim=0)
            # world = torch.cat([world, surf], dim=0)
            weight = torch.ones(surf.shape[:-1], device=surf.device)
            world = surf
            
            query_points = world.requires_grad_(True) #.squeeze(0)
            # sdf = nef(coords=query_points, channels=["sdf"])[0].squeeze(1)
            sdf = self.sdf(coords=query_points).squeeze()
            mask = sdf.abs() < truncation
            
            # if mask.any().item():
            if mask.sum()>1000:
                try:
                    # grads = F.normalize(self.gradient(query_points).squeeze()[mask,:],dim=1)
                    grads = self.gradient(query_points).squeeze()[mask,:]/self.scale

                    # Sample points inside unit circle orthogonal to gradient direction
                    n = F.normalize(grads, dim=-1)
                    u = F.normalize(n[...,[1,0,2]] * torch.tensor([1., -1., 0.], device=n.device), dim=-1)
                    v = torch.cross(n, u, dim=-1)
                    phi = torch.rand(list(grads.shape[:-1]) + [1], device=grads.device) * 2. * np.pi
                    w = torch.cos(phi) * u + torch.sin(phi) * v
                    
                    world2 = world[mask] + w * smoothness_std
                    pidx = self.octree.blas.query(world2, self.octree.active_lods[-1], with_parents=True).pidx[...,-1]
                    query_points = world2.requires_grad_(True)
                
                    # grads2 = F.normalize(self.gradient(query_points),dim=1)
                    grads2 = self.gradient(query_points)/self.scale
                    
                    normal_regularisation_loss = ((grads - grads2).norm(dim=-1) * weight[mask])[pidx>-1].sum() / weight[mask][pidx>-1].sum()
                    # normal_regularisation_loss = (grads - grads2).norm(dim=-1)[pidx>-1].sum()/(pidx>-1).shape[0]
                except:
                    print("grad_cuda_error")
        return normal_regularisation_loss

    def get_smoothing_loss(self, rays_o, rays_d, rendered_depth):
        truncation = 0.2 #0.16 0.2
        smoothness_std = 0.004/self.scale
        normal_regularisation_loss = torch.zeros([]).to(self.device)

        # sample = 2 #4
        # corners = spc_ops.unbatched_get_level_points(self.grid.blas.points, self.grid.blas.pyramid, self.active_lods[-1])
        # # coords = wisp_spc_ops.sample_spc(corners, self.active_lods[0], sample)
        # corners = (corners[...,None,:3]+torch.rand(corners.shape[0], sample, 3, device=self.device)).reshape(-1,3)
        # coords = (corners/2**self.active_lods[-1])*2-1
        # world = coords # + torch.rand_like(coords) #((coords + torch.rand_like(coords)) * voxel_sizes[1] + volume_origin).unsqueeze(0)
        # # world = 2*(torch.rand(Nx*Nx*Nx,3)-0.5).to(self.device)
        surf = rays_o + rays_d * rendered_depth[:,None]
        # points_o3d = o3d.geometry.PointCloud()
        # points_o3d.points = o3d.utility.Vector3dVector(surf.cpu().detach().numpy())
        # o3d.io.write_point_cloud(os.path.join("01_loam_map_init", "ply/surf.ply"), points_o3d)
        # surf_mask = torch.logical_and(((surf + 1)>=0).sum(1)==3, ((surf - 1)<=0).sum(1)==3).reshape(-1)
        pidx = self.octree.blas.query(surf.reshape(-1, 3), self.octree.active_lods[-1], with_parents=True).pidx[...,-1]
        surf_mask = pidx>-1
        if surf_mask.sum() > 50: #surf_mask.any().item():
            surf = surf[surf_mask,:] #.unsqueeze(0)

            # weight = torch.cat([torch.ones(world.shape[:-1], device=world.device) * 0.5, torch.ones(surf.shape[:-1], device=surf.device)], dim=0)
            # world = torch.cat([world, surf], dim=0)
            weight = torch.ones(surf.shape[:-1], device=surf.device)
            world = surf
            
            query_points = world.requires_grad_(True) #.squeeze(0)
            # sdf = nef(coords=query_points, channels=["sdf"])[0].squeeze(1)
            sdf = self.sdf(coords=query_points).squeeze()
            mask = sdf.abs() < truncation
            
            if mask.any().item():
                # grads = F.normalize(self.gradient(query_points).squeeze()[mask,:],dim=1)
                grads = self.gradient(query_points).squeeze()[mask,:]/self.scale
                
                # pc_surf = surf * self.scale + self.origin #world pc
                # uv = pc_surf
                # surf = surf.requires_grad_(True)
                # grads_suf = self.gradient(surf).squeeze()
                # # sdf = self.sdf(coords=grads_suf, lod_idx=self.grid.num_lods - 1).squeeze()
                # # mask_surf = sdf.abs() < truncation
                # normal_buffer = F.normalize(grads_suf, p=2, dim=-1, eps=1e-5)
                # normal_img = np.zeros([376*1408,3])
                # normal_img[..., :3] = (normal_buffer.detach().cpu().numpy() + 1.0) / 2.0
                # savepath = os.path.join("grad_test.png")
                # cv2.imwrite(savepath, normal_img.reshape(376,1408,3))

                # Sample points inside unit circle orthogonal to gradient direction
                n = F.normalize(grads, dim=-1)
                u = F.normalize(n[...,[1,0,2]] * torch.tensor([1., -1., 0.], device=n.device), dim=-1)
                v = torch.cross(n, u, dim=-1)
                phi = torch.rand(list(grads.shape[:-1]) + [1], device=grads.device) * 2. * np.pi
                w = torch.cos(phi) * u + torch.sin(phi) * v
                
                world2 = world[mask] + w * smoothness_std
                pidx = self.octree.blas.query(world2, self.octree.active_lods[-1], with_parents=True).pidx[...,-1]
                query_points = world2.requires_grad_(True)
            
                # grads2 = F.normalize(self.gradient(query_points),dim=1)
                grads2 = self.gradient(query_points)/self.scale
                
                normal_regularisation_loss = ((grads - grads2).norm(dim=-1) * weight[mask])[pidx>-1].sum() / weight[mask][pidx>-1].sum()
                # normal_regularisation_loss = (grads - grads2).norm(dim=-1)[pidx>-1].sum()/(pidx>-1).shape[0]
        return normal_regularisation_loss

    def get_sdf_loss2(self, z_vals, target_d, ridx, boundary, predicted_sdf, samples):
        '''
        z_vals[block_num, sample_per_block]
        target_d[rays_num]
        ridx[block_num]
        '''
        truncation = 0.15 #1. # 0.16
        ray_block_idx = torch.div(torch.nonzero(boundary).squeeze(), z_vals.shape[-1], rounding_mode='trunc')
        ray_block_num = torch.hstack((ray_block_idx[1:] - ray_block_idx[:-1], boundary.shape[0]//z_vals.shape[-1]-ray_block_idx[-1])) 
        ray_boundry = torch.zeros_like(ridx)
        ray_boundry[ray_block_idx.squeeze()] = 1
        # front_mask = (z_vals < (target_d.expand(-1,16) - truncation))
        target = target_d[ridx.tolist()][:,None].expand(-1,z_vals.shape[-1])  
        depth_mask = target > 0.
        front_mask = (z_vals < (target - truncation))
        # bask_mask = (z_vals > (target_d + truncation)) & depth_mask
        # front_mask = (front_mask | ((target < 0.) & (z_vals < 3.5)))
        bound = (target - z_vals)
        bound[target_d[ridx.tolist()] < 0., :] = 100. #10  TODO: maybe use noisy depth for bound?
        sdf_mask = (bound.abs() <= truncation) & depth_mask
       
        idx = torch.hstack((ray_block_idx[1:].squeeze()-1,torch.IntTensor([-1]).to(self.device)))
        sum_of_samples = front_mask.sum(dim=-1) + 1e-8 # + sdf_mask.sum(dim=-1) 
        sum_of_samples = spc_render.cumsum(sum_of_samples[:,None],ray_boundry)[idx] 
        rays_w_depth = torch.count_nonzero(target_d)
        
        fs_loss = (torch.max(torch.exp(-3 * predicted_sdf) - 1., predicted_sdf - bound).clamp(min=0.) * front_mask)
        fs_loss = spc_render.cumsum(fs_loss.sum(dim=-1)[:,None], ray_boundry)[idx]
        fs_loss = (fs_loss.squeeze() / sum_of_samples.squeeze()).sum() / rays_w_depth
        if torch.isnan(fs_loss).sum()>0:
            yx = 1

        sum_of_samples = sdf_mask.sum(dim=-1)  + 1e-8 
        sum_of_samples = spc_render.cumsum(sum_of_samples[:,None],ray_boundry)[idx] 
        sdf_loss = torch.abs(predicted_sdf - bound) * sdf_mask
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(20, 20))
# N = 4
# plt.scatter(z_vals[:N,:].reshape(-1,1).detach().cpu().numpy(), predicted_sdf[:N,:].reshape(-1,1).detach().cpu().numpy())
# plt.scatter(z_vals[:N,:].reshape(-1,1).detach().cpu().numpy(), sdf_loss[:N,:].reshape(-1,1).detach().cpu().numpy())
# plt.scatter(z_vals[:N,:].reshape(-1,1).detach().cpu().numpy(), bound[:N,:].reshape(-1,1).detach().cpu().numpy())
# plt.axhline(y=0,color='r')
# plt.axvline(x=target_d[0].item(),color='b')
# plt.savefig("test")
        sdf_loss = spc_render.cumsum(sdf_loss.sum(dim=-1)[:,None], ray_boundry)[idx]
        sdf_loss = (sdf_loss.squeeze() / sum_of_samples.squeeze()).sum() / rays_w_depth

        fs_loss = (fs_loss.sum(dim=-1) / ray_block_num).sum() / rays_w_depth
        # sdf_loss = ((torch.abs(predicted_sdf - bound) * sdf_mask).sum(dim=-1) / sum_of_samples).sum() / rays_w_depth
        return sdf_loss, fs_loss 

    def get_sdf_loss(self, z_vals, target_d, predicted_sdf):
        '''
        z_vals[block_num, sample_per_block]
        target_d[rays_num]
        ridx[block_num]
        '''
        truncation = 0.15 #0.15 #0.2
        
        # depth_mask = target_d > 0.
        front_mask = (z_vals < (target_d - truncation)) #& depth_mask
        # bask_mask = (z_vals > (target_d + truncation)) & depth_mask
        # front_mask = (front_mask | ((target < 0.) & (z_vals < 3.5)))
        bound = (target_d - z_vals)
        # bound[target_d[ridx.tolist()] < 0., :] = 100. #10  TODO: maybe use noisy depth for bound?
        sdf_mask = (bound.abs() <= truncation) #& depth_mask

        sum_of_samples = front_mask.sum(dim=-1) + sdf_mask.sum(dim=-1) + 1e-8
        rays_w_depth = torch.count_nonzero(target_d)
        
        fs_loss = (torch.max(torch.exp(-1. * predicted_sdf) - 1., torch.abs(predicted_sdf - bound) ) * ~sdf_mask) #.clamp(min=0.,max=5) front_mask
        # fs_loss = (torch.abs(predicted_sdf - bound) * ~sdf_mask)
        fs_loss = (fs_loss.sum(dim=-1) / (sum_of_samples+ 1e-8)).sum() / (rays_w_depth+ 1e-8)
        sdf_loss = ((torch.abs(predicted_sdf - bound) * sdf_mask).sum(dim=-1) / sum_of_samples).sum() / rays_w_depth
        if torch.isnan(fs_loss).sum()>0:
                yx = 1
       
        # if self.iter_n%10==0:
        #     fig = plt.figure(figsize=(20, 20))
        #     N = 260
        #     plt.scatter(z_vals[N,:].reshape(-1,1).detach().cpu().numpy(), predicted_sdf[N,:].reshape(-1,1).detach().cpu().numpy())
        #     # plt.scatter(z_vals[N,:].reshape(-1,1).detach().cpu().numpy(), sdf_loss[N,:].reshape(-1,1).detach().cpu().numpy())
        #     # plt.scatter(z_vals[N,:].reshape(-1,1).detach().cpu().numpy(), (torch.abs(predicted_sdf - bound) * sdf_mask)[N,:].reshape(-1,1).detach().cpu().numpy())
        #     # plt.scatter(z_vals[N,:].reshape(-1,1).detach().cpu().numpy(),  (torch.max(torch.exp(-3. * predicted_sdf) - 1., torch.abs(predicted_sdf - bound) ) * ~sdf_mask)[N,:].reshape(-1,1).detach().cpu().numpy())
        #     plt.scatter(z_vals[N,:].reshape(-1,1).detach().cpu().numpy(), bound[N,:].reshape(-1,1).detach().cpu().numpy())
        #     plt.axhline(y=0,color='r')
        #     plt.axvline(x=target_d[0].item(),color='b')
        #     plt.savefig("test_")
       
        return sdf_loss, fs_loss

    # compute normal loss
    def get_normal_loss(self, normal_, normal_ob_, render_mask=None): # (n_rays,3)
        if render_mask != None:
            normal = normal_[render_mask,:]
            normal_ob = normal_ob_[render_mask,:]
        else:
            normal = normal_
            normal_ob = normal_ob_
        del normal_, normal_ob_
        normal_loss_l1 = (normal-normal_ob).norm(1,1)
        # normal_loss_l1 = torch.min((normal-normal_ob).norm(1,1), (normal+normal_ob).norm(1,1))#(n_rays,3)-(n_rays,3)--->norm--->(n_rays, )
        if torch.sum(torch.isnan(normal_loss_l1))!=0:
            print('normal loss l1 nan!!!')
        normal_loss_l1 = torch.where(torch.isnan(normal_loss_l1), torch.full_like(normal_loss_l1, 0), normal_loss_l1)

        normal_loss_ang = abs(1-(normal * normal_ob).sum(1))
        # normal_loss_ang = torch.min(abs(1-(normal * normal_ob).sum(1)), abs(1-(normal * (-normal_ob)).sum(1)))#(n_rays,3)*(n_rays,3)--->sum--->(n_rays, )
        if torch.sum(torch.isnan(normal_loss_ang))!=0:
            print('normal loss angular nan!!!')
        normal_loss_ang = torch.where(torch.isnan(normal_loss_ang), torch.full_like(normal_loss_ang, 0), normal_loss_ang)
        normal_loss = torch.log(normal_loss_l1 + normal_loss_ang + 1)
        return normal_loss.mean()
        # return normal_loss_l1.mean(),normal_loss_ang.mean()
#---------------------------------------------------------------

    def get_weight(self, dirs, shape, sdfs, grads, deltas, hit_ray_d, boundaries, inv_s, cos_anneal_ratio, exclusive=True):
        if len(sdfs.shape)==3:
            sdfs = sdfs.squeeze(2)
      
        dirs = dirs[:, None, :].expand(shape).reshape(-1, 3)
        # n = F.normalize(grads, dim=-1)
        true_cos = (dirs * grads.reshape(-1, 3)/self.scale).sum(-1, keepdim=True)
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
            + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive
        deltas = deltas.reshape(-1, 1).clamp_max(0.5)
        estimated_next_sdf = sdfs.reshape(-1, 1) + iter_cos * deltas * 0.5
        estimated_prev_sdf = sdfs.reshape(-1, 1) - iter_cos * deltas * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alphas = ((p + 1e-5) / (c + 1e-5)).reshape(shape[0], shape[1]).clip(0.0, 1.0)
        if boundaries == None:
            transmittance = torch.cumprod(
                torch.cat([torch.ones([alphas.size()[0], 1], device=alphas.device),1.0 - alphas + 1e-7,], -1,), -1)[:, :-1]
            weight = alphas* transmittance
        else:
            transmittance = torch.exp(spc_render.cumsum(torch.log(1. - alphas.contiguous() + 1e-7).reshape(-1,1).contiguous(), boundaries.contiguous(), exclusive=exclusive))
            weight = alphas.reshape(-1,1) * transmittance
        
        return weight,alphas


    #----------------------------------query-------------------------------
    def sdf(self, coords, onlysdf=True):
            shape = coords.shape
            
            if shape[0] == 0:
                return dict(sdf=torch.zeros_like(coords)[...,0:1])
            
            if len(shape) == 2:
                coords = coords[:, None]
            num_samples = coords.shape[1]
            coords = (coords+1.0)/2
            # coords = coords*2
            # TODO(ttakikawa): this should return [batch, ns, f] but it returns [batch, f]
            feats = self.grid['sdf'](coords.reshape(-1, 3)) #.reshape(-1, self.grid['sdf'].encoding_config['n_features_per_level'], self.grid['sdf'].encoding_config['n_levels']).sum(-1)
            # mask = feats.sum(-1)==0
            # sdfs = self.decoder['sdf'].forward(feats)[:,0:1]
            # if self.decoder['sdf'].__class__==SemanticSDFDecoder:
            #     sdfs = self.decoder['sdf'].forward_sdf(feats) if onlysdf else self.decoder['sdf'].forward(feats) 
            # else:
            sdfs = self.decoder['sdf'].forward(feats) 
            # if type(sdfs)==tuple:
            #     sdf = sdfs[0].reshape(-1,num_samples,1) 
            #     sem = sdfs[1]
            #     if len(shape) == 2:
            #         sdf = sdf[:,0]
            #     return sdf, sem
            # else:
            sdfs = sdfs.reshape(-1,num_samples,1)
    
            if torch.isnan(sdfs).sum()>0:
                yx = 1

            if len(shape) == 2:
                sdfs = sdfs[:,0]
            return sdfs



    def get_output(self, coords, channels=['sdf'], view_dirs=None, embedding=None, normals=None, image_mask=None):
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
            # coords = coords*2
            # TODO(ttakikawa): this should return [batch, ns, f] but it returns [batch, f]
            # sdf_feats = self.grid['sdf'](coords.reshape(-1, 3)) #.reshape(-1, self.grid['sdf'].encoding_config['n_features_per_level'], self.grid['sdf'].encoding_config['n_levels']).sum(-1)
            # output = self.decoder['sdf'].forward(sdf_feats)
            if 'sdf' in channels:
                sdf_feats = self.grid['sdf'](coords.reshape(-1, 3))
                # if self.decoder['sdf'].__class__ == SemanticSDFDecoder:
                #     sdfs,semantics = self.decoder['sdf'].forward(sdf_feats)
                #     semantics = semantics.reshape(-1,num_samples,self.sem_class_num)
                #     out['semantic'] = semantics
                # else:
                #     if 'semantic' in channels and self.pretrain:                 #         sdfs, sdf_last_layer = self.decoder['sdf'].forward(sdf_feats,return_last=True)
                #         # with torch.no_grad():
                #         #     rgb_last_layer_ = rgb_last_layer
                #         #     sdf_last_layer_ = sdf_last_layer
                #         semantic_feat = self.grid['semantic'](coords.reshape(-1, 3))
                #         # input = torch.hstack((semantic_feat, sdf_last_layer_))
                #         # input = torch.hstack((input, coords.reshape(-1, 3)))
                #         semantics = self.decoder['semantic'].forward(semantic_feat)
                #         out['semantic'] = semantics
                #     else:
                sdfs = self.decoder['sdf'].forward(sdf_feats)

                sdfs = sdfs.reshape(-1,num_samples,1)
                if torch.isnan(sdfs).sum()>0:
                    yx = 1
                if len(shape) == 2:
                    sdfs = sdfs[:,0]
                out['sdf'] = sdfs

            if image_mask is not None and image_mask.sum()>0:
                coords = coords[image_mask]

                if 'rgb' in channels:
                    if self.model_type == '3dgs':
                        out['rgb'] = None
                    else:
                        if self.model_type == 'separate':
                            radiance_feats = self.grid['rgb'](coords.reshape(-1, 3)) #.reshape(-1, self.grid['rgb'].encoding_config['n_features_per_level'], self.grid['rgb'].encoding_config['n_levels']).sum(-1)
                        elif self.model_type == 'shared': 
                            radiance_feats = output[:,1:]

                        rgbs = self.decoder['rgb'](radiance_feats=radiance_feats, view_dirs=view_dirs.reshape(-1,3),appearance_embedding=embedding, grads=normals)
                        # rgbs = self.decoder['rgb'](radiance_feats)
                        rgbs = torch.sigmoid(rgbs)
                        rgbs = rgbs.reshape(-1,num_samples,3)
                        out['rgb'] = rgbs
                
                if 'semantic' in channels:
                    semantic_feat = self.grid['semantic'](coords.reshape(-1, 3))
                    semantics = self.decoder['semantic'].forward(semantic_feat)
                    out['semantic'] = semantics

                if 'feature' in channels:
                    radiance_feats = self.grid['semantic'](coords.reshape(-1, 3)) #.reshape(-1, self.grid['feature'].encoding_config['n_features_per_level'], self.grid['feature'].encoding_config['n_levels']).sum(-1)
                    feats = self.decoder['feature'](radiance_feats)
                    # if 'semantic' in channels:
                    #     semantics = self.decoder['semantic'](feats)
                    #     semantics = semantics.reshape(-1,num_samples,self.sem_class_num)
                    #     out['semantic'] = semantics
                    feats = feats.reshape(-1,num_samples,self.feat_dim)
                    out['feature'] = feats
                if 'instance' in channels:
                    ## hash_pc                    
                    self.instance_corners = [(coords*2-1).reshape(-1, 3)]
                    self.ins_feat = [self.grid['instance'](coords.reshape(-1, 3))] #self.grid['instance']
                    # self.ins_feat = [self.decoder['instance'](self.ins_feat[0])]
                    instance_heat_map, self.thing_semantic_heat_map, self.query_position, self.query_loss = self.instance_model(
                                                                                                        self.ins_feat, 
                                                                                                        self.instance_corners,
                                                                                                        self.scale
                                                                                                    ) 
                    self.corner_id = instance_heat_map
                    
                    instance = instance_heat_map
                    out['instance'] = instance

                if 'panoptic' in channels:
                    stuff_mask = torch.ones(out['semantic'].shape[-1]).bool().to(self.device)
                    stuff_mask[self.Thing_class] = False
                    # thing_mask = torch.ones(out['semantic'].shape[-1]).bool().to(self.device)
                    # thing_mask[self.Stuff_class] = False
                    sem_g_norm = F.softmax(out['semantic'])
                    panoptic_instance_prob = (1 - sem_g_norm[:,stuff_mask].sum(-1))[:,None] * (out['instance']/out['instance'].sum(-1)[:,None]) # * out['instance']
                    # weight_ins = out['instance'].max(dim=-1).values
                    out['panoptic'] = torch.hstack((sem_g_norm[:,stuff_mask], panoptic_instance_prob)) #weight_ins[:,None]*
                    # out['panoptic'] = out['semantic'][:,stuff_mask]
                    # out['panoptic'] = out['semantic']
                    self.panoptic_num = out['panoptic'].shape[-1]

            return out
    
    def instance_interpolate(self, coords, feats, lod, instance_model):
        query_results = instance_model.blas.query(coords[:,0], lod, with_parents=False)
        pidx = query_results.pidx
        fs = spc_ops.unbatched_interpolate_trilinear(coords, pidx.int(), instance_model.blas.points, instance_model.trinkets.int(),
                                                        feats.half(), lod).float()
        return fs.reshape(coords.shape[0], feats.shape[-1])
    
    # kaolin render 
    @torch.no_grad()
    def query_sdf(self, coords):
        shape = coords.shape
        lod_idx = self.grid['sdf'].num_lods - 1
        result = self.grid['sdf'].blas.query(torch.FloatTensor(coords.reshape(-1, 3)).to(self.device), self.grid['sdf'].active_lods[0], with_parents=True)
        # mask = pidx.pidx[:,self.grid.active_lods[0]:].max(1).values<0
        mask = result.pidx[:,self.grid['sdf'].active_lods[0]]<0
        # mask = pidx.pidx[:,self.grid.active_lods[1]:].max(1).values<0
        # num_samples = coords.shape[1]
        if len(shape) == 2:
                coords = torch.FloatTensor(coords[:, None]).to(self.device)
        feats = self.grid['sdf'].interpolate(coords, lod_idx)
        sdfs = self.decoder['sdf'].forward(feats) #/self.scale 
        sdfs[mask] = 2.0 #torch.abs(sdfs[mask])
        return sdfs.squeeze().cpu().numpy(), ~mask.cpu().numpy()     


    @torch.no_grad()
    def pose_update(self):
        T = self.R.log()
        T = SE3.exp(T.cuda()).vec()
        # T_cam = self.R_cam.log()
        # T_cam = SE3.exp(T_cam.cuda()).vec()
        T_cam = torch.cat([self.R_cam[:,0:3], axis_angle_to_quaternion(self.R_cam[:,3:])],dim=-1)
        idx = 0
        if self.mode==1:
            self.vec_es[f"{self.data_list[-1]}"] = T[0,:] if torch.isnan(T[0,:]).sum()==0 else self.vec_es[f"{self.data_list[-1]}"]
            self.vec_cam[f"{self.data_list[-1]}"] = T_cam[0,:] if torch.isnan(T_cam[0,:]).sum()==0 else self.vec_cam[f"{self.data_list[-1]}"]
            # if torch.isnan(T[0,:]).sum()>0:
            #     print("nan!!!")
            #     sys.exit(1)
        else:
            # self.vec_es[f"{self.data_list[-1]}"] = 1
            idx2 = 0
            for i in self.data_list:
                self.vec_cam[f"{i}"] = T_cam[idx2,:] if torch.isnan(T_cam[idx2,:]).sum()==0 else self.vec_cam[f"{i}"]
                idx2 += 1
            for i in self.data_list[self.span:]:
                    self.vec_es[f"{i}"] = T[idx,:] if torch.isnan(T[idx,:]).sum()==0 else self.vec_es[f"{i}"]
                    # if torch.isnan(T[0,:]).sum>0:
                    #     print("nan!!!")
                    #     sys.exit(1)
                    idx += 1    

    def get_rays_World(self, rays):
        rays_w = rays.clone()
        frame_id = torch.zeros(rays.size(0), rays.size(1), dtype=int).to(rays.device)
        if self.mode == 1:
            pose = self.R[0].matrix()[:3,:]
            rays_w[-1,:,3:6] = rays[-1,:,3:6] @ pose[:3, :3].T # (H, W, 3) rat_d
            rays_w[-1,:,0:3] = pose[:, 3].expand(rays[-1,:,3:6].shape) # (H, W, 3) rat_o 
            for i in range(rays.size(0)-1):   
                pose = self.T_fix[i].matrix()[:3,:]
                rays_w[i,:,3:6] = rays[i,:,3:6] @ pose[:3, :3].T # (H, W, 3) rat_d
                rays_w[i,:,0:3] = pose[:, 3].expand(rays[i,:,3:6].shape) # (H, W, 3) rat_o 
                if self.normal_flag:
                    rays_w[i,:,6:9] = rays[i,:,6:9] @ pose[:3, :3].T # (H, W, 3) normal
        elif self.mode == 2:  
            for i in range(rays.size(0)):   
                # # pose = self.pose[i]
                # pose = self.R[i-1].matrix()[:3,:] if i>0 else self.first_R.matrix()[:3,:]
                # rays_w[i,:,3:6] = rays[i,:,3:6] @ pose[:3, :3].T # (H, W, 3) rat_d
                # # rays_w[i,:,0:3] = pose[:, 3].expand(rays[i,:,3:6].shape) # (H, W, 3) rat_o 
                # rays_w[i,:,0:3] = rays[i,:,0:3]@pose[:3, :3].T+pose[:, 3].expand(rays[i,:,3:6].shape) # (H, W, 3) rat_o 
                # if self.normal_flag:
                #     rays_w[i,:,6:9] = rays[i,:,6:9] @ pose[:3, :3].T # (H, W, 3) normal
                
                # pose = self.pose[i]
                lidar_pose = self.R[i-1].matrix()[:3,:] if i>0 else self.first_R.matrix()[:3,:]
                cam_pose = self.R_cam[i].matrix()[:3,:]
                if self.depth_completion:
                    ray_mask = rays[i, :, -2]==1
                else:
                    ray_mask = rays[i, :, -1]>=0
                frame_id[i] = torch.ones_like(rays[i, :, -1], dtype=int)*i
                rays_w[i,ray_mask,3:6] = rays[i,ray_mask,3:6] @ lidar_pose[:3, :3].T # (H, W, 3) rat_d
                rays_w[i,ray_mask,0:3] = lidar_pose[:, 3].expand(rays[i,ray_mask,3:6].shape) # (H, W, 3) rat_o 
                if self.normal_flag:
                    rays_w[i, ray_mask, 6:9] = rays[i,ray_mask,6:9] @ lidar_pose[:3, :3].T
                # rays_w[i,ray_mask,0:3] = rays[i,ray_mask,0:3]@lidar_pose[:3, :3].T+lidar_pose[:, 3].expand(rays[i,ray_mask,3:6].shape) # (H, W, 3) rat_o 
                rays_w[i,~ray_mask,3:6] = rays[i,~ray_mask,3:6] @ cam_pose[:3, :3].T # (H, W, 3) rat_d
                # rays_w[i,~ray_mask,0:3] = rays[i,~ray_mask,0:3]@cam_pose[:3, :3].T+cam_pose[:, 3].expand(rays[i,~ray_mask,3:6].shape) # (H, W, 3) rat_o 
                rays_w[i,~ray_mask,0:3] = cam_pose[:, 3].expand(rays[i,~ray_mask,3:6].shape) 

        return {'rays_w':rays_w, 'frame_id':frame_id}

    def get_rays_World_random_frame(self, rays, random_list):
        rays_w = rays.clone()
        frame_id = torch.zeros(rays.size(0), rays.size(1), dtype=int).to(rays.device)
        for i, f in enumerate(random_list):   
            # pose_idx = self.data_list.index(f)
            pose_idx = f
            lidar_pose = self.R[pose_idx-1].matrix()[:3,:] if pose_idx>0 else self.first_R.matrix()[:3,:]
            cam_pose = self.R_cam[pose_idx].matrix()[:3,:]
            if self.depth_completion:
                ray_mask = rays[i, :, -2]==1
            else:
                ray_mask = rays[i, :, -1]>=0
            frame_id[i] = torch.ones_like(rays[i, :, -1], dtype=int)*f #i
            rays_w[i,ray_mask,3:6] = rays[i,ray_mask,3:6] @ lidar_pose[:3, :3].T # (H, W, 3) rat_d
            rays_w[i,ray_mask,0:3] = lidar_pose[:, 3].expand(rays[i,ray_mask,3:6].shape) # (H, W, 3) rat_o 
            if self.normal_flag:
                rays_w[i, ray_mask, 6:9] = rays[i,ray_mask,6:9] @ lidar_pose[:3, :3].T
            # rays_w[i,ray_mask,0:3] = rays[i,ray_mask,0:3]@lidar_pose[:3, :3].T+lidar_pose[:, 3].expand(rays[i,ray_mask,3:6].shape) # (H, W, 3) rat_o 
            rays_w[i,~ray_mask,3:6] = rays[i,~ray_mask,3:6] @ cam_pose[:3, :3].T # (H, W, 3) rat_d
            # rays_w[i,~ray_mask,0:3] = rays[i,~ray_mask,0:3]@cam_pose[:3, :3].T+cam_pose[:, 3].expand(rays[i,~ray_mask,3:6].shape) # (H, W, 3) rat_o 
            rays_w[i,~ray_mask,0:3] = cam_pose[:, 3].expand(rays[i,~ray_mask,3:6].shape) 

        return {'rays_w':rays_w, 'frame_id':frame_id}

    # @torch.no_grad()
    def sphere_tracing(self, rays_o, rays_d, lod_idx=None, num_steps=256, step_size=1.0, min_dis=1e-2, dist_max=100.0):
        x_buffer = torch.zeros_like(rays_o)
        depth_buffer = torch.zeros_like(rays_o[...,0:1])
        hit_buffer = torch.zeros_like(rays_o[...,0]).bool()
        normal_buffer = torch.zeros_like(rays_o)
        normal_buffer = torch.zeros(*rays_o.shape[:-1], 3, device=rays_o.device)
        rgb_buffer = torch.zeros(*rays_o.shape[:-1], 3, device=rays_o.device)
        semantic_buffer = torch.zeros(*rays_o.shape[:-1], device=rays_o.device, dtype=torch.long)
        instance_buffer = torch.zeros(*rays_o.shape[:-1], device=rays_o.device, dtype=torch.long)
        # instance_buffer = torch.zeros([rays_o.shape[0], self.instance_model.num_queries], device=rays_o.device)
        panoptic_buffer = torch.zeros(*rays_o.shape[:-1], device=rays_o.device, dtype=torch.long)

        if lod_idx is None:
            lod_idx = self.octree.num_lods-1

        octree, points, pyramid, prefix = self.octree.blas.octree, self.octree.blas.points,self.octree.blas.pyramid,self.octree.blas.prefix

        # res = float(2**(lod_idx+self.base_lod))
        #invres = 1.0 / res
        invres = 0.2

        # Trace SPC
        # ridx, pidx, depth = self.grid.raytrace(rays, self.grid.active_lods[lod_idx], with_exit=True)
        ridx, pidx, depth = spc_render.unbatched_raytrace(
                octree, points, pyramid, prefix,
                rays_o, rays_d, self.octree.active_lods[lod_idx], return_depth=True, with_exit=True)
        depth *= self.scale
        depth[...,0:1] += 1e-5

        first_hit = spc_render.mark_pack_boundaries(ridx)
        curr_idxes = torch.nonzero(first_hit)[...,0].int()

        first_ridx = ridx[first_hit].long()
        nug_o = rays_o[first_ridx]
        nug_d = rays_d[first_ridx]/self.scale

        mask = torch.ones([first_ridx.shape[0]], device=nug_o.device).bool()
        hit = torch.zeros_like(mask).bool()

        t = depth[first_hit][...,0:1]
        x = torch.addcmul(nug_o, nug_d, t)
        dist = torch.zeros_like(t)
        
        curr_pidx = pidx[first_hit].long()

        if mask.any().item():
            with torch.no_grad():
                # Calculate SDF for current set of query points   
                dist[mask] = self.sdf(coords=x[mask]) * invres * step_size #, pidx=curr_pidx[mask]
                dist[~mask] = 10
                dist_prev = dist.clone()
                # timer.check("first")

                for i in range(num_steps):
                    # Two-stage Ray Marching
                    # points_o3d = o3d.t.geometry.PointCloud()
                    # points_o3d.point["positions"] = o3d.core.Tensor(x.cpu().numpy(), o3d.core.float32)
                    # o3d.t.io.write_point_cloud("sphere_test.ply", points_o3d, write_ascii=True, compressed=False)
                    
                    # Step 1: Use SDF to march
                    t += dist
                    x = torch.where(mask.view(mask.shape[0], 1), torch.addcmul(nug_o, nug_d, t), x)

                    hit = torch.where(mask, torch.abs(dist)[...,0] < min_dis * invres, hit)
                    hit |= torch.where(mask, 
                                    torch.abs(dist+dist_prev)[...,0] * 0.5 < (min_dis*5) * invres, hit)
                    mask = torch.where(mask, (t < dist_max)[...,0], mask)
                    mask &= ~hit
                    if not mask.any():
                        break
                    dist_prev = torch.where(mask.view(mask.shape[0], 1), dist, dist_prev)
                    
                    # Step 2: Use AABBs to march
                    next_idxes = find_depth_bound(t, depth, first_hit, curr_idxes=curr_idxes)
                    mask &= (next_idxes != -1)
                    aabb_mask = (next_idxes != curr_idxes)
                    curr_idxes = torch.where(mask, next_idxes, curr_idxes)

                    t = torch.where((mask & aabb_mask).view(mask.shape[0], 1), depth[curr_idxes.long(), 0:1], t)
                    x = torch.where(mask.view(mask.shape[0], 1), torch.addcmul(nug_o, nug_d, t), x)
                    
                    curr_pidx = torch.where(mask, pidx[curr_idxes.long()].long(), curr_pidx)
                    if not mask.any():
                        break
                    dist[mask] = self.sdf(coords=x[mask]) * invres * step_size #, pidx=curr_pidx[mask]

                if hit.any().item():
                    hit_buffer[first_ridx] = hit
                    x_buffer[hit_buffer] = x[hit]
                    depth_buffer[hit_buffer] = t[hit]
                    
                    # grad = finitediff_gradient(x[hit], self.sdf, eps=0.005/self.scale) #.squeeze()
                    # print(f"x[hit].shape:{x[hit].shape}")
                    grad = self.gradient(x[hit]).detach()
                    if self.rgb_flag and self.model_type!='3dgs':
                        # embedding = self.embedding_a(torch.zeros(x[hit].shape[0],dtype=int, device=x.device)) if self.appearance_embedding else None
                        # rgb_buffer[hit_buffer] = self.get_output(coords=x[hit], channels=['rgb'], view_dirs=nug_d[hit]*self.scale, embedding=embedding, normals=(grad/self.scale).reshape(-1,3))['rgb'].squeeze()
                        embedding = self.embedding_a(torch.zeros(x[hit].shape[0],dtype=int, device=x.device)) if self.appearance_embedding else None
                        output = self.get_output(coords=x[hit], channels=self.channels, view_dirs=nug_d[hit]*self.scale, embedding=embedding, normals=(grad/self.scale).reshape(-1,3))
                        rgb_buffer[hit_buffer] = output['rgb'].squeeze()
                        if self.semantic_flag and (self.net=='rgb_sem' or self.net=='geo_rgb_sem'):
                            semantic_buffer[hit_buffer] = torch.argmax(torch.nn.functional.softmax(output['semantic'].squeeze() , dim=-1), -1)
                    normal_buffer[hit_buffer] = (F.normalize(grad, p=2, dim=-1, eps=1e-5) + 1.0) / 2.0
                    if self.semantic_flag:
                        semantic_buffer[hit_buffer] = torch.argmax(torch.nn.functional.softmax(
                            self.get_output(coords=x[hit], channels=['semantic'], view_dirs=nug_d[hit]*self.scale, normals=(grad/self.scale).reshape(-1,3), image_mask=torch.ones(x[hit].shape[0]).bool().to(self.device))['semantic'].squeeze() , dim=-1), -1)
                    if self.instance_flag:
                        instance_buffer[hit_buffer] = torch.argmax(torch.nn.functional.softmax(
                            self.get_output(coords=x[hit], channels=['instance'], view_dirs=nug_d[hit]*self.scale, normals=(grad/self.scale).reshape(-1,3), image_mask=torch.ones(x[hit].shape[0]).bool().to(self.device))['instance'].squeeze(), dim=-1), -1)
                        # instance_buffer[hit_buffer] = self.get_output(coords=x[hit], channels=['instance'], view_dirs=nug_d[hit]*self.scale, normals=(grad/self.scale).reshape(-1,3), image_mask=torch.ones(x[hit].shape[0]).bool().to(self.device))['instance'].squeeze()
                    if self.panoptic_flag:
                        # semantic_buffer[hit_buffer] = torch.argmax(torch.nn.functional.softmax(
                        #     self.get_output(coords=x[hit], channels=['sdf','semantic'], view_dirs=nug_d[hit]*self.scale, normals=(grad/self.scale).reshape(-1,3))['semantic'].squeeze(), dim=-1), -1)
                        panoptic_buffer[hit_buffer] = torch.argmax(torch.nn.functional.softmax(
                            self.get_output(coords=x[hit], channels=self.channels, view_dirs=nug_d[hit]*self.scale, normals=(grad/self.scale).reshape(-1,3), image_mask=torch.ones(x[hit].shape[0]).bool().to(self.device))['panoptic'].squeeze(), dim=-1), -1)
                        

        return x_buffer.cpu(), 255*normal_buffer.cpu(), depth_buffer.cpu(), 255*rgb_buffer.cpu(), semantic_buffer.cpu(), instance_buffer.cpu(), panoptic_buffer.cpu()
    
    @torch.no_grad()
    def render_nerf(self, idx):
        BATCH = 8000
        ## get rays info
        data = self.dataset_test[idx]
        ray_directions_cam = data['direction_img'].squeeze()
        ray_origin_cam = data['origin'].squeeze()  
        rays_all_num2 = ray_directions_cam.size(0) 
        render_pose_c2w = data['pose_cam'].squeeze() 

        ## render image
        normal_list, depth_list = [], []
        for i in range(0, rays_all_num2, BATCH):
            i_next = i+BATCH if (i+BATCH <rays_all_num2) else rays_all_num2
            rays_d_cam_batch = ray_directions_cam[i:i_next].clone()
            res = self.render_core_test(ray_origin_cam, ray_directions_cam[i:i_next,:], render_pose_c2w) #, depth_method='median'
            normal_list.append(res['normal']) #.cpu().numpy()
            depth_list.append((res['depth']*rays_d_cam_batch[:,2:])) #.cpu().numpy()
                
        normal_img = torch.vstack(normal_list)
        depth_img = torch.vstack(depth_list)
        # normal_images.append(normal_img.astype(np.uint8).reshape(-1, self.width, 3)) 
        w = int(self.cfg['img']['width']/self.render_factor)
        normal = normal_img.reshape(-1, w, 3).permute(2,0,1)
        depth = depth_img.reshape(-1, w)
        # resize = torchvision.transforms.Resize([self.cfg['img']['height'], self.cfg['img']['width']])
        # normal = resize(normal)
        # depth = resize(depth[None,...]).squeeze()
        return normal, depth
    
    def render_core_test(self, rays_o, rays_d, c2w, depth_method='expected'):
                rays_cam_o, rays_cam_d = rays_o, rays_d
                rays_o, rays_d = get_rays(rays_o, rays_d, c2w)
                rays_go = get_rays_sfm(rays_o[None,:,:], self.scale, self.origin)[0]
                rays_g = Rays(origins=rays_go, dirs=rays_d)
                rgb_buffer = torch.zeros(*rays_o.shape[:-1], 3, device=rays_o.device)
                result = {}

                with torch.no_grad():
                        ridx, samples, depth_samples, deltas, boundary, valid_mask, depths_outside = self.raymarch_ray_image(rays_g, rays_cam_o, rays_cam_d) 
                        
                        # background
                        if depths_outside is not None:
                                ret_outside = self.render_core_outside(rays_g[~valid_mask].origins, rays_g[~valid_mask].dirs, depths_outside)

                                color_outside = ret_outside['color']

                        deltas = deltas*self.scale
                        deltas = deltas.clip(0, 0.3)
                        depth_samples = depth_samples*self.scale
                        if ridx.shape[0]>0:
                                ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
                                hit_ray_d = (rays_g.dirs[valid_mask]).index_select(0, ridx.long())
                                # sdf_value, rgbs = self.get_output(coords=samples, view_dirs=hit_ray_d, lod_idx=self.grid.num_lods - 1, return_rgb=self.rgb_flag)
                                channels = ['sdf']
    
                                embeddings = self.embedding_a((torch.ones([1],dtype=int)*frame_id).to(self.device)).expand(samples.shape[0], -1) if self.appearance_embedding else None
                                grads = autodiff_gradient(samples, self.sdf, self.origin, self.scale) /self.scale
                                # grads = finitediff_gradient(samples, self.sdf, eps=0.005/self.scale)/self.scale
                                out = self.get_output(samples, channels=channels, view_dirs=hit_ray_d, embedding=embeddings, normals=grads.reshape(-1,3))

                                weights = self.get_weight_2(out['sdf'], deltas, self.s, 1, boundary) 

                                ray_depth = spc_render.sum_reduce(depth_samples.reshape(-1,1) * weights.reshape(-1,1), boundary)
                                depth = torch.zeros(rays_g.shape[0], 1, device=self.device)
                                buffer = torch.zeros_like(depth[valid_mask],device=depth.device)
                                buffer[ridx_hit.long(),:] = ray_depth
                                depth[valid_mask] = buffer
                                result['depth'] = depth

                                surface_points = rays_go + rays_d * depth
                                if ridx_hit.shape[0]>1000:
                                        grad = autodiff_gradient(surface_points[valid_mask][ridx_hit.long(), :], self.sdf, self.origin, self.scale) /self.scale
                                #        grad = finitediff_gradient(surface_points[ridx_hit.long(), :], self.sdf, eps=0.01/self.scale)/self.scale
                                else:
                                        grad = torch.zeros_like(surface_points[valid_mask][ridx_hit.long(), :])
                              
                                normal_buffer = F.normalize(grad, p=2, dim=-1, eps=1e-5)
                                buffer = torch.zeros_like(rgb_buffer[valid_mask],device=rgb_buffer.device)
                                buffer[ridx_hit.long(),:] = normal_buffer
                                rgb_buffer[valid_mask] = buffer
                                result['normal'] = rgb_buffer 
                                # # seq05_5760 0.15 seq04_4200 2.0
                                # height_mask = torch.logical_or((rays_d*depth)[:,2]*self.scale>0.15, depth.squeeze()==0)
                                # for key in result.keys():
                                #         result[key][height_mask] = torch.zeros_like(result[key][height_mask])
                        else:
                                depth = torch.zeros(rays_g.shape[0], 1, device=self.device)
                                semantic = torch.zeros(rays_g.shape[0], 1, device=self.device).long()
                                instance = torch.zeros(rays_g.shape[0], 1, device=self.device).long()
                                feat = torch.zeros(rays_g.shape[0], self.feat_dim, device=self.device)
                                rgb_buffer = torch.zeros(*rays_o.shape[:-1], 3, device=rays_o.device)
                                result = {'depth': depth, 'semantic': semantic, 'instance': instance, 'feature': feat, 'normal': rgb_buffer}
                                # normal_buffer = torch.zeros(rays_g.shape[0], 3, device=self.device)
                                
                return result

    def raymarch_ray_image(self, rays, rays_cam_o, rays_cam_d):
            # mask camera rays and lidar rays
            depth_cam = self.get_img_depth(rays, num_samples=5, level=self.octree.active_lods[-1]).squeeze()
            valid_mask = depth_cam>0.5
            valid_depth = depth_cam[valid_mask]
            valid_rays = rays[valid_mask]
            outside_rays = rays[~valid_mask]
            # set near and far
            rays_far = (valid_depth + 0.15)/self.scale #0.2
            rays_near = (valid_depth - 0.15)/self.scale #0.2
        
            if valid_mask.sum()>0:
                    with torch.no_grad():
                            num_samples_lidar = [25,0,0]
                            n_important = 10 #20 #8
                            up_sample_steps = 2 #4
                            num_samples_sur = [5,10] #5 
                            if sum(num_samples_sur) > 0:
                                    rays_near_est = rays_near[:,None]#/self.scale
                                    z_vals = torch.linspace(0, 1.0, num_samples_sur[0], device=valid_rays.origins.device)[None] + \
                                            (torch.zeros(valid_rays.origins.shape[0], num_samples_sur[0], device=valid_rays.origins.device) / num_samples_sur[0])
                                    z_vals *= (rays_far - rays_near)[:,None]#/self.scale
                                    z_vals += rays_near[:,None]#/self.scale
                                    n_samples = num_samples_sur[0]
                                    #new
                                    pts = (valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * z_vals[..., :, None])  # N_rays, N_samples, 3
                                    sdf = self.sdf(pts).squeeze(-1)
                                    for i in range(int(num_samples_sur[1]/5)):
                                            new_z_vals = self.up_sample(valid_rays.origins, valid_rays.dirs, z_vals, sdf, 5, 64 * 2 ** i, i)
                                            # z_vals = torch.cat([z_vals, new_z_vals+0.05*torch.rand_like(new_z_vals)/self.scale], dim=-1)
                                            z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
                                            z_vals, index = torch.sort(z_vals, dim=-1)
                                            n_samples = n_samples + 5 

                                            pts = valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * new_z_vals[..., :, None]
                                            new_sdf = self.sdf(pts).squeeze(-1)
                                            sdf = torch.cat([sdf, new_sdf], dim=-1)
                                            xx = (
                                                    torch.arange(z_vals.shape[0])[:, None]
                                                    .expand(-1, n_samples)
                                                    .reshape(-1)
                                            )
                                            index = index.reshape(-1)
                                            sdf = sdf[(xx, index)].reshape(-1, n_samples)
                                            
                            lidar_sample_lod = self.octree.active_lods[:1]
                            z_val_new, rays_near_est = self.grid_sample(valid_rays, lidar_sample_lod, num_samples_lidar)
                            z_vals = torch.cat([z_vals, z_val_new], dim=-1)
                            z_vals, index = torch.sort(z_vals, dim=-1)
                            n_samples = n_samples + sum(num_samples_lidar)                
                            
                            # z_vals, rays_near_est = self.grid_sample(valid_rays, lidar_sample_lod, num_samples_lidar)
                            # n_samples = sum(num_samples_lidar)

                            pts = (valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * z_vals[..., :, None])  # N_rays, N_samples, 3
                            sdf = self.sdf(pts).squeeze(-1)
                            for i in range(up_sample_steps):
                                    new_z_vals = self.up_sample(valid_rays.origins, valid_rays.dirs, z_vals, sdf, n_important // up_sample_steps,
                                    64 * 2 ** i, i)
                                    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
                                    z_vals, index = torch.sort(z_vals, dim=-1)
                                    n_samples = n_samples + n_important // up_sample_steps
                                    pts = valid_rays.origins[:, None, :] + valid_rays.dirs[:, None, :] * new_z_vals[..., :, None]
                                    new_sdf = self.sdf(pts).squeeze(-1)
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
                            query_results = self.octree.blas.query(samples.reshape(-1, 3), self.octree.active_lods[0])
                            pidx = query_results.pidx.reshape(-1, n_samples)
                            mask = pidx>-1
                            z_vals = z_vals[mask][:,None]
                            num_hit_samples = z_vals.shape[0]
                            deltas = deltas[mask].reshape(num_hit_samples, 1)
                            samples = samples[mask]
                            ridx = torch.arange(0, pidx.shape[0], device=pidx.device)
                            ridx = ridx[..., None].repeat(1, n_samples)[mask]
                            boundary = spc_render.mark_pack_boundaries(ridx.int())
            else:
                    samples = torch.zeros([0,3], device=rays.origins.device)
                    z_vals = torch.zeros([0,1], device=rays.origins.device)
                    deltas = torch.zeros([0,1], device=rays.origins.device)
                    boundary = torch.zeros([0], device=rays.origins.device)
                    ridx = torch.zeros([0], device=rays.origins.device)

            ## outside octree -- background
            z_vals_outside = None

            return ridx, samples, z_vals, deltas, boundary, valid_mask, z_vals_outside #, samples, z_vals, deltas #samples_d, z_vals_d, deltas_d
                
    def init_instance_model(self, dataset_train):
        pcd_all = o3d.io.read_point_cloud(os.path.join(self.cfg['path']['proj_dir'], 'ply/token_init.ply'))
        if np.array(pcd_all.points).shape[0] == 0:
            pcd_all = o3d.geometry.PointCloud()
            for idx in tqdm(dataset_train.data_list):
                color_raw1 = cv2.imread(os.path.join(dataset_train.instance_2d_data_path, f'{idx}.png'), -1) #%006d.jpg" % idx
                color_raw1 = cv2.resize(color_raw1, dsize=(dataset_train.w_d, dataset_train.h_d), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
                depth_raw1 = (cv2.imread(os.path.join(dataset_train.data_path, f'depth/{idx}.png'), -1) / 1000).reshape(-1,1)

                # pix = np.arange(0, dataset_train.h_d * dataset_train.w_d, 1)
                # pix = np.nonzero(color_raw1.reshape(-1))
                pix = []
                for id in np.unique(color_raw1):
                    if id == 0:
                        continue
                    pixel_idx = np.where(color_raw1.reshape(-1)==id)
                    select_idxs = np.random.permutation(pixel_idx[0])[:3] #select 2
                    for select_idx in select_idxs:
                        if depth_raw1[select_idx] > 0:
                            pix.append(select_idx)
                if len(pix)==0:
                    continue
                pix = np.stack(pix)
                v = np.floor_divide(pix, dataset_train.w_d)
                u = pix % dataset_train.w_d   
                rays_d = np.stack([(u - dataset_train.cx_d.numpy()) / dataset_train.fx_d.numpy(), (v - dataset_train.cy_d.numpy()) / dataset_train.fy_d.numpy(), np.ones_like(u)], axis=-1) #.to(self.device)
                rays_o = np.array([0,0,0])
                pc = rays_o + rays_d * depth_raw1[pix]

                pose = SE3(self.vec_cam[f'{idx}']).matrix().cpu().numpy()
                if pose[3,3]<0:
                    continue
                R1 = pose[0:3, 0:3]
                T1 = pose[0:3, 3]

                points_world = pc @ R1.T + T1

                pcd_world = o3d.geometry.PointCloud()
                pcd_world.points = o3d.utility.Vector3dVector(points_world)
                pcd_all = pcd_all + pcd_world

            pcd_all = pcd_all.voxel_down_sample(voxel_size = 0.8)
            if np.array(pcd_all.points).shape[0] > 150: #150:
                random_idx = np.random.randint(0,np.array(pcd_all.points).shape[0],size=[150])
                pcd_all.points = o3d.utility.Vector3dVector(np.array(pcd_all.points)[random_idx])
            print(pcd_all)
            o3d.io.write_point_cloud(os.path.join(self.cfg['path']['proj_dir'], 'ply/token_init.ply'), pcd_all)
            # pcd_all.points = o3d.utility.Vector3dVector(pcd_all.points)
            # o3d.io.write_point_cloud(os.path.join(self.cfg['path']['proj_dir'], 'ply/token_init_norm.ply'), pcd_all)

        pc_all = torch.FloatTensor(pcd_all.points).to(self.device)
        self.super_point = (pc_all - self.origin) / self.scale
        self.super_point.requires_grad=True
        # self.radius = 1.*torch.ones(self.super_point.shape[0]).to(self.device)
        # self.radius.requires_grad=True
        self.instance_model = Mask3D(**self.cfg['instance'], query_position=self.super_point, device=self.device)
        self.query_assignment_record = torch.zeros(self.instance_model.num_queries).to(self.device)
        self.query_assignment_record_add = torch.zeros(self.instance_model.num_queries).to(self.device)
        self.query_assignment_record_add_history = torch.zeros(self.instance_model.num_queries).to(self.device)
        self.query_update_map = {i: i for i in range(self.instance_model.num_queries)}
        self.instance_model.to(self.device)
        self.instance_model.add_query(self.super_point, self.scale)

def neus_weights(sdf, dists, inv_s, cos_val, z_vals=None):    
    estimated_next_sdf = sdf + cos_val * dists * 0.5
    estimated_prev_sdf = sdf - cos_val * dists * 0.5
    
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
    weights = alpha * torch.cumprod(torch.cat([torch.ones([sdf.shape[0], 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    
    if z_vals is not None:
        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0., torch.ones_like(signs), torch.zeros_like(signs))
        # This will only return the first zero-crossing
        inds = torch.argmax(mask, dim=1, keepdim=True)
        z_surf = torch.gather(z_vals, 1, inds)
        return weights, z_surf
    
    return weights

def neus_weights2(sdf, dists, inv_s, cos_val, boundaries, exclusive, z_vals=None):    
    estimated_next_sdf = sdf + cos_val * dists * 0.5
    estimated_prev_sdf = sdf - cos_val * dists * 0.5
    
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
    transmittance = torch.exp(spc_render.cumsum(torch.log(1. - alpha.contiguous() + 1e-7).reshape(-1,1).contiguous(), boundaries.contiguous(), exclusive=exclusive))
    weights = alpha.reshape(-1,1) * transmittance  #torch.cumprod(torch.cat([torch.ones([sdf.shape[0], 1], device=alpha.device), 1. - alpha ], -1), -1)[:, :-1]
    
    if z_vals is not None:
        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0., torch.ones_like(signs), torch.zeros_like(signs))
        # This will only return the first zero-crossing
        inds = torch.argmax(mask, dim=1, keepdim=True)
        z_surf = torch.gather(z_vals, 1, inds)
        return weights, z_surf
    
    return weights
