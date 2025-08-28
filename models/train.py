'''
Author: yuxuan1206 610939662@qq.com
Date: 2022-11-09 13:06:17
LastEditors: yuxuan1206 610939662@qq.com
LastEditTime: 2022-12-05 01:11:30
FilePath: /occuSLAM3D_indoor_KITTI/train.py
Description: 

Copyright (c) 2022 by yuxuan1206 610939662@qq.com, All Rights Reserved. 
'''
import torch
import sys
from lietorch import SE3
from utils.scannet_data_tools.depth2point3D import init_pointcloud2, bigger_pc, depth2pcd
from render.render_helper import *
from models.rays import *
from wisp.core import Rays
from itertools import product
import time
import torch.nn.functional as F
# import wisp.ops.spc as wisp_spc_ops
# from utils.scannet_data_tools.kitti360Viewer3DRaw import Kitti360Viewer3DRaw
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import math
from instance.models.mask3d import Mask3D
import open3d as o3d
import subprocess

class Train():
        def __init__(self, mode, device, config, data_list, vec_es, vec_cam, cubes_num):
                self.mode = mode
                self.config = config
                self.seq = config['path']['seq']
                self.velo = Kitti360Viewer3DRaw(mode='velodyne', seq=self.seq) if config['path']['dataset_type'] == 'kitti360' else None
                self.init_train(device, config, data_list, vec_es, vec_cam, cubes_num)
                self.pre_time, self.hit_time, self.back_time = [], [], []
                self.config = config
               
        def init_train(self, device, config, data_list, vec_es, vec_cam, cubes_num):
                if config['grid_otype']=='HashGrid':
                        from models.SDFGrid_hash import SDFGrid
                elif config['grid_otype']=='OctreeGrid':
                        from models.SDFGrid import SDFGrid
                else:
                        sys.exit("Wrong grid type. Please use HashGrid or OctreeGrid")
                self.nef = SDFGrid(device, config, data_list, vec_es, vec_cam, self.mode)
                self.set_train_cfg(config)
                # self.vec_es = vec_es
                pose_list = [SE3(vec_es[f"{ii}"]).matrix() for ii in data_list]
                

                if self.mode>=1 and cubes_num==0 and not self.config['pretrain']:# :
                        data_list_octree_init = data_list #range(data_list[0],data_list[-1]) 
                        pose_list = [SE3(vec_es[f"{ii}"]).matrix() for ii in data_list_octree_init]
                        self.nef.world_dims, self.nef.volume_origin, self.nef.voxel_dims, world_points = init_pointcloud2(
                                config, self.velo, data_list_octree_init, pose_list, self.nef.voxel_sizes, device, is_first=True) 
                        self.nef.scale = self.nef.world_dims.max()/2
                        self.nef.origin = self.nef.volume_origin+self.nef.world_dims/2
                        pointcloud = (world_points - self.nef.origin) / self.nef.scale # grid corrd
                        if self.depth_completion:
                                cam_pose_list = [SE3(vec_cam[f"{ii}"]).matrix() for ii in data_list_octree_init]
                                cam_points = depth2pcd(config, data_list_octree_init, cam_pose_list, seq=self.seq, cam_id=0)
                       
                        # points_o3d = o3d.t.geometry.PointCloud()
                        # points_o3d.point["positions"] = o3d.core.Tensor(pointcloud.cpu().numpy(), o3d.core.float32)
                        # o3d.t.io.write_point_cloud("pointcloud_all.ply", points_o3d, write_ascii=True, compressed=False)
                        # self.nef.grid.init_from_pointcloud(torch.FloatTensor(pointcloud_ex).to(pointcloud.device))
                        # self.nef.grid.init_from_pointcloud(pointcloud)
                        if config['path']['dataset_type'] == 'kitti360':
                                self.nef.init_from_pointcloud(pointcloud,vox_down_m=0.05/self.nef.scale,dilate=2)
                        else:
                                self.nef.init_from_pointcloud(pointcloud,vox_down_m=0.05/self.nef.scale) #2
                        del world_points, pointcloud
                
                elif self.config['pretrain']:
                        from models.decoder import PositionDecoder
                        # path = f"/mnt/northcn4/ywx1261068/exp_data/kitti360/rgb_result/v{cubes_num}/grid/optimed_grid_{cubes_num}_49.pth" 
                        path = self.config['ckpt_path']
                        self.nef.load_grid(path)
                        for key in self.nef.channels:
                                if key not in self.nef.decoder.keys() and (key in self.config['decoder'].keys()):
                                        cfg = config['decoder'][key]
                                        self.nef.chose_decoder(key, cfg, config)
                                if (key not in self.nef.grid.keys()) and (key in self.config['grid'].keys()):
                                        cfg = config['grid'][key]
                                        self.nef.chose_grid(key, cfg)
                                
                                # instance
                                data_list_octree_init = data_list #range(data_list[0],data_list[-1]) 
                                pose_list = [SE3(vec_es[f"{ii}"]).matrix() for ii in data_list_octree_init]
                                _, _, _, world_points = init_pointcloud2(
                                        config, self.velo, data_list_octree_init, pose_list, self.nef.voxel_sizes, device, is_first=True) 
                                pointcloud = (world_points - self.nef.origin) / self.nef.scale # grid corrd


                elif self.mode==2:
                        # path = os.path.join(config['path']['proj_dir'], f"grid/optimed_grid_{cubes_num-1}.pth")
                        path = os.path.join(config['path']['proj_dir'], f"../v0/grid/optimed_grid_0_49.pth")
                        self.nef.read_last_grid(path)
                        data_list_octree_init = range(data_list[0]-5,data_list[-1]+5) 
                        # data_list_octree_init = data_list #range(data_list[0],data_list[-1]) #TODO

                        pose_list = [SE3(vec_es[f"{ii}"]).matrix() for ii in data_list_octree_init]
                        world_points, bounds = init_pointcloud2(config, self.velo, data_list_octree_init, pose_list, self.nef.voxel_sizes, device, is_first=False) 
                        self.nef.volume_origin = bounds[:,0] - (self.nef.world_dims - bounds[:,1] + bounds[:,0]) / 2
                        self.nef.origin = self.nef.volume_origin+self.nef.world_dims/2
                        pointcloud = (world_points - self.nef.origin) / self.nef.scale
                        self.nef.init_from_pointcloud(pointcloud,vox_down_m=0.05/self.nef.scale)
                        del world_points, pointcloud


        # def segment_pc(self, pc, sh_path, args):
        #         save_path = './instance/all_pc.ply'
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
        #         o3d.io.write_point_cloud(save_path, pcd)
        #         args = [save_path] + args
        #         subprocess.run(["bash", sh_path] + args, check=True)
                

        def set_train_cfg(self, config):
                self.train_info = config['train'][f'mode{self.mode}']
                
                # Set Optimizer
                self.lr = {}
                self.lr['occ_lr'] = self.train_info['occ_lr'] if self.train_info['depth_optim'] else 0
                self.lr['pose_lr'] = self.train_info['pose_lr'] if self.train_info['pose_optim'] else 0
                self.lr['s_lr'] = self.train_info['s_lr']
                self.lr['rgb_lr'] = self.train_info['rgb_lr']
                self.lr['mlp_lr'] = self.train_info['mlp_lr']
                # self.nef.set_optimer(self.mode, occ_lr, 0, s_lr, sem_lr, 0.001)
                self.pose_epoch = self.train_info['pose_epoch']
                self.nef.geometry_iter = self.train_info['geometry_iter']
                # Get lamuda
                self.lamudas = {}
                self.lamudas['depth_lamuda'] = self.train_info['depth_lamuda'] if self.train_info['depth_optim'] else 0
                self.lamudas['depth_lamuda_decay'] = self.train_info['depth_lamuda_decay'] if self.train_info['depth_optim'] else 0
                
                #self.lamudas['eik_lamuda'] = train_info['eik_lamuda'] 
                #self.lamudas['smoothing_lamuda'] = train_info['smoothing_lamuda']  

                # self.depth = self.train_info['depth_lamuda']
                # self.eik = self.train_info['eik_lamuda'] 
                # self.smooth = self.train_info['smoothing_lamuda'] 

                self.lamudas['sdf_near_lamuda'] = self.train_info['sdf_near_lamuda'] 
                self.lamudas['sdf_far_lamuda'] = self.train_info['sdf_far_lamuda'] 
                self.lamudas['normal_lamuda'] = self.train_info['normal_lamuda']
                self.lamudas['rgb_lamuda'] = self.train_info['rgb_lamuda']
                self.lamudas['semantic_lamuda'] = self.train_info['semantic_lamuda']
                self.lamudas['feature_lamuda'] = self.train_info['feature_lamuda']
                self.lamudas['instance_lamuda'] = self.train_info['instance_lamuda']
                # Get Loss terms
                self.optim_flag =  config["loss_term"]
                self.normal_flag = config["loss_term"]["normal"]
                self.semantic_flag = config["loss_term"]["semantic2d"] or config["loss_term"]["semantic3d"]  #config["loss_term"]["semantic"]
                self.instance_flag = config["loss_term"]["instance"]
                self.panoptic_flag = config["loss_term"]["panoptic"]
                self.depth_completion = config["loss_term"]["depth_completion"]
                self.rgb_flag = config["loss_term"]["rgb"] 
                self.feature_flag = config["loss_term"]["feature"] 
                if config['path']['dataset_type'] == 'kitti360':
                        self.ray_long = 7 + self.semantic_flag * 1 + self.instance_flag * 1 + self.normal_flag * 3 + self.rgb_flag * 3 + self.depth_completion #self.feature_flag * 64 +
                else:
                        if ~self.normal_flag and ~self.semantic_flag:
                                self.ray_long = 7
                        elif ~self.normal_flag and self.semantic_flag:
                                self.ray_long = 8
                        else:
                                self.ray_long = 11
        
        def set_frame_optimer(self, path_base, data_list, vec_gt, vec_es_odom, key_idx):
                self.nef.data_list = data_list
                self.nef.init_pose(path_base, data_list, vec_gt, vec_es_odom, key_idx)
                self.nef.set_optimer(self.lr['occ_lr'], self.lr['pose_lr'], self.lr['s_lr'], self.lr['rgb_lr'], self.lr['mlp_lr']) #0.001
                self.optim_pose = self.train_info['pose_optim']
                # self.lamudas['depth_lamuda'] = 1 
                # self.lamudas['eik_lamuda'] = 20
                # self.lamudas['smoothing_lamuda'] = 5
                
                self.lamudas['depth_lamuda'] = self.train_info['depth_lamuda']
                self.lamudas['eik_lamuda'] = self.train_info['eik_lamuda'] 
                self.lamudas['smoothing_lamuda'] = self.train_info['smoothing_lamuda']  
                print(self.lamudas)
                print(self.lr)
        
        def update_optimer(self):
                self.nef.set_optimer(self.lr['occ_lr'], self.lr['pose_lr'], self.lr['s_lr'], self.lr['rgb_lr'], self.lr['mlp_lr']) #0.001
                self.optim_pose = self.lr['pose_lr']>0
                self.lamudas['depth_lamuda'] = self.train_info['depth_lamuda']
                self.lamudas['eik_lamuda'] = self.train_info['eik_lamuda'] 
                self.lamudas['smoothing_lamuda'] = self.train_info['smoothing_lamuda'] 

        def init_loss(self):
                self.losses = {}
                keys = ['depth', 'eikonal', 'smooth', 'near', 'far', 'normal', 'rgb', 'vit', 'semantic', 'instance', 'entropy', 'query', 'assignment', 'pose', 'gt_pose', 'brightness', 'img_depth']
                for key in keys:
                        self.losses[key] = []
                self.losses_occ = []
                # self.losses_depth = []
                # self.losses_tv = []
                # self.losses_eik = []
                # self.losses_near = []
                # self.losses_far = []
                # self.losses_normal1 = []
                # self.losses_normal2 = []
                # self.losses_rgb = []
                # self.losses_sem = []
                # self.losses_feat = []
                # self.losses_pose = []
                # self.losses_gt_pose = []

        def freeze_setting(self, epoch, rgb_epoch=2, all_epoch=5):
                if epoch == 0:
                        if not self.config['pretrain']:
                                if self.config['loss_term']['rgb']:
                                        self.rgb_loss_flag, self.nef.rgb_flag = False, False
                                        self.nef.freeze_param(channels=['rgb'])
                                if self.config['loss_term']['instance']:
                                        self.nef.freeze_param(channels=['semantic','instance','panoptic'])
                        else:
                                self.rgb_loss_flag, self.nef.rgb_flag = False, False
                                self.nef.freeze_param(channels=['sdf','rgb'])
                elif epoch == rgb_epoch:
                        if not self.config['pretrain'] and self.config['model_type']!='3dgs':
                                self.nef.freeze_param(channels=['sdf'])
                        self.rgb_loss_flag, self.nef.rgb_flag = True, True
                        self.nef.unfreeze_geometry(channels=self.nef.channels[1:])
                        # self.nef.freeze_param(channels=['sdf'])
                elif epoch == all_epoch:
                        if not self.config['pretrain'] and self.config['model_type']!='3dgs':
                                self.nef.unfreeze_geometry(channels=['sdf'])
                        # if self.panoptic_flag:
                        #         self.nef.freeze_query_param()
                


        def record_loss(self, loss_occ, loss): #, nor_error1, nor_error2
                self.losses_occ.append(loss_occ)
                for key in loss.keys():
                        if loss[key]>0:
                                self.losses[key].append(loss[key])
                # self.losses_depth.append(loss['depth'])
                # self.losses_tv.append(loss['smooth'])
                # self.losses_eik.append(loss['eikonal'])
                # self.losses_near.append(loss['near'])
                # self.losses_far.append(loss['far'])
                # # self.losses_normal1.append(nor_error1)
                # # self.losses_normal2.append(nor_error2)
                # if loss['rgb']>0:
                #         self.losses_rgb.append(loss['rgb']) 
                # self.losses_sem.append(loss['semantic'])
                # if loss['feature']>0:: #self.feature_flag
                #         self.losses_feat.append(loss['feature'])
                # if loss['normal']>0:
                #         self.losses_normal.append(loss['normal'])
                # self.losses_pose.append(loss['pose'])
                # self.losses_gt_pose.append(loss['gt_pose'])

        def mean_loss(self, e, show=False):
                losses_mean = {}
                for key in self.losses.keys():
                        if len(self.losses[key])>0:
                                losses_mean[key] = np.stack(self.losses[key]).mean()
                losses_occ_mean = np.stack(self.losses_occ).mean()
                rgb_psnr = mse2psnr_np(losses_mean['rgb']) if len(self.losses['rgb'])>0 else 0
                losses_mean['psnr'] = rgb_psnr
                log_depth = f"Train Report : epoch={e}, occ_loss={losses_occ_mean}, occ_lr={self.nef.optim_occ.state_dict()['param_groups'][0]['lr']}"
                for key in losses_mean.keys():
                        log_depth += ', '+key+f'_mean={losses_mean[key]}'
                print(log_depth)
                # logs.append(log_depth+"\n")
                if show:
                        self.loss = losses_occ_mean
                        self.losses_mean = losses_mean
        
        def tensorboard_show(self, writer, its):
                writer.add_scalar("Loss", self.loss, its)
                for key in self.losses_mean.keys():
                        writer.add_scalar(key, self.losses_mean[key], its)


        def train(self, train_data, train_list_key_frame, e, tqdm_t, train_data_vit=None, cos_anneal_ratio=0, random_list=None): #, nor_error1, nor_error2, sem_error \
                self.nef.iter_n += 1
                self.nef.optim_occ.zero_grad()
                # if 'instance' in self.nef.grid.keys():
                #         self.nef.instance_optimizer.zero_grad()

                # else:
                # =================reconstruction=================
                loss_all, Loss_val, loss, Loss_ins = self.get_rays_loss(
                                                train_data,
                                                train_list_key_frame,
                                                # multi_cube,
                                                epoch_n=e, 
                                                cos_anneal_ratio=cos_anneal_ratio,
                                                random_list=random_list)
                # self.record_loss(Loss_val, loss)
                postfix = {'s':self.nef.s.data.item()}
                
                if self.optim_flag['feature'] : #and e>=5
                        # =================SAM ViT=================
                        loss_vit = self.get_vit_loss(
                                                                        train_data_vit, 
                                                                        train_list_key_frame,
                                                                        epoch_n=e, 
                                                                        cos_anneal_ratio=cos_anneal_ratio)
                        # postfix = {}
                        loss['vit'] = loss_vit.item()
                        postfix['vit'] = loss_vit.item()
                        loss_all += loss_vit
                
                self.record_loss(Loss_val, loss)                        
                loss_all.backward(retain_graph=True)
                self.nef.optim_occ.step()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()

                # if 'instance' in self.nef.grid.keys() and Loss_ins>0:
                #         postfix['instance'] = Loss_ins.cpu().item()
                #         # print(f"instance_loss: {Loss_ins.cpu().item()}")    
                #         torch.autograd.set_detect_anomaly(True)                   
                #         Loss_ins.backward(retain_graph=True)
                #         self.nef.instance_optimizer.step()
                
                if self.optim_pose:
                        if self.nef.epoch>=self.pose_epoch:
                                # torch.nn.utils.clip_grad_norm_(self.nef.R_cam, 1, norm_type=2)
                                self.nef.pose_optimizer.step()
                                self.nef.pose_optimizer.zero_grad()
                                self.nef.pose_update()
                                with torch.no_grad():
                                        self.nef.plot_traj()
                        else:
                                self.nef.pose_optimizer.zero_grad()
                
                for key in loss.keys():
                        if loss[key]>0:
                                postfix[key] = loss[key]
                tqdm_t.set_postfix(postfix)


        #----------------------------------Loss------------------------------- 
        from PIL import Image
        import numpy as np
        @torch.no_grad()
        def get_vit_loss(self, train_data, data_list, epoch_n=0, cos_anneal_ratio=1):
                ray_chunk_test = 10000 #32768 #self.config['params']['ray_chunk_test']

                # self.nef.iter_n += 1
                # self.nef.optim_occ.zero_grad()

                loss = {}
                loss_ = 0 #torch.FloatTensor([]).to(self.nef.device)
                # with tqdm(total=len(train_data)) as t:
                        # t.set_description('Render vit img')
                for ii, data in enumerate(train_data):

                        with torch.no_grad():
                                # gt_samvit = data['gt_samvit'][0].to(self.nef.device)  # [1,256,64,64]
                                # DINOv2 feature ground truth
                                img_path = os.path.join(os.environ['KITTI360_DATASET']+"/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect", '%010d.png' % data['idx'])
                                # gt_samvit = self.dino2.get_feature(img_path)     
                                gt_samvit = self.dino2.get_feature_PCA(data_list.index(data['idx']))  
                                # gt_samvit = gt_samvit[:,:,:int(64*376/1408)+1,:] #sam
                                h, w = data['vit_patch'][0], data['vit_patch'][1]

                        render_pose_c2w = data['pose_cam'] #[0]
                        # rays_depth_gt = data['depth'].squeeze()
                        #cam
                        ray_origin_cam = data['origin'].squeeze()
                        ray_directions_cam = data['direction_vit_img'].squeeze()
                        # ray_directions_cam = data['direction_img'].squeeze()

                        normalize_uv = data['vu_vit']
                        normalize_uv[:,0] /= data['vit_patch'][0]
                        normalize_uv[:,1] /= data['vit_patch'][1]

                        # debug
                        # pts_test = ray_origin_cam + 5*ray_directions_cam
                        # o3d_pts = o3d.geometry.PointCloud()
                        # o3d_pts.points = o3d.utility.Vector3dVector(pts_test.cpu().numpy())
                        # o3d.io.write_point_cloud('test_vit.ply', o3d_pts)
                       
                        rays_all_num = h * w #ray_directions_cam.size(0)
                        # rays_all_num = 376*1408 

                        # always use 64x64 features as SAM default to 1024x1024
                        # h, w = data['h'], data['w']
                        # rays_o_hw = data['rays_o_lr']
                        # rays_d_hw = data['rays_d_lr']
                        # outputs = self.model.render(rays_o_hw, rays_d_hw, staged=False, index=index, bg_color=bg_color, perturb=False, cam_near_far=cam_near_far, return_feats=1, H=h, W=w)
                        pred_samvit = torch.FloatTensor([]).to(self.nef.device)
                        pred_pemap = torch.FloatTensor([]).to(self.nef.device)
                        valid_mask_img = torch.FloatTensor([]).to(self.nef.device)
                        pred_normal = torch.FloatTensor([]).to(self.nef.device)
                        # rgb = torch.FloatTensor([]).to(self.nef.device)
                        for num in range(0, math.ceil(rays_all_num/ray_chunk_test)):
                                i = num * ray_chunk_test
                                i_next = i+ray_chunk_test if (i+ray_chunk_test <rays_all_num) else rays_all_num

                                rays_o, rays_d = get_rays(ray_origin_cam, ray_directions_cam[i:i_next,:], render_pose_c2w) 
                                rays_go = get_rays_sfm(rays_o[None,:,:], self.nef.scale, self.nef.origin).squeeze()
                                # z_near, z_far = instection(rays_go, rays_d, ground_truth['depth'])
                                rays_g = Rays(origins=rays_go, dirs=rays_d) #, dist_min=z_near, dist_max=z_far
                                
                                info = {'frame_id':(data['frame_id']*torch.ones(rays_g.shape[-1])).to(self.nef.device),
                                        'vu':normalize_uv[i:i_next,:]}
                                outputs, valid_mask = self.nef.Render( 
                                                rays_g, 
                                                info,
                                                self.nef.iter_n,
                                                epoch_n,
                                                cos_anneal_ratio,
                                                flag=1, 
                                                output='pre'
                                                ) 
                                pred_samvit = torch.vstack((pred_samvit, outputs['samvit'])) if pred_samvit.shape[0]>0 else outputs['samvit'] 
                                pred_pemap = torch.vstack((pred_pemap, outputs['dino_pe'])) if pred_pemap.shape[0]>0 else outputs['dino_pe'] 
                                valid_mask_img = torch.vstack((valid_mask_img, valid_mask)) if valid_mask_img.shape[0]>0 else valid_mask
                                # rgb = torch.vstack((rgb, outputs['rgb'])) if rgb.shape[0]>0 else outputs['rgb']
                                # pred_normal = torch.vstack((pred_normal, outputs['normal'])) if pred_normal.shape[0]>0 else outputs['normal'] 
                                # clean_cache()

                        features = pred_samvit.reshape(h, w, -1)
                        pemap = pred_pemap.reshape(h, w, -1)
                        pred_samvit = pred_samvit.reshape(1, h, w, -1).permute(0, 3, 1, 2).contiguous()
                        pred_pemap = pred_pemap.reshape(1, h, w, -1).permute(0, 3, 1, 2).contiguous()
                        gt_samvit_inter = F.interpolate(gt_samvit,pred_samvit.shape[-2:], mode='bilinear') #gt_samvit[None,:]
                        valid_mask_img = valid_mask_img.reshape(1, h, w, 1).permute(0, 3, 1, 2).contiguous()
                        # pred_samvit_inter = F.interpolate(pred_samvit, gt_samvit.shape[-2:], mode='bilinear')
                        # valid_mask_img = F.interpolate(valid_mask_img.float(), gt_samvit.shape[-2:], mode='bilinear')
                        # pred_pemap_inter = F.interpolate(pred_pemap, gt_samvit.shape[-2:], mode='bilinear')
                        valid_mask_img = F.interpolate(valid_mask_img.float(), pred_samvit.squeeze().shape[-2:], mode='bilinear')
                        valid_mask_img = valid_mask_img.squeeze()>0.5
                        
                        # with torch.no_grad():
                        # #         #vis
                        #         features_pca = self.PCA.fit_transform(features.reshape(-1,features.shape[-1]).detach().cpu().numpy())
                        #         features_pca = minmax_scale(features_pca).reshape(h,w,3) * 255.
                        #         cv2.imwrite('/sdb1/yx/IROS2024/Reconstruction/scannet/0087_02/1_geo_rgb_sem_ins/v0/dino_render_img/%05d.png'%data_list[int(data['frame_id'])], features_pca)   

                        #         # features_pca = self.PCA.fit_transform(pemap.reshape(-1,pemap.shape[-1]).detach().cpu().numpy())
                        #         # features_pca = minmax_scale(features_pca).reshape(h,w,3) * 255.
                        #         # # cv2.imwrite('test.jpg', features_pca)   
                        #         # cv2.imwrite('/sdb1/yx/IROS2024/Reconstruction/scannet/0087_02/1_geo_rgb_sem_ins/v0/dino_render_img/%05d.png'%data_list[int(data['frame_id'])], features_pca)   

                        # loss
                        criterion = torch.nn.MSELoss(reduction='mean')
                        loss_img = criterion((pred_samvit+pred_pemap).squeeze()[:,valid_mask_img], gt_samvit_inter.squeeze()[:,valid_mask_img])#.mean()
                        # loss_img = criterion((pred_samvit_inter+pred_pemap_inter).squeeze()[:,valid_mask_img], gt_samvit.squeeze()[:,valid_mask_img])#.mean()
                        if loss_img.isnan():
                                yx =1 
                        loss_ += loss_img #torch.hstack((loss_.squeeze(), loss_img.squeeze())) if loss_.shape[0]>0 else loss_img.squeeze()
                        # t.update(1)       
                loss["vit"] = loss_/len(train_data) #.mean()
                # Loss = 0
                # if loss['vit'] > 0:
                #         Loss += self.lamudas['feature_lamuda'] * loss['vit'] 
                # # Loss += outputs['eik_loss']

                # # print(self.occ_grid.grad)
                # if torch.isnan(Loss).sum()>0:
                #         yx = 1
                # # with torch.autograd.detect_anomaly():
                # Loss.backward(retain_graph=True)
                # self.nef.optim_occ.step()

                # return pred_samvit_inter, gt_samvit_inter, loss
                return self.lamudas['feature_lamuda'] * loss["vit"]


        def get_rays_loss(self, rayses, data_list, epoch_n=0, cos_anneal_ratio=0, random_list=None):
                t1 = time.time()
                random_list_cam = random_list[1]
                random_list = torch.hstack((random_list[0], random_list[1])).int()
                if random_list is None:
                        rays_w = self.nef.get_rays_World(rayses) 
                else:
                        rays_w = self.nef.get_rays_World_random_frame(rayses, random_list) 
                frame_id = rays_w['frame_id'].reshape(-1)
                rays_w = rays_w['rays_w']
                rays_g = get_rays_sfm(rays_w, self.nef.scale, self.nef.origin)
                
                ## =============== read ovservaion (GT) ===============
                # Ray：[o1,o2,o3,d1,d2,d3,n1,n2,n3,c1,c2,c3,s,i,deph]
                ## ====================================================
                # rays_g_all = torch.reshape(rays_g,((-1,self.ray_long)))
                rays_g_all = torch.reshape(rays_g,((-1, rays_g.shape[-1])))

                ground_truth = {'depth':rays_g_all[:,-2]}
                ground_truth['uv'] = rays_g_all[:,-1]
                if self.depth_completion:
                        ground_truth['confidence'] = rays_g_all[:,-3]
                # if self.feature_flag:
                #         ground_truth['feature'] = rays_g_all[:, (6+self.rgb_flag*3+self.normal_flag*3):(6+self.rgb_flag*3+self.normal_flag*3+self.feature_flag*self.nef.feat_dim)] #TODO
                if self.rgb_flag:
                        if self.config['model_type'] != '3dgs':
                                ground_truth['rgb'] = rays_g_all[:, 6+self.normal_flag*3:9+self.normal_flag*3]
                if self.semantic_flag:
                        semantic_gt = rays_g_all[:,-3-self.instance_flag:-2-self.instance_flag].clone().detach().int()
                        
                        # ## delete sky
                        # not_sky_mask = (semantic_gt!=23).squeeze() # sky=23
                        # semantic_gt = semantic_gt[not_sky_mask,:].squeeze()
                        # rays_g_all = rays_g_all[not_sky_mask,:]
                        # frame_id = frame_id[not_sky_mask]

                        ground_truth['semantic'] = semantic_gt.squeeze()  
                if self.instance_flag:
                        instance_gt = rays_g_all[:,-3:-2].clone().detach().int()
                        ground_truth['instance'] = instance_gt  
                if self.panoptic_flag:
                        semantic_gt = rays_g_all[:,-3-self.panoptic_flag:-2-self.panoptic_flag].clone().detach().int()
                        instance_gt = rays_g_all[:,-3:-2].clone().detach().int()
                        self.stuff_map = torch.zeros([len(self.config['Stuff_class'])+len(self.config['Thing_class'])+1]).to(semantic_gt.device)
                        for label_new, label_raw in enumerate(self.config['Stuff_class']):
                                self.stuff_map[label_raw] = label_new+1
                        panoptic_gt = self.stuff_map[semantic_gt.clone().long()].int()
                        Thing_flag = torch.isin(semantic_gt, torch.tensor(self.config['Thing_class']).to(semantic_gt.device))
                        panoptic_gt[Thing_flag] = instance_gt[Thing_flag]+len(self.config['Stuff_class'])
                        # panoptic_gt = semantic_gt.clone()
                        ground_truth['panoptic'] = panoptic_gt 
                        ground_truth['semantic'] = semantic_gt.squeeze()  
                        ground_truth['instance'] = instance_gt  
                if self.normal_flag:
                        ground_truth['normal'] = rays_g_all[:, 6:9]
                ground_truth['frame_id'] = frame_id

                rays_go, rays_gd = rays_g_all[:,:3], rays_g_all[:,3:6]
                z_near, z_far = instection(rays_go, rays_gd, ground_truth['depth'])

                rays_g = Rays(origins=rays_go, dirs=rays_gd, dist_min=z_near, dist_max=z_far)
                # t2 = time.time()
                # loss_depth, loss_smooth, loss_eik, loss_near, loss_far, loss_rgb, loss_semantic, loss_feat = \
                loss = \
                        self.nef.Render( 
                                rays_g, 
                                ground_truth,
                                random_list_cam,
                                self.nef.iter_n,
                                epoch_n,
                                cos_anneal_ratio,
                                flag=1, 
                                output='loss'
                                ) 
               
                # t3 = time.time()
                #--------------------------------------------
                Loss = 0
                if loss['depth'] > 0:
                        # if self.nef.iter_n>self.nef.geometry_iter:
                        Loss += self.lamudas['depth_lamuda'] * loss['depth']
                        # Loss += self.lamudas['smoothing_lamuda'] * loss['smooth']
                if self.depth_completion:
                        Loss += 1.0 * loss['img_depth']
                Loss += self.lamudas['eik_lamuda'] * loss['eikonal']
                if loss['near'] > 0 and loss['far'] > 0:  #self.nef.iter_n<=self.nef.geometry_iter:
                        Loss += self.lamudas['sdf_near_lamuda'] * loss['near'] + self.lamudas['sdf_far_lamuda'] * loss['far']
                if self.rgb_flag and loss['rgb'] > 0 and self.rgb_loss_flag:
                        # if self.nef.iter_n>self.nef.geometry_iter:
                        Loss += self.lamudas['rgb_lamuda'] * loss['rgb']
                                # Loss += 5*loss['brightness']
                if self.feature_flag and loss['feature'] > 0:
                        Loss += self.lamudas['feature_lamuda'] * loss['feature']
                if self.normal_flag and loss['normal']>0:
                        if self.nef.iter_n>self.nef.geometry_iter:
                                Loss += self.lamudas['normal_lamuda'] * loss['normal']
                if self.semantic_flag and loss['semantic'] > 0:
                        Loss += self.lamudas['semantic_lamuda'] * loss['semantic']
                if self.instance_flag and loss['instance'] > 0:
                        # Loss += self.lamudas['instance_lamuda'] * loss['instance']
                        Loss_ins = loss['instance'] #+ 10 * loss['query']#\
                                #    +loss['assignment']
                                #    +loss['entropy']# \
                        # Loss_ins = 0
                        Loss += 30*loss['instance'] + 10 * loss['query']
 
                else:
                        Loss_ins = 0
              
              
                # print(self.occ_grid.grad)
                if torch.isnan(Loss).sum()>0:
                        yx = 1
                # # with torch.autograd.detect_anomaly():
                # Loss.backward(retain_graph=True)
         
                # self.nef.optim_occ.step()
                # if self.optim_pose:
                #         if self.nef.epoch>=self.pose_epoch:
                #                 # torch.nn.utils.clip_grad_norm_(self.nef.R_cam, 1, norm_type=2)
                #                 self.nef.pose_optimizer.step()
                #                 self.nef.pose_optimizer.zero_grad()
                #                 self.nef.pose_update()
                #                 with torch.no_grad():
                #                         self.nef.plot_traj()
                #         else:
                #                 self.nef.pose_optimizer.zero_grad()
                
                # loss_rgb = loss_rgb.cpu().item() if torch.is_tensor(loss_rgb) else loss_rgb
                Loss_val = Loss.cpu().item() if torch.is_tensor(Loss) else Loss

                for key in loss.keys():
                        loss[key] = loss[key].cpu().item() if torch.is_tensor(loss[key]) else loss[key]
                
                
                return Loss, Loss_val, loss, Loss_ins# loss_depth, loss_smooth, loss_eik, loss_near, loss_far, loss_rgb, loss_semantic, loss_feat #, normal_loss1, normal_loss2, sematic_loss

        #-------------------------------------------------------------
        def get_depth_lidar(self, rays_d, c2w, depth_gt):
                rays_o, rays_d = get_rays(rays_d, c2w) 
                rays_go = get_rays_sfm(rays_o[None,:,:], self.nef.scale, self.nef.origin)
                with torch.no_grad():
                        depth = self.tracer.get_hit(self.nef, rays_go[0,:,:], rays_d, depth_gt, None, self.nef.iter_n, flag=0)
                        points = rays_o + rays_d * depth
                        points_gt = rays_o + rays_d * depth_gt[:,None]
                return depth, points, points_gt
        
        def get_sdf_reander(self, rays_o, rays_d, c2w):
                rays_o, rays_d = get_rays(rays_o, rays_d, c2w) 
                rays_go = get_rays_sfm(rays_o[None,:,:], self.nef.scale, self.nef.origin)
                with torch.no_grad():
                        surface_point, normal, depth_SF, rgb, semantic, instance, panoptic = self.nef.sphere_tracing(rays_go[0,:,:], rays_d)

                        # depth_gt = rays_g_all[:,-1]
                        # z_near, z_far = instection(rays_go[0,:,:], rays_d, depth_gt)

                        # rays_g = Rays(origins=rays_go[0,:,:], dirs=rays_d, dist_min=z_near, dist_max=z_far)
                        # depth_render = self.nef.render(rays_g)

                return surface_point, normal, depth_SF, rgb, semantic, instance, panoptic

        def update_lamuda(self):
                # if self.lamudas['smoothing_lamuda']  < 10:
                #         self.lamudas['rgb_lamuda'] *= 1.2
                # self.lamudas['sdf_near_lamuda'] *= 0.8
                # self.lamudas['sdf_far_lamuda'] *= 0.8
                if self.lamudas['smoothing_lamuda']  > 10:
                        self.lamudas['smoothing_lamuda'] *= 0.9




def clean_cache():
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

@torch.no_grad()
def instection(rays_o,rays_d, depth_gt):
        z_near = 1.0 * torch.ones(depth_gt.size(0)).to(rays_o.device)
        # z_far = torch.zeros_like(depth_gt).to(rays_o.device)+80
        z_far = depth_gt+2
        z_far[depth_gt<0] = -depth_gt[depth_gt<0]+80
        return z_near, z_far


def expand_points(points, voxel_size):
    """
    A naive version of the sparse dilation.
    """
    # a cube with size=3 and step=1.
    cube_grids_3 = list(product(*zip([0, 0, -1], [0, 0, 0]))) #[-1, -1, -1],  [1, 1, 1]
    # add the offsets to the points.
    points_expanded = [
        points + np.array(grid_point) * voxel_size for grid_point in cube_grids_3
    ]
    points_expanded = np.concatenate(points_expanded, axis=0)
    return np.unique(points_expanded, axis=0)