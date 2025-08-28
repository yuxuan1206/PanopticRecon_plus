'''
Author: yuxuan1206 610939662@qq.com
Date: 2022-06-23 15:27:59
LastEditors: yuxuan1206 610939662@qq.com
LastEditTime: 2022-10-30 22:25:49
FilePath: /occuSLAM3D_indoor_KITTI/kitti_dataset.py
Description: 

Copyright (c) 2022 by yuxuan1206 610939662@qq.com, All Rights Reserved. 
'''
import imp
import os
from posixpath import split
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from torchvision import transforms as T
import sys
sys.path.append("..")
from models.rays import *

import open3d as o3d
from lietorch import SE3
# from .read_write_model import *
from PIL import Image
import cv2
from tqdm import tqdm



class ScannetDataset(Dataset):
    def __init__(self, config, data_list, device, vec_es, split, factor=1, optim_flag=None, normal_flag=False, semantic_flag=False):
        # self.base_dir = base_dir
        self.data_list = data_list
        self.device = device
        self.vec_es = vec_es
        self.split = split
        self.optim_flag = optim_flag
        self.factor = factor
        self.depth_completion = optim_flag['depth_completion']
        self.normal_flag = normal_flag
        self.semantic_flag = self.optim_flag['semantic2d'] or self.optim_flag['semantic3d'] or self.optim_flag['panoptic'] #semantic_flag
        self.instance_flag = self.optim_flag['instance'] or self.optim_flag['panoptic']
        self.rgb_flag = self.optim_flag['rgb'] 
        self.h, self.w = config['img']['height'], config['img']['width']
        self.Guassion_flag = config['3DGS']
        self.patch_size = 1 #8
        self.factor = factor
        self.sequence = config['path']['dataset_dir'].split('/')[-1]
        intrinsic_rgb = torch.from_numpy(np.loadtxt(os.path.join(config['path']['dataset_dir'], "intrinsic/intrinsic_color.txt")))
        if self.Guassion_flag:
            self.h_d, self.w_d = self.h, self.w 
            intrinsic_depth = intrinsic_rgb
        else:
            self.h_d, self.w_d = config['img']['height_d'], config['img']['width_d']
            intrinsic_depth = torch.from_numpy(np.loadtxt(os.path.join(config['path']['dataset_dir'], "intrinsic/intrinsic_depth.txt")))
        self.fx, self.fy, self.cx, self.cy = intrinsic_rgb[0, 0], intrinsic_rgb[1, 1], intrinsic_rgb[0, 2], intrinsic_rgb[1, 2]
        self.fx_d, self.fy_d, self.cx_d, self.cy_d = intrinsic_depth[0, 0], intrinsic_depth[1, 1], intrinsic_depth[0, 2], intrinsic_depth[1, 2]

        self.define_transforms()
        # self.velo = Kitti360Viewer3DRaw(mode='velodyne', seq=0)
        self.data_path = config['path']['dataset_dir']
        self.proj_dir = config['path']['proj_dir']
        if self.semantic_flag:
            self.semantic_2d_data_path = config['path']['semantic_2d_data_path']
        if self.instance_flag:
            self.instance_2d_data_path = config['path']['instance_2d_data_path']
            self.Thing_class = config['Thing_class']
            self.Stuff_class = config['Stuff_class']

        self.pc_radius = config['ray']['pc_radius']
        self.min_z = config['ray']['min_z']
        self.max_z = config['ray']['max_z']
        self.depth_min = config['ray']['depth_min']
        # print(f"Get Dataset from : {self.data_path} - {data_list}")

        if self.split=="train":
            self.init_rays()


    def init_rays(self):
        self.rgb_list, self.semantic_list, self.instance_list, self.depth_list = [], [], [], []

        # self.init_img_depth()
        ## depth ray
        pix = torch.arange(0, self.h_d*self.w_d, 1, dtype=torch.float32)
        v = pix.div(self.w_d, rounding_mode='floor')
        u = pix % self.w_d  
        rays_d = torch.stack([(u - self.cx_d) / self.fx_d, (v - self.cy_d) / self.fy_d, torch.ones_like(u)], dim=-1) #.to(self.device)
        # rays_d = rays_d / rays_d.norm(2,1)[:, None]
        rays_o = torch.FloatTensor([0,0,0]).expand(rays_d.shape)

        rays_all = []
        self.semantic_list = []
        for i in tqdm(self.data_list):  
        # for i in self.data_list:  
            if self.Guassion_flag:
                imagePath = os.path.join("/sdb1/xieyx/code/GaussianPro/scannet/scene0088_00/depth_3dgs", f'{i}.png')
            else:
                imagePath = os.path.join(self.data_path, f'depth/{i}.png')
            depth = Image.open(imagePath) #[176:,:] # sky=23
            self.depth_list.append(torch.from_numpy(np.array(depth.resize((self.w, self.h), Image.NEAREST))/1000.))
            depth = torch.FloatTensor(np.array(depth)/1000.) 
            # depth += torch.rand(depth.shape)/20
            # # depth to range
            # depth = depth.reshape(-1)/rays_d[:,-1]

            # mask = torch.logical_and(depth.reshape(-1)>0.8, depth.reshape(-1)<10) #TODO
            mask = torch.ones_like(depth.reshape(-1)).bool()
            depth_, rays_o_, rays_d_ = depth.reshape(-1)[mask], rays_o[mask.reshape(-1)], rays_d[mask.reshape(-1)]
            # depth_, rays_o_, rays_d_ = depth.reshape(-1), rays_o, rays_d
            rays_lidar = [rays_o_, rays_d_]

            if self.optim_flag['rgb']:
                rays_lidar.append(torch.zeros_like(rays_d_))
            # if self.optim_flag['feature']:
            #     rays_lidar.append(torch.zeros([rays_o.shape[0], self.feat_dim]))
            if self.semantic_flag:
                rays_lidar.append(torch.zeros_like(depth_.reshape(-1,1)).to(int))
            if self.instance_flag:
                rays_lidar.append(torch.zeros_like(depth_.reshape(-1,1)).to(int))

            rays_lidar.append(depth_.reshape(-1,1))
            rays_lidar.append(torch.IntTensor(range(depth.shape[0]*depth.shape[1]))[mask].reshape(-1,1)) #uv
            rays = torch.cat(rays_lidar,dim=1)    
            rays_all.append(rays.reshape(-1,rays.shape[-1]))

            ## -------------------------- 2D-image 10hz --------------------------

            # semantic_imgs
            if self.optim_flag['semantic2d'] or self.optim_flag['panoptic']:
                imagePath = os.path.join(self.semantic_2d_data_path, f'{i}.png')
                try:
                    semantic_img = np.array(Image.open(imagePath))
                except:
                    print(f"No such file: {imagePath}")
                    semantic_img = 0 * np.ones([self.h, self.w], dtype=int)
                # edge = edge_detector(imagePath)
                # semantic_img[edge>0] = 0
                self.semantic_list.append(torch.from_numpy(semantic_img))
            
            # 2d_rgb_imgs
            if self.optim_flag['rgb']:
                rgb_img_path = os.path.join(self.data_path, f'color/{i}.jpg')
                rgb_img = np.array(Image.open(rgb_img_path)) / 255.0 #[176:,:] # sky=23
                # rgb_img[:20,:,:] = 0
                # rgb_img[-20:,:,:] = 0
                # rgb_img[:,:10,:] = 0
                # rgb_img[:,-10:,:] = 0
                self.rgb_list.append(torch.from_numpy(rgb_img))

            # instance_imgs
            if self.instance_flag:
                imagePath = os.path.join(self.instance_2d_data_path, f'{i}.png') 
                if os.path.exists(imagePath):
                    # instance_img = np.array(Image.open(imagePath))
                    instance_img = cv2.imread(imagePath, -1) # >255 ID np.int16
                else:
                    print(f"No such file: {imagePath}")
                    instance_img = 0 * np.ones([self.h, self.w], dtype=int)
                if instance_img.dtype==np.uint16:
                    instance_img = instance_img.astype(np.int16)
                # if instance_img.max() > 10:
                #     yx = 1
                self.instance_list.append(torch.from_numpy(instance_img))

        self.imgs = rays_all
        self.rays_all = torch.vstack(rays_all) #.to(self.device)
        self.semantic_list = torch.stack(self.semantic_list, dim=0) if len(self.semantic_list)>0 else None
        self.rgb_list = torch.stack(self.rgb_list, dim=0) if len(self.rgb_list)>0 else None
        self.instance_list = torch.stack(self.instance_list, dim=0) if len(self.instance_list)>0 else None
        self.depth_list = torch.stack(self.depth_list, dim=0) if len(self.depth_list)>0 else None
        # print("Get all rays num = ", self.rays_all.shape[0])


    def init_img_depth(self):
        self.depth_list = []
        for i in self.data_list:
            #semantic_imgs
            imagePath = os.path.join(self.data_path, f'depth/{i}.png')
            depth_img = np.array(Image.open(imagePath)) #[176:,:] # sky=23

            self.depth_list.append(torch.from_numpy(depth_img).reshape(-1))
        self.depth_list = torch.stack(self.depth_list, dim=0)

    def define_transforms(self):
        self.transform = T.ToTensor()
    
    def __len__(self):
        if self.split == "train":
            return self.rays_all.shape[0]
        else:
            return len(self.data_list)
    
    # def max_len(self, max_len_):
    #     self.max_len = max_len_
        
    def load_color(self, path):
        img = PILImage.open(path)
        img = img.resize((self.w, self.h), PILImage.LANCZOS) # same size with depth imgs
        return self.transform(img).permute(1,2,0).to(self.device)
    
    def set_random_list(self, random_list=None):
        self.random_list_geo = random_list[0]
        self.random_list_rgb = random_list[1]


    def __getitem__(self, idxes):
        #         #         if self.split == "train":
            idx = idxes[0]
            idx2 = idxes[1]
            # return self.rays_all[idx, :]
            # return torch.stack([self.imgs[imgnum, idx, :] for imgnum in range(self.imgs.size(0))])
            sample_batch = []
            batch_cam, batch_lidar = [], []
            max_len = max(i.size(0) for i in idx)
            
            frames_list = self.imgs if self.random_list_geo is None else self.random_list_geo
            for i,imgnum in enumerate(frames_list):
    
                ## -------------------------- 3D-lidar --------------------------
                sample = self.imgs[imgnum][idx[i], :]
                pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, max_len-sample.size(0)))
                # batch_list = [pad(sample)]
                batch_lidar.append(pad(sample))
                
            for i,imgnum in enumerate(self.random_list_rgb):
                ## -------------------------- 2D-image --------------------------
                # semantic_val = self.semantic_list[imgnum,idx2[imgnum]] #.to(self.device, non_blocking=True)
                v = idx2[i].div(self.w//self.patch_size, rounding_mode='floor')
                u = idx2[i] % (self.w // self.patch_size)
                
                if self.optim_flag['rgb'] | self.optim_flag['semantic2d']:
                    patch_u, patch_v = torch.meshgrid(torch.arange(self.patch_size), torch.arange(self.patch_size))
                    patch_u = (u[:,None]*self.patch_size+patch_u.reshape(-1)).reshape(-1)
                    patch_v = (v[:,None]*self.patch_size+patch_v.reshape(-1)).reshape(-1)
                    rays_d_patch = torch.stack([(patch_u - self.cx) / self.fx, (patch_v - self.cy) / self.fy, torch.ones_like(patch_u)], dim=-1).to(torch.float32) #.to(self.device)
                    # rays_d_patch = rays_d_patch / rays_d_patch.norm(2,1)[:, None]
                    # get patch rays in velodyne frame 
                    # rays_o_patch = self.TrCam0ToVelo[:3,3].expand(patch_u.size(0),-1)
                    # rays_d_patch = rays_d_patch @ self.TrCam0ToVelo[:3, :3].T #.squeeze()
                    # get patch rays in camera frame
                    rays_o_patch = torch.FloatTensor([0,0,0]).expand(patch_u.size(0),-1)
                    rays_patch = [rays_o_patch, rays_d_patch] #  /rays_d_patch.norm(2,1)[:, None]!!!
                    
                    if self.optim_flag['normal']:
                        rays_patch.append(torch.zeros_like(rays_d_patch))
                    if self.optim_flag['rgb']:
                        rgb_patch = self.rgb_list[imgnum, patch_v.long(), patch_u.long()].float()
                        rays_patch.append(rgb_patch)
                    # if self.optim_flag['feature']:
                    #     rays_patch.append(torch.zeros([rays_d_patch.shape[0], self.feat_dim]))
                    if self.optim_flag['semantic2d'] or self.optim_flag['panoptic']:
                        sem_patch = self.semantic_list[imgnum, patch_v, patch_u] 
                        rays_patch.append(sem_patch[:,None])
                    if self.instance_flag:
                        ins_patch = self.instance_list[imgnum, patch_v, patch_u] 
                        rays_patch.append(ins_patch[:,None])
                    if self.depth_completion:
                        depth_patch = self.depth_list[imgnum, patch_v.long(), patch_u.long()].float() * rays_d_patch.norm(2,1)
                        conf_patch = self.conf_list[imgnum, patch_v.long(), patch_u.long()].float()
                        rays_patch.append(conf_patch[:,None])
                        rays_patch.append(depth_patch[:,None])
                    else:
                        depth_patch = - self.depth_list[imgnum, patch_v.long(), patch_u.long()].float() - 1e-5
                        rays_patch.append(depth_patch.reshape(-1,1))
                        # rays_patch.append(-1.0 * torch.ones(rays_d_patch.shape[0])[:,None]) # image has no depth observation
                    rays_patch.append(idx2[i].reshape(-1,1)) #uv TODO
                    rays_patch = torch.cat(rays_patch, dim=1)
                    batch_cam.append(rays_patch)
                    
            if len(batch_cam)==0:
                sample_batch = batch_lidar
            else:
                batch_lidar = [torch.vstack((ray, torch.zeros(batch_cam[0].shape[0]-ray.shape[0], ray.shape[1]))) for ray in batch_lidar]
                sample_batch = batch_lidar + batch_cam
            return torch.stack(sample_batch).to(self.device) #, torch.stack(sample_batch_sem).to(self.device)
        else:
            idx = idxes
            data = {}  
            data['frame_id'] = idxes      
            data_idx = int(self.data_list[idx])
            data['idx'] = data_idx

            if f"{data_idx}" not in self.vec_es.keys():
                return {}

            pose = SE3(self.vec_es[f"{data_idx}"]).matrix().to(self.device)                   
            data['pose_es'] = pose
            data['pose_gt'] = pose
            data['pose_cam'] = pose
            
            # points = self.lidar_data[idx].to(self.device)
            # data['points'] = points
            # depth = torch.norm(points,dim=1).to(self.device)[:,None]
            # depth = self.imgs[idx][:,-1:].to(self.device)       
            # data['depth'] = depth

            # data['direction'] = (points/depth).to(self.device) 
            # data['semantic'] = np.zeros(depth.shape, dtype=int)

            w,h = self.w/self.factor, self.h/self.factor
            fx, fy, cx, cy = self.fx/self.factor, self.fy/self.factor, self.cx/self.factor, self.cy/self.factor,
            pix = torch.arange(0, h * w, 1)
            v = pix.div(w, rounding_mode='floor')
            u = pix % w
            rays_d_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1).float() #.to(self.device)
            rays_d_cam = rays_d_cam / rays_d_cam.norm(2,1)[:, None]
            # rays_o = self.TrCam0ToVelo[:3,3].expand(u.size(0),-1)
            # rays_o = self.TrCam0ToVelo[:3,3]
            # rays_d = rays_d_cam @ self.TrCam0ToVelo[:3, :3].T #.squeeze()
            rays_o = torch.FloatTensor([0,0,0])
            rays_d = rays_d_cam
            data['origin'] = rays_o.to(self.device)
            data['direction_img'] = rays_d.to(self.device)

            if self.optim_flag['feature']:
                # vit_scale = 16 * self.h // 1024 #[16 * self.h // 1024,
                vit_scale = 3
                data['vit_patch'] = [self.h // vit_scale, self.w // vit_scale]
                # data['gt_samvit'] = self.feat_list[idx]
                intrinsics = torch.FloatTensor([self.fx, self.fy, self.cx, self.cy]).to(self.device)

                data['direction_vit_img'], data['vu_vit'] = get_patch_rays(intrinsics/vit_scale, data['vit_patch'][0], data['vit_patch'][1], device=self.device)
                # data['intrinsics'] = intrinsics

            return data
 
    def get_every_img_batch(self):
        # return self.imgs.size(1)
        sample_num = [self.imgs[imgnum].size(0) for imgnum in range(len(self.imgs))]
        return torch.Tensor(sample_num)

from packaging import version as pver
def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

@torch.cuda.amp.autocast(enabled=False)
def get_patch_rays(intrinsics, H, W, patch_size=1, device='cpu'):
    ''' get rays
    Args:
        poses: [N/1, 4, 4], cam2world
        intrinsics: [N/1, 4] tensor or [4] ndarray
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        i, j: [N]
    '''
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    results = {}

    zs = torch.ones_like(i) 
    xs = (i - cx) / fx
    ys = (j - cy) / fy 

    directions = torch.stack((xs, ys, zs), dim=-1) # [N, 3]
    # do not normalize to get actual depth, ref: https://github.com/dunbar12138/DSNeRF/issues/29
    rays_d = directions / torch.norm(directions, dim=-1, keepdim=True) 

    return rays_d, torch.hstack((j[:,None],i[:,None])) #results


if __name__=='__main__':
    yaml_path = 'config/render_scannet_few.yaml'
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda:0")
    all_pose_list = range(cfg['train']['all_pose'][0], cfg['train']['all_pose'][-1])
    from utils.scannet_data_tools.init_data import init_all_poses,  init_poses_from_gt
    vec_es, vec_cam, vec_gt, traj_gt, traj_loam, odom, vec_loam  = init_all_poses(cfg['path']['dataset_dir'], int(cfg['path']['proj_dir'][-1]), all_pose_list, device) 
    frame_list = list(range(cfg['train']['all_pose'][0],cfg['train']['all_pose'][-1],cfg['frame']['step'])) 

    frame_start = 0 #0
    frame_end = frame_start +  cfg['frame']['num'] #16
    train_list = frame_list[frame_start:frame_end]
    optim_flag = {'rgb':True, 'semantic':False, 'normal':True, 'feature':False}
    dataset_train = KITTIDataset(cfg, train_list, device, vec_gt, vec_es, vec_cam, "train", optim_flag=optim_flag)