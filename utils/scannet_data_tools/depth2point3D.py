# coding=utf-8

from random import random
import numpy as np
from argparse import Namespace
import os
import math
import imageio
import open3d as o3d
import sys

from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torchvision import transforms as T
# from zmq import device
from tqdm import tqdm

def bigger_pc(pc, delta=0.2, N=3):
    pc_bigger = []
    # num = int(pc.shape[0] * 0.5)
    pc_bigger.extend(pc)
    for i in range(N):
        # idx = torch.LongTensor(random.sample(pc.shape[0], num))
        # pc_new = pc[idx] + delta*torch.rand_like(pc[idx])
        pc_new = pc
        pc_new[:,1] += delta*torch.rand_like(pc[:,1])
        pc_bigger.extend(pc_new)
    pc_bigger = torch.stack(pc_bigger)
    return pc_bigger


def read_point_cloud(filename: str, min_range, z_th):
    # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
    if ".bin" in filename:
        points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3]
    elif ".ply" in filename or ".pcd" in filename:
        pc_load = o3d.io.read_point_cloud(filename)
        points = np.asarray(pc_load.points)
    else:
        sys.exit("The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)")
    z = points[:, 2]
    points = points[z > z_th]
    points = points[np.linalg.norm(points, axis=1) >= min_range]
    pc_out = o3d.geometry.PointCloud()
    pc_out.points = o3d.utility.Vector3dVector(points)
    return pc_out
    
def depths2points(points, poses):
    # intrinsic_inv = np.linalg.inv(intrinsic[:3,:3])
    points_all = []
    for point, pose in zip(points, poses):
        points_world = (pose[:3, :3] @ point.T).T + pose[:3,3] 
        points_all.append(points_world)
        if torch.isnan(points_world.max()):
                yx = 1
    return np.vstack(points_all)

def get_points_bounds(p):
    bounds = np.array( [ [p[:,0].min(), p[:,0].max()],
                         [p[:,1].min(), p[:,1].max()],
                         [p[:,2].min(), p[:,2].max()] ])
    return bounds

def scale_to_unit_cube(points, y_level=-0.5):

    bbox = get_points_bounds(points)
    loc = (bbox[0] + bbox[1]) / 2
    # scale = 2. / (bbox[1] - bbox[0]).max()
    scale = 2. / ((bbox[1] - bbox[0]).max() * 1.05)
    vertices_t = (points - loc.reshape(-1, 3)) * scale
    y_min = min(vertices_t[:, 1])

    # create_transform_matrix
    S_loc = np.eye(4)
    S_loc[:-1, -1] = -loc
    # create scale mat
    S_scale = np.eye(4) * scale
    S_scale[-1, -1] = 1
    # create last translate matrix
    S_loc2 = np.eye(4)
    S_loc2[1, -1] = -y_min + y_level

    S = S_loc2 @ S_scale @ S_loc
    
    points_trans = (S[:3,:3] @ points.transpose()).transpose() + S[:3,3]
    return points_trans, S

def cal_T(points, pose_t):

    bbox = get_points_bounds(points)
    loc = (bbox[0] + bbox[1]) / 2
    # scale = 2. / (bbox[1] - bbox[0]).max()
    scale = 2. / ((bbox[1] - bbox[0]).max() * 1.2)
    vertices_t = (points - loc.reshape(-1, 3)) * scale
    y_min = min(vertices_t[:, 1])

    # create_transform_matrix
    S_loc = np.eye(4)
    # S_loc[:-1, -1] = -loc
    S_loc[:-1, -1] = -pose_t.cpu().numpy()

    # create scale mat
    S_scale = np.eye(4) * scale
    S_scale[-1, -1] = 1
    # create last translate matrix
    # S_loc2 = np.eye(4)
    # S_loc2[1, -1] = -y_min + y_level

    # S = S_loc2 @ S_scale @ S_loc
    S = S_scale @ S_loc
    
    # points_trans = (S[:3,:3] @ points.transpose()).transpose() + S[:3,3]
    return S

def genarate_points(points, S):    
    points_trans = (S[:3,:3] @ points.transpose()).transpose() + S[:3,3]
    return points_trans

def cal_world2grid_update(path_base, data_idx, pose, scale, y_level=-0.5):
    
    point_ = o3d.io.read_point_cloud(os.path.join(path_base,f"{data_idx}.ply"))
    points = torch.FloatTensor(point_.points)
    device = "cuda:0"
    p = points.to(device) @ pose[:3, :3].T + pose[:3,3]  #world

    bbox = torch.FloatTensor( [ [p[:,0].min(), p[:,1].min(), p[:,2].min()],
                         [p[:,0].max(), p[:,1].max(), p[:,2].max()]]).to(device)
    loc = (bbox[0] + bbox[1]) / 2

    # create_transform_matrix
    S_loc = torch.eye(4).to(device)
    S_loc[:-1, -1] = -loc
    # create scale mat
    S_scale = torch.eye(4).to(device) * scale
    S_scale[-1, -1] = 1

    S = S_scale @ S_loc
    
    return S




####main

def init_pointcloud(path_cfg, data_list, pose_list, center_frame, pose_t):

    # center_frame = torch.IntTensor(center_frame)
    POINT_PATH = os.path.join(path_cfg['data_basedir'], path_cfg['dataset_dir'])
    T_PATH = os.path.join(path_cfg['data_basedir'], path_cfg['proj_dir'])
    have_nearfar_bd = None
    nearfar_path = ""

    depths = []
    poses = []
    points = []
    point_center_list = []
    pose_center_list = []
    i = 0
    # for test_idx in data_list:
    for test_idx in center_frame:
        # if have_nearfar_bd:
        #     nearfar = np.loadtxt(os.path.join(nearfar_path, f"{test_idx}.txt"))
        #     nearfar = np.uint16(nearfar*1000)
        # else:
        #     nearfar = np.array([depth.min(), depth.max()])        
        # depth[np.where( depth < nearfar[0])] = 0
        # depth[np.where( depth > nearfar[1])] = 0

        # pose = np.loadtxt(os.path.join(pose_path, f"{test_idx}.txt"))
        pose = pose_list[i].cpu()
        # point_ = o3d.io.read_point_cloud(os.path.join(POINT_PATH,f"{test_idx}.ply"))
        # point = torch.FloatTensor(point_.points)
        point_cloud = np.reshape(np.fromfile(os.path.join(POINT_PATH, f"{test_idx:010}.bin"), dtype=np.float32), (-1, 4))
        # point_ = o3d.geometry.PointCloud()
        # point_.points = o3d.utility.Vector3dVector(point_cloud[:, 0:3])
        # point_.normals = o3d.utility.Vector3dVector(point_cloud_with_normal[:, 3:6])
        point = torch.FloatTensor(o3d.utility.Vector3dVector(point_cloud[:, 0:3]))
        

        mask = torch.logical_and(torch.norm(point,dim=1) > 0.5, torch.norm(point,dim=1) < 45)   
        point = point[mask]

        if torch.prod(torch.IntTensor(data_list)-test_idx)==0:
            poses.append(pose)
            points.append(point)
            
        pose_center_list.append(pose)
        # point = point[torch.where(torch.norm(point, p=2, dim=1, keepdim=True).squeeze(1)<20)[0].tolist(),:]
        point_center_list.append(point)

        i = i + 1
        
        # poses.append(pose)
        # points.append(point)

    #-----------------------
    # world_points_0 = depths2points([point_center], [pose_center]).astype(np.float16)
    world_points_0 = depths2points(point_center_list, pose_center_list).astype(np.float16)
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(world_points_0)
    o3d.io.write_point_cloud(os.path.join(T_PATH, "ply/gt_points_test.ply"), points_o3d)

    # trans_S = cal_T(depths2points([point_center], [pose_center]).astype(np.float16))
    trans_S = cal_T(depths2points(point_center_list, pose_center_list).astype(np.float16), pose_t)
    points_all = genarate_points(depths2points(points, poses).astype(np.float16), trans_S)


    np.savetxt(os.path.join(T_PATH, "T_w_g/pointcloud_trans_1.txt"), trans_S, fmt='%.5f',delimiter='\t')

    points_all = np.array(points_all)

    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(points_all)
    o3d.io.write_point_cloud(os.path.join(T_PATH, "ply/test_list.ply"), points_o3d)

    return trans_S


def cal_frame_o(path_base, frame, pose):

    # point_center_list = []
    # pose_center_list = []
    point_ = o3d.io.read_point_cloud(os.path.join(path_base,f"{frame}.ply"))
    point = torch.FloatTensor(point_.points)
    mask = torch.logical_and(torch.norm(point,dim=1) > 0.5, torch.norm(point,dim=1) < 45)  
    point = point[mask]
    # pose_center_list.append(pose.cpu())
    # point_center_list.append(point)
 
    points = depths2points([point], [pose.cpu()]).astype(np.float16)
    bbox = get_points_bounds(points)
    loc = (bbox[0] + bbox[1]) / 2
    return torch.FloatTensor(loc)



def cal_frame_o2(path_base, frame, poses):

    points = []
    # poses = []
    for i in range(len(frame)):
        point_ = o3d.io.read_point_cloud(os.path.join(path_base,f"{frame[i]}.ply"))
        point = torch.FloatTensor(point_.points)
        # mask = torch.logical_and(torch.norm(point,dim=1) > 0.5, torch.norm(point,dim=1) < 40)  
        # point = point[mask]
        # poses.append(pose.cpu())
        points.append(point)
 
    points = depths2points(points, poses).astype(np.float16)
    bbox = get_points_bounds(points)
    loc = (bbox[0] + bbox[1]) / 2
    return torch.FloatTensor(loc)


def cal_T2(points):

    bbox = get_points_bounds(points)
    loc = (bbox[0] + bbox[1]) / 2 + np.array([-2,6,0])
    # scale = 2. / (bbox[1] - bbox[0]).max()
    scale = 2. / ((bbox[1] - bbox[0]).max() * 1.25)
    vertices_t = (points - loc.reshape(-1, 3)) * scale
    y_min = min(vertices_t[:, 1])

    # create_transform_matrix
    S_loc = np.eye(4)
    S_loc[:-1, -1] = -loc
    # S_loc[:-1, -1] = -pose_t.cpu().numpy()

    # create scale mat
    S_scale = np.eye(4) * scale
    S_scale[-1, -1] = 1
    # create last translate matrix
    # S_loc2 = np.eye(4)
    # S_loc2[1, -1] = -y_min + y_level

    # S = S_loc2 @ S_scale @ S_loc
    S = S_scale @ S_loc
    
    # points_trans = (S[:3,:3] @ points.transpose()).transpose() + S[:3,3]
    return S



# go-surf
def compute_world_dims(pts, voxel_sizes, n_levels, margin=0):
    bounds = torch.tensor(get_points_bounds(pts))
    coarsest_voxel_dims = ((bounds[:,1] - bounds[:,0] + margin*2) / voxel_sizes[-1])
    coarsest_voxel_dims = torch.ceil(coarsest_voxel_dims) + 1
    coarsest_voxel_dims = coarsest_voxel_dims.int()
    world_dims = (coarsest_voxel_dims - 1) * voxel_sizes[-1]
    
    # Center the model in within the grid
    volume_origin = bounds[:,0] - (world_dims - bounds[:,1] + bounds[:,0]) / 2
    
    # Multilevel dimensions
    voxel_dims = (coarsest_voxel_dims.view(1,-1).repeat(n_levels,1) - 1) * (voxel_sizes[-1] / torch.tensor(voxel_sizes).unsqueeze(-1)).int() + 1

    return world_dims, volume_origin, voxel_dims

# from utils.scannet_data_tools.kitti360Viewer3DRaw import Kitti360Viewer3DRaw, init_camera
def init_pointcloud2(cfg, velo, data_list, pose_list, voxel_sizes, device, is_first=False):
  # center_frame = torch.IntTensor(center_frame)
    POINT_PATH = os.path.join(cfg['path']['dataset_dir'])
    T_PATH = os.path.join(cfg['path']['proj_dir'])
    have_nearfar_bd = None
    nearfar_path = ""

    pc_radius = cfg['ray']['pc_radius']
    min_z = cfg['ray']['min_z']
    max_z = cfg['ray']['max_z']
    self_x, self_y, self_z = cfg['ray']['self_x'], cfg['ray']['self_y'], cfg['ray']['self_z']
    depth_min = cfg['ray']['depth_min']

    if cfg['path']['dataset_type'] == 'scannet':
        if cfg['3DGS']:
            h, w = cfg['img']['height'], cfg['img']['width']
            intrinsic_depth = torch.from_numpy(np.loadtxt(os.path.join(POINT_PATH, "intrinsic/intrinsic_color.txt")))
        else:
            ## camera intrinsic_depth
            h, w = cfg['img']['height_d'], cfg['img']['width_d']
            intrinsic_depth = torch.from_numpy(np.loadtxt(os.path.join(POINT_PATH, "intrinsic/intrinsic_depth.txt")))
        fx, fy, cx, cy = intrinsic_depth[0, 0], intrinsic_depth[1, 1], intrinsic_depth[0, 2], intrinsic_depth[1, 2]
       
        ## image ray
        pix = torch.arange(0, h*w, 1)
        v = pix.div(w, rounding_mode='floor')
        u = pix % w   
        rays_d = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1) #.to(self.device)
        # rays_d = rays_d / rays_d.norm(2,1)[:, None]
        rays_o = torch.FloatTensor([0,0,0])

    depths = []
    poses = []
    points = []
    point_center_list = []
    pose_center_list = []
    i = 0
    for test_idx in tqdm(data_list):
    # for test_idx in center_frame:
        pose = pose_list[i].cpu()
        #point_cloud = np.reshape(np.fromfile(os.path.join(POINT_PATH, f"{test_idx:010}.bin"), dtype=np.float32), (-1, 4))
        if cfg['path']['dataset_type'] == 'maicity':
            point_ = read_point_cloud(os.path.join(POINT_PATH,f"{test_idx:06}.ply"), cfg['params']['min_range'], cfg['params']['z_th'])
            # bbx_min = np.array([-pc_radius, -pc_radius, min_z])
            # bbx_max = np.array([pc_radius, pc_radius, max_z])
            # bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)
            # point_ = point_.crop(bbx)
            point = torch.FloatTensor(np.array(point_.points))
        elif cfg['path']['dataset_type'] == 'newer_college':
            point_ = read_point_cloud(os.path.join(POINT_PATH,f"{test_idx:05}.pcd"), cfg['params']['min_range'], cfg['params']['z_th'])
            point = torch.FloatTensor(np.array(point_.points))
        elif cfg['path']['dataset_type'] == 'kitti360':
            # range filter: crop the points clouds by lidar range
            data = velo.loadVelodyneData(test_idx)
            if cfg['curl']:
                # pcd = o3d.io.read_point_cloud(os.path.join(POINT_PATH,f"{test_idx}.ply"))
                # data = np.asarray(pcd.points)
                data = velo.curlVelodyneData(test_idx, data)
            # else: 
            #     data = np.reshape(np.fromfile(os.path.join(POINT_PATH, f"{test_idx:010}.bin"), dtype=np.float32), (-1, 4))[:, :3]
            data = data[:, :3]
            data = data[np.logical_and(data[:, 2] > min_z, data[:, 2] < max_z)]
            x, y, z = data[:, 0], data[:, 1], data[:, 2]
            car_bbox_mask = np.logical_and(z < self_z, z > -self_z)
            car_bbox_mask = np.logical_and(np.logical_and(y < self_y, y > -self_y), car_bbox_mask)
            car_bbox_mask = np.logical_and(np.logical_and(x < self_x, x > -self_x), car_bbox_mask)
            data = data[~car_bbox_mask]
            behind_mask = np.logical_and(data[:, 2] > 0, np.logical_and(data[:, 1] < 0.5, data[:, 1] > -0.5))
            data = data[~behind_mask]
            # data = data[np.logical_or(data[:, 1] > min_x, data[:, 1] < -min_x)]
            depth = np.linalg.norm(data, axis=1)
            data = data[np.logical_and(depth <= pc_radius, depth >= depth_min)]
            point = torch.FloatTensor(data)

            # block filter: crop the point clouds into a cube 
            # point_cloud = np.reshape(np.fromfile(os.path.join(POINT_PATH, f"{test_idx:010}.bin"), dtype=np.float32), (-1, 4)) 
            # depth = np.linalg.norm(point_cloud[:,:3],axis=1)
            # mask = depth>depth_min
            # point_ = o3d.geometry.PointCloud()
            # point_.points = o3d.utility.Vector3dVector(point_cloud[mask, 0:3])
            # bbx_min = np.array([-pc_radius, -pc_radius, min_z])
            # bbx_max = np.array([pc_radius, pc_radius, max_z])
            # bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)
            # point_ = point_.crop(bbx)
            # point = torch.FloatTensor(np.array(point_.points))
            # point = torch.FloatTensor(np.array(o3d.utility.Vector3dVector(point_cloud[:, 0:3])))
            poses.append(pose)
            points.append(point)
        elif cfg['path']['dataset_type'] == 'scannet':
            # if test_idx % 50 ==0 :
            # # depth image
            # if cfg['3DGS']:
            #     imagePath = os.path.join("/sdb1/xieyx/code/GaussianPro/scannet/scene0088_00/depth_3dgs", f'{test_idx}.png')
            # else:
            imagePath = os.path.join(POINT_PATH, f'depth/{test_idx}.png')
            depth_img = np.array(Image.open(imagePath))/1000.0
            # # depth to range
            # depth_img = depth_img.reshape(-1)/rays_d[:,-1]

            
            point = (rays_o + rays_d * depth_img.reshape(-1,1)).float()
            mask = np.logical_and(depth_img.reshape(-1)>0.3, depth_img.reshape(-1)<10)
            point = point[mask]

            poses.append(pose)
            if point.shape[0]>300000:
                idx = np.random.randint(0,point.shape[0],size=[300000])
                point = point[idx]
            points.append(point)
        else:
            sys.exit("Wrong dataset type. Please use maicity, newer_college or kitti-360")
             

        i = i + 1
        
        # poses.append(pose)
        # points.append(point)

    #-----------------------
    # world_points_0 = depths2points([point_center], [pose_center]).astype(np.float16)
    world_points = depths2points(points, poses) #.astype(np.float16)
    if world_points.shape[0]<1e7:
        points_o3d = o3d.geometry.PointCloud()
        points_o3d.points = o3d.utility.Vector3dVector(world_points)
        o3d.io.write_point_cloud(os.path.join(T_PATH, "ply/gt_points_test.ply"), points_o3d)
    if is_first:
        world_dims, volume_origin, voxel_dims = compute_world_dims(world_points, voxel_sizes, len(voxel_sizes),margin=0.1)
        origin = volume_origin + world_dims/2

        return world_dims.to(device), volume_origin.to(device), voxel_dims.to(device), torch.FloatTensor(world_points).to(device) #(world_points - origin.numpy()) / (world_dims.max().numpy()/2)
    else:
        bounds = torch.tensor(get_points_bounds(world_points))
        return torch.FloatTensor(world_points).to(device), bounds.to(device)
    
import open3d as o3d
import numpy as np

def depth2pcd(cfg, data_list, pose_list, seq, cam_id):
    sequence = '2013_05_28_drive_%04d_sync'%seq
    depth_completion_path = os.path.join(os.environ['KITTI360_DATASET'], 'depth_completion', sequence)
    camera, _ = init_camera(seq=seq, cam_id=cam_id)
    fx, fy = camera.K[0, 0], camera.K[1, 1]
    cx, cy = camera.K[0, 2], camera.K[1, 2]
    i=0
    points = []
    for idx in data_list:
        depth_path = os.path.join(depth_completion_path, 'depth/%010d.npy' % idx)
        depth_image = np.load(depth_path)
        conf_path = os.path.join(depth_completion_path, 'confidence/%010d.npy' % idx)
        conf_image = np.load(conf_path)
        pose = pose_list[i].cpu().numpy()
        height, width = depth_image.shape
        # intrinsic = o3d.camera.PinholeCameraIntrinsic(depth_image.shape[0], depth_image.shape[1], fx, fy, cx, cy)
        # pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_image), intrinsic, np.linalg.inv(pose))
        # point = np.asarray(pcd.points)
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - cx) * depth_image / fx
        y = (y - cy) * depth_image / fy
        z = depth_image
        point = np.vstack((x.reshape(-1), y.reshape(-1), z.reshape(-1), np.ones((height * width)))).T
        point = np.dot(point, pose.T)[:,0:3]
        
        mask = np.logical_and(conf_image > 0.98, depth_image<25).reshape(-1)
        point = point[mask]
        points.append(point)
        i += 1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point)
        # o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(os.path.join('./semantic_result_kitti/00', 'ply/%010d.ply' % idx), pcd)
    points = np.vstack(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(0.05)
    points = np.asarray(pcd.points)
    o3d.io.write_point_cloud(os.path.join('./semantic_result_kitti/00', 'ply/cam_points.ply'), pcd)
    return points
    