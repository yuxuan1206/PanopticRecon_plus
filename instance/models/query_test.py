import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('instance/')

import torch
import numpy as np
import open3d as o3d
import math
from query_model import QueryModel
from arguments import ModelParams, PipelineParams, OptimizationParams

pc_path = "/sde1/yx/result/panoptic_recon++/scannetpp/scen_1ada7a0617/panoptic_test/v0/ply//token_init.ply"
# "/sde1/yx/dataset/replica/result/apartment_2/rgb_depth_new/v0/ply/token_init.ply"

def cov_vec2cov_matrix(cov_vec):
    batch = cov_vec.shape[0]
    n = int(math.sqrt(2*cov_vec.shape[-1]))
    cov_matrix = torch.zeros([batch, n, n]).to(cov_vec.device)
    idx = 0
    for i in range(n):
        for j in range(i,n):
            cov_matrix[:,i,j] = cov_vec[:,idx]
            cov_matrix[:,j,i] = cov_vec[:,idx]
            idx += 1
    return cov_matrix

def gaussian_3d_probability(mean, cov, point):
    diff = point[:,None,:] - mean[None,...]
    cov_matrices = cov_vec2cov_matrix(cov)
    indices = torch.arange(mean.shape[0]).to(cov.device)
    inv_cov = torch.inverse(cov_matrices.index_select(0,indices))
    d2 = (diff[:,:,None,:] @ inv_cov[None,...]) @ diff[:,:,:,None]
    det_cov = torch.det(cov_matrices.index_select(0,indices))
    prob = torch.exp(-0.5*d2.squeeze()) / (torch.sqrt( (2*torch.pi)**3 * det_cov ))

    return prob #[N_points, N_queries]

def computeCov_2D(mean_3D, K, Cov_3D, pose):
    # pc form world to camera
    mean_cam = pose[:3,:3] @ mean_3D + pose[:,3]
    # 

def get_radius_2D():


    # perspective projection matrix
    near = 1.5
    far = 80 # infinite
    # aspect = self.W / self.H
    x = self.W / (2.0 * intrinsic[0].item())
    y = self.H / (2.0 * intrinsic[1].item()) # fl_y
    cx = (intrinsic[2].item()-self.W/2)/self.W
    cy = (intrinsic[3].item()-self.H/2)/self.H
    proj = np.array([[1/x, 0, -2*cx, 0],  #1/(y*aspect)
                    [0, 1/y, -2*cy, 0],
                    [0, 0, -(far+near)/(far-near), -(2*far*near)/(far-near)],
                    [0, 0, -1, 0]], dtype=np.float32)
    mvps = proj[None,:] @ np.linalg.inv(mvps)
    # radius_2D = 
    return radius_2D #[N_queries]



opt = OptimizationParams()
gaussians = QueryModel()

pcd = o3d.io.read_point_cloud(pc_path)
pc = torch.tensor(np.asarray(pcd.points)).float().cuda()
# center, diagonal = get_center_and_diag(cam_centers)
# radius = diagonal * 1.1
cameras_extent = 15
gaussians.create_from_pcd(pc, cameras_extent)
gaussians.training_setup(opt)

# Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
try:
    screenspace_points.retain_grad()
except:
    pass

means3D = gaussians.get_xyz
means2D = screenspace_points
# opacity = pc.get_opacity
scales = gaussians.get_scaling
rotations = gaussians.get_rotation
cov3D = gaussians.get_covariance()

p = torch.FloatTensor([1,1,0.6]).to(screenspace_points.device)
gaussian_3d_probability(means3D, cov3D, means3D)



# shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
# dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
# dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
# sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
# colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

# gaussians.update_learning_rate(iteration)