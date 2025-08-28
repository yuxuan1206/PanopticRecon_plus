import torch
from lietorch import SO3, SE3, LieGroupParameter
import os
import sys
import numpy as np
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from numpy.linalg import inv
from tqdm import tqdm

@torch.no_grad()
def read_time(path):
        data = np.loadtxt(path)
        time = data[:,0]*60*60 + data[:,1]*60 + data[:,2]
        # # time_big = data[:,0]*60*60 + data[:,1]*60
        # # time_small = data[:,2]
        # # time_txt = np.stack((time_big, time_small),1)
        # np.savetxt("/media/yx/Elements/mydata/kitti/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/timestamps_sec.txt", time)
        return time

@torch.no_grad()
def init_all_poses(config,  data_list, device):
        import re
        ## pose
        # seq = config.split('/')[-1]
        pose_folder_path = config + "/pose"
        pose_files = os.listdir(pose_folder_path)
        convert = lambda text: int(text) if text.isdigit() else text
        sorted_by_number_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        pose_files = sorted(pose_files, key=sorted_by_number_key)
        data_list = list(data_list)
        # data = {}
        # with open(os.path.join(config, seq+".txt")) as f:
        #     for line in f:
        #         key, value = line.strip().split(' = ')
        #         data[key] = value
        # colorToDepthExtrinsics = np.array(data['colorToDepthExtrinsics'].split(' '),dtype=np.float32).reshape(4,4)

        vecs, vecs_cam = {}, {}
        for p in tqdm(pose_files):
            pose_file = os.path.join(pose_folder_path, p)
            # rgb
            pose = np.loadtxt(pose_file) #c2w

            p = p.strip('DSC')
            if pose[3,3]<0 and int(p.split('.')[0]) in data_list:
                data_list.remove(int(p.split('.')[0]))
                continue
            r = R.from_matrix(pose[0:3, 0:3])
            quat = r.as_quat()
            trans = pose[0:3, 3:].squeeze()
            vecs_cam[f"{int(p.split('.')[0])}"] = torch.from_numpy(np.append(trans, quat)).to(device).to(torch.float32) 
            # depth
            vecs[f"{int(p.split('.')[0])}"] = vecs_cam[f"{int(p.split('.')[0])}"]
            # pose_depth = pose @ np.linalg.inv(colorToDepthExtrinsics) #d2w = c2w @ d2c
            # r = R.from_matrix(pose_depth[0:3, 0:3])
            # quat = r.as_quat()
            # trans = pose_depth[0:3, 3:].squeeze()
            # vecs[f"{int(p.split('.')[0])}"] = torch.from_numpy(np.append(trans, quat)).to(device).to(torch.float32) 
     
        return vecs, vecs_cam, data_list

def init_all_poses_custom(path, device):
        data = np.load(path)
        data[:,2] = 1.3
        vecs, vecs_cam = {}, {}

        # cam_rotation = (R.from_euler('y', -20, degrees=True) * R.from_euler('x', 90, degrees=True) * R.from_euler('y', -90, degrees=True)).as_matrix()
        cam_rotation = (R.from_euler('x', 5, degrees=True) * R.from_euler('z', 90, degrees=True) * R.from_euler('y', -90, degrees=True)).as_matrix()
        cam_matrix = np.eye(4)
        cam_matrix[:3,:3] = cam_rotation
        trans_list = []
        for i, pose in tqdm(enumerate(data)):
                r = R.from_quat(pose[3:]).as_matrix() # @  cam_rotation
                t = pose[:3]
                t[-1] = -t[-1]
                matrix = np.eye(4)
                matrix[:3,:3] = r
                matrix[:3,3] = t
                pose = matrix @ np.linalg.inv(cam_matrix)
                quat = R.from_matrix(pose[0:3, 0:3]).as_quat()
                trans = pose[0:3, 3:].squeeze()
                trans_list.append(trans)
                vecs_cam[f"{i}"] = torch.from_numpy(np.append(trans, quat)).to(device).to(torch.float32) 
                # depth
                vecs[f"{i}"] = vecs_cam[f"{i}"]
        return vecs, vecs_cam

        
def addVOnoise(vec, noise_old, scale_t=0.1, scale_r=0.1):
        noise = torch.hstack((scale_t*(torch.rand_like(vec[:3])-0.), scale_r*(torch.rand_like(vec[3:])-0.))) + noise_old
        vec = vec + noise
        return vec, noise


def matrix_to_lie(trans,quat):
        # quat = transforms.matrix_to_quaternion(matrix[:3, :3]) # wxyz
        # quat = torch.cat((quat[1:], quat[0][None]), 0)  # swap real first to real last xyzw
        # trans = matrix[:3, 3]

        vec = torch.cat((trans, quat), 0)
        # Ps = SE3.InitFromVec(vec)
        return vec


def init_T():
        t_lidar2imu = torch.FloatTensor([0.014, -0.012, -0.015])
        q_lidar2imu = torch.FloatTensor([1.0, 0.0, 0.0, 0.0]) # wxyz
        T_lidar2imu = torch.hstack((transforms.quaternion_to_matrix(q_lidar2imu), t_lidar2imu.unsqueeze(1))) # wxyz

        # T_imu2mask = torch.FloatTensor([])
        return T_lidar2imu

def get_lidar_pose(trans, quat, T_lidar2imu):
        trans_l = torch.FloatTensor(trans @ T_lidar2imu[:3, :3].T + T_lidar2imu[:3,3])
        q_ls = []
        for i in range(quat.shape[0]):
                rotm_i = R.from_quat(quat[i,:])
                rotm_l = T_lidar2imu[:3, :3] @ rotm_i.as_matrix()
                q_l = R.from_matrix(rotm_l)
                q_ls.append(q_l.as_quat())
        quat_l = torch.FloatTensor(q_ls)
        vec_lidar = torch.hstack((trans_l, quat_l))
        return vec_lidar

def read_calib_file(filename):
    """ 
        read calibration file (with the kitti format)
        returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    calib_file = open(filename)
    key_num = 0

    for line in calib_file:
        # print(line)
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))

        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    return calib
    
