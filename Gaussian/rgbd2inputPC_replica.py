import numpy
import os
os.environ['QT_QPA_PLATFORM']="offscreen"
import sys
from PIL import Image
import numpy as np
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.gaussian_model import BasicPointCloud
from plyfile import PlyData, PlyElement
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

import json
from pathlib import Path
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import open3d as o3d
from typing import NamedTuple
import cv2

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    K: np.array
    sky_mask: np.array
    normal: np.array
    depth: np.array

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, sky_seg=False, load_normal=False, load_depth=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)

        # #sky mask
        if sky_seg:
            sky_path = image_path.replace("images", "mask")[:-4]+".npy"
            sky_mask = np.load(sky_path).astype(np.uint8)
        else:
            sky_mask = None
            
        if load_normal:
            normal_path = image_path.replace("images", "normals")[:-4]+".npy"
            normal = np.load(normal_path).astype(np.float32)
            normal = (normal - 0.5) * 2.0
        else:
            normal = None

        if load_depth:
            # depth_path = image_path.replace("images", "metricdepth")[:-4]+".npy"
            # depth = np.load(depth_path).astype(np.float32)
            depth_path = image_path.replace("images", "depth_nerf")[:-4]+".png"
            depth = cv2.imread(depth_path, -1).reshape(968, 1296).astype(np.float32)
            depth = depth / 100
        else:
            depth = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, 
                              K=intr.params, sky_mask=sky_mask, normal=normal, depth=depth)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

## colmap 
# path = '/sdb1/xieyx/code/GaussianPro/scannet/scene0088_00'
# cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
# cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)

# cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
# cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

# images_folder = '/sdb1/xieyx/code/GaussianPro/scannet/scene0088_00/images'

# cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=images_folder)
# cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
#############################
scene = "apartment_2"
base_path = f"/sdc1/yx/dataset/replica/habitat/{scene}"
image_path = os.path.join(base_path, "color")
image_files = os.listdir(image_path)
intrinsic = np.loadtxt(os.path.join(base_path, "intrinsic/intrinsic_color.txt"))
fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
img = cv2.imread(os.path.join(base_path, "color/0.png"))
h, w = img.shape[0], img.shape[1]

pcd_all = o3d.geometry.PointCloud()

# # "f6659a3107"
for file_name in image_files:
    idx = int(file_name.split('.')[0])
    if idx % 1 == 0:

# # # "5748ce6f01"
# for file_name in image_files[:-11]:
#     idx = int(file_name.split('.')[0])
#     if idx % 3 == 0:

# # "1ada7a0617"
# for file_name in image_files[:-17]:
#     idx = int(file_name.split('.')[0])
#     if idx % 1 == 0:

# # "bcd2436daf"
# for file_name in image_files[:-15]:
#     idx = int(file_name.split('.')[0])
#     if idx % 1 == 0:

        color_raw1 = o3d.io.read_image(f"/sdc1/yx/dataset/replica/habitat/{scene}/color/{idx}.png") #%006d.jpg" % idx
        uint16_img1 = cv2.imread(f'/sdc1/yx/dataset/replica/habitat/{scene}/depth/{idx}.png', -1)
        uint16_img1 = cv2.resize(uint16_img1, dsize=(w, h), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        uint16_img1[uint16_img1<0.5*1000] = 100*1000
        depth_raw1 = o3d.geometry.Image(uint16_img1)
        rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw1, depth_raw1, depth_scale = 1000, depth_trunc = 20, convert_rgb_to_intensity=False)

        inter = o3d.camera.PinholeCameraIntrinsic()
        # inter.set_intrinsics(1274, 952, 1169.35914615, 1155.46032225,  637.,  476.)
        inter.set_intrinsics(w, h, fx, fy, cx, cy)
        pcd1 = o3d.geometry.PointCloud().create_from_rgbd_image(rgbd_image1, inter)

        # R1 = cam_infos[idx].R
        # T1 = cam_infos[idx].T
        pose_path = os.path.join(base_path, "pose")
        pose_file = os.path.join(pose_path, file_name.replace('png', 'txt'))
        pose = np.loadtxt(pose_file) #c2w

        # extrinsics = []
        # with open(pose_file) as file_object:
        #     lines = file_object.readlines()
        #     for line in lines:
        #         line = line.strip('\n')
        #         nums = line.split(' ')
        #         nums = list(map(float, nums))
        #         nums = np.array(nums)
        #         extrinsics.append(nums)
        # pose = np.array(extrinsics)

        if pose[3,3]<0:
            continue
        R1 = pose[0:3, 0:3]
        T1 = pose[0:3, 3]

        # trans = np.array([[1, -1, -1], [1, -1, -1], [1, -1, -1]])
        # R1 = np.multiply(R1, trans)

        positions = np.asarray(pcd1.points, dtype=np.float32)

        # 3
        # points_world = np.dot(np.linalg.inv(R1), (positions-T1).T).T

        # 2
        # points_world = np.dot(R1, (positions-T1).T).T  # 8

        # 1
        # points_world = np.dot(R1, positions.T).T + T1
        points_world = positions @ R1.T + T1

        colors_world = np.asarray(pcd1.colors, dtype=np.float32)

        pcd_world = o3d.geometry.PointCloud()
        pcd_world.points = o3d.utility.Vector3dVector(points_world)
        pcd_world.colors = o3d.utility.Vector3dVector(colors_world)
        # os.makedirs(f'/home/xieyuxuan/Data/Project/result/panoptic_recon++/scannetpp/scen_{scene}/0_input_ply', exist_ok=True)
        # o3d.io.write_point_cloud(f'/home/xieyuxuan/Data/Project/result/panoptic_recon++/scannetpp/scen_{scene}/0_input_ply/{idx}.ply', pcd_world)

        pcd_world = pcd_world.voxel_down_sample(voxel_size = 0.05) #0.03
        print(pcd_world)
        pcd_all = pcd_all + pcd_world

print(pcd_all)
os.makedirs(f'/sdc1/yx/dataset/replica/habitat/{scene}/0_input_ply', exist_ok=True)
o3d.io.write_point_cloud(f'/sdc1/yx/dataset/replica/habitat/{scene}/0_input_ply/pointcloud_gt_open3d.ply', pcd_all)
