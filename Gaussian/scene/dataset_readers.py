#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import open3d as o3d
import cv2
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

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

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

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
            depth = depth / 1000
        else:
            depth = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, 
                              K=intr.params, sky_mask=sky_mask, normal=normal, depth=depth)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapCamerasNew(cam_extrinsics_dir, cam_intrinsics_path, images_folder, sky_seg=False, load_normal=False, load_depth=False, custom_path=None):
    cam_infos = []

    height = 968
    width = 1296
    # height = 779
    # width = 1168

    intrinsics = []
    with open(cam_intrinsics_path) as file_object:
        lines = file_object.readlines()
        for line in lines:
            line = line.strip('\n')
            nums = line.split(' ')
            nums = list(map(float, nums))
            nums = np.array(nums)
            intrinsics.append(nums)

    intrinsics = np.array(intrinsics)

    focal_length_x = intrinsics[0, 0]
    focal_length_y = intrinsics[1, 1]
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)

    K = np.array([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]])

    if custom_path is None:
        cam_extrinsics_name = os.listdir(cam_extrinsics_dir)
        # cam_extrinsics_name = os.listdir("/mnt/meadow/yx/dataset/scannet/scans/scene0088_00/depth_test")
        for idx, name in enumerate(cam_extrinsics_name):  #[1230:1550]
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics_name)))
            sys.stdout.flush()

            uid = 1

            extrinsics = []
            cam_extrinsics_path = os.path.join(cam_extrinsics_dir, name[:-4]+".txt")
            with open(cam_extrinsics_path) as file_object:
                lines = file_object.readlines()
                for line in lines:
                    line = line.strip('\n')
                    nums = line.split(' ')
                    nums = list(map(float, nums))
                    nums = np.array(nums)
                    extrinsics.append(nums)
            
            extrinsics = np.array(extrinsics)
            if extrinsics[-1,-1] < 1:
                continue
            # print(extrinsics)
            R = extrinsics[0:3, 0:3]
            T = -1 * np.dot(np.linalg.inv(R), extrinsics[0:3, 3]).transpose()

            image_path = os.path.join(images_folder, name[:-4]+".jpg")
            image_name = os.path.basename(image_path).split(".")[0]

            image = Image.open(image_path)

            # #sky mask
            if sky_seg:
                sky_path = image_path.replace("images", "mask")[:-4]+".npy"
                sky_mask = np.load(sky_path).astype(np.uint8)
            else:
                sky_mask = None
                
            if load_normal:
                normal_path = os.path.join("/sdc1/xyx/result/panoptic_recon++/scannetpp/scen_f6659a3107/rgb_depth/v0/normal_render_img", name[:-4].zfill(10) + ".png")
                # print(normal_path)
                try:
                    normal = cv2.imread(normal_path, -1).astype(np.float32)
                except:
                    print(f"no {normal_path}")
                normal = normal / 255.0
                normal = (normal - 0.5) * 2.0
                normal = normal.transpose(2, 0, 1)
                # normal_path = image_path.replace("images", "normals")[:-4]+".npy"
                # normal = np.load(normal_path).astype(np.float32)
                # normal = (normal - 0.5) * 2.0
            else:
                normal = None

            if load_depth:
                ## GT
                # depth_path = os.path.join("/home/yx/Data/yx/dataset/scannet/scans/scene0087_02/depth", name[:-4] + ".png") #name[:-4].zfill(10)
                ## nerf
                # depth_path = os.path.join("/home/yx/Data/yx/exp_result/panoptic_recon++/hash_pc/v0/depth_render_value", name[:-4].zfill(10) + ".png")
                depth_path = image_path.replace("color", "depth")[:-4]+".png"
                depth = cv2.imread(depth_path, -1).astype(np.float32)
                # depth = depth.reshape(968, 1296)
                depth = depth / 1000

                # depth_path = image_path.replace("color", "depth")[:-4]+".png"
                # depth = cv2.imread(depth_path, -1).astype(np.float32)
                # depth = cv2.resize(depth, dsize=(1296, 968), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
                # depth = depth / 1000
                # # depth[depth > 6] = 0
                # # print(depth.shape)
            else:
                depth = None

            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height, 
                                K=K, sky_mask=sky_mask, normal=normal, depth=depth)
            cam_infos.append(cam_info)
        
    else: #custom
        data = np.load(custom_path)
        data[:,2] = 1.3 
        cam_rotation = (Rotation.from_euler('x', 5, degrees=True) * Rotation.from_euler('z', 90, degrees=True) * Rotation.from_euler('y', -90, degrees=True)).as_matrix()  #20
        cam_matrix = np.eye(4)
        cam_matrix[:3,:3] = cam_rotation
        for i, pose in enumerate(data):
            uid = 1
            r = Rotation.from_quat(pose[3:]).as_matrix()
            t = pose[:3]
            t[-1] = -1*t[-1]
            extrinsics = np.eye(4)
            extrinsics[:3,:3] = r
            extrinsics[:3,3] = t
            extrinsics = extrinsics @ np.linalg.inv(cam_matrix)
            R = extrinsics[0:3, 0:3]
            T = -1 * np.dot(np.linalg.inv(R), extrinsics[0:3, 3]).transpose()
            image_path = os.path.join(images_folder, f"0.jpg") #{i}
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)

            image_name = f'{i}'
            sky_mask = None
            normal = None
            depth = None
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height, 
                            K=K, sky_mask=sky_mask, normal=normal, depth=depth)
            cam_infos.append(cam_info)
        

    sys.stdout.write('\n')
    return cam_infos

def readColmapCameras_roma(images_folder, sky_seg=False, load_normal=False, load_depth=False):
    cam_infos = []

    height = 1920
    width = 1920

    focal_length_x = 967.532829
    focal_length_y = 964.832367
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)

    K = np.array([967.532829, 964.832367, 960, 960])

    with open("/mnt/meadow/yx/exp_data/xgrids/gro_result/geo_result_xgrids_roma/pose/pose_optim.json", "r") as f:
        content = json.load(f)
    
    for idx in range(len(content['images_data'])):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(content['images_data'])))
        sys.stdout.flush()

        uid = 1

        content_idx = content['images_data'][idx]
        basename = content_idx['name'].replace('_29_0.jpg', '')
        # print(basename)

        qw = content_idx['qw']
        qx = content_idx['qx']
        qy = content_idx['qy']
        qz = content_idx['qz']
        R = Quaternion([qw, qx, qy, qz]).rotation_matrix
        R = np.linalg.inv(R)

        tx = content_idx['tx']
        ty = content_idx['ty']
        tz = content_idx['tz']
        T = np.array([tx, ty, tz])
        T = T + 0.1

        image_path = os.path.join(images_folder, basename + '_29_0.png')
        image_name = basename.zfill(10)
        image = Image.open(image_path)

        # #sky mask
        if sky_seg:
            sky_path = image_path.replace("images", "mask")[:-4]+".npy"
            sky_mask = np.load(sky_path).astype(np.uint8)
        else:
            sky_mask = None
            
        if load_normal:
            normal_path = os.path.join("/mnt/meadow/yx/exp_data/xgrids/gro_result/geo_result_xgrids_roma/normal_render_img/tensor([0])", str(idx).zfill(10) + '.png')
            normal = cv2.imread(normal_path, -1).astype(np.float32)
            normal = normal / 255.0
            normal = (normal - 0.5) * 2.0
            normal = normal.transpose(2, 0, 1)
            # normal_path = image_path.replace("images", "normals")[:-4]+".npy"
            # normal = np.load(normal_path).astype(np.float32)
            # normal = (normal - 0.5) * 2.0
        else:
            normal = None

        if load_depth:
            depth_path = os.path.join("/mnt/meadow/yx/exp_data/xgrids/gro_result/geo_result_xgrids_roma/depth_render_img/tensor([0])", str(idx).zfill(10) + '.png')
            depth = cv2.imread(depth_path, -1).astype(np.float32)
            depth = depth / 100

            # depth_path = image_path.replace("color", "depth")[:-4]+".png"
            # depth = cv2.imread(depth_path, -1).astype(np.float32)
            # depth = cv2.resize(depth, dsize=(1296, 968), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            # depth = depth / 1000
            # # depth[depth > 6] = 0
            # # print(depth.shape)
        else:
            depth = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, 
                              K=K, sky_mask=sky_mask, normal=normal, depth=depth)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, input, eval, llffhold=8, sky_seg=False, load_normal=False, load_depth=False, custom_path=None):    # llffhold=8
    # try:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    #     cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    #     cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    # except:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
    #     cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    #     cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # reading_dir = "images" if images == None else images

    # cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), 
    #                                        sky_seg=sky_seg, load_normal=load_normal, load_depth=load_depth)
    
    cam_extrinsics_dir = os.path.join(path, 'pose')
    cam_intrinsics_path = os.path.join(path, 'intrinsic/intrinsic_color.txt')
    images_folder = os.path.join(path, 'color')
    cam_infos_unsorted = readColmapCamerasNew(cam_extrinsics_dir, cam_intrinsics_path, images_folder, sky_seg=False, load_normal=load_normal, load_depth=load_depth, custom_path=custom_path)

    # images_folder = '/sdc1/xgrids/roma_playground/reconstruct/image_0'
    # cam_infos_unsorted = readColmapCameras_roma(images_folder, sky_seg=False, load_normal=True, load_depth=True)

    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.strip('DSC')))
    try:
        with open(os.path.join(path, 'split.json')) as f:
            split_setting = json.load(f)
        train_set = split_setting['train']
        test_set = split_setting['test']
        if eval:
            # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx % llffhold == 0) and idx in train_set]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_set]
            # if 'waymo' in path:
            #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != (llffhold-1)]
            #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == (llffhold-1)]
            # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % (llffhold * 3) >= 3]
            # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % (llffhold * 3) < 3]
        else:
            # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            # train_cam_infos = cam_infos
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx % llffhold == 0) and idx in train_set]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_set]
    except:
        if eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = "/sdb1/xieyx/code/GaussianPro/pointcloud_roma_open3d.ply"
    # ply_path = "/home/yx/Data/yx/exp_result/panoptic_recon++/0_input_ply/scene0087_02/pointcloud_gt_open3d.ply"
    ply_path = input

    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_train=True):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            sky_mask = np.ones_like(image)[:, :, 0].astype(np.uint8)

            if is_train:
                normal_path = image_path.replace("train", "normals")[:-4]+".npy"
                normal = np.load(normal_path).astype(np.float32)
                normal = (normal - 0.5) * 2.0
                # normal[2, :, :] *= -1
            else:
                normal = np.zeros_like(image).transpose(2, 0, 1)

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], 
                            K=np.array([1, 2, 3, 4]), sky_mask=sky_mask, normal=normal))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, is_train=False)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}