import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
# from sensor_data import SensorData

# data extracted using SensorData.py
DATA_PATH = "/mnt/northcn4/ywx1261068/dataset/scannet/scans/scene0087_02"
RGB_PATH = "/mnt/northcn4/ywx1261068/dataset/scannet/scans/scene0087_02/color" 
DEPTH_PATH = "/mnt/northcn4/ywx1261068/dataset/scannet/scans/scene0087_02/depth"  
POSE_PATH = "/mnt/northcn4/ywx1261068/dataset/scannet/scans/scene0087_02/pose"   
MAX_INDEX = 500 # take up to this index of images
SKIP = 10 # take one image of every SKIP to speed up the processing
SAMPLE_RATIO = 0.05 # only take a subsample of pixels. Otherwise, it will take too long to render


intrinsic_depth = np.loadtxt(os.path.join(DATA_PATH, 'intrinsic/intrinsic_depth.txt'))
poses = [np.loadtxt(os.path.join(POSE_PATH, f'{i}.txt')) for i in range(0, MAX_INDEX, SKIP)]

def load_image(path):
    image = Image.open(path)
    return np.array(image)

rgb_images = [load_image(os.path.join(RGB_PATH, f'{i}.jpg')) for i in range(0, MAX_INDEX, SKIP)]
depth_images = [load_image(os.path.join(DEPTH_PATH, f'{i}.png'))for i in range(0, MAX_INDEX, SKIP)]

def convert_from_uvd(u, v, d, intr, pose):
    extr = np.linalg.inv(pose)
    if d == 0:
        return None, None, None
    
    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    depth_scale = 1000
    
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    world = (pose @ np.array([x, y, z, 1]))
    return world[:3] / world[3]

x_data, y_data, z_data, c_data = [], [], [], []
for idx in range(len(depth_images)):
    d = depth_images[idx]
    c = rgb_images[idx]
    p = poses[idx]
    
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if random.random() < SAMPLE_RATIO:
                x, y, z = convert_from_uvd(j, i, d[i, j], intrinsic_depth, p)
                if x is None:
                    continue
                    
                x_data.append(x)
                y_data.append(y)
                z_data.append(z)
                
                ci = int(i * c.shape[0] / d.shape[0])
                cj = int(j * c.shape[1] / d.shape[1])
                c_data.append(c[ci, cj] / 255.0)

def plot_3d(xdata, ydata, zdata, color=None, b_min=2, b_max=8, view=(45, 45)):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=200)
    ax.view_init(view[0], view[1])

    ax.set_xlim(b_min, b_max)
    ax.set_ylim(b_min, b_max)
    ax.set_zlim(b_min, b_max)

    ax.scatter3D(xdata, ydata, zdata, c=color, cmap='rgb', s=0.1)

# plot_3d(x_data, y_data, z_data, color=c_data)

# plt.show()

p = np.hstack((np.array(x_data)[:,None],np.array(y_data)[:,None],np.array(z_data)[:,None]))
c = np.array(c_data)
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(p)
pcd.colors = o3d.utility.Vector3dVector(c)
o3d.io.write_point_cloud(os.path.join(DATA_PATH,  'points.ply'), pcd)
