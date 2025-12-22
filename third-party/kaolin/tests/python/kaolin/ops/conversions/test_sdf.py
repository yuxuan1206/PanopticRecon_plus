# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import sys

import torch

from kaolin.ops.conversions import sdf


class TestSdfToVoxelgrids:

    def sphere(self, points, center=0, radius=0.5):
        return torch.sum((points - center) ** 2, 1) ** 0.5 - radius

    def two_spheres(self, points):
        dis1 = self.sphere(points, 0.1, 0.4)
        dis2 = self.sphere(points, -0.1, 0.4)
        dis = torch.zeros_like(dis1)
        mask = (dis1 > 0) & (dis2 > 0)
        dis[mask] = torch.min(dis1[mask], dis2[mask])
        mask = (dis1 < 0) ^ (dis2 < 0)
        dis[mask] = torch.max(-torch.abs(dis1[mask]), -torch.abs(dis2[mask]))
        mask = (dis1 < 0) & (dis2 < 0)
        dis[mask] = torch.min(torch.abs(dis1[mask]), torch.abs(dis2[mask]))
        return dis

    def sdf_to_voxelgrids_naive(self, sdf, res):
        outputs = []
        for i_batch in range(len(sdf)):
            output = torch.ones((res, res, res))
            grid_pts = torch.nonzero(output).float() / (res - 1) - 0.5
            outputs.append((sdf[i_batch](grid_pts) <= 0).float().reshape(output.shape))
        return torch.stack(outputs)

    def test_sdf_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected sdf to be list "
                                 r"but got <class 'int'>."):
            sdf.sdf_to_voxelgrids(0)
    def test_each_sdf_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected sdf\[0\] to be callable "
                                 r"but got <class 'int'>."):
            sdf.sdf_to_voxelgrids([0])
    def test_bbox_center_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected bbox_center to be int or float "
                                 r"but got <class 'str'>."):
            sdf.sdf_to_voxelgrids([self.sphere], bbox_center=' ')

    def test_bbox_dim_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected bbox_dim to be int or float "
                                 r"but got <class 'str'>."):
            sdf.sdf_to_voxelgrids([self.sphere], bbox_dim=' ')

    def test_init_res_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected init_res to be int "
                                 r"but got <class 'float'>."):
            sdf.sdf_to_voxelgrids([self.sphere], init_res=0.5)

    def test_upsampling_steps_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected upsampling_steps to be int "
                                 r"but got <class 'float'>."):
            sdf.sdf_to_voxelgrids([self.sphere], upsampling_steps=0.5)

    @pytest.mark.parametrize('init_res', [4, 8, 32])
    @pytest.mark.parametrize('upsampling_steps', [0, 2, 4])
    def test_sphere(self, init_res, upsampling_steps):
        final_res = init_res * 2 ** upsampling_steps + 1
        assert(torch.equal(sdf.sdf_to_voxelgrids([self.sphere], init_res=init_res, upsampling_steps=upsampling_steps), 
                           self.sdf_to_voxelgrids_naive([self.sphere], final_res)))

    @pytest.mark.parametrize('init_res', [4, 8, 32])
    @pytest.mark.parametrize('upsampling_steps', [0, 2, 4])
    def test_two_spheres(self, init_res, upsampling_steps):
        final_res = init_res * 2 ** upsampling_steps + 1
        assert(torch.equal(sdf.sdf_to_voxelgrids([self.two_spheres], init_res=init_res, upsampling_steps=upsampling_steps), 
                           self.sdf_to_voxelgrids_naive([self.two_spheres], final_res)))



def sphere(points, center=0, radius=0.5):
        return np.sum((points - center) ** 2, 1) ** 0.5 - radius

def two_spheres(points):
        dis1 = sphere(points, 0.1, 0.4)
        dis2 = sphere(points, -0.1, 0.4)
        dis = np.zeros_like(dis1)
        mask = (dis1 > 0) & (dis2 > 0)
        dis[mask] = np.minimum(dis1[mask], dis2[mask])
        mask = (dis1 < 0) ^ (dis2 < 0)
        dis[mask] = np.maximum(-np.abs(dis1[mask]), -np.abs(dis2[mask]))
        mask = (dis1 < 0) & (dis2 < 0)
        dis[mask] = np.minimum(np.abs(dis1[mask]), np.abs(dis2[mask]))
        return dis

from kaolin.ops.conversions import voxelgrid
import open3d as o3d
import numpy as np
import os
import trimesh
import skimage
import skimage.measure
from multiprocessing.pool import ThreadPool
from skimage import measure
import multiprocessing
import itertools
from sdf import progress, stl, mesh
from functools import partial
# from sdf import *

def trimesh_to_open3d(src):
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst

WORKERS = multiprocessing.cpu_count()
SAMPLES = 2 ** 22
BATCH_SIZE = 32
x0,y0,z0 = -1,-1,-1
x1,y1,z1 = 1,1,1
verbose = True
sparse=True

volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
step = (volume / SAMPLES) ** (1 / 3)
dx = dy = dz = step
print('min %g, %g, %g' % (x0, y0, z0))
print('max %g, %g, %g' % (x1, y1, z1))
print('step %g, %g, %g' % (dx, dy, dz))
X = np.arange(x0, x1, dx)
Y = np.arange(y0, y1, dy)
Z = np.arange(z0, z1, dz)

s = BATCH_SIZE
Xs = [X[i:i+s+1] for i in range(0, len(X), s)]
Ys = [Y[i:i+s+1] for i in range(0, len(Y), s)]
Zs = [Z[i:i+s+1] for i in range(0, len(Z), s)]

batches = list(itertools.product(Xs, Ys, Zs))
num_batches = len(batches)
num_samples = sum(len(xs) * len(ys) * len(zs)
    for xs, ys, zs in batches)
print('%d samples in %d batches with %d workers' %
            (num_samples, num_batches, WORKERS))
points = []
skipped = empty = nonempty = 0
bar = progress.Bar(num_batches, enabled=verbose)
pool = ThreadPool(WORKERS)
f = partial(mesh._worker, two_spheres, sparse=sparse)
for result in pool.imap(f, batches):
    bar.increment(1)
    if result is None:
        skipped += 1
    elif len(result) == 0:
        empty += 1
    else:
        nonempty += 1
        points.extend(result)
bar.done()
stl.write_binary_stl('mesh_test.stl', points)
mesh = mesh._mesh(points)

points = generate(*args, **kwargs)
stl.write_binary_stl(path, points)

# init_res = 2**7
# upsampling_steps = 2
# final_res = init_res * 2 ** upsampling_steps + 1
# grid = (sdf.sdf_to_voxelgrids([two_spheres], init_res=init_res, upsampling_steps=upsampling_steps)).type(torch.uint8)
# # vertices, faces, normals, values = skimage.measure.marching_cubes(
# #             (1-grid).squeeze(0).numpy(), level=0.0, spacing=[r for r in [2/final_res,2/final_res,2/final_res]]
# #         )
# # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
# # o3d_mesh_canonical  = trimesh_to_open3d(mesh)
# # o3d.io.write_triangle_mesh('mesh_test.ply', o3d_mesh_canonical)
# # yx = 1

# vertices, faces = voxelgrid.voxelgrids_to_trianglemeshes(grid.cuda())
# mesh = trimesh.Trimesh(vertices=vertices[0].cpu(), faces=faces[0].cpu())
# o3d_mesh_canonical  = trimesh_to_open3d(mesh)
# o3d.io.write_triangle_mesh('mesh_test.ply', o3d_mesh_canonical)
yx = 1