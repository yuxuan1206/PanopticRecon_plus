import json
import numpy as np
import open3d as o3d

scene = "scene0088_00"
path = "data/scannet/"+scene+"/semantic_result/mesh/edit_mesh.0.100000.segs.json"
mesh_path = "data/scannet/"+scene+"/semantic_result/mesh/edit_mesh.ply"
# path = "data/kitti360/semantic_result/mesh/edit_mesh_world.0.100000.segs.json"
# mesh_path = "data/kitti360/semantic_result/mesh/edit_mesh_world.ply" # world

with open(path, 'r') as f:
        data = json.load(f)

segid = data['segIndices']
color_id = np.random.random( [len(data['segIndices']), 3] ) #list(set(data['segIndices']))
color = color_id[segid]


mesh_raw = o3d.io.read_triangle_mesh(mesh_path)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
mesh_raw.vertex_colors = o3d.utility.Vector3dVector(color)
o3d.io.write_triangle_mesh('data/scannet/'+scene+'/instance_result/1_superpoint/mesh_segment_0.1.ply', mesh_raw)

yx=1