import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
from sdf import stl as ss
# from stl import mesh as stl_mesh
import numpy as np
from multiprocessing.pool import ThreadPool
import itertools
from sdf import progress,mesh
from functools import partial
import torch

from data.label import labels, id2label
import open3d as o3d
from tqdm import tqdm
from wisp.ops.differential import finitediff_gradient, tetrahedron_gradient, autodiff_gradient
from models.decoder import SemanticSDFDecoder
import kaolin.ops.spc as spc_ops
from instance.models.query_model import QueryModel

num_lods = 3
# device = 'cuda1'
@torch.no_grad()
def query_sdf(coords):
        shape = coords.shape
        if len(shape) == 2:
                coords = torch.FloatTensor(coords[:, None]).to(device)
        dis = torch.norm(coords - origin_all,dim=-1)
        cube_idx = dis.min(-1).indices + START#[N,CUBE,3]
        grids = []
        decodes = []
        masks_all = torch.zeros([coords.shape[0]],dtype=torch.bool).to(coords.device)
        sdfs_all = torch.zeros_like(coords[:,:,0:1])
        if torch.unique(cube_idx).tolist()[-1]>1:
            yx = 1
        for i in torch.unique(cube_idx).tolist():
            # data = torch.load(os.path.join(result_dir, f"optimed_grid_0_{i}.pth")) #
            data = torch.load(os.path.join(result_dir, FILE)) #
            # data = torch.load(os.path.join(result_dir, f'{FILE}')) #
            grid = data['grid']
            # grids.append(grid)
            decoder = data['decoder']
            octree = data['octree']
            if INSTANCE:
                from instance.models.mask3d import Mask3D
                import yaml
                yaml_path = "config/ijrr/scannet0000_02/render_scannet_hash_++_all.yaml"
                with open(yaml_path, "r") as f:
                    cfg = yaml.safe_load(f)
                # instance_corners = data['instance_corners']
                instance_model = Mask3D(**cfg['instance'], query_position=data['query_position'], device=device) #, radius=data['radius']
                instance_model.to(device)
                instance_model.load_state_dict(data['instance_model'])
                # instance_model.update_network(data['query_mask'], data['query_update_map'])
                instance_model.gaussians = QueryModel()
                instance_model.gaussians.load_ply(os.path.join(cfg['path']['proj_dir'], f'v0/grid/optimed_query_19.ply'))
                
            # decodes.append(decoder)
            mask_cube = cube_idx==i
            # masks.append(mask_cube)
            points = coords[mask_cube] - origin_all[i - START]
            points_normalized = (points+1.0)/2
            feats = grid['sdf'](points_normalized.reshape(-1, 3))
            if decoder['sdf'].__class__==SemanticSDFDecoder:
                sdfs = decoder['sdf'].forward_sdf(feats)[:,0:1]
            else:
                sdfs = decoder['sdf'].forward(feats)[:,0:1]

            query_results = octree.blas.query(points.reshape(-1, 3), octree.active_lods[-1], with_parents=True)
            mask = query_results.pidx[:,octree.active_lods[0]]<0

            sdfs[mask] = 10.0 
            masks_all[mask_cube] = mask
            sdfs_all[mask_cube] = sdfs.reshape(points_normalized.shape[0],points_normalized.shape[1],1)

        return sdfs_all.squeeze().detach().cpu().numpy(), ~masks_all.detach().cpu().numpy()  

@ torch.no_grad()
def extract_mesh(idx):
    WORKERS = 1 #multiprocessing.cpu_count()
    SAMPLES = 2 ** 25 #31 #34 23
    BATCH_SIZE = 64
    
    # # bcd2436daf
    # x0,y0,z0 = -0.6,-0.7,-0.4
    # x1,y1,z1 = 0.6,0.7,0.4

    # # 5748ce6f01
    # x0,y0,z0 = -0.7,-0.8,-0.5
    # x1,y1,z1 = 0.7,0.8,0.5

    # x0,y0,z0 = -0.6,-1.7,-1.7
    # x1,y1,z1 = 6.8,9,1.5

    x0,y0,z0 = 5.5,2,0
    x1,y1,z1 = 8.5,7.5,3.2

    x0 = x0/scale.cpu().numpy()-world_origin[0].cpu().numpy()
    x1 = x1/scale.cpu().numpy()-world_origin[0].cpu().numpy()
    y0 = y0/scale.cpu().numpy()-world_origin[1].cpu().numpy()
    y1 = y1/scale.cpu().numpy()-world_origin[1].cpu().numpy()
    z0 = z0/scale.cpu().numpy()-world_origin[2].cpu().numpy()
    z1 = z1/scale.cpu().numpy()-world_origin[2].cpu().numpy()

    verbose = True
    sparse=False #True

    volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
    step = (volume / SAMPLES) ** (1 / 3)
    dx = dy = dz = step
    # dz = 0.5*step
    # dx = 3*step
    # dy = (1.5*step)//1
  
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
    f = partial(mesh._worker, query_sdf, sparse=sparse)
    for result in pool.imap(f, batches):
        bar.increment(1)
        if result is None:
            skipped += 1
        elif len(result) == 0:
            empty += 1
        else:
            nonempty += 1
            # points.extend(result)
            points.extend((result+world_origin.cpu().numpy())*scale.cpu().numpy())
            # if nonempty > 30:
            #     break
            # if nonempty%100==0:
            #     stl.write_binary_stl(f'mesh_test_part{nonempty}.stl', points)
    bar.done()
    # ss.write_binary_stl(f'01_loam_map_init/mesh/mesh_fenge{idx}.stl', points)
    
    ss.write_binary_stl(STLPATH, points)
    
    #-------------------------------------------------------------
@ torch.no_grad()
def inference_mesh(mesh_path, cfg, channel=['sem','ins','rgb']):
    if len(channel)>0:
        mymesh= o3d.io.read_triangle_mesh(mesh_path) 
        # mymesh= o3d.io.read_triangle_mesh(STLPATH) 
        vertices = torch.FloatTensor( np.asarray(mymesh.vertices) )[:,None].cuda()

        # vertices = np.asarray(o3d.io.read_point_cloud("/sdf1/yx/dataset/scannet++/gt_pc/1ada7a0617_sem.ply").points)
        # vertices[:, -1:] *= -1 
        # vertices = torch.FloatTensor( vertices[:,[1,0,2]] )[:,None].cuda()

        vertices = (vertices/scale) - world_origin

        sem_batch = 50000
        semantic = torch.zeros(vertices.shape[0])
        semantic_raw = torch.zeros(vertices.shape[0])
        instance = torch.zeros(vertices.shape[0])
        panoptic = torch.zeros(vertices.shape[0])
        color = torch.zeros([vertices.shape[0],3])
        normal = torch.zeros([vertices.shape[0],3])
        pidxs = torch.zeros(vertices.shape[0])
        for i in tqdm(range(0, vertices.shape[0], sem_batch), desc='get semantic labels'):
                next_ind = min(i+sem_batch, vertices.shape[0])
                v = vertices[i:next_ind]
                dis = torch.norm(v - origin_all,dim=-1)
                cube_idx = dis.min(-1).indices + START#[N,CUBE,3]
                sem = torch.zeros(v.shape[0], dtype=torch.int64).cuda()
                ins = torch.zeros(v.shape[0], dtype=torch.int64).cuda()
                pan = torch.zeros(v.shape[0], dtype=torch.int64).cuda()
                rgb = torch.zeros([v.shape[0],3], dtype=torch.float32).cuda()
                nor = torch.zeros([v.shape[0],3], dtype=torch.float32).cuda()
                pidx = torch.zeros(v.shape[0], dtype=torch.int64).cuda()
                masks_all = torch.zeros([v.shape[0]],dtype=torch.bool).cuda()
                # semantic_lists = torch.zeros_like(vertices[:,:,0:1])
                if torch.unique(cube_idx).tolist()[-1]>1:
                    yx = 1
                for j in torch.unique(cube_idx).tolist():
                    # data = torch.load(os.path.join(result_dir, f"optimed_grid_{j}.pth")) #optimed_grid_{i}.pth
                    data = torch.load(os.path.join(result_dir, FILE)) #optimed_grid_{i}.pth
                    grid = data['grid']
                    # grids.append(grid)
                    decoder = data['decoder']
                    octree = data['octree']
                    if INSTANCE:
                        # instance_corners = data['instance_corners']
                        instance_model = Mask3D(**cfg['instance'], query_position=data['query_position'], device=device) #, radius=data['radius']
                        instance_model.to(device)
                        instance_model.load_state_dict(data['instance_model'], strict=False)
                        # instance_model.update_network(data['query_mask'], data['query_update_map'])
                        instance_model.gaussians = QueryModel()
                        instance_model.gaussians.load_ply(os.path.join(result_dir, f'optimed_query_19.ply'))
                    mask_cube = cube_idx==j
                    # masks.append(mask_cube)
                    p_g = v[mask_cube] - origin_all[j - START]
                    p_g = (p_g+1.0)/2
                    # normal
                    grad = autodiff_gradient(p_g.reshape(-1, 3), sdf) / scale
                    normal_buffer = torch.nn.functional.normalize(grad, p=2, dim=-1, eps=1e-5)
                    nor = (normal_buffer + 1.0) / 2.0
                    if 'rgb' in channel:
                        # feats = grid['sdf'](p_g.reshape(-1, 3))
                        radiance_feats = grid['rgb'](p_g.reshape(-1, 3))
                        # rgb_g = decoder['rgb'](radiance_feats)
                        grad = autodiff_gradient(p_g.reshape(-1, 3), sdf) / scale
                        normal = grad / grad.norm(2, 1)[:, None]
                        embedding = data['embedding_a'][0] if 'embedding_a' in data.keys() else None
                        rgb_g = decoder['rgb'](radiance_feats=radiance_feats, view_dirs=-grad/grad.norm(2, 1)[:, None], appearance_embedding=embedding, grads=normal)
                        # rgbs = self.decoder['rgb'](radiance_feats=radiance_feats, view_dirs=view_dirs.reshape(-1,3),appearance_embedding=embedding, grads=normals)
                        rgb_g = torch.sigmoid(rgb_g)
                        rgb[mask_cube] = rgb_g
                    if 'sem' in channel:
                        # feats = grid['sdf'](p_g.reshape(-1, 3))
                        # _, sem_g = decoder['sdf'](feats)
                        # sem_g = torch.argmax(torch.nn.functional.softmax(sem_g.squeeze(), dim=-1), -1)
                        # # 
                        feats = grid['sdf'](p_g.reshape(-1, 3))
                        _, sdf_last_layer = decoder['sdf'](feats,return_last=True)

                        # radiance_feats = grid['rgb'](p_g.reshape(-1, 3))
                        # grad = autodiff_gradient(p_g.reshape(-1, 3), sdf) / scale
                        # normal = grad / grad.norm(2, 1)[:, None]
                        # embedding = data['embedding_a'][0] if 'embedding_a' in data.keys() else None
                        # _, rgb_last_layer = decoder['rgb'](radiance_feats=radiance_feats, view_dirs=-grad/grad.norm(2, 1)[:, None], appearance_embedding=embedding, grads=normal, return_last=True)
                        
                        semantic_feat = grid['semantic'](p_g.reshape(-1, 3))
                        # input = torch.hstack((semantic_feat, sdf_last_layer))
                        # input = torch.hstack((input, rgb_last_layer))
                        # if j==2: #                        # input = torch.hstack((input, p_g.reshape(-1, 3)))
                        sem_g_ = decoder['semantic'].forward(semantic_feat)
                        sem_g = torch.argmax(torch.nn.functional.softmax(sem_g_.squeeze(), dim=-1), -1)
                        # sem[mask_cube] = sem_g
                    if 'ins' in channel:
                        # instance_feat = grid['instance'](p_g.reshape(-1, 3))
                        # ins_g = decoder['instance'].forward(instance_feat)
                        instance_corners = [(p_g*2-1).reshape(-1, 3)]
                        # ins_feat = [instance_interpolate((p_g*2-1).reshape(-1, 3)[:,None], grid['instance'].features[0], grid['instance'].active_lods[-1], grid['instance'])]
                        ins_feat = [grid['instance'](p_g.reshape(-1, 3))]
                        instance_heat_map, thing_semantic_heat_map, query_position, query_loss = instance_model(
                                                                                                    ins_feat, 
                                                                                                    instance_corners,
                                                                                                    scale
                                                                                                ) 
                        ins_g_ = instance_heat_map
                        ins_g = torch.argmax(torch.nn.functional.softmax(ins_g_.squeeze(), dim=-1), -1)
                        ins[mask_cube] = ins_g

                        stuff_mask = torch.ones(sem_g_.shape[-1]).bool().to(device)
                        THING = cfg['Thing_class']
                        stuff_mask[THING] = False
                        sem_g_norm = torch.nn.functional.softmax(sem_g_)
                        panoptic_instance_prob = (1 - sem_g_norm[:,stuff_mask].sum(-1))[:,None] * (ins_g_/ins_g_.sum(-1)[:,None]) 
                        pan_g_ = torch.hstack((sem_g_norm[:,stuff_mask], panoptic_instance_prob))
                        pan_g = torch.argmax(torch.nn.functional.softmax(pan_g_.squeeze(), dim=-1), -1)
                        pan[mask_cube] = pan_g

                        stuff_label = cfg['Stuff_class']
                        thing_label = torch.argmax(thing_semantic_heat_map,-1)
                        thing_map = [0]+ cfg['Thing_class']
                        pan2sem_mapping = { **{pan_id+1: sem_id for pan_id, sem_id in enumerate(stuff_label)}, \
                                        **{pan_id+1+len(stuff_label): thing_map[sem_id] for pan_id, sem_id in enumerate(thing_label)} }
                        sem = torch.zeros_like(pan)
                        for key, val in pan2sem_mapping.items():
                                sem[pan==key] = val
                        semantic[i:next_ind] = sem.cpu()

                        # sem[mask_cube] = sem_g

                semantic[i:next_ind] = sem.cpu()
                semantic_raw[i:next_ind] = sem_g.cpu()
                color[i:next_ind] = rgb.cpu()
                instance[i:next_ind] = ins.cpu()
                panoptic[i:next_ind] = pan.cpu()
                normal[i:next_ind] = nor.cpu()
                # pidxs[i:next_ind] = pidx.cpu()

    return {'rgb':color, 'sem':semantic, 'sem_raw':semantic_raw, 'ins':instance, 'pan':panoptic, 'normal':normal}
        
def save_mesh(mymesh, result, cfg):
    if 'sem' in result.keys():
        save_file = savedir+f'/mesh/mesh_semantic_{START}_{CUBE}.ply'
        v_colors_sem = np.vstack([id2label[semID].color for semID in result['sem'].tolist()]) / 255.0
        mymesh.vertex_colors = o3d.utility.Vector3dVector(v_colors_sem.reshape(-1,3)) 
        o3d.io.write_triangle_mesh(save_file, mymesh)
    if 'sem_raw' in result.keys():
        save_file = savedir+f'/mesh/mesh_semantic_raw_{START}_{CUBE}.ply'
        v_colors_sem = np.vstack([id2label[semID].color for semID in result['sem_raw'].tolist()]) / 255.0
        mymesh.vertex_colors = o3d.utility.Vector3dVector(v_colors_sem.reshape(-1,3)) 
        o3d.io.write_triangle_mesh(save_file, mymesh)
    # if 'rgb' in result.keys():
    #     save_file = savedir+f'/mesh/mesh_rgb_{START}_{CUBE}.ply'
    #     v_colors = result['rgb'].detach().numpy()
    #     mymesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.reshape(-1,3)) 
    #     o3d.io.write_triangle_mesh(save_file, mymesh)
    if 'pan' in result.keys():
        color_id = np.random.random( [1000, 3] )
        save_file = savedir+f'/mesh/mesh_panoptic_{START}_{CUBE}.ply'
        v_colors_pan = color_id[np.array(result['pan'], dtype=np.int16)]
        stuff_mask = np.array(result['pan'], dtype=np.int16) < len(cfg['Stuff_class'])+1
        v_colors_pan[stuff_mask,:] = v_colors_sem[stuff_mask,:]
        mymesh.vertex_colors = o3d.utility.Vector3dVector(v_colors_pan.reshape(-1,3)) 
        o3d.io.write_triangle_mesh(save_file, mymesh)
    if 'ins' in result.keys():
        save_file = savedir+f'/mesh/mesh_instance_{START}_{CUBE}.ply'
        # np.random.seed(42)
        # color_id = np.random.random( [800, 3] )
        # color_id[0] = np.zeros([1,3])
        v_colors_ins = v_colors_pan.copy()
        v_colors_ins[stuff_mask,:] = 100/255. * np.ones([1,3])
        mymesh.vertex_colors = o3d.utility.Vector3dVector(v_colors_ins.reshape(-1,3)) 
        o3d.io.write_triangle_mesh(save_file, mymesh)

    #normal
    save_file = savedir+f'/mesh/mesh_normal_{START}_{CUBE}.ply'
    v_colors_nor = result['normal']
    mymesh.vertex_colors = o3d.utility.Vector3dVector(v_colors_nor.reshape(-1,3)) 
    o3d.io.write_triangle_mesh(save_file, mymesh)
                    
                    

def sdf(coords):
    shape = coords.shape
    if shape[0] == 0:
        return dict(sdf=torch.zeros_like(coords)[...,0:1])

    feats = grid['sdf'](coords.reshape(-1, 3))
    # mask = feats.sum(-1)==0
    sdfs = decoder['sdf'](feats)[:,0:1]
    return sdfs

def instance_interpolate(coords, feats, lod, instance_model):
        query_results = instance_model.blas.query(coords[:,0], lod, with_parents=False)
        pidx = query_results.pidx
        fs = spc_ops.unbatched_interpolate_trilinear(coords, pidx.int(), instance_model.blas.points, instance_model.trinkets.int(),
                                                        feats.half(), lod).float()
        return fs.reshape(coords.shape[0], feats.shape[-1])



if __name__ == "__main__":
    # origins = []
    START = 0 #25
    CUBE = 1 #20 #3 #2 #38 #36
    print(f'extracting mesh of CUBE_{CUBE}')
    # FILE = f"optimed_grid_{START}.pth" #"with_sem.pth" # 
    # FILE = "optimed_grid_0_49.pth"
    FILE = "optimed_grid_0_19.pth"
    SEMANTIC = 1 #0 #
    INSTANCE = 1 #0 #
    # THING = [2,5,6]    # bcd2436daf
    # THING = [2,3,5,6,7]    # 5748ce6f01
    # THING = [2,3,5,6,7,10,11,12]    # 1ada7a0617
    RGB = 0 #1
    savedir = '/mnt/nas_new/yx/pami/exp/scannet/0000_02/v0' #'/sdf1/yx/pami/draw_fig/1ada7a0617_test/v0' #"/sde1/yx/dataset/replica/result/apartment_2/rgb_depth_new/v0" #
    result_dir = savedir + '/grid' #'semantic_result_kitti/round2_wonull_lr0002_00'
    # data = torch.load(os.path.join(result_dir, f'grid/optimed_grid_0.pth'))
    data = torch.load(os.path.join(result_dir, f'{FILE}'))
    grid = data['grid']
    decoder = data['decoder']
    os.makedirs(os.path.join(savedir, 'mesh'), exist_ok=True)
    STLPATH= os.path.join(savedir, f'mesh/mesh_test_{START}_{CUBE}_32.stl') #mesh_grid_{CUBE}.stl
    origin_all = []
    scale = data['scale']
    world_origin = data['origin']/scale
    origin0 = data['origin']/scale 
    origin_all.append(origin0-world_origin) 
    # origin_all.append((origin0-world_origin+1.0)/2)
    device = scale.device
    idx = 0
    # for idx in range(1,CUBE):
    #     origin = torch.load(os.path.join(result_dir, f'grid/optimed_grid_{idx}.pth'))['origin']/scale
    #     origin_all.append(origin-origin0)
    for idx in range(START+1,START+CUBE):
        origin = torch.load(os.path.join(result_dir, f'optimed_grid_0_{idx}.pth'))['origin']/scale
        origin_all.append(origin-world_origin)
        # origin_all.append((origin-world_origin+1.0)/2)
    origin_all = torch.stack(origin_all)

    # extract_mesh(idx)

    np.random.seed(25) #60 scannet
    # mesh_path = "/sde1/yx/dataset/replica/result/apartment_2/rgb_depth_new/v0/mesh/edit_mesh_remove_faces.ply" #"/sdf1/yx/pami/draw_fig/final/scannet++/1ada7a0617/ours/mesh/mesh_semantic_0_1.ply" #
    mesh_path = "/mnt/nas_new/yx/pami/exp/scannet/0000_02/v0/mesh/mesh_test_0_1_32.stl" #"/sdf1/yx/pami/draw_fig/final/scannet++/1ada7a0617/ours/mesh/mesh_semantic_0_1.ply" #
    savedir = '/mnt/nas_new/yx/pami/exp/scannet/0000_02/v0' #'/sdf1/yx/pami/draw_fig/final/replica/ours' #'/sdf1/yx/pami/draw_fig/final/scannet++/1ada7a0617/ours' #
    os.makedirs(os.path.join(savedir, 'mesh'), exist_ok=True)
    # mesh_path = STLPATH
    yaml_path = "config/ijrr/scannet0000_02/render_scannet_hash_++_all.yaml"
    # yaml_path = "config/ijrr/replica_apartment_2/render_scannet_hash_++_all.yaml"
    # yaml_path = "config/ijrr/scannetpp_1ada7a0617/render_scannet_hash_++_all.yaml"
    from instance.models.mask3d import Mask3D
    # from instance.models.mask3d_old import Mask3D
    import yaml
    # yaml_path = "config/ijrr/scannet0000_02/render_scannet_hash_++.yaml"
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    result = inference_mesh(mesh_path, cfg, channel=['sem','ins'])
    mymesh= o3d.io.read_triangle_mesh(mesh_path) 
    save_mesh(mymesh, result, cfg)
