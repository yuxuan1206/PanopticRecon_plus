import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KITTI360_DATASET"] = "/sdf1/kitti360"
os.environ["XGRIDS_DATASET"] = "/xgrids/roma_playground" #"/sdf1/xgrids/reconstruct"
os.environ['QT_QPA_PLATFORM']="offscreen"
import torch
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./")
import open3d as o3d
import cv2
import time
import torch.nn.functional as F
from scripts.labels_scannet import labels, id2label
import imageio
from tqdm import tqdm
import time
import pandas
from autolabel.dataset import SceneDataset
from autolabel import model_utils
from autolabel import visualization
from autolabel.utils.feature_utils import get_feature_extractor
from pathlib import Path
from sklearn import decomposition
from sklearn.metrics.pairwise import cosine_similarity
from torch_ngp.nerf.utils import custom_meshgrid
import mcubes
import trimesh
import h5py
import pickle

# # scannet
# scene_idx = '0628_02' #'0420_01' #'0088_00' #'0087_02' #'0628_02' #
# # MODEL = '/sdf1/yx/exp/all/scannet/'+scene_idx+'/3dgs_depth_more/v0/grid/optimed_grid_0_19.pth'
# result_path = f'/home/yx/Data/yx/exp/mesh/pvlff' #{scene_idx}'
# # yaml_path = "config/exp/scannet"+scene_idx+"/render_scannet_hash_++_all.yaml"

# mesh_PATH = '/mnt/meadow/yx/dataset/scannet/scans/scene0087_02/scene0087_02_vh_clean_2.labels.ply'
# # GT_PATH = '/mnt/data/yx/dataset/scannet/gt_pc/scene'+scene_idx+'_GT.npy'
# label_map_path = os.path.join("/home/yx/Data/yx/IROS2024/pvlff/scannet", f"scene{scene_idx}/label_map.csv")
# # MESH_PATH = f'/home/yx/Data/yx/dataset/scannet/scans/scene{scene_idx}/scene{scene_idx}_vh_clean_2.ply'
# MESH_PATH = f'/home/yx/Data/yx/exp/mesh/pvlff/{scene_idx}.ply'

# # scannet++
# scene_idx = 'f6659a3107' #'5748ce6f01' #'1ada7a0617' #'1ada7a0617' #
# # MODEL = f'/sdc1/xyx/result/panoptic_recon++/scannetpp/scen_{scene_idx}/rgb_depth/v0/grid/optimed_grid_0_19.pth'
# result_path = '/home/yx/Data/yx/exp/mesh/pvlff' #f'/home/yx/Data/yx/IROS2024/pvlff/scannet++/{scene_idx}'
# # yaml_path = f"config/exp/scannetpp_{scene_idx}/render_scannet_hash_++_all.yaml"
# # GT_PATH = f'/mnt/data/yx/dataset/scannet++/gt_pc/{scene_idx}_GT.npy'
# label_map_path = os.path.join("/home/yx/Data/yx/IROS2024/pvlff/scannet++", f"{scene_idx}/label_map.csv")
# # MESH_PATH = f'/home/yx/Data/yx/dataset/scannet++/{scene_idx}/mesh_aligned_0.05.ply'
# MESH_PATH = f'/home/yx/Data/yx/exp/mesh/pvlff/{scene_idx}.ply'

# replica
scene_idx = 'apartment_2' 
result_path = 'home/yx/Data/yx/exp/mesh/pvlff'
GT_PATH = '/mnt/data/yx/dataset/replica/gt_pc/'+scene_idx+'_GT.npy'
label_map_path = os.path.join("/mnt/data/yx/dataset/replica/label_csv", f"{scene_idx}.csv")
MESH_PATH = '/home/yx/Data/yx/exp/mesh/pvlff/apartment_2_edit_mesh_remove_faces.ply'


label_map = pandas.read_csv(label_map_path)
thing = np.unique(label_map['new_label'][np.array(label_map['type'], dtype=np.bool8)].values)
evaluated_labels = np.array(label_map['new_label'])[np.array(label_map['evaluated'], dtype=np.bool8)]
label_id_map = np.array(list(label_map['new_label'].values))
print(label_id_map)


def read_args():
    parser = model_utils.model_flag_parser()
    parser.add_argument('scene')
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument(
        '--max-depth',
        type=float,
        default=7.5,
        help="The maximum depth used in colormapping the depth frames.")
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--out',
                        type=str,
                        required=True,
                        help="Where to save the video.")
    parser.add_argument('--classes',
                        default=None,
                        type=str,
                        nargs='+',
                        help="Which classes to segment the scene into.")
    parser.add_argument('--label-map',
                        default=None,
                        type=str,
                        help="Path to list of labels.")
    return parser.parse_args()

@ torch.no_grad()
def compute_3d_labels(model, xyzs, feature_transform, classes):
    aabb = model.aabb_infer
    xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.
    # query density and RGB
    density_outputs = model.density(xyzs.reshape(-1, 3))
    geometric_features = density_outputs['geo_feat']
    xyz_feature_encoding = model.feature_encoder(xyzs.reshape(-1, 3), bound=model.bound)
    # semantic
    semantic_features = model.semantic(
        geometric_features.view(-1, geometric_features.shape[-1]))
    # semantic_features = semantic_features.view(
    #     (geometric_features.shape[0], geometric_features.shape[1],
    #         semantic_features.shape[-1]))

    # contrastive
    contrastive_ema = None
    contrastive_features = model.contrastive(xyz_feature_encoding, contrastive_ema)
    contrastive_features = F.normalize(contrastive_features)
    batch = 50000
    if classes is not None:
        similarities, p_semantic = [], []
        for i in range(0,semantic_features.shape[0],batch):
            features = semantic_features[i:i+batch,:]
            features = (features / torch.norm(features, dim=-1, keepdim=True))
            text_features = feature_transform.text_features
            C = text_features.shape[0]
            similarities_ = torch.zeros((features.shape[0], C), dtype=features.dtype)
            similarities_ = (features[:, None] * text_features).sum(dim=-1).cpu()
            p_semantic_ = similarities_.argmax(dim=-1).cpu().numpy()
            similarities.append(similarities_)
            p_semantic.append(p_semantic_)
        
    similarities = torch.vstack(similarities)
    p_semantic = np.concatenate(p_semantic) 

    instance_feature = contrastive_features.cpu().numpy()
    sim_mat = cosine_similarity(instance_feature, model.instance_centers)
    pred_instance = np.argmax(sim_mat, axis=1)
    

    # denoised semantic
    # print(similarities.shape)
    pred_semantic_denoiseds = np.copy(p_semantic)
    instance_ids = np.unique(pred_instance)
    for ins_id in instance_ids:
        if ins_id == 0:
            continue
        
        # if method == 'majority_voting':
        #     semantic_ids = p_semantic[pred_instance == ins_id]
        #     ids, cnts = np.unique(semantic_ids, return_counts=True)
        #     pred_semantic_denoiseds[pred_instance == ins_id] = ids[np.argmax(cnts)]
        # elif method == 'average_similarity':
        sim = similarities.cpu().numpy()[pred_instance == ins_id]
        sim = np.mean(sim, axis=0)
        s_id = np.argmax(sim, axis=-1)
        pred_semantic_denoiseds[pred_instance == ins_id] = s_id
        # elif method == 'average_feature':
        #     # currently unavailable due to the memory size.
        #     # need better implementation strategy
        #     # TODO
        #     raise NotImplementedError()
        #     feats = semantic_features[pred_instance == ins_id]
        #     feats = np.mean(feats, dim=0)
        #     feats = feats / np.norm(feats, order=2, axis=-1)
        #     sim = feats @ text_features.T
        #     s_id = np.argmax(sim, axis=-1)
        #     pred_semantic_denoiseds[pred_instance == ins_id] = s_id
        # else:
        #     raise NotImplementedError()
    
    pred_semantic_denoiseds = label_id_map[pred_semantic_denoiseds]

    p_panoptic = p_semantic.copy()
    instance_mask = np.isin(p_semantic, Thing_class)
    p_panoptic[instance_mask] = pred_instance[instance_mask]
    
    return pred_semantic_denoiseds, pred_instance, p_panoptic

class FeatureTransformer:

    def __init__(self, scene_path, feature_name, classes, checkpoint=None, without_features=False):
        if not without_features:
            with h5py.File(os.path.join(scene_path, 'features.hdf'), 'r') as f:
                features = f[f'features/{feature_name}']
                blob = features.attrs['pca'].tobytes()
                self.pca = pickle.loads(blob)
                self.feature_min = features.attrs['min']
                self.feature_range = features.attrs['range']
            self.first_fit = False
        else:
            self.pca = decomposition.PCA(n_components=3)
            self.feature_min = None
            self.feature_range = None
            self.first_fit = True


        if feature_name is not None:
            extractor = get_feature_extractor(feature_name, checkpoint)
            self.text_features = self._encode_text(extractor, classes)

    def _encode_text(self, extractor, text):
        return extractor.encode_text(text)

    def __call__(self, p_features):
        H, W, C = p_features.shape
        if self.first_fit:
            features = self.pca.fit_transform(p_features.reshape(H * W, C))
            self.first_fit = False
        else:
            features = self.pca.transform(p_features.reshape(H * W, C))

        if (self.feature_min is not None) and (self.feature_range is not None):
            features = np.clip((features - self.feature_min) / self.feature_range,
                            0., 1.)
        else:
            features = np.clip((features - np.min(features)) / (np.max(features) - np.min(features)),
                            0., 1.)
        return (features.reshape(H, W, 3) * 255.).astype(np.uint8)


def load_pointscloud(mesh_path):

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pc = torch.FloatTensor(np.asarray(mesh.vertices))
    sem_gt = np.int64(255 * np.asarray(mesh.vertex_colors))
    # pcd = o3d.io.read_point_cloud("feature_result_kitti/00/semantic_points_gt.ply")
    # lidar_pts = torch.FloatTensor(pcd.points)
    return pc, sem_gt

def eval_semantic(sem_gt, sem):
        # label_map_path = os.path.join(result_folder, "label_map.csv")
        # label_map = pandas.read_csv(label_map_path) 

        iou, acc = {}, {}
        # self.model.eval()
        p_semantic = sem
        gt_semantic = sem_gt
        # p_instance = self._predict_instance(point_cloud)
        mask = np.isin(gt_semantic, evaluated_labels)
        intersection = np.bitwise_and(p_semantic == gt_semantic, mask).sum()

        union = mask.sum()
        # if self.debug:
        #         pc_vis = point_cloud.cpu().numpy()[mask]
        #         pc_vis = o3d.utility.Vector3dVector(pc_vis)
        #         pc_vis = o3d.geometry.PointCloud(pc_vis)

        #         p_sem = self.label_to_color_id[p_semantic][mask]
        #         colors = COLORS[p_sem % COLORS.shape[0]] / 255.
        #         pc_vis.colors = o3d.utility.Vector3dVector(colors)

        #         o3d.visualization.draw_geometries([pc_vis])

        #         gt_sem = self.label_to_color_id[gt_semantic[mask]]
        #         gt_colors = COLORS[gt_sem % COLORS.shape[0]] / 255.
        #         pc_vis.colors = o3d.utility.Vector3dVector(gt_colors)
        #         o3d.visualization.draw_geometries([pc_vis])

        label = np.unique(label_map['new_label'].values)
        for i in label:
                if i not in evaluated_labels:
                        continue
                object_mask = gt_semantic[mask] == i
                if object_mask.sum() == 0:
                        continue
                p_mask = p_semantic[mask]
                true_positive = np.bitwise_and(p_mask == i, object_mask).sum()
                true_negative = np.bitwise_and(p_mask != i, object_mask == False).sum()
                false_positive = np.bitwise_and(p_mask == i, object_mask == False).sum()
                false_negative = np.bitwise_and(p_mask != i, object_mask).sum()

                class_iou = float(true_positive) / ( true_positive + false_positive + false_negative)
                prompt = label_map[label_map['new_label']==i]['prompt'].values[0]
                iou[prompt] = class_iou
                acc[prompt] = float(true_positive) / (true_positive + false_negative)
        
        iou['total'] = np.mean(list(iou.values()))
        acc['total'] = np.mean(list(acc.values()))

        print(f"mIoU = {iou['total']*100} | mAcc = {acc['total']*100}")

        return iou, acc

def eval_instance(ins_gt, ins):
        from scipy.optimize import linear_sum_assignment
        def compute_iou(pred, gt):
                intersection = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                if union == 0:
                        return 0 
                return intersection/ union

        iou, acc = {}, {}
        
        ious = np.zeros((len(np.unique(ins)), len(np.unique(ins_gt))))
        for i, pred in enumerate(np.unique(ins)):
                for j, gt in enumerate(np.unique(ins_gt)):
                        pred_amsk = (ins == pred).flatten()
                        gt_mask = (ins_gt == gt).flatten()
                        ious[i,j] = compute_iou(pred_amsk, gt_mask)

        row_ind, col_ind = linear_sum_assignment(-ious)

        mCov = np.mean([ious[i,j] for i, j in zip(row_ind, col_ind)])
        print(f"ins: {[ious[i,j] for i, j in zip(row_ind, col_ind)]}")
        print(f"ins_gt: {np.unique(ins_gt)}")
        weights = [(ins_gt==np.unique(ins_gt)[j]).sum()/ins_gt.shape[0] for j in col_ind]
        print(f"weights: {weights}")
        mW_cov = np.average([ious[i,j] for i, j in zip(row_ind, col_ind)], weights=weights)

        print(f"mCov = {mCov*100} | mW_cov = {mW_cov*100}")
        return mCov, mW_cov



OUTPUT = 'all' #'per' #'all'
MODE = 'val' #'test' #'NVS'#

if __name__ == "__main__":

        
        flags = read_args()
        model_params = model_utils.read_params(flags.model_dir)

        view_size = (480, 360) #scannet
        # view_size = (1408, 376) #kitti360
        dataset = SceneDataset('test',
                                flags.scene,
                                size=view_size,
                                batch_size=16384,
                                features=model_params.features,
                                load_semantic=False,
                                lazy=True)
        classes = flags.classes
        if flags.label_map is not None:
                label_map = pandas.read_csv(flags.label_map)
                classes = label_map['prompt'].values
                Thing_class = np.unique(label_map['new_label'][np.array(label_map['type'], dtype=np.bool8)].values)
        global semantic_color_map
        semantic_color_map = (np.random.rand(len(classes), 3) * 255).astype(np.uint8)

        feature_transform = None
        if model_params.features is not None:
                feature_transform = FeatureTransformer(flags.scene,
                                                model_params.features, classes,
                                                flags.checkpoint)
        n_classes = dataset.n_classes if dataset.n_classes is not None else 2
        model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds, n_classes, model_params).cuda()
        model = model.eval()
        model_utils.load_checkpoint(model, os.path.join(flags.model_dir, 'checkpoints'))
        global instance_color_map
        instance_color_map = (np.random.rand(model.instance_centers.shape[0], 3) * 255).astype(np.uint8)
        
        # GT = np.load(GT_PATH)
        # pts, sem_gt, ins_gt = GT[:,:3], GT[:,3], GT[:,4].astype(int)

        mesh = o3d.io.read_triangle_mesh(MESH_PATH)
        GT = np.asarray(mesh.vertices).copy()
        pts = GT[:, :3]

        # # # #only scannet++ & replica
        pts[:, -1:] *= -1 
        pts = pts[:,[1,0,2]]
        
        # render = Render(0, yaml_path, MODEL, mode=MODE)         
        # sem, pan = render.query_pts_sem(torch.FloatTensor(pts).to(render.device))
        pts_norm = pts - dataset.scene_center
        T_w02w = np.array([[0,1,0,0.0],[0,0,1,0],[1,0,0,0],[0,0,0,1]])
        pts_norm = pts_norm @ T_w02w[:3,:3].T
        pts_norm = torch.FloatTensor(pts_norm).cuda()
        
        sem, ins, pan = [], [], []
        for b in range(0,pts_norm.shape[0],1000000):
            sem_, ins_, pan_ = compute_3d_labels(model, pts_norm[b:b+1000000,:], feature_transform, classes)
            # print(sem_.shape)
            sem.append(sem_)
            ins.append(ins_)
            pan.append(pan_)
        sem = np.hstack(sem)
        ins = np.hstack(ins)
        pan = np.hstack(pan)
        # print(np.unique(sem))

        # o3d_pts = o3d.geometry.PointCloud()
        # o3d_pts.points = o3d.utility.Vector3dVector(pts)
        sem_color = np.vstack([id2label[semID].color for semID in sem.tolist()])
        # o3d_pts.colors = o3d.utility.Vector3dVector(sem_color/255.0)
        # o3d.io.write_point_cloud(os.path.join(result_path, 'pred_3d_sem.ply'), o3d_pts)

        mesh.vertex_colors = o3d.utility.Vector3dVector(sem_color/255.0)
        o3d.io.write_triangle_mesh(os.path.join(result_path, f'{scene_idx}_mesh_sem.ply'), mesh)
        
        
        color_id = np.random.random( [1000, 3] )
        # o3d_pts.colors = o3d.utility.Vector3dVector(color_id[ins])
        # o3d.io.write_point_cloud(os.path.join(result_path, 'pred_3d_ins.ply'), o3d_pts)

        mesh.vertex_colors = o3d.utility.Vector3dVector(color_id[ins])
        o3d.io.write_triangle_mesh(os.path.join(result_path, f'{scene_idx}_mesh_ins.ply'), mesh)

        ins_color = color_id[ins]
        pan_color = sem_color/255.0
        instance_mask = np.isin(sem, Thing_class)
        pan_color[instance_mask] = ins_color[instance_mask]
        # o3d_pts.colors = o3d.utility.Vector3dVector(pan_color)
        # o3d.io.write_point_cloud(os.path.join(result_path, 'pred_3d_pan.ply'), o3d_pts)

        # o3d_pts.colors = o3d.utility.Vector3dVector(color_id[ins_gt])
        # o3d.io.write_point_cloud(os.path.join(result_path, 'pred_3d_ins_gt.ply'), o3d_pts)

        mesh.vertex_colors = o3d.utility.Vector3dVector(pan_color)
        o3d.io.write_triangle_mesh(os.path.join(result_path, f'{scene_idx}_mesh_pan.ply'), mesh)


        # iou, acc = eval_semantic(sem_gt.astype(np.int16()), sem)
        # print(f"iou, acc = {iou}, {acc}")

        # # ins = pan
        # # ins[~np.isin(sem, thing)] = 0
        # mask = np.logical_and(np.isin(sem_gt, evaluated_labels), ins_gt>0)
        # mCov, mW_cov = eval_instance(ins_gt[mask], ins[mask])
        # print(f"mCov = {mCov}, mW_cov = {mW_cov}")
        # # v_colors = np.vstack([id2label[semID].color for semID in sem.tolist()])



       
