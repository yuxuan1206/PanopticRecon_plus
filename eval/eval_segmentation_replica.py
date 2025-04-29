import csv
import os
os.environ['QT_QPA_PLATFORM']="offscreen"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import cv2
from pathlib import Path
import pandas
from PIL import Image
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from rich.table import Table
from rich.console import Console
from rich import print as rprint
from pathlib import Path
import json
import pickle

scene_idx = "apartment_2" 
scene_name = scene_idx

gt_folder = "/mnt/data/yx/dataset/replica/"+scene_idx+"/"
gt_semantic_path = gt_folder + "semantic"
gt_instance_path = gt_folder + "instance"

# PVLFF
result_folder = f"/home/yx/Data/yx/IROS2024/pvlff/replica/{scene_idx}" 
pre_semantic_path = os.path.join(result_folder, "render_semantic_label")
pre_instance_path = os.path.join(result_folder, "render_instance_label")
names = os.listdir(gt_semantic_path)

# # our
# # result_folder = "/sdc1/xyx/result/panoptic_recon++/scannetpp/" + scene_name + "/seg_wo_weight/v0" #ours
# result_folder = "/sdc1/xyx/result/panoptic_recon++/scannetpp/" + scene_name + "/PanopticRecon/v0" #PanopticRecon
# pre_semantic_path = os.path.join(result_folder, "semantic_render_label")
# pre_instance_path = os.path.join(result_folder, "instance_render_label")
# names = os.listdir(gt_semantic_path)

# # PanopticLifting
# result_folder = f"/home/yx/code/panoptic-lifting/runs/{scene_idx}_test_scannetpp_{scene_idx}" 
# pre_semantic_path = os.path.join(result_folder, "pred_semantics")
# pre_instance_path = os.path.join(result_folder, "pred_surrogateid")
# names = os.listdir(gt_semantic_path)

debug_flag = True #False #
H, W = 968, 1296
stride = 1
vis_path = os.path.join(result_folder, "visualizations")
label_map_path = os.path.join("eval/label_map", f"{scene_idx}_map.csv")
label_map = pandas.read_csv(label_map_path)
factor = 4


# -----
# Adapt from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
from collections import defaultdict
OFFSET = 256 * 256 * 256
VOID = 0 # or -1

class PanopticStatCat():
        def __init__(self):
            # panoptic segmentation evaluation
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

            # semantic segmentation evaluation
            self.semantic = {'iou': 0.0, 'acc': 0.0}
            self.semantic_denoised = {'iou': 0.0, 'acc': 0.0}
            self.semantic_n = 0

        def __iadd__(self, panoptic_stat_cat):
            self.iou += panoptic_stat_cat.iou
            self.tp += panoptic_stat_cat.tp
            self.fp += panoptic_stat_cat.fp
            self.fn += panoptic_stat_cat.fn
            self.semantic['iou'] += panoptic_stat_cat.semantic['iou']
            self.semantic['acc'] += panoptic_stat_cat.semantic['acc']
            self.semantic_denoised['iou'] += panoptic_stat_cat.semantic_denoised['iou']
            self.semantic_denoised['acc'] += panoptic_stat_cat.semantic_denoised['acc']
            self.semantic_n += panoptic_stat_cat.semantic_n
            return self


class PanopticStat():
    def __init__(self):
        self.panoptic_per_cat = defaultdict(PanopticStatCat)
        self.instance_stat = {
            'coverage': [],
            'gt_inst_area': [],
            'num_pred_inst': 0,
            'num_gt_inst': 0,
        }
        self.panoptic_miou = 0

    def __getitem__(self, i):
        return self.panoptic_per_cat[i]

    def __iadd__(self, panoptic_stat):
        for label, panoptic_stat_cat in panoptic_stat.panoptic_per_cat.items():
            self.panoptic_per_cat[label] += panoptic_stat_cat
        self.instance_stat['coverage'].extend(panoptic_stat.instance_stat['coverage'])
        self.instance_stat['gt_inst_area'].extend(panoptic_stat.instance_stat['gt_inst_area'])
        self.instance_stat['num_pred_inst'] += panoptic_stat.instance_stat['num_pred_inst']
        self.instance_stat['num_gt_inst'] += panoptic_stat.instance_stat['num_gt_inst']
        return self

    def pq_average(self, categories, label_thing_mapping, instance_type='all', verbose=False):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        tp_all, fp_all, fn_all = 0, 0, 0
        for label in categories:
            iou = self.panoptic_per_cat[label].iou
            tp = self.panoptic_per_cat[label].tp
            fp = self.panoptic_per_cat[label].fp
            fn = self.panoptic_per_cat[label].fn
            if tp + fp + fn == 0:
                n += 1
                if verbose:
                    per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'valid': False, 'tp': tp, 'fp': fp, 'fn': fn}
                else:
                    per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'valid': False}
                continue
            
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            if verbose:
                per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'valid': True, 'tp': tp, 'fp': fp, 'fn': fn}
            else:
                per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'valid': True}
            
            # only evaluate instances of "thing" type
            if label_thing_mapping is not None:
                if instance_type == 'thing' and label_thing_mapping[label] != 1:
                    continue
                if instance_type == 'stuff' and label_thing_mapping[label] != 0:
                    continue

            pq += pq_class
            sq += sq_class
            rq += rq_class
            tp_all += tp
            fp_all += fp
            fn_all += fn
            n += 1

        if verbose:
            return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n, 
                    'tp': tp_all / n, 'fp': fp_all / n, 'fn': fn_all / n}, per_class_results
        else:
            return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n, 'miou': self.panoptic_miou}, per_class_results
    
    def instance_average(self, iou_threshold=0.5):
        stat_coverage = np.array(self.instance_stat['coverage'])
        stat_gt_inst_area = np.array(self.instance_stat['gt_inst_area'])
        coverage = np.mean(stat_coverage)
        weighted_coverage = np.sum((stat_gt_inst_area / stat_gt_inst_area.sum()) * stat_coverage)
        prec = (stat_coverage > iou_threshold).sum() / self.instance_stat['num_pred_inst']
        rec = (stat_coverage > iou_threshold).sum() / self.instance_stat['num_gt_inst']
        return {'mCov': coverage, 'mWCov': weighted_coverage, 'mPrec': prec, 'mRec': rec}
    
    def semantic_average(self, categories):
        iou, acc, iou_d, acc_d, n = 0, 0, 0, 0, 0
        per_class_results = {}
        for label in categories:
            if self.panoptic_per_cat[label].semantic_n == 0:
                per_class_results[label] = {'iou': 0.0, 'acc': 0.0, 'iou_d': 0.0, 'acc_d': 0.0, 'valid': False}
                n += 1 # 以防漏检的不算
                continue
            n += 1
            iou_class = self.panoptic_per_cat[label].semantic['iou'] / self.panoptic_per_cat[label].semantic_n
            acc_class = self.panoptic_per_cat[label].semantic['acc'] / self.panoptic_per_cat[label].semantic_n
            iou_d_class = self.panoptic_per_cat[label].semantic_denoised['iou'] / self.panoptic_per_cat[label].semantic_n
            acc_d_class = self.panoptic_per_cat[label].semantic_denoised['acc'] / self.panoptic_per_cat[label].semantic_n
            per_class_results[label] = {'iou': iou_class, 'acc': acc_class, 'iou_d': iou_d_class, 'acc_d': acc_d_class, 'valid': True}
            iou += iou_class
            acc += acc_class
            iou_d += iou_d_class
            acc_d += acc_d_class
        return {'iou': iou / n, 'acc': acc / n, 'iou_d': iou_d / n, 'acc_d': acc_d / n, 'n': n}, per_class_results


class OpenVocabEvaluator:

    def __init__(self,
                 device='cuda:0',
                 name="model",
                 features=None,
                 checkpoint=None,
                 debug=False,
                 stride=1,
                 save_figures=None,
                 time=False):
        self.device = device
        self.name = name
        self.debug = debug
        self.stride = stride
        self.model = None
        self.label_id_map = None
        self.label_map = None
        self.features = features
        # self.extractor = get_feature_extractor(features, checkpoint)
        self.save_figures = save_figures
        self.time = time

    def reset(self, label_map, figure_path):
        self.label_map = label_map
        self.label_id_map = torch.tensor(self.label_map['idx'].values).to(
            self.device)
        # self.text_features = self._infer_text_features()
        self.label_mapping = {0: 'void'}
        self.label_to_color_id = np.zeros((label_map['idx'].max() + 1),
                                          dtype=int)
        self.our_label = np.zeros((label_map['idx'].max() + 1),
                                          dtype=int)
        self.label_thing_mapping = None
        if 'thing' in self.label_map:
            self.label_thing_mapping = {0: -1}
            for index, (i, prompt, thing, new_label) in enumerate(
                    zip(label_map['idx'], label_map['class'], label_map['thing'], label_map['label'])):
                self.label_mapping[new_label] = prompt
                self.label_to_color_id[new_label] = index + 1
                self.label_thing_mapping[new_label] = thing
                self.our_label[i] = new_label
        else:
            for index, (i, prompt) in enumerate(
                    zip(label_map['idx'], label_map['class'])):
                self.label_mapping[i] = prompt
                self.label_to_color_id[i] = index + 1
        self.save_figures = figure_path
        os.makedirs(self.save_figures, exist_ok=True)
        if 'evaluated' in self.label_map:
            self.evaluated_labels = np.unique(label_map[label_map['evaluated']==1]['label'].values)
        else:
            self.evaluated_labels = label_map['id'].values

    def _infer_text_features(self):
        return self.extractor.encode_text(self.label_map['class'].values)

    def eval(self, dataset, visualize=False):
        raise NotImplementedError()


class OpenVocabInstancePQEvaluator(OpenVocabEvaluator):

    def __init__(self, 
                 device='cuda:0', 
                 name="model", 
                 debug=False, 
                 stride=1, 
                 save_figures=None, 
                 time=False):
        super().__init__(device, name, debug, stride, save_figures, time)

    def eval(self):
        self.debug = False #True #
        self.panoptic_stat = PanopticStat()

        # process frames        
        pred_semantics, pred_instances, gt_semantics, gt_instances, indices = [], [], [], [], []
        for frame in tqdm(os.listdir(pre_semantic_path)):
            # pred_semantic = cv2.imread(os.path.join(pre_semantic_path, frame), -1)
            # pred_instance = cv2.imread(os.path.join(pre_instance_path, frame), -1)
            pred_semantic = np.array(Image.open(os.path.join(pre_semantic_path, frame)).resize((W//factor,H//factor), Image.NEAREST)).astype(np.int64)
            pred_instance = np.array(Image.open(os.path.join(pre_instance_path, frame)).resize((W//factor,H//factor), Image.NEAREST)).astype(np.int64)
            if scene_name == 'kitti360':
                pred_semantic = pred_semantic[126:,:]
                pred_instance = pred_instance[126:,:]
                # cv2.imwrite(os.path.join(os.path.join(result_folder, "vis_instance"), frame), color_map[pred_instance])
                # pred_semantic = self.our_label[pred_semantic] #panoptic_nerf
            pred_semantics.append(pred_semantic)
            pred_instances.append(pred_instance)

            idx = int(frame.split('.')[0])
            indices.append(idx)
            if scene_name == 'kitti360':
                gt_semantic = np.array(Image.open(os.path.join(gt_semantic_path, "%010d.png"%idx)).resize((W//factor,H//factor), Image.NEAREST)).astype(np.int64)
                gt_instance = np.array(Image.open(os.path.join(gt_instance_path, "%010d.png"%idx)).resize((W//factor,H//factor), Image.NEAREST)).astype(np.int64)
                # gt_semantic = np.array(Image.open(os.path.join(gt_semantic_path, "%010d.png"%idx)).resize((W//2,H//2), Image.NEAREST)).astype(np.int64)
                # gt_instance = np.array(Image.open(os.path.join(gt_instance_path, "%010d.png"%idx)).resize((W//2,H//2), Image.NEAREST)).astype(np.int64)
                gt_semantic = gt_semantic[126:,:]
                gt_instance = gt_instance[126:,:]
                gt_instance[gt_instance%1000==0]=0
            else:
                # gt_semantic = cv2.imread(os.path.join(gt_semantic_path, f"{idx}.png"), -1)
                # gt_instance = cv2.imread(os.path.join(gt_instance_path, f"{idx}.png"), -1)
                gt_semantic = np.array(Image.open(os.path.join(gt_semantic_path, f"{idx}.png")).resize((W//factor,H//factor), Image.NEAREST)).astype(np.int64)
                gt_instance = np.array(Image.open(os.path.join(gt_instance_path, f"{idx}.png")).resize((W//factor,H//factor), Image.NEAREST)).astype(np.int64)
                gt_instance[gt_instance>1e4]=0
            gt_semantic_remapping = self.our_label[gt_semantic] #ours
            # gt_semantic_remapping = gt_semantic #  panoptic lifting
            gt_semantics.append(gt_semantic_remapping)
            gt_instances.append(gt_instance)
        pred_semantics = np.stack(pred_semantics, axis=0)
        pred_instances = np.stack(pred_instances, axis=0)
        gt_semantics = np.stack(gt_semantics, axis=0)
        gt_instances = np.stack(gt_instances, axis=0)
        indices = np.array(indices)

        # evaluate semantic segmentation
        self._evaluate_semantic(gt_semantics, pred_semantics, indices)

        # label remapping for gt
        gt_instances, gt_thing_ids = self._instance_label_remapping(gt_instances, gt_semantics)

        # evaluate instance segmentation
        self._evaluate_instance(gt_instances, gt_thing_ids, pred_instances, indices)

        # label remapping for prediction
        pred_instances, pred_thing_ids = self._instance_label_remapping(pred_instances, pred_semantics)

        # evaluate panoptic segmentation
        self._evaluate_panoptic(
            pred_instances=pred_instances,
            pred_semantics=pred_semantics,
            # gt_images=gt_images,
            gt_semantics=gt_semantics,
            gt_instances=gt_instances,
            indices=indices
        )

        return self.panoptic_stat

    def _process_frames(self, dataset):
        pred_instances = []
        semantic_similarities = []
        gt_images = []
        gt_semantics = []
        gt_instances = []
        indices = []
        gt_instance_paths = dataset.scene.gt_instance()
        gt_semantic_paths = dataset.scene.gt_semantic()
        for i, (gt_semantic_path, gt_instance_path) in enumerate(
                tqdm(list(zip(gt_semantic_paths, gt_instance_paths)), desc="Processing")):
            if i % self.stride != 0:
                continue
            indices.append(i)
            batch = dataset._get_test(i)
            gt_images.append(batch['pixels'])

            # read gt semantic and gt instance
            gt_semantic = np.array(
                Image.open(gt_semantic_path).resize(dataset.camera.size, Image.NEAREST)).astype(np.int64)
            gt_instance = np.array(
                Image.open(gt_instance_path).resize(dataset.camera.size, Image.NEAREST)).astype(np.int64)
            gt_semantics.append(gt_semantic)
            gt_instances.append(gt_instance)

            # get instance and semantic features
            rays_o = torch.tensor(batch['rays_o']).to(self.device)
            rays_d = torch.tensor(batch['rays_d']).to(self.device)
            direction_norms = torch.tensor(batch['direction_norms']).to(self.device)
            outputs = self.model.render(rays_o,
                                        rays_d,
                                        direction_norms,
                                        staged=True,
                                        perturb=False)
            instance_feature = outputs['contrastive_features'].cpu().numpy()
            image_height, image_width, feature_dim = instance_feature.shape
            instance_feature = instance_feature.reshape(-1, feature_dim)
            sim_mat = cosine_similarity(instance_feature, self.model.instance_centers)
            pred_instance = np.argmax(sim_mat, axis=1)
            pred_instance = pred_instance.reshape(image_height, image_width) + 1 # start from 1, 0 means noise
            pred_instances.append(pred_instance)

            semantic_feature = outputs['semantic_features']
            semantic_feature = (semantic_feature / torch.norm(semantic_feature, dim=-1, keepdim=True))
            similarity = semantic_feature @ self.text_features.T
            similarity = similarity.cpu().numpy()
            semantic_similarities.append(similarity)
        
        pred_instances = np.stack(pred_instances, axis=0)
        semantic_similarities = np.stack(semantic_similarities, axis=0)
        gt_images = np.stack(gt_images, axis=0)
        gt_semantics = np.stack(gt_semantics, axis=0)
        gt_instances = np.stack(gt_instances, axis=0)
        indices = np.array(indices)
        return pred_instances, semantic_similarities, gt_images, gt_semantics, gt_instances, indices

    
    def _instance_label_remapping(self, instances, semantics):
        if 'thing' not in self.label_map:
            return instances
        
        stuff_id_mapping = {}
        thing_id_list = []
        instance_ids = np.unique(instances)
        new_instance_id = np.max(instance_ids) + 1

        void_mask = np.isin(instances, [VOID])
        if void_mask.sum() != 0:
            s_labels = np.unique(semantics[void_mask])
            for s_id in s_labels:
                if s_id not in self.evaluated_labels:
                    continue
                else:
                    instances[np.logical_and(
                        void_mask, semantics == s_id
                    )] = new_instance_id
                    new_instance_id += 1

        for ins_id in instance_ids:
            if ins_id == VOID:
                continue
            s_labels = semantics[instances == ins_id]
            s_ids, cnts = np.unique(s_labels, return_counts=True)
            s_id = s_ids[np.argmax(cnts)]
            
            if s_id not in self.evaluated_labels:
                instances[instances == ins_id] = VOID

            elif s_id in self.evaluated_labels and self.label_thing_mapping[s_id] == 0:
                if s_id not in stuff_id_mapping.keys():
                    stuff_id_mapping[s_id] = ins_id
                else:
                    instances[instances == ins_id] = stuff_id_mapping[s_id]
            
            elif s_id in self.evaluated_labels and self.label_thing_mapping[s_id] == 1:
                thing_id_list.append(ins_id)
        return instances, thing_id_list
    
    def _read_gt_panoptic_segmentation(self, semantic, instance):
        gt_segms = {}
        labels, labels_cnt = np.unique(instance, return_counts=True)

        for label, label_cnt in zip(labels, labels_cnt):
            if label == VOID:
                continue
            semantic_ids = semantic[instance == label]
            ids, cnts = np.unique(semantic_ids, return_counts=True)
            gt_segms[label] = {
                'area': label_cnt,
                'category_id': ids[np.argmax(cnts)]
            }
        return gt_segms
    
    def _predict_panoptic_segmentation(self, pred_instance, pred_semantic):
        # construct panoptic segmentation
        pred_segms = {}
        labels, labels_cnt = np.unique(pred_instance, return_counts=True)

        for label, label_cnt in zip(labels, labels_cnt):
            if label == VOID:
                continue
            semantic_ids = pred_semantic[pred_instance == label]
            ids, cnts = np.unique(semantic_ids, return_counts=True)
            pred_segms[label] = {
                'area': label_cnt,
                'category_id': ids[np.argmax(cnts)]
            }

        return pred_segms

    def _evaluate_semantic(self, gt_semantics, pred_semantics, indices):

        if self.debug:
            semantic_label_color_mapping = {}
            labels = np.unique(
                np.append(
                    np.unique(gt_semantics), np.unique(pred_semantics)
                )
            )
            for label in labels:
                color = np.random.rand(3, )
                semantic_label_color_mapping[label] = color
        
        for gt_semantic, pred_semantic, index in tqdm(
            list(zip(gt_semantics, pred_semantics, indices)), desc="Evaluating semantic segmentation"):

            mask = np.isin(gt_semantic, self.evaluated_labels)
            labels = np.unique(gt_semantic)
            for label in labels:
                if label not in self.evaluated_labels:
                    continue
                object_mask = gt_semantic[mask] == label

                # semantic
                pred_mask = pred_semantic[mask] == label
                true_positive = np.bitwise_and(pred_mask, object_mask).sum()
                false_positive = np.bitwise_and(pred_mask,
                                                object_mask == False).sum()
                false_negative = np.bitwise_and(pred_mask == False,
                                                object_mask).sum()

                class_iou = float(true_positive) / (
                    true_positive + false_positive + false_negative)
                self.panoptic_stat[label].semantic['iou'] += class_iou
                self.panoptic_stat[label].semantic['acc'] += float(true_positive) / (true_positive + false_negative)

                # # denoised semantic
                # pred_mask_denoised = pred_semantic_denoised[mask] == label
                # true_positive = np.bitwise_and(pred_mask_denoised, object_mask).sum()
                # false_positive = np.bitwise_and(pred_mask_denoised,
                #                                 object_mask == False).sum()
                # false_negative = np.bitwise_and(pred_mask_denoised == False,
                #                                 object_mask).sum()

                # class_iou = float(true_positive) / (
                #     true_positive + false_positive + false_negative)
                # self.panoptic_stat[label].semantic_denoised['iou'] += class_iou
                # self.panoptic_stat[label].semantic_denoised['acc'] += float(true_positive) / (true_positive + false_negative)

                self.panoptic_stat[label].semantic_n += 1
            
            if self.debug:
                plt.figure(figsize=(30, 10))
                axis = plt.subplot2grid((1, 2), loc=(0, 0))
                p_s = np.zeros((pred_semantic.shape[0], pred_semantic.shape[1], 3))
                labels = np.unique(pred_semantic)
                s_patches = []
                for label in labels:
                    color = semantic_label_color_mapping[label]
                    p_s[pred_semantic == label] = color
                    s_patches.append(mpatches.Patch(color=color, label=self.label_mapping[label][:10]))
                axis.imshow(p_s)
                axis.set_title("Predicted Semantic")
                axis.axis('off')
                axis.legend(handles=s_patches[:20])

                # axis = plt.subplot2grid((1, 3), loc=(0, 1))
                # p_sd = np.zeros((pred_semantic_denoised.shape[0], pred_semantic_denoised.shape[1], 3))
                # labels = np.unique(pred_semantic_denoised)
                # s_patches = []
                # for label in labels:
                #     color = semantic_label_color_mapping[label]
                #     p_sd[pred_semantic_denoised == label] = color
                #     s_patches.append(mpatches.Patch(color=color, label=self.label_mapping[label][:10]))
                # axis.imshow(p_sd)
                # axis.set_title("Predicted Denoised Semantic")
                # axis.axis('off')
                # axis.legend(handles=s_patches[:20])

                axis = plt.subplot2grid((1, 2), loc=(0, 1))
                gt_s = np.zeros((gt_semantic.shape[0], gt_semantic.shape[1], 3))
                labels = np.unique(gt_semantic)
                s_patches = []
                for label in labels:
                    color = semantic_label_color_mapping[label]
                    gt_s[gt_semantic == label] = color
                    s_patches.append(
                        mpatches.Patch(
                            color=color, 
                            label=self.label_mapping[label][:10] if label in self.label_mapping.keys() else "otherprop"
                        )
                    )
                axis.imshow(gt_s)
                axis.set_title("GT Semantic")
                axis.axis('off')
                axis.legend(handles=s_patches[:20])
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_figures, '{:06}_semantic.png'.format(index)))
                plt.close()
    
    def _evaluate_instance(self, gt_instances, gt_thing_ids, pred_instances, indices):
        if self.debug:
            pred_instance_label_color_mapping = {}
            gt_instance_label_color_mapping = {}

        print("Evaluating instance segmentation ...")
        gt_inst_ids, gt_inst_areas = np.unique(gt_instances, return_counts=True)
        for gt_inst_id, gt_inst_area in zip(gt_inst_ids, gt_inst_areas):
            if gt_inst_id not in gt_thing_ids:
                continue
            gt_inst_mask = gt_instances == gt_inst_id
            pred_inst_ids, pred_gt_intersections = np.unique(pred_instances[np.logical_and(gt_inst_mask, pred_instances>0)], return_counts=True)
            if len(pred_gt_intersections) == 0:
                self.panoptic_stat.instance_stat['coverage'].append(0)
            else:
                index = np.argmax(pred_gt_intersections)
                matched_pred_inst_id = pred_inst_ids[index]
                matched_pred_gt_intersection = pred_gt_intersections[index]
                matched_pred_inst_mask = pred_instances == matched_pred_inst_id
                iou = matched_pred_gt_intersection / (np.sum(matched_pred_inst_mask) + np.sum(gt_inst_mask) - matched_pred_gt_intersection)
                self.panoptic_stat.instance_stat['coverage'].append(iou)
            self.panoptic_stat.instance_stat['gt_inst_area'].append(gt_inst_area)
            
            if self.debug:
                    color = np.random.rand(3, )
                    pred_instance_label_color_mapping[matched_pred_inst_id] = color
                    gt_instance_label_color_mapping[gt_inst_id] = color
        
        gt_inst_mask = np.isin(gt_instances, gt_thing_ids)
        pred_inst_ids = np.unique(pred_instances[gt_inst_mask])
        self.panoptic_stat.instance_stat['num_pred_inst'] += len(pred_inst_ids)
        self.panoptic_stat.instance_stat['num_gt_inst'] += len(gt_thing_ids)

        if self.debug:
            for gt_instance, pred_instance, index in tqdm(
                list(zip(gt_instances, pred_instances, indices)), desc="[DEBUG] visualizing"):
                
                plt.figure(figsize=(20, 10))
                axis = plt.subplot2grid((1, 2), loc=(0, 0))
                p_ins = np.zeros((pred_instance.shape[0], pred_instance.shape[1], 3))
                labels = np.unique(pred_instance)
                for label in labels:
                    p_ins[pred_instance == label] = pred_instance_label_color_mapping.get(label, np.zeros((3, )))
                axis.imshow(p_ins)
                axis.set_title("Predicted Instance")
                axis.axis('off')

                axis = plt.subplot2grid((1, 2), loc=(0, 1))
                gt_ins = np.zeros((gt_instance.shape[0], gt_instance.shape[1], 3))
                labels = np.unique(gt_instance)
                for label in labels:
                    gt_ins[gt_instance == label] = gt_instance_label_color_mapping.get(label, np.zeros((3, )))
                axis.imshow(gt_ins)
                axis.set_title("GT Instance")
                axis.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(self.save_figures, '{:06}_instance.png'.format(index)))
                plt.close()
    
    def _evaluate_panoptic(self, pred_instances, pred_semantics, gt_semantics, gt_instances, indices):
        # self.debug = True
        print("Evaluating panoptic quality ...")
        miou, nn = 0, 0
        gt_segms = self._read_gt_panoptic_segmentation(gt_semantics, gt_instances)
        pred_segms = self._predict_panoptic_segmentation(pred_instances, pred_semantics)

        if self.debug: #True: #self.debug: #True: #self.debug:
            pred_panoptic_label_color_mapping = {}
            gt_panoptic_label_color_mapping = {}
        
        ### evaluate panoptic segmentation
        # confusion matrix calculation
        gt_pred_instance = gt_instances.astype(np.uint64) * OFFSET + pred_instances.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(gt_pred_instance, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue

            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            # print(iou)
            if iou > 0.3:
                miou += iou
                nn += 1
                self.panoptic_stat[gt_segms[gt_label]['category_id']].tp += 1
                self.panoptic_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

                if self.debug: #True: #self.debug: #True: #self.debug:
                    color = np.random.rand(3, )
                    pred_panoptic_label_color_mapping[pred_label] = color
                    gt_panoptic_label_color_mapping[gt_label] = color

        self.panoptic_stat.panoptic_miou = miou / nn if nn>0 else 0

        # count false negatives
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            self.panoptic_stat[gt_info['category_id']].fn += 1

            if self.debug: #True: #self.debug: #True: #
                color = np.random.rand(3, )
                gt_panoptic_label_color_mapping[gt_label] = color

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID region
            if intersection / pred_info['area'] > 0.5:
                continue
            self.panoptic_stat[pred_info['category_id']].fp += 1

            if self.debug: #True: #self.debug: #
                color = np.random.rand(3, )
                pred_panoptic_label_color_mapping[pred_label] = color
        
        if self.debug: #True: #
            for i, index in enumerate(tqdm(indices, desc="[DEBUG] visualizing")):
                plt.figure(figsize=(20, 10))
                # axis = plt.subplot2grid((1, 3), loc=(0, 0))
                # gt_image = gt_images[i]
                # rgb = (gt_image * 255).astype(np.uint8)
                # axis.imshow(rgb)
                # axis.set_title("GT Image")
                # axis.axis('off')
                
                axis = plt.subplot2grid((1, 2), loc=(0, 0))
                pred_instance = pred_instances[i]
                p_panop = np.zeros((pred_instance.shape[0], pred_instance.shape[1], 3))
                labels = np.unique(pred_instance)
                pred_panop_patches = []
                for label in labels:
                    if label == VOID:
                        continue
                    color = pred_panoptic_label_color_mapping.get(label, np.zeros((3, )))
                    p_panop[pred_instance == label] = color
                    pred_panop_patches.append(
                        mpatches.Patch(color=color, label=self.label_mapping[pred_segms[label]['category_id']][:10])
                    )
                axis.imshow(p_panop)
                axis.set_title("Predicted Panoptic")
                axis.axis('off')
                axis.legend(handles=pred_panop_patches[:30])

                axis = plt.subplot2grid((1, 2), loc=(0, 1))
                gt_instance = gt_instances[i]
                gt_panop = np.zeros((gt_instance.shape[0], gt_instance.shape[1], 3))
                labels = np.unique(gt_instance)
                gt_panop_patches = []
                for label in labels:
                    if label == VOID:
                        continue
                    color = gt_panoptic_label_color_mapping.get(label, np.zeros((3, )))
                    gt_panop[gt_instance == label] = color
                    gt_panop_patches.append(
                        mpatches.Patch(color=color, label=self.label_mapping[gt_segms[label]['category_id']][:10])
                    )
                axis.imshow(gt_panop)
                axis.set_title("GT Panoptic")
                axis.axis('off')
                axis.legend(handles=gt_panop_patches[:30])

                plt.tight_layout()
                plt.savefig(os.path.join(self.save_figures, '{:06}_panoptic.png'.format(index)))
                plt.close()

def print_panoptic_results(panoptic_stat, categories, label_mapping, label_thing_mapping, verbose=False):

    json_result = {}
    print_tables = []

    def percentage_to_string(num):
        if num is None:
            return "N/A"
        else:
            v = num * 100
            return f"{v:.1f}"

    console = Console()
    # panoptic segmentation
    pq_total_result, pq_per_class_result = panoptic_stat.pq_average(categories, label_thing_mapping, verbose=verbose)
    table = Table(show_lines=True, caption_justify='left')
    table.add_column('Class')
    table.add_column('PQ')
    table.add_column('SQ')
    table.add_column('RQ')
    table.add_column('mIoU')
    if verbose:
        table.add_column('tp')
        table.add_column('fp')
        table.add_column('fn')

    table.title = "Panoptic Evaluation"
    json_result['panoptic'] = {}
    per_class_result = {}
    for category_id in categories:
        pq_info = pq_per_class_result[category_id]
        if pq_info['valid']:
            if verbose:
                table.add_row(label_mapping[category_id], 
                        percentage_to_string(pq_info['pq']),
                        percentage_to_string(pq_info['sq']),
                        percentage_to_string(pq_info['rq']),
                        str(pq_info['tp']),
                        str(pq_info['fp']),
                        str(pq_info['fn']))
                per_class_result[label_mapping[category_id]] = {
                    'PQ': pq_info['pq'] * 100, 'SQ': pq_info['sq'] * 100, 'RQ': pq_info['rq'] * 100,
                    'tp': pq_info['tp'], 'fp': pq_info['fp'], 'fn': pq_info['fn']
                }
            
            else:
                table.add_row(label_mapping[category_id], 
                        percentage_to_string(pq_info['pq']),
                        percentage_to_string(pq_info['sq']),
                        percentage_to_string(pq_info['rq']))
                per_class_result[label_mapping[category_id]] = {
                    'PQ': pq_info['pq'] * 100, 'SQ': pq_info['sq'] * 100, 'RQ': pq_info['rq'] * 100
                }
    json_result['panoptic']['per_class_result'] = per_class_result
    if verbose:
        table.add_row('Total:\n{} valid panoptic categories.'.format(
                        pq_total_result['n']),
                  percentage_to_string(pq_total_result['pq']), 
                  percentage_to_string(pq_total_result['sq']), 
                  percentage_to_string(pq_total_result['rq']),
                  '{:.1f}'.format(pq_total_result['tp']),
                  '{:.1f}'.format(pq_total_result['fp']),
                  '{:.1f}'.format(pq_total_result['fn']))
        json_result['panoptic']['total'] = {
            'PQ': pq_total_result['pq'] * 100, 'SQ': pq_total_result['sq'] * 100, 'RQ': pq_total_result['rq'] * 100,
            'tp': pq_total_result['tp'], 'fp': pq_total_result['fp'], 'fn': pq_total_result['fn']
        }
    else:
        table.add_row('Total:\n{} valid panoptic categories.'.format(
                        pq_total_result['n']),
                  percentage_to_string(pq_total_result['pq']), 
                  percentage_to_string(pq_total_result['sq']), 
                  percentage_to_string(pq_total_result['rq']),
                  percentage_to_string(pq_total_result['miou']))
        json_result['panoptic']['total'] = {
            'PQ': pq_total_result['pq'] * 100, 'SQ': pq_total_result['sq'] * 100, 'RQ': pq_total_result['rq'] * 100, 'mIoU': pq_total_result['miou'] * 100
        }
    console.print(table)
    print_tables.append(table)

    # semantic segmentation
    semantic_total_result, semantic_per_class_result = panoptic_stat.semantic_average(categories)
    table = Table(show_lines=True, caption_justify='left')
    table.add_column('Class')
    table.add_column('S_iou')
    table.add_column('S_acc')
    table.add_column('S_iou_d')
    table.add_column('S_acc_d')

    table.title = "Semantic Evaluation"

    json_result['semantic'] = {}
    per_class_result = {}
    for category_id in categories:
        semantic = semantic_per_class_result[category_id]
        if semantic['valid']:
            table.add_row(label_mapping[category_id],
                    percentage_to_string(semantic['iou']),
                    percentage_to_string(semantic['acc']),
                    percentage_to_string(semantic['iou_d']),
                    percentage_to_string(semantic['acc_d']))
            per_class_result[label_mapping[category_id]] = {
                'S_iou': semantic['iou'] * 100, 'S_acc': semantic['acc'] * 100, 'S_iou_d': semantic['iou_d'] * 100, 'S_acc_d': semantic['acc_d'] * 100
            }
    json_result['semantic']['per_class_result'] = per_class_result

    table.add_row('Total:\n{} valid semantic categories'.format(
                    semantic_total_result['n']),
                percentage_to_string(semantic_total_result['iou']),
                percentage_to_string(semantic_total_result['acc']),
                percentage_to_string(semantic_total_result['iou_d']),
                percentage_to_string(semantic_total_result['acc_d']))
    json_result['semantic']['total'] = {
        'S_iou': semantic_total_result['iou'] * 100, 'S_acc': semantic_total_result['acc'] * 100, 
        'S_iou_d': semantic_total_result['iou_d'] * 100, 'S_acc_d': semantic_total_result['acc_d'] * 100
    }
    console.print(table)
    print_tables.append(table)

    # instance segmentation
    instance_result = panoptic_stat.instance_average(iou_threshold=0.1)
    table = Table(show_lines=True, caption_justify='left')
    table.add_column('mCov')
    table.add_column('mWCov')
    table.add_column('mPrec')
    table.add_column('mRec')

    table.title = "Instance Evaluation"
    table.add_row(
        percentage_to_string(instance_result['mCov']),
        percentage_to_string(instance_result['mWCov']),
        percentage_to_string(instance_result['mPrec']),
        percentage_to_string(instance_result['mRec']))
    json_result['instance'] = {
        'mCov': instance_result['mCov'] * 100, 'mWCov': instance_result['mWCov'] * 100,
        'mPrec': instance_result['mPrec'] * 100, 'mRec': instance_result['mRec'] * 100
    }
    console.print(table)
    print_tables.append(table)
    return print_tables, json_result


def print_iou_acc_results(ious, accs, table_title="Direct"):
    table = Table()
    table.add_column('Class')
    table.add_column('mIoU')
    table.add_column('mAcc')
    table.title = table_title

    def percentage_to_string(iou):
        if iou is None:
            return "N/A"
        else:
            v = iou * 100
            return f"{v:.1f}"

    reduced_iou = {}
    for iou in ious:
        for key, value in iou.items():
            if key not in reduced_iou:
                reduced_iou[key] = []
            if value is None:
                continue
            reduced_iou[key].append(value)
    reduced_acc = {}
    for acc in accs:
        for key, value in acc.items():
            if key not in reduced_acc:
                reduced_acc[key] = []
            if value is None:
                continue
            reduced_acc[key].append(value)
    for key, values in reduced_iou.items():
        if key == 'total':
            continue
        mIoU = np.mean(values)
        mAcc = np.mean(reduced_acc[key])
        table.add_row(key, percentage_to_string(mIoU),
                      percentage_to_string(mAcc))

    scene_total = percentage_to_string(
        np.mean([r['total'] for r in ious if 'total' in r]))
    scene_total_acc = percentage_to_string(
        np.mean([r['total'] for r in accs if 'total' in r]))
    table.add_row('Total', scene_total, scene_total_acc)

    console = Console()
    console.print(table)
    return table

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write_results(out, tables, json_result, panoptic_stat=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dumped = json.dumps(json_result, cls=NumpyEncoder, indent=2)
    with open(out / 'results.json', 'w') as f:
        f.write(dumped)

    with open(out / 'table.txt', 'w') as f:
        for table in tables:
            rprint(table, file=f)
            rprint('\n\n', file=f)
    
    # if panoptic_stat is not None:
    #     with open(out / 'panoptic_stat.pkl', 'wb') as outp:
    #         pickle.dump(panoptic_stat, outp, pickle.HIGHEST_PROTOCOL)



if __name__=='__main__':
    ## panoptic 
    evaluator = OpenVocabInstancePQEvaluator(
                name=scene_name,
                debug=debug_flag,
                stride=stride,
                save_figures=vis_path,
                time=False
    )

    evaluator.reset(label_map, vis_path)
    panoptic_stat = evaluator.eval()
    # panoptic_stats += panoptic_stat
    print(f"semantic评估类别: {evaluator.evaluated_labels}")
    tables, json_result = print_panoptic_results(panoptic_stat, 
                                categories=evaluator.evaluated_labels,
                                label_mapping=evaluator.label_mapping,
                                label_thing_mapping=evaluator.label_thing_mapping,
                                verbose=False)
    write_results(os.path.join(result_folder, "evaluation"), tables, json_result, panoptic_stat)
