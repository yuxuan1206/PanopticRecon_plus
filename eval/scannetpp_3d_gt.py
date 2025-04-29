import numpy as np
# from ..dataset import DatasetTemplate
import glob
import os
import glob, plyfile, numpy as np, multiprocessing as mp, torch, json, argparse
import open3d as o3d
import pandas

root_path = '/sdf1/yx/dataset/scannet++/data'

def get_raw2scannetv2_label_map(g_label_names):
    lines = [line.rstrip() for line in open("/mnt/meadow/yx/dataset/scannetpp_full/metadata/semantic_benchmark/map_benchmark.csv")]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(g_label_names)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        raw_label = elements[0]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = raw_label
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet




if __name__ == "__main__":

    scans = ['1ada7a0617', '5748ce6f01', 'f6659a3107'] 
    THING = [[2,3,5,6,7,10,11,12], [2,3,5,6,7], [2,3,5,6,9]]
    STUFF = [[1,4,8,9,13], [1,4,8,9], [1,4,7,8,10]]

    min_xyz = np.ones(3) * 999.
    max_xyz = np.ones(3) * -999.
    ins_cnt_all = np.zeros(20)
    for idx, scan in enumerate(scans):

        label_map_path = os.path.join("/sdb2/xyx/scannetpp", f"{scan}_3d.csv")
        label_map = pandas.read_csv(label_map_path) 
        file_object = open("/sdf1/yx/dataset/scannet++/metadata/semantic_classes.txt") 
        try:
            gt_csv = file_object.read()
        finally:
            file_object.close()
        gt_csv = gt_csv.split("\n")

        remapper = -1*np.ones(1500, dtype=int)
        for index, (i, new_label) in enumerate(zip(label_map['idx'], label_map['label'])):
            remapper[i] = new_label


        print(idx, scan)
        pts_ply = "scans/mesh_aligned_0.05.ply"
        labels_ply = "scans/mesh_aligned_0.05_semantic.ply"
        segs_json = "scans/segments.json"
        aggr_json = "scans/segments_anno.json"

        f = plyfile.PlyData().read(os.path.join(root_path, scan, pts_ply))
        points = np.array([list(x) for x in f.elements[0]])
        coords = points[:, :3]
        min_co = np.min(coords, axis=0)
        max_co = np.max(coords, axis=0)
        min_xyz = np.minimum(min_xyz, min_co)
        max_xyz = np.maximum(max_xyz, max_co)
        
        f2 = plyfile.PlyData().read(os.path.join(root_path, scan, labels_ply))
        sem_labels = remapper[np.array(f2.elements[0]['label'])]
        sem_labels[sem_labels<0] = 0

        color_map = np.random.randint(0,255,size=(20000,3), dtype=np.uint8)
        color_map[0] = np.array([0,0,0])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords.reshape(-1,3))
        color = color_map[sem_labels.astype(np.int32)]
        pcd.colors = o3d.utility.Vector3dVector(color/255.)
        o3d.io.write_point_cloud( "/sdc1/yx/dataset/scannet++/gt_pc/"+scan+"_sem.ply", pcd)
        coords.astype(np.float32).tofile("/sdc1/yx/dataset/scannet++/gt_pc/"+scan+"_pc.npy")
        sem_labels.astype(np.int32).tofile("/sdc1/yx/dataset/scannet++/gt_pc/"+scan+"_sem.npy")

        with open(os.path.join(root_path, scan, segs_json)) as jsondata:
            d = json.load(jsondata)
            seg = d['segIndices']
        segid_to_pointid = {}
        for i in range(len(seg)):
            if seg[i] not in segid_to_pointid:
                segid_to_pointid[seg[i]] = []
            segid_to_pointid[seg[i]].append(i)

        instance_segids = []
        labels = []
        raw_mesh_id = []
        with open(os.path.join(root_path, scan, aggr_json)) as jsondata:
            d = json.load(jsondata)
            for x in d['segGroups']:
                instance_segids.append(x['segments'])
                # labels.append(name2semId[g_raw2scannetv2[x['label']]])
                if x['label'] == 'REMOVE' or x['label'] not in gt_csv:
                    labels.append(0)
                    continue
                raw_id = np.array(gt_csv.index(x['label']))
                if x['label'] not in raw_mesh_id:
                    raw_mesh_id.append(x['label']+f"_{raw_id}")
                ours_id = remapper[raw_id]
                labels.append(ours_id)
        # if(labels_ply == 'scene0217_00_vh_clean_2.labels.ply' and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]):
        #     instance_segids = instance_segids[: int(len(instance_segids) / 2)]

        instance_labels = np.ones(sem_labels.shape[0]) * -1
        sem_ins_cnt = np.zeros(ins_cnt_all.shape[0])
        for i in range(len(instance_segids)):
            if labels[i] == -1:
                continue
            segids = instance_segids[i]
            globalid = labels[i] * 100 + sem_ins_cnt[labels[i]]
            for segid in segids:
                instance_labels[segid_to_pointid[segid]] = globalid
            sem_ins_cnt[labels[i]] += 1
        
        instance_labels[~np.isin(sem_labels, THING[idx])] = 0
        instance_labels[instance_labels<0] = 0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords.reshape(-1,3))
        color = color_map[instance_labels.astype(np.int32)]
        pcd.colors = o3d.utility.Vector3dVector(color/255.)
        o3d.io.write_point_cloud( "/sdc1/yx/dataset/scannet++/gt_pc/"+scan+"_ins.ply", pcd)
        instance_labels.astype(np.int32).tofile("/sdc1/yx/dataset/scannet++/gt_pc/"+scan+"_ins.npy")

        ins_cnt_all += sem_ins_cnt / sem_ins_cnt.sum()

        GT = np.hstack(( np.hstack((coords, sem_labels[:,None])), instance_labels[:,None]))
        np.save("/sdc1/yx/dataset/scannet++/gt_pc/"+scan+"_GT.npy", GT)


    print(min_xyz, max_xyz)
    print(ins_cnt_all)