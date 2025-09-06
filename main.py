import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TCNN_CUDA_ARCHITECTURES']="1"
os.environ['QT_QPA_PLATFORM']="offscreen"

import shutil
# import warnings 
from xml.etree.ElementTree import PI
from cv2 import log
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import sys
import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from render.render_helper import *
from torch.utils.tensorboard import SummaryWriter
import time
from utils.scannet_data_tools.init_scannet_pose import init_all_poses

import open3d as o3d
from data.label import get_labels #id2label
id2label = get_labels("0420_01")

from models.train import Train, clean_cache
from utils.mkdir import mkdir_result_folder


if __name__ == '__main__':
        # warnings.filterwarnings("error")
        torch.manual_seed(0)  
        np.random.seed(250)
        torch.set_printoptions(precision=20)
        np.set_printoptions(precision=20)
        # writer = SummaryWriter()
        # get current time and create folder
        current_time = time.localtime()
        date = time.strftime('%Y-%m-%d', current_time)
        hour_minute = time.strftime('%H:%M', current_time)
        log_dir = os.path.join(os.getcwd(), 'logs', f"{date} {hour_minute}")
        folder_name = os.path.join(log_dir, 'events')
        if os.path.exists(folder_name):
                shutil.rmtree(folder_name)
        os.makedirs(folder_name)
        writer = SummaryWriter(folder_name)
        # writer = None
        tensorboard_flag = False if writer == None else True

        # yaml_path = "config/scannet0087_02/render_scannet_hash_++_all.yaml"
        # yaml_path = "config/scannet0420_01/render_scannet_hash_++_all.yaml"
        yaml_path = "config/scannet0420_01/render_scannet_hash_++_3dgs_test.yaml"
        # yaml_path = "config/scannetpp_1ada7a0617/render_scannet_hash_++_all.yaml"
        
        config_file = yaml_path.split("/")[-1]

        with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
        with open(os.path.join(log_dir, config_file), "w") as f:
            yaml.dump(cfg, f, allow_unicode=True, indent=4, default_flow_style=None, sort_keys=False)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:0")

        # Get grid
        grid_model_name = cfg['model']['model_name']
        print(f"Grid Model = {grid_model_name}")
        assert( grid_model_name in GRIDMODEL )

        # train
        batch_size = int(cfg['params']['ray_chunk'])
        batch_size_2 = int(cfg['params']['ray_chunk_mode2'])
        batch_size_img = int(cfg['params']['ray_chunk_img'])
        # batch_size_3 = int(cfg['params']['ray_chunk_doulbe_cubes'])
        loss_occ = 0
        frame_step = cfg['frame']['step']

        # read pose_noise
        all_pose_list = range(cfg['train']['all_pose'][0], cfg['train']['all_pose'][-1]) #cfg['train']['all_pose']
        # read_bag()
        with torch.no_grad():
                if cfg['path']['dataset_type'] == 'scannet':
                        vec_es, vec_cam, all_pose_list = init_all_poses(cfg['path']['dataset_dir'], all_pose_list, device) 
                        vec_es_init = vec_es
                        vec_gt = vec_es
                        frame_list = list(range(cfg['train']['all_pose'][0],cfg['train']['all_pose'][-1],frame_step))  
                else:
                        sys.exit("Wrong dataset type. Please use maicity, newer_college or kitti360")
                # vec_es_odom = qury_pose.PoseInit(vec_es, frame_list)

        logs = []
        multi_cube = {"volumes":[], "span_idx":[-1]}
        first_idx = frame_list[0]
        mode = 2 #1
        cubes_num = cfg['start_block_idx'] #0
        frame_num = cfg['frame']['num']
        frame_half = int(frame_num/2)
        frame_start = cubes_num*frame_half #0
        frame_end = frame_start + frame_num #16
        proj_dir_root = cfg['path']['proj_dir']
        #-------------STEP-------------
        for si in range(1):
                cfg['path']['proj_dir'] = proj_dir_root + f"/v{cubes_num}" #volume folder
                mkdir_result_folder(cfg['path']['proj_dir'])
                train_info = cfg['train'][f'mode{mode}'] #[f'step{si+1}']

                step = 2 if mode==1 else 1
                # frame_start = frame_start-4 if (frame_start >= 4 and mode==1) else frame_start # ADD
                train_list = frame_list[frame_start:frame_end:step]
                if cfg['path']['dataset_type'] == 'scannet':
                        train_list = [i for i in train_list if i in all_pose_list]
                # train_list = frame_list
                train_list_last = frame_list[frame_start-frame_half:frame_end-frame_half:1] if train_list[0]>first_idx else None
                # train_list_last = frame_list[frame_start-6:frame_end-6:1] if train_list[0]>0 else None
                if mode == 1:
                        p_list = [train_list[0]] #train_info['pointcloud_list'] 
                
                pose_t = vec_es[f"{train_list[0]}"][:3]
                pose_t_last = vec_es[f"{train_list_last[0]}"][:3] if train_list[0]>first_idx else 0
                # delta_pose_t = get_new_grid_frame(pose_t-pose_t_last, multi_cube["volumes"][-1].trans_world_to_scale, cfg['params']['grid_resolution']) if train_list[0]>first_idx else 0 #'grid' in vars()

                print(f"====================No. {cubes_num} CUBE=====================")

                Pipline = Train(mode, device, cfg, train_list, vec_es, vec_cam, cubes_num)
                # Pipline.nef.set_optimer(0.02, 0, 0, 0, 0.0006)
               
                # iter num
                its = 0
                its_2 = 0
                print(f"Train Step {si+1} : mode = {mode}")

                key_num = len(train_list) if mode==1 else 1
                # key_num = 1
                key_start = train_info['start_frame'] if (mode==1 and train_list[0]>first_idx) else 0
                # key_start = train_info['start_frame']+4 if (mode==1 and train_list[0]>0) else 0 # ADD

                #-------------optimer-------------
                for k in range(key_start, key_num):
                        # clean_cache()

                        # key frame
                        train_list_key_frame = train_list[:k+1] if mode==1 else train_list
                        # grid.change_key_frame(cfg['path']['proj_dir'], train_list_key_frame, vec_loam, k+1, traj_gt, traj_loam)
                        
                        #read data 
                        if cfg['path']['dataset_type'] == 'scannet':
                                from utils.scannet_data_tools.scannet_dataset import ScannetDataset
                                dataset_train = ScannetDataset(cfg, train_list_key_frame, device, vec_es, "train", optim_flag=Pipline.optim_flag, normal_flag=Pipline.normal_flag,semantic_flag=Pipline.semantic_flag)
                                if cfg['loss_term']['feature']:
                                        vit_list = train_list_key_frame
                                        dataset_vit = list(ScannetDataset(cfg, vit_list, device, vec_es, "val", optim_flag=Pipline.optim_flag))
                        else:
                                sys.exit("Wrong dataset type. Please use maicity, newer_college or kitti360")

                        ## panoptic
                        if cfg['loss_term']['instance'] and not cfg['pretrain']:
                                Pipline.nef.init_instance_model(dataset_train)
                        
                        Pipline.set_frame_optimer(cfg['path']['proj_dir'], train_list_key_frame, vec_gt, vec_es, k+1)
                        OCCExpLR = torch.optim.lr_scheduler.ExponentialLR(Pipline.nef.optim_occ, gamma=train_info['occ_lr_decay'])
                        # if cfg['loss_term']['instance']:
                        #         INSExpLR = torch.optim.lr_scheduler.ExponentialLR(Pipline.nef.instance_optimizer, gamma=0.8)
                        if train_info['pose_optim']:
                                pose_lr_optim = torch.optim.lr_scheduler.ExponentialLR(Pipline.nef.pose_optimizer, gamma=train_info['occ_lr_decay'])

                        ## init sampler
                        train_sampler = SimpleSampler(train_list_key_frame, dataset_train, cfg['path']['dataset_type'], total=dataset_train.get_every_img_batch(), batch=[batch_size_2, batch_size_img])

                        
                        # depthloss_epoch = 1e5
                        poseloss = 0
                        transloss = 0                        
                        #-------------batch-------------
                        t1 = time.time()
                        e_start = 0 #1 #
                        Pipline.nef.epoch = e_start
                        Pipline.rgb_loss_flag = True
                        rgb_epoch = 1
                        all_epoch = 5
                        for e in range(e_start, train_info['epoch']):

                                if e == all_epoch:
                                        train_sampler = SimpleSampler(train_list_key_frame, dataset_train, cfg['path']['dataset_type'], total=dataset_train.get_every_img_batch(), batch=[batch_size_2, batch_size_img*2])
                                Pipline.freeze_setting(e, rgb_epoch=rgb_epoch, all_epoch=all_epoch) 
                             
                                print(f"Start Epoch {e} : ")
                                depthloss_epoch = 0
                                data_rand_frame_geo, data_rand_frame_rgb = torch.randperm(len(train_list)), torch.randperm(len(train_list))
                                curr_geo, curr_rgb = 0, 0
                                per_iter_geo = 100 if e>=rgb_epoch else 50
                                per_iter_rgb = 10 #5
                                
                                train_sampler.epoch_iter = 300
                                with tqdm(total=train_sampler.epoch_iter) as t:
                                        t.set_description('Optim: ')
                                        Pipline.init_loss()
                                        for i in range(train_sampler.epoch_iter):
                                                cos_anneal_ratio = min([1.0, its / (train_sampler.epoch_iter) ]) #cfg['train']['mode2']['epoch']*
                                                train_data_vit = [dataset_vit[torch.randint(0, len(dataset_vit), size=[1])]] if cfg['loss_term']['feature'] else None
                                                
                                                random_list_rgb = data_rand_frame_rgb[curr_rgb:curr_rgb+per_iter_rgb]
                                                random_list_geo = data_rand_frame_geo[curr_geo:curr_geo+per_iter_geo]
                                                # random_list_rgb = torch.IntTensor([61,62,63,64,65])
                                                # random_list_geo = torch.IntTensor([23])
                                                
                                                ## only geometry epoch
                                                if e<rgb_epoch and not cfg['pretrain']:
                                                        random_list_rgb = torch.tensor([])

                                                dataset_train.set_random_list(random_list=[random_list_geo, random_list_rgb])
                                                idx = train_sampler.nextids(dataset_train, random_list=[random_list_geo, random_list_rgb], need_img=((cfg['model_type']!='3dgs') & Pipline.rgb_flag) | cfg["loss_term"]["semantic2d"]) # Pipline.feature_flag|
                                                
                                                Pipline.nef.uv = idx[1]

                                                Pipline.train(dataset_train[idx], train_list_key_frame, e, t, train_data_vit=train_data_vit, cos_anneal_ratio=cos_anneal_ratio, random_list=[random_list_geo, random_list_rgb])
                                                # p.step()
                                                t.update(1)
                                                its += 1

                                                if curr_geo+2*per_iter_geo < len(data_rand_frame_geo):
                                                        curr_geo += per_iter_geo  
                                                else:
                                                        curr_geo = 0
                                                        data_rand_frame_geo = torch.randperm(len(train_list))
                                                if curr_rgb+2*per_iter_rgb < len(data_rand_frame_rgb):
                                                        curr_rgb += per_iter_rgb
                                                else:
                                                        curr_rgb = 0
                                                        data_rand_frame_rgb = torch.randperm(len(train_list))

                                Pipline.mean_loss(e,show=tensorboard_flag)
                                if tensorboard_flag:
                                        Pipline.tensorboard_show(writer, its) 
                                OCCExpLR.step()
                                # if cfg['loss_term']['instance']:
                                #         INSExpLR.step()
                                if Pipline.nef.epoch>=train_info['pose_epoch'] and train_info['pose_optim']:
                                        pose_lr_optim.step()
                                Pipline.nef.epoch += 1
                                # clean_cache()


                                with torch.no_grad():
                                        # test_list = list(range(train_list[0], train_list[0]+1,1)) #+20, 10
                                        test_list = [30,197,350] #[1230,1430,1530] #
                                        if cfg['path']['dataset_type'] == 'kitti360':
                                                dataset_test = KITTIDataset(cfg, test_list, device, vec_gt, vec_es, Pipline.nef.vec_cam, "val", optim_flag=Pipline.optim_flag)
                                        elif cfg['path']['dataset_type'] == 'scannet':
                                                dataset_test = ScannetDataset(cfg, test_list, device, vec_es, "val", optim_flag=Pipline.optim_flag, normal_flag=Pipline.normal_flag,semantic_flag=Pipline.semantic_flag) 
                                        dataloader_test = DataLoader(dataset = dataset_test, batch_size=1)
                                        _ = render_depth(dataloader_test, Pipline.nef, cfg, Pipline.get_sdf_reander) 

                                grid_save_path = os.path.join(cfg['path']['proj_dir'], f'grid/optimed_grid_{cubes_num}_{e}.pth')
                                Pipline.nef.save_grid(grid_save_path, e)
                                if e > 2 and e!= 21:
                                        try:
                                                os.remove(os.path.join(cfg['path']['proj_dir'], f'grid/optimed_grid_{cubes_num}_{e-2}.pth')) 
                                        except:
                                                print("no "+os.path.join(cfg['path']['proj_dir'], f'grid/optimed_grid_{cubes_num}_{e-2}.pth'))
                               
                                # Pipline.update_lamuda()

                        # vec_es_odom.update_pose(vec_es)
                        t2 = time.time()
                        print(f"training time = {(t2-t1)} s") #/(e_max*train_sampler.epoch_iter)
                        with torch.no_grad():
                                if cfg['vis']:
                                        extract_mesh_kaolin(Pipline.nef,cubes_num,cfg['path']['proj_dir'],29,-1,1,-1,1,-0.1,0.2)
                                        # if Pipline.semantic_flag:
                                        mesh= o3d.io.read_triangle_mesh(cfg['path']['proj_dir']+f'/mesh/mesh_test{cubes_num}.stl')
                                        vertices = np.asarray(mesh.vertices)
                                        # vertices = (mesh.vertices-Pipline.nef.origin.cpu().numpy())/Pipline.nef.scale.cpu().numpy()
                                        sem_batch = 65536
                                        rgb_img = torch.zeros(vertices.shape[0])
                                        for i in tqdm(range(0, vertices.shape[0], sem_batch), desc='get semantic labels'):
                                                next_ind = min(i+sem_batch, vertices.shape[0])
                                                out = Pipline.nef.get_output(torch.from_numpy(vertices[i:next_ind]).float().cuda(), channels=['sdf','rgb'])
                                                rgb = out['rgb']
                                                # sem = torch.argmax(torch.nn.functional.softmax(sem.squeeze(), dim=-1), -1)
                                                rgb_img[i:next_ind] = rgb.cpu()
                                        
                                        v_colors = np.vstack([id2label[semID].color for semID in semantic.tolist()]) #todo
                                        mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors/255.0)
                                        save_file = cfg['path']['proj_dir']+f'/mesh/mesh_semantic{cubes_num}.ply'
                                        o3d.io.write_triangle_mesh(save_file, mesh)

                                        if Pipline.semantic_flag:
                                                mesh= o3d.io.read_triangle_mesh(cfg['path']['proj_dir']+f'/mesh/mesh_test{cubes_num}.stl')
                                                vertices = np.asarray(mesh.vertices)
                                                # vertices = (mesh.vertices-Pipline.nef.origin.cpu().numpy())/Pipline.nef.scale.cpu().numpy()
                                                sem_batch = 65536
                                                semantic = torch.zeros(vertices.shape[0])
                                                for i in tqdm(range(0, vertices.shape[0], sem_batch), desc='get semantic labels'):
                                                        next_ind = min(i+sem_batch, vertices.shape[0])
                                                        _, sem = Pipline.nef.get_output(torch.from_numpy(vertices[i:next_ind]).float().cuda(), semantic_flag=True)
                                                        sem = torch.argmax(torch.nn.functional.softmax(sem.squeeze(), dim=-1), -1)
                                                        semantic[i:next_ind] = sem.cpu()
                                                
                                                v_colors = np.vstack([id2label[semID].color for semID in semantic.tolist()])
                                                mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors/255.0)
                                                save_file = cfg['path']['proj_dir']+f'/mesh/mesh_semantic{cubes_num}.ply'
                                                o3d.io.write_triangle_mesh(save_file, mesh)
                                        # extract_mesh_kaolin(Pipline.nef,cubes_num,cfg['path']['proj_dir'],30,-0.53,0.43,-0.069,0.068,-0.04,0.009)


                # # NEXT
                if mode==2: #and train_list[0]==first_idx:
                        frame_start_last = frame_start
                        frame_start = frame_start + frame_half
                        frame_end = frame_start + frame_num


                # # writer.close()
                cubes_num = cubes_num + 1
