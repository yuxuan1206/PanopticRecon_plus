import time
import torch
import numpy as np
import os
import imageio
from tqdm import tqdm
import cv2
import open3d as o3d

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
mse2psnr_np = lambda x : -10. * np.log(x) / np.log(10)
GRIDMODEL = ["SH", "RGB", "SDF(sparse)"]            
        
class SimpleSampler:
    # def __init__(self, data_list, dataset_train, dataset, total, batch): #(self, data_list, w, h, dataset, total, batch):
    #     w, h = dataset_train.w, dataset_train.h
    #     # every img
    #     self.patch_size = 1 #8
    #     self.w = w // self.patch_size
    #     #    #     self.h = 250 // self.patch_size if dataset=='kitti360' else h // self.patch_size
    #     self.h_raw = h // self.patch_size
    #     self.total = total 
    #     self.curr = total.int()
    #     self.curr2 = self.w * self.h_raw
        

    #     self.ids = []
    #     self.ids2 = []
    #     # self.epoch_iter = (total.max()//batch).int().item()
    #     self.epoch_iter = torch.div(total[-1], batch[0], rounding_mode='trunc').int().item()
    #     self.batch = torch.div(total, self.epoch_iter, rounding_mode='trunc').int()
    #     # epoch_iter2 = ((total//batch2).max()).int().item()
    #     self.img_uvs = self.w*self.h
    #     # self.batch2 = self.img_uvs//self.epoch_iter
    #     self.batch2 = batch[1]
        
    # def nextids(self, dataset_train, random_list=None, need_img=False): #(self, img_num, need_img)
    #     img_num = len(dataset_train.imgs)
    #     self.lidar_ids = []
    #     self.pixel_ids = []
    #     self.curr2 += self.batch2
    #     img_random_flag = 0
    #     if self.curr2 + self.batch2 > self.img_uvs:
    #         img_random_flag = 1
    #         self.curr2 = 0 
    #     for i in range(img_num):
    #         if i != img_num-1:
    #             # batch = [512,512] # past frames batch
    #             batch = [self.batch[i], self.batch2]
    #         else:
    #             batch = [self.batch[i], self.batch2]
    #         self.curr[i] += batch[0]
    #         if self.curr[i] + batch[0] > self.total[i]:
    #             if len(self.ids) < i+1:
    #                 self.ids.append(torch.LongTensor(np.random.permutation(self.total[i].int().item())))
    #             else:
    #                 self.ids[i] = torch.LongTensor(np.random.permutation(self.total[i].int().item()))
    #             self.curr[i] = 0        
            
    #         self.lidar_ids.append(self.ids[i][self.curr[i]:self.curr[i]+batch[0]])

    #         # imgs
    #         if need_img:
    #             if img_random_flag:
    #                 if len(self.ids2) < i+1:
    #                     self.ids2.append(torch.LongTensor(np.random.permutation(self.img_uvs))+(self.h_raw-self.h)*self.w)
    #                 else:
    #                     self.ids2[i] = torch.LongTensor(np.random.permutation(self.img_uvs)+(self.h_raw-self.h)*self.w)
                
    #             self.pixel_ids.append(self.ids2[i][self.curr2:self.curr2+self.batch2])

    #     return self.lidar_ids , self.pixel_ids

    def __init__(self, data_list, dataset_train, dataset, total, batch):
        # every img
        w, h = dataset_train.w, dataset_train.h
        self.patch_size = 1 #8
        self.w = w // self.patch_size
        #        self.h = 250 // self.patch_size if dataset=='kitti360' else h // self.patch_size
        self.h_raw = h // self.patch_size
        self.total = total 
        self.curr = total.int()
        # self.curr2 = self.w * self.h_raw

        self.semantic_flag = dataset_train.semantic_flag
        self.instance_flag = dataset_train.instance_flag
        if dataset_train.semantic_list is not None:
            self.non_zero_semantic_img = [torch.isin(image.reshape(-1), torch.IntTensor(dataset_train.Stuff_class)) for image in dataset_train.semantic_list] #[torch.nonzero(image.reshape(-1)) for image in dataset_train.semantic_list]
            self.shuffled_sem_idx_list = [indices[torch.randperm(indices.size(0))].squeeze() for indices in self.non_zero_semantic_img]
        if dataset_train.instance_list is not None:
            self.non_zero_instance_img = [torch.nonzero(image.reshape(-1)) for image in dataset_train.instance_list]
            self.shuffled_ins_idx_list = [indices[torch.randperm(indices.size(0))].squeeze() for indices in self.non_zero_instance_img]
        self.curr2, self.curr2_sem, self.curr2_ins = [0] * len(dataset_train.imgs), [0] * len(dataset_train.imgs), [0] * len(dataset_train.imgs)

        self.ids = [0] * len(dataset_train.imgs)
        self.ids2 = [0] * len(dataset_train.imgs)
        # self.epoch_iter = (total.max()//batch).int().item()
        self.epoch_iter = torch.div(total[-1], batch[0], rounding_mode='trunc').int().item()
        self.batch = torch.div(total, self.epoch_iter, rounding_mode='trunc').int()
        # epoch_iter2 = ((total//batch2).max()).int().item()
        self.img_uvs = self.w*self.h
        # self.batch2 = self.img_uvs//self.epoch_iter
        self.batch2 = batch[1]
        self.sem_batch, self.ins_batch = int(self.batch2*0.5), int(self.batch2*0.2) #0.2
        self.sem_ids, self.ins_ids = [], []
        
    def nextids(self, dataset_train, random_list=None, need_img=False):
        img_num = len(dataset_train.imgs)
        self.lidar_ids, self.pixel_ids = [],[]
        # self.curr2 += self.batch2
        # img_random_flag = 0
        # if self.curr2 + self.batch2 > self.img_uvs:
        #     img_random_flag = 1
        #     self.curr2 = 0 
        data = random_list[0] if random_list[0] is not None else range(img_num)
        for i in data:

            batch = [self.batch[i], self.batch2]
  
            self.curr[i] += batch[0]
            if self.curr[i] + batch[0] > self.total[i]:
                self.ids[i] = torch.LongTensor(np.random.permutation(self.total[i].int().item()))
                self.curr[i] = 0        
            
            self.lidar_ids.append(self.ids[i][self.curr[i]:self.curr[i]+batch[0]])

        # imgs
        for i in random_list[1]:

            self.ids2[i] = torch.LongTensor(np.random.permutation(self.img_uvs))+(self.h_raw-self.h)*self.w
            
            if self.semantic_flag:
                if self.curr2_sem[i] + self.sem_batch > self.shuffled_sem_idx_list[i].shape[0]:
                    self.shuffled_sem_idx_list[i] = self.non_zero_semantic_img[i][torch.randperm(self.non_zero_semantic_img[i].size(0))]
                    self.curr2_sem[i] = 0
                pixel_id = self.shuffled_sem_idx_list[i][self.curr2_sem[i]:self.curr2_sem[i]+self.sem_batch] if len(self.shuffled_sem_idx_list[i])>0 else []
                self.curr2_sem[i] += self.sem_batch
            if self.instance_flag:
                if self.curr2_ins[i] + self.ins_batch > self.shuffled_ins_idx_list[i].shape[0]:
                    self.shuffled_ins_idx_list[i] = self.non_zero_instance_img[i][torch.randperm(self.non_zero_instance_img[i].size(0))]
                    self.curr2_ins[i] = 0
                pixel_id  = torch.cat((pixel_id, self.shuffled_ins_idx_list[i][self.curr2_ins[i]:self.curr2_ins[i]+self.ins_batch].squeeze())) if len(self.shuffled_ins_idx_list[i])>0 else pixel_id
                self.curr2_ins[i] += self.ins_batch

            if self.semantic_flag or self.instance_flag:
                batch_img = self.batch2-len(pixel_id)
                pixel_id = torch.cat((pixel_id, self.ids2[i][torch.randperm(self.img_uvs)[:batch_img]]))
            else:
                pixel_id = self.ids2[i][torch.randperm(self.img_uvs)[:self.batch2]]
            
            self.pixel_ids.append(pixel_id)

        return self.lidar_ids , self.pixel_ids

def get_mix_image(depth_torch, rgb_torch, h, w):
    depth_np = ( (depth_torch/depth_torch.max()).reshape(h,w,-1).cpu().numpy()*255).astype(np.uint8)
    rgb_np = (rgb_torch.reshape(h,w,-1).cpu().numpy()*255).astype(np.uint8)[:, :, ::-1]
    depth_color_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
    mix_image = cv2.addWeighted(depth_color_np, 0.5, rgb_np, 0.5, 0)
    return mix_image[:,:,::-1]

def get_mix_image_np(depth, rgb):
    rgb = rgb[:,:,::-1]
    depth_color_np = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    mix_image = cv2.addWeighted(depth_color_np, 0.5, rgb, 0.5, 0)
    return mix_image[:,:,::-1]

def render_depth(dataloader, grid, config, fn):
    ray_chunk_test = config['params']['ray_chunk_test']
    # savedir = config['path']['proj_dir']
    # print("************************Render Test Ray!************************")
    # render_points_frames = []
    color_id = np.random.random( [100000, 3] )*255.
    color_id[0,:] = 0
    with tqdm(total=len(dataloader)) as t:
        t.set_description('Render test rays ')
        for ii, data in enumerate(dataloader):
            # depth_gt_tensor = data["depth"].reshape(-1)

            # grid_idx = max(torch.where((data["idx"] - torch.Tensor(multi_cube["span_idx"]))>0)[0])
            grid_idx=0
            # grid = multi_cube["volumes"][grid_idx]
            render_pose_c2w = data['pose_cam'][0]
            render_points = []
            obs_points = []
            obs_points_frames = []
            normal_list, rgb_list, semantic_list, instance_list, panoptic_list = [], [], [], [], []
            depth_list = []
            images_test = []
            
            # #LIDAR
            # ray_directions_lidar = data['direction'].squeeze()
            # mask = torch.norm(ray_directions_lidar,dim=1)>0.1
            # ray_directions_lidar = ray_directions_lidar[mask]
            # # rays_all_num2 = ray_directions_lidar.size(0)
            # ob_points = data['points'].squeeze()
            # ob_points = ob_points[mask]
            # rays_depth_gt = data['depth'].squeeze()

            #cam
            ray_origin_cam = data['origin'].squeeze()
            ray_directions_cam = data['direction_img'].squeeze()
            rays_all_num2 = ray_directions_cam.size(0)

            # if grid.mode==2:
            #     grid.set_debug()
            for i in range(0, rays_all_num2, ray_chunk_test):
                i_next = i+ray_chunk_test if (i+ray_chunk_test <rays_all_num2) else rays_all_num2

                # points_grid, depth_chunk, points_obs, semantic = grid.get_lidar_depth_chunk(ray_directions_lidar[i:i_next,:],
                #                                         render_pose_c2w, rays_depth_gt[i:i_next], ob_points[i:i_next,:])
                # depth_chunk, points_grid, points_gt = fn(ray_directions_lidar[i:i_next,:],
                #                                         render_pose_c2w, rays_depth_gt[i:i_next])
                points_obs, normal_rgb, depth_chunk, rgb_chunk, semantic, instance, panoptic = fn(ray_origin_cam, ray_directions_cam[i:i_next,:], render_pose_c2w)
                # render_points.append(points_grid)
                normal_list.append(normal_rgb)
                rgb_list.append(rgb_chunk)
                obs_points.append(points_obs)
                semantic_list.append(semantic[:,None].cpu().numpy())
                instance_list.append(instance[:,None].cpu().numpy())
                panoptic_list.append(panoptic[:,None].cpu().numpy())
                depth_list.append(depth_chunk[:,None].cpu().numpy())

            # render_points_frames.append(torch.cat(render_points))
            obs_points_frames.append(torch.cat(obs_points))

            normal_img = np.vstack(normal_list)
            draw_rgb_normal(normal_img, data['idx'], config, 'normal_st')
            rgb_img = np.vstack(rgb_list)
            draw_rgb_normal(rgb_img, data['idx'], config, 'rgb_st')
            draw_semantic_rgb(np.vstack(semantic_list), data['idx'], config)
            draw_instance_rgb(np.vstack(instance_list), color_id, data['idx'], config)
            draw_instance_rgb(np.vstack(panoptic_list), color_id, data['idx'], config, folder="panoptic_st")
            draw_depth_render(np.vstack(depth_list), data['idx'], config)
            t.update(1)

            # images_test.append(normal_img.astype(np.uint8).reshape(-1, 1408, 3))            
        
        # rgb_vid_path = os.path.join(savedir, f"normal_render_img/normal.mov")
        # imageio.mimwrite(rgb_vid_path, images_test, fps=25, quality=8)
        # print(f"Test RGB video saved in : {rgb_vid_path}")

        # points = torch.cat(obs_points_frames).cpu().numpy()
        # points_o3d = o3d.t.geometry.PointCloud()
        # points_o3d.point["positions"] = o3d.core.Tensor(points, o3d.core.float32)
        # # points_o3d.points = o3d.utility.Vector3dVector(np.array(np.where(occupancy_grid>0)).T)
        # o3d.t.io.write_point_cloud("render_test.ply", points_o3d, write_ascii=True, compressed=False)

        # ply = o3d.io.read_point_cloud("samples/voxel_vis_source.ply")
        # points_gt = np.array(ply.points)
        # accuracy, accuracy_normals = metrics_evl(points, points_gt)
    # return  torch.cat(obs_points_frames).detach().cpu().numpy(), torch.cat(render_points_frames).cpu().numpy() #, semantic
    return None


# from tools.mesh_metrics import distance_p2p
# def metrics_evl(pointcloud_pred, pointcloud_tgt):
#     #     pointscloud = o3d.geometry.PointCloud()
#     pointscloud.points = o3d.utility.Vector3dVector(pointcloud_pred)
#     pointscloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))  #     normals_pred = np.array(pointscloud.normals) #.to(self.device)

#     pointscloud = o3d.geometry.PointCloud()
#     pointscloud.points = o3d.utility.Vector3dVector(pointcloud_pred)
#     pointscloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))  #     normals_tgt = np.array(pointscloud.normals)

#     # Accuracy: how far are th points of the predicted pointcloud
#     # from the target pointcloud
#     # accuracy, accuracy_normals = distance_p2p(
#     #     pointcloud_pred, normals_pred, pointcloud_tgt, normals_tgt
#     # )
#     accuracy, accuracy_normals = distance_p2p(
#         pointcloud_pred, None, pointcloud_tgt, None
#     )
#     return accuracy.mean(), accuracy_normals


from PIL import Image
import matplotlib.pyplot as plt
# from labels_ours import id2label
from labels_scannet import id2label
# from labels import id2label
def draw_rgb_normal(rgb, i, config, save_dir):
    # rgb_normal = np.vstack((np.zeros([176*1408,3]),rgb)).squeeze()
    rgb_normal = rgb.squeeze()
    savepath = os.path.join(config['path']['proj_dir'], save_dir, "%03d.png"%i)
    rgb_normal = rgb_normal.reshape(config['img']['height'],config['img']['width'],3)
    rgb = rgb_normal.copy()
    rgb[:,:,0] = rgb_normal[:,:,2]
    rgb[:,:,2] = rgb_normal[:,:,0]
    cv2.imwrite(savepath, rgb)
    # cv2.imshow('sdf_render', rgb_normal.reshape(-1,1408,3))
    # savepath = os.path.join(config['path']['proj_dir'], "normal_render_img/%03d.png"%i)
    # cv2.imwrite(savepath, rgb_normal.reshape(-1,1408,3))

def draw_depth_render(depth, i, config):
    depth = depth*256/(depth.max()-depth.min())
    # depth = np.vstack((np.zeros([176*1408,1]),depth.squeeze(-1))).squeeze()
    depth = depth.squeeze()
    # show
    # cv2.imshow('depth_render', depth.reshape(-1,1408,1))
    # save
    savepath = os.path.join(config['path']['proj_dir'], "depth_st/%03d.png"%i)
    cv2.imwrite(savepath, depth.reshape(config['img']['height'],config['img']['width'],1))

def draw_semantic_rgb(sem, i, config):
    savepath = os.path.join(config['path']['proj_dir'], "semantic_st/%03d.png"%i)
    sem = sem.squeeze()
    sem = np.vstack([id2label[semID].color for semID in sem.tolist()]) 
    sem = sem.reshape(config['img']['height'],config['img']['width'],3)
    rgb = sem.copy()
    rgb[:,:,0] = sem[:,:,2]
    rgb[:,:,2] = sem[:,:,0]
    cv2.imwrite(savepath, rgb)

def draw_instance_rgb(ins, color_id, i, config, folder="instance_st"):
    if ins.shape[-1] == 1:
        savepath = os.path.join(config['path']['proj_dir'], folder, "%03d.png"%i)
        ins = color_id[ins].reshape(config['img']['height'],config['img']['width'],3)
        cv2.imwrite(savepath, ins[:,:,::-1])
    else:
        ins = ins.squeeze()
        for ii in range(ins.shape[-1]):
            ins_ = ins[:,ii]
            ins_ = 255 * np.int32(ins_>0.5)
            savepath = os.path.join(config['path']['proj_dir'], folder, f"%03d_{ii}.png"%i)
            ins_ = ins_.reshape(config['img']['height'],config['img']['width'],1)
            cv2.imwrite(savepath, ins_[:,:,::-1])

def draw_semantic_proj(semantic, i, config):
    imagePath = os.path.join(os.environ['KITTI360_DATASET'], 'data_2d_semantics/train', '2013_05_28_drive_%04d_sync'%0, 'image_%02d/semantic_rgb'%0, '%010d.png' % i)
    img = Image.open(imagePath)
    plt.figure("Semantic error")
    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('raw')
    plt.subplot(3,1,2)
    sem_IDs= np.vstack((np.zeros([176*1408,1]),semantic)).squeeze() #.reshape(-1,1408)
    img2 = np.vstack([id2label[semID].color for semID in sem_IDs.tolist()]).reshape(-1,1408,3)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('eval')
    plt.subplot(3,1,3)
    img3 = (np.array(img.convert('RGB'))-img2)
    img3[:176,:] = img2[:176,:]
    plt.imshow(img3.astype(np.uint8))
    plt.axis('off')
    # plt.colorbar()
    plt.title('error')
    # plt.show() 
    plt.savefig(os.path.join(config['path']['data_basedir'], config['path']['proj_dir'], "semantic_img/%03d.png"%i), dpi=500)
    plt.close('all')
   
    yx = 1

def render_video(dataloader, multi_cube, ray_directions_rgb, config, step, epoch_n, render_rgb=False, render_depth=False, save_gt_depth=False, render_depth_error=True):
    ray_chunk_test = config['params']['ray_chunk_test']
    # near = config['params']['near']
    # far = config['params']['far']
    savedir = config['path']['savedir']
    h, w = int(config['params']['h']), int(config['params']['w'])
    h = int(h / config['params']['rescale'])
    w = int(w / config['params']['rescale'])
    
    print("************************Render Test image to Video!************************")
    print("render_rgb = ", render_rgb, " , render_depth = ", render_depth)
    t1 = time.time()
    rgb_loss, rgb_psnr = [], []
    depth_loss, depth_acc = [], []
    images_test = []
    depths_test = []
    # images_gt = []
    depths_gt = []
    mix_images= []
    depths_error = []

    with tqdm(total=len(dataloader)) as t:
        t.set_description('Render test video ')
        for ii, data in enumerate(dataloader):
            # depth_gt_tensor = data["depth"].reshape(-1)

            # grid_idx = max(torch.where((data["idx"] - torch.Tensor(multi_cube["span_idx"]))>0)[0])
            grid_idx=0
            grid = multi_cube["volumes"][grid_idx]
            render_pose_c2w = data['pose'][0]
            ray_directions_rgb = ray_directions_rgb.reshape(-1,3)
            rays_all_num = ray_directions_rgb.shape[0]
            rgb= []
            depth = []
            #CAM
            for i in range(0, rays_all_num, ray_chunk_test):
                i_next = i+ray_chunk_test if (i+ray_chunk_test <rays_all_num) else rays_all_num

                rgb_chunk, depth_chunk = grid.get_rgb_depth_chunk(ray_directions_rgb[i:i_next,:],
                                                        render_pose_c2w)
                rgb.append(rgb_chunk)
                depth.append(depth_chunk)
            
            #LIDAR
            ray_directions_lidar = ray_directions_lidar.reshape(-1,3)
            rays_all_num2 = ray_directions_lidar.shape[0]
            for i in range(0, rays_all_num2, ray_chunk_test):
                i_next = i+ray_chunk_test if (i+ray_chunk_test <rays_all_num2) else rays_all_num2

                rgb_chunk, depth_chunk = grid.get_rgb_depth_chunk(ray_directions_lidar[i:i_next,:],
                                                        render_pose_c2w)
                rgb.append(rgb_chunk)
                depth.append(depth_chunk)
                
            if render_depth:
                depth_torch = torch.cat(depth)
                # depth_gt_tensor = data["depth"].reshape(-1)
                # # depth_mask = torch.logical_and(depth_gt_tensor > grid.z_near, depth_gt_tensor < grid.z_far)
                # depth_mask = depth_gt_tensor > 0
                # depth_abs_loss = torch.abs(depth_gt_tensor - depth_torch)[depth_mask]
                # # depth_abs_loss = torch.abs(depth_gt_tensor - depth_torch)
                # depth_acc_e = torch.count_nonzero(depth_abs_loss < 0.05) / depth_abs_loss.shape[0]
                # depth_loss_e = torch.mean(depth_abs_loss)
                # depth_loss.append(depth_loss_e.item())
                # depth_acc.append(depth_acc_e.item())
                depth_raw = depth_torch.cpu().numpy()
                depth = depth_raw/depth_raw.max()
                depth = (depth*255).astype(np.uint8).reshape(h, w, -1)
                depths_test.append(depth)
                # if ~render_rgb:
                #     t.set_postfix(test_psnr=mse2psnr(depth_loss_e.cpu()).item())
                t.update(1)
            if render_rgb:
                rgb_gt = data['color'].reshape(-1,3)
                color_bg = data['color_bg'].reshape(-1)
                rgb_torch = torch.cat(rgb)
                rgb_loss_e = torch.mean((rgb_gt - rgb_torch)[color_bg] ** 2)        
                rgb_psnr.append(mse2psnr(rgb_loss_e.cpu()).item())
                rgb_loss.append(rgb_loss_e.item())            
                rgb_torch[~color_bg]=0
                rgb = (rgb_torch.cpu().numpy()*255).astype(np.uint8).reshape(h, w, -1)
                images_test.append(rgb)
                t.set_postfix(test_psnr=mse2psnr(rgb_loss_e.cpu()).item())
                t.update(1)
            if render_rgb and render_depth:
                mix_images.append(get_mix_image_np(depth, rgb))

            # For save gt depth
            if save_gt_depth:
                depth_gt = data["depth"][0,0].cpu().numpy()
                depth_gt = depth_gt / depth_gt.max()
                depth_gt = (depth_gt*255).astype(np.uint8)
                depths_gt.append(depth_gt)
            
            # if render_depth_error and render_depth:
            #     # rgb_gt = data['color'].reshape(h, w, 3)
            #     # rgb_gt = (rgb_gt.cpu().numpy()*255).astype(np.uint8)

            #     depth_gt = data["depth"][0,0].cpu().numpy()
            #     mask = (depth_gt==0)
            #     depth_es = depth_raw.reshape(h, w)
            #     depth_es[mask] = 0
            #     depth_error = abs(depth_gt - depth_es)
            #     depth_error = depth_error / depth_error.max()
            #     depth_error = (depth_error*255).astype(np.uint8)

            #     # depths_error.append(get_mix_image_np(depth_error,rgb_gt))
            #     depths_error.append(depth_error)
            
                
    t2 = time.time()
    logs = ""
    if render_rgb:
        rgb_vid_path = os.path.join(savedir, f"rgb_test_{step}_{epoch_n}.mov")
        imageio.mimwrite(rgb_vid_path, images_test, fps=25, quality=8)
        print(f"Test RGB video saved in : {rgb_vid_path}")
        print(f"Epoch {epoch_n} test RGB report : loss = {np.array(rgb_loss).mean()} PSNR = {np.array(rgb_psnr).mean()}")
        logs += f"Epoch {epoch_n} test report : loss = {np.array(rgb_loss).mean()} PSNR = {np.array(rgb_psnr).mean()}\n"
    
    if render_depth:
        depth_vid_path = os.path.join(savedir, f"depth_test_{step}_{epoch_n}.mov")
        imageio.mimwrite(depth_vid_path, depths_test, fps=25, quality=8)
        print(f"Test Depth video saved in : {depth_vid_path}")
        # print(f"Epoch {epoch_n} test Depth report : Abs loss = {np.array(depth_loss).mean()} ACC = {np.array(depth_acc).mean()}")
        # logs += f"Epoch {epoch_n} test Depth report : Abs loss = {np.array(depth_loss).mean()} ACC = {np.array(depth_acc).mean()}\n"

    if render_depth and render_rgb:
        mix_vid_path = os.path.join(savedir, f"mix_test_{step}_{epoch_n}.mov")
        imageio.mimwrite(mix_vid_path, mix_images, fps=25, quality=8)
        print(f"Test Mix video saved in : {mix_vid_path}")
    
    if save_gt_depth:
        # For save gt depth
        imageio.mimwrite(os.path.join(savedir, f"depth_gt.mov"), depths_gt, fps=25, quality=8)
        print("Gt Depth video saved in : ", os.path.join(savedir, f"depth_gt.mov"))

    # if render_depth_error:
    #     depth_vid_path = os.path.join(savedir, f"depth_error_{step}_{epoch_n}.mov")
    #     imageio.mimwrite(depth_vid_path, depths_error, fps=25, quality=8)
    #     print(f"Test Depth_error video saved in : {depth_vid_path}")

    

    print(f"Render {len(dataloader)} images total time = {t2-t1} s")
    print("************************Render END!!!************************")
    
    return logs