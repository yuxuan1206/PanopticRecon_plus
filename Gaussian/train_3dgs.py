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
import sys
sys.path.append('Gaussian/')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import vis_depth, read_propagted_depth
from gaussian_renderer import render, network_gui
from utils.graphics_utils import depth_propagation, check_geometric_consistency
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, load_pairs_relation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import imageio
import numpy as np
import torchvision

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    record_file = os.path.join(args.model_path, 'loss.txt')

    gaussians = GaussianModel(dataset.sh_degree) # load model
    scale = 1.0 # 
    scene = Scene(dataset, gaussians, resolution_scales=[scale]) # read dataset
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    loss_list = []

    for iteration in range(first_iter, opt.iterations + 1):  
        # print(gaussians.get_xyz.shape[0])  
        # gaussians_num.append(gaussians.get_xyz.shape[0])

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # randidx = randint(0, len(viewpoint_stack)-1)
        # viewpoint_cam = viewpoint_stack[randidx]
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale=scale).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        #render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_normal=args.normal_loss)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                            return_normal=opt.normal_loss, return_opacity=True, return_depth=opt.depth_loss or opt.depth2normal_loss)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image[opacity_mask], gt_image[opacity_mask])
        # ssim_loss = 1.0 - ssim(image, gt_image, mask=opacity_mask)
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = 1.0 - ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        # print('!!!')
        # print(f'Ll1: {torch.any(torch.isnan(Ll1))}')
        # print(f'ssim_loss: {torch.any(torch.isnan(ssim_loss))}')

        # flatten loss
        if opt.flatten_loss:
            scales = gaussians.get_scaling
            min_scale, _ = torch.min(scales, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss += opt.lambda_flatten * flatten_loss

        # # opacity loss
        # if opt.sparse_loss:
        #     opacity = gaussians.get_opacity
        #     opacity = opacity.clamp(1e-6, 1-1e-6)
        #     log_opacity = opacity * torch.log(opacity)
        #     log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
        #     sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
        #     loss += opt.lambda_sparse * sparse_loss

        if opt.normal_loss:
            rendered_normal = render_pkg['render_normal']
            # print('normal')
            # print(rendered_normal.cpu())
            if viewpoint_cam.normal is not None:
                normal_gt = viewpoint_cam.normal.cuda()
                if viewpoint_cam.sky_mask is not None:
                    filter_mask = viewpoint_cam.sky_mask.to(normal_gt.device).to(torch.bool)
                    normal_gt[~(filter_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)
                l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
                cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()
                normal_loss = opt.lambda_l1_normal * l1_normal + opt.lambda_cos_normal * cos_normal
                loss += 10 * normal_loss

        if opt.depth_loss and viewpoint_cam.depth is not None:
            render_depth = render_pkg['render_depth']
            depth_gt = viewpoint_cam.depth

            filter_mask = (depth_gt != 0).to(torch.bool)
            l1_depth = torch.abs(render_depth.cpu() - depth_gt)[filter_mask].mean()
            # loss += l1_depth.cuda()
            loss += 0.1 * l1_depth.cuda()

            # loss_l2 = torch.nn.MSELoss()              # l2_depth = loss_l2(render_depth.cpu()[filter_mask], depth_gt[filter_mask])
            # loss += 0.1 * l2_depth.cuda()


        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not torch.isnan(loss):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                loss_list.append(loss.item())
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            with open(record_file, 'a') as f:
                print('iteration:', iteration, 
                      '  loss:', loss.item(),
                      '  Ll1:', (1.0 - opt.lambda_dssim) * Ll1.item(),
                      '  ssim_loss:', opt.lambda_dssim * ssim_loss.item(),
                    #   '  depth_loss:', 0.1 * l1_depth.item(),
                      '  flatten_loss:', opt.lambda_flatten * flatten_loss.item(),
                    #   '  normal_loss', 10 * normal_loss.item(),
                      file=f)
                
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if iteration > 10000:
                    try:
                            os.remove(scene.model_path + "/chkpnt" + str(iteration-10000) + ".pth") 
                    except:
                            print("no " + scene.model_path + "/chkpnt" + str(iteration-10000) + ".pth")
                            
    with open(record_file, 'a') as f:
        print('loss_list:', loss_list, file=f)
    with open('12.txt', 'a') as f:
        print(loss_list, file=f)
    # with open('gaussians_num.txt', 'a') as f:
    #     print(gaussians_num, file=f)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, flatten_loss, l1_depth, normal_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/flatten_loss', flatten_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', l1_depth.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
    # if iteration%10==0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    image_normal = torch.clamp(renderFunc(viewpoint, scene.gaussians, return_normal=True, *renderArgs)["render_normal"], -1.0, 1.0)
                    image_normal = (image_normal+1)/2
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_normal".format(viewpoint.image_name), image_normal[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

class Pipeline_3dgs():
    def __init__(self):
        self.init_3dgs_model()

    def init_3dgs_model(self):
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        parser.add_argument('--ip', type=str, default="127.0.0.1")
        parser.add_argument('--port', type=int, default=6009)
        parser.add_argument('--debug_from', type=int, default=-1)
        parser.add_argument('--detect_anomaly', action='store_true', default=False)
        parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(0,60000,1000)))
        parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(0,60000,5000)))
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=list(range(0,60000,5000)))
        parser.add_argument("--start_checkpoint", type=str, default = None)
        
        self.args = parser.parse_args(sys.argv[1:])
        self.args.save_iterations.append(self.args.iterations)
        
        print("Optimizing " + self.args.model_path)

        ## Initialize system state (RNG)
        safe_state(self.args.quiet)
        torch.autograd.set_detect_anomaly(self.args.detect_anomaly)
        # training(self.args, lp.extract(self.args), op.extract(self.args), pp.extract(self.args), self.args.test_iterations, self.args.save_iterations, self.args.checkpoint_iterations, self.args.start_checkpoint, self.args.debug_from)

        ## load model and dataset
        self.dataset, self.opt, self.pipe = lp.extract(self.args), op.extract(self.args), pp.extract(self.args)
        self.testing_iterations = self.args.test_iterations
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = Scene(self.dataset, self.gaussians, resolution_scales=[1])
        self.gaussians.training_setup(self.opt)

        ## record
        self.tb_writer = prepare_output_and_logger(self.dataset)
        self.record_file = os.path.join(self.args.model_path, 'loss.txt')

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iteration = 0
        self.viewpoint_stack = None
        self.loss_list = []

        self.viewpoint_stack = self.scene.getTrainCameras(scale=1).copy()

    def select_frame(self,random_idx):
        # Pick a random Camera
        # if not self.viewpoint_stack:
        #     self.viewpoint_stack = self.scene.getTrainCameras(scale=1).copy()
        # random_idx = randint(0, len(self.viewpoint_stack)-1)
        self.viewpoint_cam = self.viewpoint_stack[random_idx]
        # real_frame_idx = int(self.viewpoint_cam.uid)
        real_frame_idx = int(self.viewpoint_cam.image_name)
        need_nerf = (self.viewpoint_cam.depth is None)
        return random_idx, real_frame_idx, need_nerf

    def train_step(self, random_idx, normal=None, depth=None):
        # for i in range(10):
        self.iteration += 1 
        iteration, opt = self.iteration, self.opt

        self.gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 5000 == 0:
            self.gaussians.oneupSHdegree()

        

        # viewpoint_cam = self.viewpoint_stack.pop(random_idx)

        ## add normal & depth GT
        # if normal is not None:
        #     self.viewpoint_cam.normal = normal
        if self.viewpoint_cam.depth is not None:
            depth = self.viewpoint_cam.depth.cuda()
        #     self.viewpoint_cam.depth = depth
        if self.viewpoint_cam.normal is not None:
            normal = self.viewpoint_cam.normal.cuda()

        bg = torch.rand((3), device="cuda") if opt.random_background else self.background

        render_pkg = render(self.viewpoint_cam, self.gaussians, self.pipe, bg, 
                                return_normal=opt.normal_loss, return_opacity=True, return_depth=opt.depth_loss or opt.depth2normal_loss)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = self.viewpoint_cam.original_image.cuda()
        
        h, w = gt_image.shape[1], gt_image.shape[2]
        filter_mask = torch.ones([h, w]).bool()
        # filter_mask[:20,:] = False
        # filter_mask[-20:,:] = False
        # filter_mask[:,:20] = False
        # filter_mask[:,-20:] = False
        # image = image[:,filter_mask].reshape(3, h-40, w-40)
        # gt_image = gt_image[:,filter_mask].reshape(3, h-40, w-40)
        # opacity mask
        if iteration < opt.propagated_iteration_begin and opt.depth_loss:
            opacity_mask = render_pkg['render_opacity'] > 0.2 #0.999
            # opacity_mask = opacity_mask[filter_mask].reshape(h-40, w-40).unsqueeze(0).repeat(3, 1, 1)
            opacity_mask = opacity_mask[filter_mask].reshape(h, w).unsqueeze(0).repeat(3, 1, 1)
        else:
            opacity_mask = render_pkg['render_opacity'] > 0.0
            # opacity_mask = opacity_mask[filter_mask].reshape(h-40, w-40).unsqueeze(0).repeat(3, 1, 1)
            opacity_mask = opacity_mask[filter_mask].reshape(h, w).unsqueeze(0).repeat(3, 1, 1)
        
        Ll1 = l1_loss(image[opacity_mask], gt_image[opacity_mask])
        ssim_loss = 1.0 - ssim(image, gt_image, window_size=7, mask=opacity_mask)
        # Ll1 = l1_loss(image, gt_image)
        # ssim_loss = 1.0 - ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        # flatten loss
        flatten_loss = torch.zeros(1).squeeze().to(loss.device)
        if opt.flatten_loss:
            scales = self.gaussians.get_scaling
            min_scale, _ = torch.min(scales, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss += opt.lambda_flatten * flatten_loss
        
        # opacity loss
        if opt.sparse_loss:
            opacity = gaussians.get_opacity
            opacity = opacity.clamp(1e-6, 1-1e-6)
            log_opacity = opacity * torch.log(opacity)
            log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
            sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
            loss += opt.lambda_sparse * sparse_loss


        l1_depth = torch.zeros(1).squeeze().to(loss.device)
        if opt.depth_loss and depth is not None:
            render_depth = render_pkg['render_depth']
            if render_depth.shape[0] != depth.shape[0]:
                resize = torchvision.transforms.Resize([depth.shape[0], depth.shape[1]])
                render_depth = resize(render_depth[None,...]).squeeze()
            depth_gt = depth#.cuda()
            filter_mask = (depth_gt > 0.5).to(torch.bool)
            l1_depth = torch.abs(render_depth - depth_gt)[filter_mask].mean()
            # loss += l1_depth.cuda()
            loss += 0.1 * l1_depth #.cuda()

        normal_loss = torch.zeros(1).squeeze().to(loss.device)
        if opt.normal_loss:
            # print('normal')
            # print(rendered_normal.cpu())
            if normal is not None:
                rendered_normal = render_pkg['render_normal']
                if rendered_normal.shape[1] != normal.shape[1]:
                    resize = torchvision.transforms.Resize([normal.shape[1], normal.shape[2]])
                    rendered_normal = resize(rendered_normal)
                normal_gt = normal #.cuda()
                if self.viewpoint_cam.sky_mask is not None:
                    filter_mask = self.viewpoint_cam.sky_mask.to(normal_gt.device).to(torch.bool)
                    normal_gt[~(filter_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                filter_mask = (normal_gt.norm(dim=0) > 0.5).to(torch.bool)
                l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
                cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()
                normal_loss = opt.lambda_l1_normal * l1_normal + opt.lambda_cos_normal * cos_normal
                loss += 10 * normal_loss

        loss.backward()
        # iter_end.record()

        with torch.no_grad():
            # # Progress bar
            # if not torch.isnan(loss):
            #     ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            #     loss_list.append(loss.item())
            # if iteration % 10 == 0:
            #     progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            #     progress_bar.update(10)
            # if iteration == opt.iterations:
            #     progress_bar.close()

            # Log and save
            with open(self.record_file, 'a') as f:
                print('iteration:', iteration, 
                    '  loss:', loss.item(),
                    '  Ll1:', (1.0 - opt.lambda_dssim) * Ll1.item(),
                    '  ssim_loss:', opt.lambda_dssim * ssim_loss.item(),
                    '  depth_loss:', 0.1 * l1_depth.item(),
                    '  flatten_loss:', opt.lambda_flatten * flatten_loss.item(),
                    '  normal_loss', 10 * normal_loss.item(),
                    file=f)
                
            training_report(self.tb_writer, iteration, Ll1, loss, l1_loss, flatten_loss, l1_depth, normal_loss, self.testing_iterations, self.scene, render, (self.pipe, self.background))
            if (iteration in self.args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                self.scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if iteration > opt.opacity_reset_interval :
                        yx = 1
                    self.gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none = True)
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()

            if (iteration in self.args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((self.gaussians.capture(), iteration), self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 2000, 7000, 14000, 20000, 25000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 2000, 7000, 14000, 20000, 25000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
