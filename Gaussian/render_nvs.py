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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
# torch.set_printoptions(profile="full")
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, vis_depth, vis_depth1
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import cv2

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_nvs")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depth_nvs")
    # normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_normal")
    depth_gray_path = os.path.join(depth_path.replace("render_depth", "render_depth_gray_nvs"))

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    makedirs(depth_path , exist_ok=True)
    # makedirs(normal_path, exist_ok=True)
    makedirs(depth_gray_path , exist_ok=True)

    depth_loss_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        renders = render(view, gaussians, pipeline, background, return_depth=True, return_normal=True)
        rendering = renders["render"]
        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        render_depth = renders["render_depth"]

        render_depth_gray = render_depth.detach().cpu().numpy()
        render_depth_gray = (render_depth_gray * 1000).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_gray_path, view.image_name + ".png"), render_depth_gray)

        if view.depth is not None:
            depth_gt = view.depth
            filter_mask = (depth_gt != 0).to(torch.bool)
            l1_depth = torch.abs(render_depth.cpu() - depth_gt)[filter_mask].mean()
            depth_loss_list.append(l1_depth.item())
            print(l1_depth.item())

        # print(f'min:{render_depth.min()}  max:{render_depth.max()}')
        # print(view.image_name)
        
        if view.sky_mask is not None:
            render_depth[~(view.sky_mask.to(render_depth.device).to(torch.bool))] = 300
        render_depth = vis_depth1(render_depth.detach().cpu().numpy())[0]
        # imageio.imwrite(os.path.join(depth_path , '{0:05d}'.format(idx) + ".png"), render_depth)
        imageio.imwrite(os.path.join(depth_path, view.image_name + ".png"), render_depth)

        render_normal = (renders["render_normal"] + 1.0) / 2.0
        if view.sky_mask is not None:
            render_normal[~(view.sky_mask.to(rendering.device).to(torch.bool).unsqueeze(0).repeat(3, 1, 1))] = -10
        # render_normal = renders["render_normal"]
        # np.save(os.path.join(normal_path, view.image_name + ".png"), renders["render_normal"].detach().cpu().numpy())
        # torchvision.utils.save_image(render_normal, os.path.join(normal_path, view.image_name + ".png"))
        # normal_gt = torch.nn.functional.normalize(view.normal, p=2, dim=0)
        if view.normal is not None:
            normal_gt = view.normal
            render_normal_gt = (normal_gt + 1.0) / 2.0
            torchvision.utils.save_image(render_normal_gt, os.path.join(normal_path, view.image_name + "_normalgt.png"))
        # exit()

    # if len(depth_loss_list) > 0:
    #     depth_loss = sum(depth_loss_list) / len(depth_loss_list)
    #     with open('/sdb1/xieyx/code/GaussianPro/output/roma/01/depth_error.txt', 'a') as f:
    #         print(name, depth_loss, file=f)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        dataset.skip = 1
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, custom_path="/sdf1/yx/ijrr/demo_8/traj_scannet_f665_fixed.npy")

        # gaussians._scaling[:, 0] = 0.001
        # gaussians._scaling[:, 1] = 0.0005
        # gaussians._scaling[:, 2] = -10000.0
        # gaussians._rotation[:, 0] = 1
        # gaussians._rotation[:, 1:] = 0
        scales = gaussians.get_scaling

        # min_scale, _ = torch.min(scales, dim=1)
        # max_scale, _ = torch.max(scales, dim=1)
        # median_scale, _ = torch.median(scales, dim=1)
        # print(min_scale)
        # print(max_scale)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)