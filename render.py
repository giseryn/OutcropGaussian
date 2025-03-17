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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import pdb

from utils_extend.loss_depth_utils import depth_loss,depth_loss_cuda,depth_to_normal
from utils_extend.image_process import get_compare_images,visualize_normal_map,visualize_depth_map_pil
from utils_extend.data_preprocessing import fit_the_depth, mean_radii,use_lidar_depth,use_fusion_depth

def save_pic(image_data, savepath):
    from PIL import Image
    tensor1 = image_data.permute(1, 2, 0)
    tensor1 = (tensor1 - tensor1.min()) / (tensor1.max() - tensor1.min()) * 255
    tensor1 = tensor1.byte()
    tensor1 = tensor1.cpu().detach().numpy()
    image1 = Image.fromarray(tensor1)
    image1.save(savepath)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")

    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gts_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    gts_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_normal")

    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)

    makedirs(gts_path, exist_ok=True)
    makedirs(gts_depth_path, exist_ok=True)
    makedirs(gts_normal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        render_pkg = render(view, gaussians, pipeline, background)
        image, depth_image = (render_pkg["render"], render_pkg["depth_image"])
        rendering = image
        rendering_depth = depth_image.squeeze().repeat(3, 1, 1).cuda()
        rendering_normal = depth_to_normal(depth_image.squeeze())

        gt = view.original_image[0:3, :, :]
        gt_depth=view.depth_image[0:3, :, :]
        gt_normal=depth_to_normal(gt_depth[0])

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(rendering_depth, os.path.join(render_depth_path, '{0:05d}'.format(idx) + ".png"))
        save_pic(rendering_depth,os.path.join(render_depth_path, '{0:05d}'.format(idx) + ".png"))
        # save_pic(depth_to_normal(depth_image.squeeze()).permute(2, 0, 1), os.path.join(render_normal_path, '{0:05d}'.format(idx) + ".png"))
        visualize_normal_map(rendering_normal, show=False,save_path=os.path.join(render_normal_path, '{0:05d}'.format(idx) + ".png"))


        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_depth, os.path.join(gts_depth_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(depth_to_normal(gt_depth[0]).permute(2, 0, 1), os.path.join(gts_normal_path, '{0:05d}'.format(idx) + ".png"))
        visualize_normal_map(gt_normal,show=False,save_path=os.path.join(gts_normal_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # gaussians._opacity.fill_(1)

        if dataset.depth_type == 'lidar':
            use_lidar_depth(gaussians, scene.getTrainCameras(), dataset)
            print('use radar depth maps')
        elif dataset.depth_type == 'fusion':
            use_fusion_depth(gaussians, scene.getTrainCameras(), dataset)
            print('use fusion depth maps')
        else:
            print('Use monocular depth maps')

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