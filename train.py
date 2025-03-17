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

import matplotlib.pyplot as plt
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np

from utils_extend.loss_depth_utils import depth_loss,depth_loss_cuda,depth_to_normal,normal_consistency_loss
from utils_extend.contribution_assessment import depth_from_points,depth_from_points_backup,remove_low_contrib_gaussian
from utils_extend.data_preprocessing import fit_the_depth, mean_radii,use_lidar_depth,use_fusion_depth
from utils_extend.image_process import get_compare_images,visualize_normal_map,visualize_depth_map_pil
from utils_extend.voxel_constraints import initialize_voxels,compare_points_with_voxels,get_mask_for_points_outside_voxels,update_voxel_labels
from utils_extend.rpc_model import rpc_camera_test,check_rpc_proj
import wandb

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# CUDA_LAUNCH_BLOCKING = 1


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    # fit_the_depth(gaussians, scene.getTrainCameras())
    # Radii, depth_range_tensor, data_extent = mean_radii(gaussians, scene.getTrainCameras())

    dataset.depth_type = ''.join(dataset.depth_type)
    dataset.depth_weight = float(''.join(dataset.depth_weight))

    if dataset.depth_type=='lidar':
        use_lidar_depth(gaussians, scene.getTrainCameras(), dataset)
        print('use radar depth maps')
    elif dataset.depth_type=='fusion':
        use_fusion_depth(gaussians, scene.getTrainCameras(), dataset)
        print('use fusion depth maps')
    else:
        print('Use monocular depth maps')

    if dataset.voxel:
        voxel_labels, min_coords, voxel_size = initialize_voxels(gaussians._xyz)

    # gaussians._scaling=torch.full_like(gaussians._scaling, -5.6470)




    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # init_voxel = gaussians.voxel_init()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # get_compare_images(scene, gaussians, opt, background, pipe, dataset, render)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):



        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
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
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        # check_rpc_proj(viewpoint_stack, gaussians, iteration)

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, depth_image, alpha_image,contrib_map \
            = (render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"],
               render_pkg["radii"], render_pkg["depth_image"], render_pkg["alpha_image"], render_pkg["contrib_map"])

        # unique_tensor = torch.unique(contrib_map)
        #
        # print(unique_tensor.size())

        # Loss
        def get_depth_weight(current_iteration, max_iterations,depth_weight):
            if current_iteration >=max_iterations:
                return 0
            # 计算当前进度的10%区间
            progress = current_iteration / max_iterations
            stage = int(progress * 10)  # 0 到 9 的整数

            if dataset.depth == "Single-view":
                return float(depth_weight)
            elif dataset.depth == "Multi-view":
                return max(depth_weight - 0.1 * stage*depth_weight, 0)
            else:
                return 0
        def get_voxel_weight(voxel_weight):

            if dataset.voxel == True:
                return voxel_weight
            else:
                return 0

        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth_image = viewpoint_cam.depth_image[0].cuda()
        gt_normal_image = depth_to_normal(gt_depth_image)

        render_normal_image = depth_to_normal(depth_image.squeeze())

        # visualize_normal_map(render_normal_image,show=True)
        # visualize_normal_map(gt_normal_image, show=True)
        # visualize_depth_map_pil(depth_image.squeeze())
        visualize_depth_map_pil(gt_depth_image)

        proportion_in_voxels=0
        if dataset.voxel:
            proportion_in_voxels = compare_points_with_voxels(gaussians._xyz, voxel_labels, min_coords, voxel_size)

        # L_depth = depth_loss(gaussians._xyz, viewpoint_cam)

        L_depth_cuda = depth_loss_cuda(depth_image.squeeze(),gt_depth_image,True)
        L_normal_cuda = normal_consistency_loss(render_normal_image,gt_normal_image)

        L_l1 = l1_loss(image, gt_image)
        L_ssim = (1.0 - ssim(image, gt_image))
        L_RGB = (1.0 - opt.lambda_dssim) * L_l1 + opt.lambda_dssim * L_ssim
        depth_weight = get_depth_weight(iteration, opt.densify_until_iter,dataset.depth_weight)
        # voxel_weight = get_voxel_weight(0.2)

        loss = (1 - depth_weight) * L_RGB + 0.5*depth_weight * (L_depth_cuda+L_normal_cuda)

        # L_depth = depth_loss(gaussians._xyz, viewpoint_cam)
        #
        # Ll1 = l1_loss(image, gt_image)
        # L_RGB = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # if L_depth>0:
        #     loss = 1 * L_RGB + 0 * L_depth
        # else:
        #     loss = L_RGB

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{ema_loss_for_log:.{7}f}", " || total_points": f"{gaussians._xyz.size()[0]}","|| proportion_in_voxels": f"{proportion_in_voxels}"})
                progress_bar.update(10)
                ema_loss_for_log = 0
            if iteration == opt.iterations:
                progress_bar.close()
            # gaussians.save_image(image, gt_image, iteration)
            # if iteration % 1000 == 0 and iteration >= 10000:
            #     gaussians.depth_from_points(scene.getTrainCameras(),depth_range_tensor,data_extent)
            # gaussians.save_image(image, gt_image, iteration)
            # if iteration % 1000 == 0:
            #     gaussians.save_image(image, gt_image, iteration)

            # Log and save

            training_report(tb_writer, iteration, loss, L_ssim, L_l1, L_depth_cuda, l1_loss, depth_loss_cuda,
                            iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)


                if iteration > opt.densify_from_iter and iteration % dataset.voxel_range == 0:
                    if dataset.voxel:
                        voxel_labels=update_voxel_labels(gaussians._xyz, voxel_labels, min_coords, voxel_size)
                        get_mask_for_points_outside_voxels(gaussians, voxel_labels, min_coords, voxel_size)


                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None


                    if dataset.contribution:
                        contrib_remove_mask = remove_low_contrib_gaussian(gaussians, scene.getTrainCameras().copy(),
                                                                          render, pipe, bg,dataset.contribution_range)
                        gaussians.prune_points(contrib_remove_mask)

                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.1, scene.cameras_extent,
                                                size_threshold,)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def init_wandb(dataset):
    wandb.init(
        # set the wandb project where this run will be logged
        project="OutcropGauss3D",
        name=dataset.model_path.split("/")[1],
        # track hyperparameters and run metadata
        config=dataset
    )


def training_report(tb_writer, iteration, loss, L_ssim, L_l1, L_depth_CUDA, l1_loss, depth_loss_cuda, elapsed,
                    testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    # if tb_writer:
    #     tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
    #     tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
    #     tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    # wandb.log(
    #     {'train_loss_patches/total_loss': loss.item(),
    #      'train_loss_patches/l1_loss': L_l1.item(),
    #      'train_loss_patches/L_ssim': L_ssim.item(),
    #      'train_loss_patches/depth': L_depth_CUDA.item(),
    #      'iteration': iteration
    #      })
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                depth_test = 0.0
                normal_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    depth = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["depth_image"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_depth_image = torch.clamp(viewpoint.depth_image.to("cuda"), 0.0, 1.0)


                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    # 计算 depth_loss_cuda 的结果
                    depth_loss = depth_loss_cuda(depth.squeeze(), gt_depth_image[0], True).mean().double()
                    normal_loss = depth_loss_cuda(depth_to_normal(depth.squeeze()), depth_to_normal(gt_depth_image[0]), True).mean().double()
                    # 检查是否为 NaN
                    if not torch.isnan(depth_loss):
                        depth_test += depth_loss

                    if not torch.isnan(normal_loss):
                        normal_test += normal_loss

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                depth_test /= len(config['cameras'])
                normal_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} DEPTH {} NORMAL{}".format(iteration, config['name'], l1_test,
                                                                                 psnr_test, depth_test,normal_test))

                # if tb_writer:
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                # if config['name'] == 'test':
                #     wandb.log(
                #         {
                #             'Evaluating test/loss_viewpoint - l1_loss': l1_test,
                #             'Evaluating test/loss_viewpoint - psnr': psnr_test,
                #             'Evaluating test/loss_viewpoint - depth': depth_test,
                #             'Evaluating test/loss_viewpoint - normal': normal_test,
                #             'iteration': iteration
                #         })
                # if config['name'] == 'train':
                #     wandb.log(
                #         {
                #             'Evaluating train/loss_viewpoint - l1_loss': l1_test,
                #             'Evaluating train/loss_viewpoint - psnr': psnr_test,
                #             'Evaluating train/loss_viewpoint - depth': depth_test,
                #             'Evaluating train/loss_viewpoint - normal': normal_test,
                #             'iteration': iteration
                #         })

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        # wandb.log(
        #     {
        #         'total_points': scene.gaussians.get_xyz.shape[0],
        #         'iteration': iteration
        #     })
        torch.cuda.empty_cache()


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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 2_000, 3_000, 4_000, 5_000,
                                                                           6_000,7_000, 8_000, 9_000, 10_000, 11_000, 12_000, 13_000, 14_000,
                                                                           15_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 20_000, 30_000,40_000, 50_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # os.environ["WANDB_MODE"] = "offline"
    # init_wandb(args)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
