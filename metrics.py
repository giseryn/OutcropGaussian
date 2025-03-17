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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from skimage.metrics import mean_squared_error
import numpy as np
from skimage import io
import traceback

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def read_depth_Images(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = io.imread(renders_dir / fname).astype(np.float32) / 255.0
        gt = io.imread(gt_dir / fname).astype(np.float32) / 255.0
        renders.append(render)
        gts.append(gt)
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                gt_depth_dir = method_dir / "gt_depth"
                gt_normal_dir = method_dir / "gt_normal"

                renders_dir = method_dir / "renders"
                renders_depth_dir = method_dir / "renders_depth"
                renders_normal_dir = method_dir / "renders_normal"

                renders, gts, image_names = readImages(renders_dir, gt_dir)
                renders_depth, gts_depth, depth_image_names = read_depth_Images(renders_depth_dir, gt_depth_dir)
                renders_normal, gts_normal, normal_image_names = read_depth_Images(renders_normal_dir, gt_normal_dir)

                ssims = []
                psnrs = []
                lpipss = []
                depth_rmses = []
                normal_rmses = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    # depth_rmses.append(depthrmse(renders_depth[idx], gts_depth[idx]))
                    # depth_rmses.append(depth_loss_cuda(renders_depth[idx],gts_depth[idx]))
                    depth_mse = mean_squared_error(gts_depth[idx], renders_depth[idx])
                    depth_rmse = np.sqrt(depth_mse)
                    depth_rmses.append(depth_rmse)

                    normal_mse = mean_squared_error(gts_normal[idx], renders_normal[idx])
                    normal_rmse = np.sqrt(normal_mse)
                    normal_rmses.append(normal_rmse)
                    # print(depth_rmse,depth_mse)
                    # print(depth_rmses)

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  Depth RMSE: {:>12.7f}".format(torch.tensor(depth_rmses).mean(), ".5"))
                print("  Normal RMSE: {:>12.7f}".format(torch.tensor(normal_rmses).mean(), ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                     "Depth RMSE": torch.tensor(depth_rmses).mean().item(),
                                                     "Normal RMSE": torch.tensor(normal_rmses).mean().item()},
                                                    )
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                         "Depth RMSE": {name: DR for DR, name in zip(torch.tensor(depth_rmses).tolist(), image_names)},
                                                         "Normal RMSE": {name: NR for NR, name in zip(torch.tensor(normal_rmses).tolist(), image_names)}},
                                                        )

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            traceback.print_exc()


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
