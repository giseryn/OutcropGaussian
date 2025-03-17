import os
from PIL import Image
import numpy as np
import shutil
import cv2
import torch


from utils_extend.loss_depth_utils import depth_loss_rmse_pic


def calculate_histogram_complexity(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    histogram = cv2.calcHist([image_np], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    complexity = -np.sum(histogram * np.log2(histogram + 1e-6))  # Entropy
    return complexity


def image_histogram(image_folder, result_folder, top_n=100):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if
                   fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]
    complexities = []

    for image_path in image_paths:
        complexity = calculate_histogram_complexity(image_path)
        complexities.append((image_path, complexity))

    # Sort by complexity and select top_n
    complexities.sort(key=lambda x: x[1], reverse=True)
    top_images = complexities[:top_n]

    for i, (image_path, complexity) in enumerate(top_images):
        shutil.copy(image_path, os.path.join(result_folder, f"{i + 1}_{os.path.basename(image_path)}"))

    print(f"Top {top_n} images copied to {result_folder}")


def test_image_histogram():
    # Example usage:
    image_histogram("data/xjsd_handle/images", "data/xjsd_handle/result")


def save_image(gaussian, image, gt_image, iteration):
    from PIL import Image
    tensor1 = gt_image.permute(1, 2, 0)
    tensor1 = (tensor1 - tensor1.min()) / (tensor1.max() - tensor1.min()) * 255
    tensor1 = tensor1.byte()
    tensor1 = tensor1.cpu().detach().numpy()
    image1 = Image.fromarray(tensor1)
    image1.save(f"{iteration}_output_gt_image.png")

    tensor2 = image.permute(1, 2, 0)
    tensor2 = torch.nan_to_num(tensor2, nan=0.0)
    tensor2 = (tensor2 - tensor2.min()) / (tensor2.max() - tensor2.min()) * 255
    tensor2 = tensor2.byte()
    tensor2 = tensor2.cpu().detach().numpy()
    image2 = Image.fromarray(tensor2)
    image2.save(f"{iteration}_output_image.png")


def save_pic(image_data, savepath):
    from PIL import Image
    tensor1 = image_data.permute(1, 2, 0)
    tensor1 = (tensor1 - tensor1.min()) / (tensor1.max() - tensor1.min()) * 255
    tensor1 = tensor1.byte()
    tensor1 = tensor1.cpu().detach().numpy()
    image1 = Image.fromarray(tensor1)
    image1.save(savepath + ".png")



def get_compare_images(scene, gaussians, opt, background, pipe, dataset, render):

    viewpoint_stack = scene.getTrainCameras().copy()
    os.mkdir(dataset.model_path + 'compareimages/')
    for viewpoint_cam in viewpoint_stack:
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, depth_image, alpha_image \
            = (render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"],
               render_pkg["radii"], render_pkg["depth_image"], render_pkg["alpha_image"])
        render_image = image.cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = viewpoint_cam.depth_image.cuda()
        render_depth=depth_image.squeeze().repeat(3, 1, 1).cuda()
        depth_rmse_image=depth_loss_rmse_pic(depth_image.squeeze(),gt_depth[0],dataset.model_path,True)

        image_name = viewpoint_cam.image_name
        render_images_savepath = dataset.model_path + 'compareimages/render_image_' + image_name
        gt_image_savepath = dataset.model_path + 'compareimages/gt_image_' + image_name
        gt_depth_savepath = dataset.model_path + 'compareimages/gt_depth_' + image_name
        depth_rmse_savepath = dataset.model_path + 'compareimages/depth_rmse_' + image_name
        render_depth_images_savepath = dataset.model_path + 'compareimages/render_depth_image_' + image_name

        save_pic(render_image, render_images_savepath)
        save_pic(render_depth, render_depth_images_savepath)
        save_pic(gt_image, gt_image_savepath)
        save_pic(gt_depth, gt_depth_savepath)
        save_pic(depth_rmse_image, depth_rmse_savepath)


def visualize_normal_map(normal_tensor, save_path=None, show=True):
    """
    使用 PIL 可视化从深度图生成的法线图。

    参数:
    - normal_tensor (torch.Tensor): 输入法线张量，形状为 (3, H, W)，值范围为 [-1, 1]。
    - save_path (str): 可选，保存图像的路径。如果为 None，则不保存。

    返回:
    - normal_image (PIL.Image): 可视化的法线图。
    """
    try:
        # 确保法线张量在 CPU 上
        normal_tensor = normal_tensor.cpu().detach()

        # 转置形状为 (H, W, 3)，以适应图像格式
        normal_array = normal_tensor.permute(1, 2, 0).numpy()

        # 将值从 [-1, 1] 映射到 [0, 255]
        normal_array = (normal_array + 1) / 2 * 255
        normal_array = normal_array.astype(np.uint8)

        # 转换为 PIL 图像
        normal_image = Image.fromarray(normal_array, mode="RGB")

        # 保存图像（如果提供保存路径）
        if save_path:
            normal_image.save(save_path)
            # print(f"法线图已保存到: {save_path}")

        # 显示图像
        if show:
            normal_image.show()

        return normal_image
    except Exception as e:
        print(f"Error visualizing normal map: {e}")
        return None

def visualize_depth_map_pil(depth_tensor, save_path=None, colormap="viridis"):
    """
    使用 PIL 可视化深度图张量（单通道），通过颜色梯度表示深度。

    参数:
    - depth_tensor (torch.Tensor): 深度图张量，形状为 (H, W)，值范围为任意浮点数。
    - save_path (str): 可选，保存图像的路径。如果为 None，则不保存。
    - colormap (str): 用于深度图的颜色映射，支持 "viridis", "plasma", "inferno", "magma" 等。

    返回:
    - depth_image (PIL.Image): 可视化的深度图。
    """
    # 确保深度图张量在 CPU 上
    depth_tensor = depth_tensor.cpu().detach()

    # 转换为 NumPy 数组
    depth_array = depth_tensor.numpy()

    # 归一化深度值到 [0, 1]，防止数据超出范围影响颜色映射
    depth_min, depth_max = depth_array.min(), depth_array.max()
    if depth_max - depth_min > 1e-5:  # 避免除零错误
        normalized_depth = (depth_array - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = np.zeros_like(depth_array)

    # 使用 NumPy 实现颜色映射 (默认 viridis)
    colormaps = {
        "viridis": [(68, 1, 84), (72, 35, 116), (62, 74, 137), (49, 104, 142), (38, 130, 142),
                    (31, 158, 137), (53, 183, 121), (109, 205, 89), (180, 222, 44), (253, 231, 37)],
        "plasma": [(13, 8, 135), (75, 3, 161), (125, 3, 168), (168, 34, 150), (203, 70, 120),
                   (229, 107, 93), (248, 148, 65), (253, 194, 38), (240, 249, 33), (245, 253, 191)]
    }

    # 获取对应的颜色映射列表
    cmap = colormaps.get(colormap, colormaps["viridis"])
    cmap = np.array(cmap) / 255.0  # 归一化到 [0, 1]

    # 将深度值映射到颜色映射的索引
    color_indices = (normalized_depth * (len(cmap) - 1)).astype(np.int32)

    # 映射深度值到 RGB
    depth_colored = cmap[color_indices]
    depth_colored = (depth_colored * 255).astype(np.uint8)  # 转为 [0, 255]

    # 转换为 PIL 图像
    depth_image = Image.fromarray(depth_colored, mode="RGB")

    # 保存图像（如果指定保存路径）
    if save_path:
        depth_image.save(save_path)
        print(f"深度图已保存到: {save_path}")

    # 显示深度图
    depth_image.show()

    return depth_image
