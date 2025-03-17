import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
import numpy as np
import pandas as pd
from PIL import Image
import cv2

def depth_loss(gaussians, viewpoint_cam):
    def focal2fov(focal_length, dimension_size):
        return 2 * np.arctan(dimension_size / (2 * focal_length))

    def project_points_to_image(points_3d, R, T, focal_length_x, focal_length_y, cam_center, image_size, gt_depth):
        """
        将三维点投影到二维图像平面并生成深度图和点数量统计图。

        参数：
        - points_3d: 三维点 (N, 3)，Tensor 类型
        - R: 旋转矩阵 (3, 3)，Tensor 类型
        - T: 平移向量 (3,)，Tensor 类型
        - focal_length_x: 水平方向的焦距
        - focal_length_y: 垂直方向的焦距
        - cam_center: 相机光心位置 (cx, cy)
        - image_size: 图像大小 (height, width)

        返回：
        - depth_image: 深度图 (height, width)
        - point_count_image: 点数量统计图 (height, width)
        """
        # 确保输入数据类型为 Tensor 并在同一设备上
        device = points_3d.device
        R = torch.tensor(R, dtype=torch.float32).to(device)
        T = torch.tensor(T, dtype=torch.float32).to(device)

        # 提取相机光心坐标
        cx, cy = cam_center

        # 初始化深度图和点数量统计图
        height, width = image_size
        depth_image = torch.full((height, width), float('inf'), device=device)

        # 将三维点变换到相机坐标系
        points_cam = torch.matmul(points_3d, R.T) + T

        # 投影到图像平面
        x = (points_cam[:, 0] * focal_length_x / points_cam[:, 2]) + cx
        y = (points_cam[:, 1] * focal_length_y / points_cam[:, 2]) + cy

        # 转换为整数索引
        ix = x.round().long()
        iy = y.round().long()

        # 创建有效点掩码
        valid_mask = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height) & (points_cam[:, 2] > 0)

        # 筛选有效点
        ix = ix[valid_mask]
        iy = iy[valid_mask]
        depth = points_cam[:, 2][valid_mask]
        # depth = normalize_vector(depth)

        # 更新深度图和点数量统计图
        depth_image[iy, ix] = torch.minimum(depth_image[iy, ix], depth)
        depth_image = torch.where(torch.isinf(depth_image), torch.tensor(float('nan'), device='cuda'), depth_image)

        # Convert to 3-channel image
        depth_image = depth_image.unsqueeze(0).repeat(3, 1, 1)
        assert depth_image.shape == gt_depth.shape, "Shapes do not match"

        # from PIL import Image
        # tensor1 = gt_depth[0, :, :]
        # tensor1 = (tensor1 - tensor1.min()) / (tensor1.max() - tensor1.min()) * 255
        # tensor1 = tensor1.byte()
        # tensor1 = tensor1.cpu().detach().numpy()
        # image1 = Image.fromarray(tensor1)
        # image1.save("output_gt_image.png")
        #
        # tensor2 = depth_image[0, :, :]
        # tensor2 = torch.nan_to_num(tensor2, nan=0.0)
        # tensor2 = (tensor2 - tensor2.min()) / (tensor2.max() - tensor2.min()) * 255
        # tensor2 = tensor2.byte()
        # tensor2 = tensor2.cpu().detach().numpy()
        # image2 = Image.fromarray(tensor2)
        # image2.save("output_image.png")

        depth_loss = normalized_cross_correlation(depth_image.float(), gt_depth)
        return depth_loss

    # 示例使用
    points_3d = gaussians
    height = viewpoint_cam.image_height  # 示例图像高度
    width = viewpoint_cam.image_width  # 示例图像宽度
    fov_x = viewpoint_cam.FoVx
    fov_y = viewpoint_cam.FoVy
    focal_length_x = width / (2 * math.tan(fov_x / 2))
    focal_length_y = height / (2 * math.tan(fov_y / 2))
    cam_center = (width / 2, height / 2)  # 示例相机光心位置
    R = viewpoint_cam.R  # 示例旋转矩阵
    T = viewpoint_cam.T  # 示例平移向量
    gt_depth = viewpoint_cam.depth_image

    depth_loss = project_points_to_image(points_3d, R, T, focal_length_x, focal_length_y, cam_center, (height, width),
                                         gt_depth)

    # 释放未使用的内存
    torch.cuda.empty_cache()

    return depth_loss

def normalized_cross_correlation(tensor1, tensor2):
    def compute_mse(depth_image1, depth_image2, mask):
        return torch.mean((depth_image1[mask] - depth_image2[mask]) ** 2)

    # 计算绝对误差 (MAE)
    def compute_mae(depth_image1, depth_image2, mask):
        return torch.mean(torch.abs(depth_image1[mask] - depth_image2[mask]))

    # 计算相关系数 (Correlation Coefficient)
    def compute_correlation(depth_image1, depth_image2, mask):
        x = depth_image1[mask]
        y = depth_image2[mask]
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        denominator = torch.sqrt(torch.sum((x - x_mean) ** 2) * torch.sum((y - y_mean) ** 2))
        return numerator / denominator

    def compute_ncc(tensor1, tensor2, mask):
        # 过滤NaN值
        valid_tensor1 = tensor1[mask]
        valid_tensor2 = tensor2[mask]

        # 剔除最大的5%和最小的5%的数据
        sorted_indices1 = torch.argsort(valid_tensor1)
        sorted_indices2 = torch.argsort(valid_tensor2)
        num_elements = valid_tensor1.numel()
        lower_bound = int(0.01 * num_elements)
        upper_bound = int(0.99 * num_elements)

        valid_indices1 = sorted_indices1[lower_bound:upper_bound]
        valid_indices2 = sorted_indices2[lower_bound:upper_bound]

        filtered_tensor1 = valid_tensor1[valid_indices1]
        filtered_tensor2 = valid_tensor2[valid_indices2]

        # 计算均值和标准差
        mean1 = torch.mean(filtered_tensor1)
        mean2 = torch.mean(filtered_tensor2)
        std1 = torch.std(filtered_tensor1)
        std2 = torch.std(filtered_tensor2)

        epsilon = 1e-10
        std1 = std1 + epsilon
        std2 = std2 + epsilon

        # 计算NCC
        ncc = torch.mean((filtered_tensor1 - mean1) * (filtered_tensor2 - mean2)) / (std1 * std2)

        return 1 - ncc

    # 排除NaN值
    mask = ~torch.isnan(tensor1)
    # 计算相似度指标
    # mse = compute_mse(tensor1, tensor2, mask)
    # mae = compute_mae(tensor1, tensor2, mask)
    # correlation = compute_correlation(tensor1, tensor2, mask)
    ncc = compute_ncc(tensor1, tensor2, mask)

    # print(f"MSE: {mse.item()}")
    # print(f"MAE: {mae.item()}")
    # print(f"Correlation: {correlation.item()}")
    # print(f"ncc: {ncc.item()}")

    return ncc


def normalize_vector(vector):
    """
    对一维向量进行归一化，忽略NaN值
    :param vector: 输入张量，形状为 (n,)
    :return: 归一化后的向量，形状为 (n,)
    """
    # 创建掩码，标识不包含NaN的元素
    mask = ~torch.isnan(vector)

    # 过滤掉包含NaN的元素
    valid_vector = vector[mask]

    # 计算范数（长度）
    norm = torch.norm(valid_vector)

    # 避免除以零
    norm = norm if norm != 0 else 1.0

    # 归一化有效元素
    normalized_vector = valid_vector / norm

    # 创建输出张量，初始化为NaN
    result = torch.full_like(vector, float('nan'))

    # 将归一化后的元素填回到输出张量中对应的位置
    result[mask] = normalized_vector

    return result


def depth_loss_cuda(render_output, gt, normalize=False, method='min-max'):
    """
    计算网络输出和真实值之间的 RMSE，并生成一张与输入尺寸一致的图。

    Args:
        render_output (torch.Tensor): 神经网络的输出张量。
        gt (torch.Tensor): 真实值张量。
        normalize (bool): 是否对输入张量进行归一化处理。默认为 False。
        method (str): 归一化方法。'min-max' 表示最小-最大归一化，
                      'z-score' 表示均值-标准差归一化。默认为 'min-max'。

    Returns:
        torch.Tensor: 每个格点的均方根误差 (RMSE) 图。
    """

    # 归一化函数
    def min_max_normalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)

    def z_score_normalize(tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std

    # 选择是否进行归一化处理
    if normalize:
        if method == 'min-max':
            render_output = min_max_normalize(render_output)
            gt = min_max_normalize(gt)
        elif method == 'z-score':
            render_output = z_score_normalize(render_output)
            gt = z_score_normalize(gt)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    # 创建一个与输入尺寸一致的张量来存储每个格点的RMSE
    rmse_map = torch.full(render_output.shape, float('nan'), device=render_output.device)

    # 计算每个像素的RMSE
    valid_mask = ~torch.isnan(render_output) & ~torch.isnan(gt)
    if valid_mask.any():
        # 计算有效像素的RMSE
        rmse_map[valid_mask] = torch.sqrt((render_output[valid_mask] - gt[valid_mask]) ** 2)

    # 计算均方根误差 (RMSE)
    rmse = torch.sqrt(torch.mean((render_output[valid_mask] - gt[valid_mask]) ** 2))

    return rmse


def normal_consistency_loss(output_normals, gt_normals):
    """
    计算法向量的一致性损失（基于夹角差的均方误差）。

    Args:
        output_normals (torch.Tensor): 网络预测的法向量张量，形状为 (N, 3)。
        gt_normals (torch.Tensor): 真实法向量张量，形状为 (N, 3)。

    Returns:
        torch.Tensor: 每个点的角度差损失图（单位：弧度）。
        torch.Tensor: 整体的均方角度损失（Global RMS Angular Difference）。
    """
    # 确保法向量单位化
    output_normals = torch.nn.functional.normalize(output_normals, dim=1)
    gt_normals = torch.nn.functional.normalize(gt_normals, dim=1)

    # 计算法向量之间的夹角（余弦相似性）
    cos_sim = torch.sum(output_normals * gt_normals, dim=1).clamp(-1.0, 1.0)

    # 转为角度差（弧度制）
    angular_difference = torch.acos(cos_sim)

    # 计算有效区域的角度差损失
    valid_mask = ~torch.isnan(angular_difference)
    angular_difference_valid = angular_difference[valid_mask]

    # 计算整体均方角度损失
    global_loss = torch.sqrt(torch.mean(angular_difference_valid ** 2))

    return global_loss


def depth_loss_rmse_pic(render_output, gt, output_path, normalize=False, method='min-max'):
    """
    计算网络输出和真实值之间的 RMSE，并生成一张与输入尺寸一致的图。

    Args:
        render_output (torch.Tensor): 神经网络的输出张量。
        gt (torch.Tensor): 真实值张量。
        normalize (bool): 是否对输入张量进行归一化处理。默认为 False。
        method (str): 归一化方法。'min-max' 表示最小-最大归一化，
                      'z-score' 表示均值-标准差归一化。默认为 'min-max'。

    Returns:
        torch.Tensor: 每个格点的均方根误差 (RMSE) 图。
    """

    # 归一化函数
    def min_max_normalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)

    def z_score_normalize(tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std

    # 选择是否进行归一化处理
    if normalize:
        if method == 'min-max':
            render_output = min_max_normalize(render_output)
            gt = min_max_normalize(gt)
        elif method == 'z-score':
            render_output = z_score_normalize(render_output)
            gt = z_score_normalize(gt)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    # 创建一个与输入尺寸一致的张量来存储每个格点的RMSE
    rmse_map = torch.full(render_output.shape, float('nan'), device=render_output.device)

    # 计算每个像素的RMSE
    valid_mask = ~torch.isnan(render_output) & ~torch.isnan(gt)
    if valid_mask.any():
        # 计算有效像素的RMSE
        rmse_map[valid_mask] = torch.sqrt((render_output[valid_mask] - gt[valid_mask]) ** 2)

    # 计算均方根误差 (RMSE)
    rmse = torch.sqrt(torch.mean((render_output[valid_mask] - gt[valid_mask]) ** 2))

    # 计算小于0.01, 0.05, 0.1的累计像素比例
    thresholds = [0.01, 0.05, 0.1]
    ratios = {}

    for threshold in thresholds:
        count = (rmse_map < threshold).sum().item()  # 计算小于阈值的元素数量
        total_pixels = rmse_map.numel()  # 总像素数量
        ratios[threshold] = count / total_pixels  # 计算比例
    df = pd.DataFrame(ratios,index=thresholds)
    df.to_csv(output_path+'depth_rmse.csv', mode='a', header=False, index=False)

    return rmse_map.unsqueeze(0).repeat(3, 1, 1)


def depth_to_normal_back(depth_image):
    try:
        depth_map = np.array(depth_image.cpu().detach().numpy()).astype(np.float32)  # 转换为 NumPy 数组并确保为浮点数

        # 如果深度图是多通道，提取单通道
        if len(depth_map.shape) == 3:  # 多通道图像 (H, W, C)
            depth_map = depth_map[:, :, 0]  # 提取第一个通道

        # 计算 Sobel 梯度
        dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)  # x 方向梯度
        dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)  # y 方向梯度

        # 初始化法向量图
        height, width = depth_map.shape
        normal_map = np.zeros((height, width, 3), dtype=np.float32)

        # 计算法向量
        normal_map[..., 0] = -dx  # X 分量
        normal_map[..., 1] = -dy  # Y 分量
        normal_map[..., 2] = 1  # Z 分量

        # 归一化
        norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
        normal_map /= (norm + 1e-8)

        # 转换为颜色编码（可视化法向量图）
        normal_map_visual = ((normal_map + 1) / 2 * 255).astype(np.uint8)

        # 保存结果
        normal_image = Image.fromarray(normal_map_visual)
        # normal_image.show()

        transform = transforms.ToTensor()  # ToTensor() 将 [0, 255] 范围的像素值转换为 [0.0, 1.0] 浮点数范围
        normal_tensor = transform(normal_image)


        return normal_tensor
    except Exception as e:
        print(f"Error processing depth image: {e}")


def depth_to_normal(depth_tensor):
    """
    Compute normal vectors from a 2D depth map using PyTorch.

    Args:
        depth_tensor (torch.Tensor): Input 2D depth map of shape (H, W).

    Returns:
        torch.Tensor: Normal map of shape (H, W, 3) with normalized vectors.
    """
    try:
        # Ensure input is a 2D Tensor
        assert depth_tensor.ndim == 2, "Input depth map must be 2D."

        # Sobel kernel for x and y gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth_tensor.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=depth_tensor.device).unsqueeze(0).unsqueeze(0)

        # Add batch and channel dimensions: (1, 1, H, W)
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)

        # Compute gradients
        grad_x = F.conv2d(depth_tensor, sobel_x, padding=1).squeeze(0).squeeze(0)  # Gradient in x direction
        grad_y = F.conv2d(depth_tensor, sobel_y, padding=1).squeeze(0).squeeze(0)  # Gradient in y direction

        scale_factor = 1
        grad_x *= scale_factor
        grad_y *= scale_factor

        # Construct normal vectors
        normal_map = torch.zeros((depth_tensor.shape[2], depth_tensor.shape[3], 3), device=depth_tensor.device)
        normal_map[..., 0] = -grad_x  # X component
        normal_map[..., 1] = -grad_y  # Y component
        normal_map[..., 2] = 1        # Z component

        # Normalize the normal map
        norm = torch.norm(normal_map, dim=2, keepdim=True)
        normal_map = normal_map / (norm + 1e-8)  # Out-of-place division

        return normal_map.permute(2, 0, 1)
    except Exception as e:
        print(f"Error computing normal map: {e}")
        return None



