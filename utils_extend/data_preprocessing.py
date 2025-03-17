import torch
from tqdm import tqdm
import math
from sklearn.linear_model import LinearRegression
from PIL import Image
import os
from pathlib import Path
import plyfile
import torch.nn.functional as F
import numpy as np
from scipy.spatial import Delaunay
import torchvision.transforms as transforms
import cv2

from utils_extend.image_process import save_pic


def use_lidar_depth(gaussian, viewpoint_stack, dataset):
    def sliding_window_knn_impute(tensor, k=2, window_size=150, step=150):
        # 将0转换为NaN
        mask = tensor == 0
        tensor[mask] = float('nan')

        # 获取张量的形状
        rows, cols = tensor.shape

        # 创建一个对应大块分区索引的二维栅格
        grid = torch.full((rows // step + 1, cols // step + 1), float('nan'), device=tensor.device)

        # 使用tqdm添加进度条
        for i in range(0, rows, step):
            for j in range(0, cols, step):
                # 定义窗口边界
                row_start = i
                row_end = min(i + window_size, rows)
                col_start = j
                col_end = min(j + window_size, cols)

                # 提取当前窗口
                window = tensor[row_start:row_end, col_start:col_end]

                # 获取非NaN的索引
                non_nan_idx = torch.nonzero(~torch.isnan(window), as_tuple=False)
                nan_idx = torch.nonzero(torch.isnan(window), as_tuple=False)

                if non_nan_idx.shape[0] == 0:
                    continue

                # 将二维索引转换为实际坐标
                non_nan_coords = torch.stack([non_nan_idx[:, 0], non_nan_idx[:, 1]], dim=1).float()
                nan_coords = torch.stack([nan_idx[:, 0], nan_idx[:, 1]], dim=1).float()

                # 计算欧氏距离
                distances = torch.cdist(nan_coords, non_nan_coords)

                # 确保k不超过非NaN值的数量
                k_adj = min(k, non_nan_idx.shape[0])

                # 获取k个最近邻的索引
                _, indices = torch.topk(distances, k=k_adj, largest=False, dim=1)

                # 获取k个最近邻的值
                nearest_values = window[non_nan_idx[indices, 0], non_nan_idx[indices, 1]]

                # 计算平均值
                imputed_values = torch.mean(nearest_values, dim=1)

                # 填充NaN值
                window[nan_idx[:, 0], nan_idx[:, 1]] = imputed_values

                # 将窗口插值结果放回原张量
                tensor[row_start:row_end, col_start:col_end] = window

                # 计算当前窗口的平均值并存储在栅格中
                grid[i // step, j // step] = torch.nanmean(window)

                # 释放未使用的内存
                torch.cuda.empty_cache()

        # 处理全为NaN的窗口
        for i in range(0, rows, step):
            for j in range(0, cols, step):
                row_start = i
                row_end = min(i + window_size, rows)
                col_start = j
                col_end = min(j + window_size, cols)

                window = tensor[row_start:row_end, col_start:col_end]

                if torch.all(torch.isnan(window)):
                    # 使用栅格中的平均值进行赋值
                    grid_row = i // step
                    grid_col = j // step
                    neighbors = []

                    if grid_row > 0:
                        neighbors.append(grid[grid_row - 1, grid_col])
                    if grid_row < grid.shape[0] - 1:
                        neighbors.append(grid[grid_row + 1, grid_col])
                    if grid_col > 0:
                        neighbors.append(grid[grid_row, grid_col - 1])
                    if grid_col < grid.shape[1] - 1:
                        neighbors.append(grid[grid_row, grid_col + 1])

                    if neighbors:
                        mean_value = torch.tensor(neighbors).nanmean()
                        window[:] = mean_value

                    tensor[row_start:row_end, col_start:col_end] = window

        return tensor

    def create_2d_array_cuda(x, y, z, width, height, picname):
        # 输入检查
        assert torch.is_tensor(x) and torch.is_tensor(y) and torch.is_tensor(z), "x, y, z should be tensors"
        assert x.dim() == 1 and y.dim() == 1 and z.dim() == 1, "x, y, z should be 1-dimensional"
        assert x.shape == y.shape == z.shape, "x, y, z should have the same shape"
        assert x.device.type == 'cuda' and y.device.type == 'cuda' and z.device.type == 'cuda', "x, y, z should be on CUDA"

        # 确保x和y是整数类型，并且在有效范围内
        x = x.long().clamp(0, width - 1)
        y = y.long().clamp(0, height - 1)

        # 创建用于存储和和计数的张量
        sum_tensor = torch.full((height, width), float(0), device='cuda')
        count_tensor = torch.zeros((height, width), device='cuda')

        # 使用scatter_add_累加z值和计数
        index = y * width + x
        sum_tensor.view(-1).scatter_add_(0, index, z)
        count_tensor.view(-1).scatter_add_(0, index, torch.ones_like(z))

        # 计算平均值，处理 NaN 的情况
        sparse_depth_map = torch.where(count_tensor > 0, sum_tensor / count_tensor, sum_tensor)

        # interpolated_depth_map = knn_impute_batched(sparse_depth_map)
        save_pic(sparse_depth_map.repeat(3, 1, 1), dataset.source_path + '/lidar_test/' + picname)
        interpolated_depth_map = sliding_window_knn_impute(sparse_depth_map)

        return interpolated_depth_map

    def ply_to_numpy(ply_file):
        # 读取PLY文件
        plydata = plyfile.PlyData.read(ply_file)

        # 假设我们要提取顶点数据
        vertices = plydata['vertex'].data

        # 将顶点数据转换为numpy数组c
        vertex_array = np.array([(v['x'], v['y'], v['z']) for v in vertices])

        return torch.from_numpy(vertex_array).cuda()

    def get_depth_data(gaussian, viewpoint_cam, dataset):
        points_3d = ply_to_numpy(dataset.source_path + '/lidar/scan2.ply')
        # points_3d = gaussian._xyz
        height = viewpoint_cam.image_height  # 示例图像高度
        width = viewpoint_cam.image_width  # 示例图像宽度
        depth_image = viewpoint_cam.depth_image
        image_size = (height, width)
        fov_x = viewpoint_cam.FoVx
        fov_y = viewpoint_cam.FoVy
        focal_length_x = width / (2 * math.tan(fov_x / 2))
        focal_length_y = height / (2 * math.tan(fov_y / 2))
        cam_center = (width / 2, height / 2)  # 示例相机光心位置
        # 提取相机光心坐标
        cx, cy = cam_center
        R = viewpoint_cam.R  # 示例旋转矩阵
        T = viewpoint_cam.T  # 示例平移向量

        # 确保输入数据类型为 Tensor 并在同一设备上
        device = depth_image.device
        points_3d = torch.tensor(points_3d, dtype=torch.float32).to(device)
        R = torch.tensor(R, dtype=torch.float32).to(device)
        T = torch.tensor(T, dtype=torch.float32).to(device)

        # 将点云从世界坐标系转换到相机坐标系
        points_camera = torch.matmul(points_3d, R.T) + T

        # 投影到图像平面
        x = points_camera[:, 0] / points_camera[:, 2]
        y = points_camera[:, 1] / points_camera[:, 2]
        z = points_camera[:, 2]  # 获取z坐标

        # 转换为像素坐标
        u = focal_length_x * x + cx
        v = focal_length_y * y + cy

        valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z > 0)
        ix = u[valid_mask].long()
        iy = v[valid_mask].long()
        sparse_depth = z[valid_mask]

        interpolated_depth_map = create_2d_array_cuda(ix, iy, sparse_depth, width, height, viewpoint_cam.image_name)
        return interpolated_depth_map

    def get_depth_data_new(viewpoint_cam, dataset):
        points_3d = ply_to_numpy(dataset.source_path + '/lidar/lidar.ply')
        height = viewpoint_cam.image_height
        width = viewpoint_cam.image_width
        depth_image = viewpoint_cam.depth_image
        image_size = (height, width)
        fov_x = viewpoint_cam.FoVx
        fov_y = viewpoint_cam.FoVy
        focal_length_x = width / (2 * math.tan(fov_x / 2))
        focal_length_y = height / (2 * math.tan(fov_y / 2))
        cam_center = (width / 2, height / 2)

        cx, cy = cam_center
        R = viewpoint_cam.R
        T = viewpoint_cam.T

        # 确保数据在CUDA上
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将点云从世界坐标系转换到相机坐标系
        points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # 增加齐次坐标
        points_camera = (R @ points_3d_homogeneous.T).T + T  # 旋转和平移

        # 投影到图像平面
        x = points_camera[:, 0] / points_camera[:, 2]
        y = points_camera[:, 1] / points_camera[:, 2]
        z = points_camera[:, 2]  # 获取z坐标

        # 转换为像素坐标
        u = focal_length_x * x + cx
        v = focal_length_y * y + cy

        valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z > 0)
        u = u[valid_mask].long()
        v = v[valid_mask].long()
        z = z[valid_mask]

        # 深度图初始化
        depth_image = torch.zeros(image_size, dtype=torch.float32, device=device)
        depth_image[v, u] = z

        # 将3D点转换为相机坐标系
        points_cam = torch.matmul(R, points_3d.T).T + T

        # 投影到2D图像平面
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        u = (focal_length_x * (x / z)) + cx
        v = (focal_length_y * (y / z)) + cy

        # 将像素坐标转换为整数索引并过滤出有效范围内的点
        valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z > 0)
        u = u[valid_mask].long()
        v = v[valid_mask].long()
        z = z[valid_mask]

        # 深度图初始化
        depth_image = torch.zeros(image_size, dtype=torch.float32, device=device)
        depth_image[v, u] = z

        # 获取投影点的2D坐标
        points_2d = torch.stack([u, v], dim=1).cpu().numpy()

        # 执行Delaunay三角剖分
        tri = Delaunay(points_2d)

        # 对每个三角形，计算平面参数
        for simplex in tri.simplices:
            pts = points_cam[simplex]
            # 使用SVD计算平面参数
            A = pts - pts[0]  # 构造平面方程
            normal = torch.cross(A[1], A[2])
            normal = normal / torch.norm(normal)
            w = -torch.dot(normal, pts[0])

            # 将平面方程参数保存在CUDA设备上
            plane_params = torch.cat([normal, torch.tensor([w], device=device)])

            # 对应的图像平面上的坐标
            u_tri = u[simplex]
            v_tri = v[simplex]

            # 将初始深度赋给对应的像素
            for i in range(3):
                depth_image[v_tri[i], u_tri[i]] = z[simplex[i]]

        return depth_image

    def get_depth_data_cuda_test(gaussian, viewpoint_camera, dataset):

        points_3d = ply_to_numpy(dataset.source_path + '/lidar/scan2.ply')

        # 相机参数
        height = viewpoint_cam.image_height  # 图像高度
        width = viewpoint_cam.image_width  # 图像宽度
        fov_x = viewpoint_cam.FoVx
        fov_y = viewpoint_cam.FoVy
        focal_length_x = width / (2 * math.tan(fov_x / 2))
        focal_length_y = height / (2 * math.tan(fov_y / 2))
        cx = width / 2
        cy = height / 2

        R = torch.tensor(viewpoint_cam.R, dtype=torch.float32).cuda()
        T = torch.tensor(viewpoint_cam.T, dtype=torch.float32).cuda()

        X_c = torch.matmul(points_3d, R.T) + T  # 形状 (N, 3)

        # 提取相机坐标系下的 X, Y, Z 分量
        X = X_c[:, 0]
        Y = X_c[:, 1]
        Z = X_c[:, 2]

        # 确保Z的值大于0
        valid_depth_mask = Z > 0

        # 应用有效深度掩码到所有相关变量
        X = X[valid_depth_mask]
        Y = Y[valid_depth_mask]
        Z = Z[valid_depth_mask]

        # 防止除以零
        epsilon = 1e-6
        Z = Z + epsilon

        # 计算归一化的图像坐标
        x_normalized = X / Z
        y_normalized = Y / Z

        # 应用相机内参，计算像素坐标
        u = x_normalized * focal_length_x + cx
        v = y_normalized * focal_length_y + cy

        # 将像素坐标堆叠成形状为 (N, 2) 的张量
        pixel_coords = torch.stack([u, v], dim=1)

        # 可选：将像素坐标限制在图像尺寸内
        pixel_coords[:, 0].clamp_(0, width - 1)
        pixel_coords[:, 1].clamp_(0, height - 1)

        # pixel_coords 即为每个 3D 点在图像上的像素坐标
        # Round pixel coordinates to the nearest integer (since pixel indices are integers)
        u = pixel_coords[:, 0].long().clamp(0, width - 1)  # X-coordinates (u)
        v = pixel_coords[:, 1].long().clamp(0, height - 1)  # Y-coordinates (v)

        # Initialize Z-buffer with inf for each pixel (size: [height, width])
        z_buffer = torch.full((height, width), float('inf')).cuda()

        # Use advanced indexing to update the z_buffer with the closest depths
        # Only update where the current depth is less than the stored depth
        mask = Z < z_buffer[v, u]  # Find where the new points are closer
        z_buffer[v[mask], u[mask]] = Z[mask]  # Update z-buffer with the smaller depth values

        # The final depth map is now contained in z_buffer (where each pixel has the closest depth)
        depth_map = z_buffer

        # Ensure depth_map is on CPU and convert to NumPy array
        depth_map[depth_map == float('inf')] = 0
        depth_map_np = depth_map.cpu().numpy()

        # Replace zeros or infinite values with NaN for better visualization
        depth_map_np[depth_map_np == 0] = np.nan
        depth_map_np[depth_map_np == float('inf')] = np.nan

        # Optionally, normalize the depth values for better contrast
        min_depth = np.nanmin(depth_map_np)
        max_depth = np.nanmax(depth_map_np)
        depth_map_normalized = (depth_map_np - min_depth) / (max_depth - min_depth)

        # Plot the depth map
        plt.figure(figsize=(10, 8))
        plt.imshow(depth_map_normalized, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Normalized Depth')
        plt.title(viewpoint_camera.image_name)
        plt.axis('off')  # Hide axis ticks and labels
        plt.show()

    for viewpoint_cam in tqdm(viewpoint_stack, desc="Processing viewpoints"):
        # interpolated_depth_map = get_depth_data_cuda_test(gaussian, viewpoint_cam, dataset)
        # interpolated_depth_map = get_depth_data(gaussian,viewpoint_cam,dataset)
        # interpolated_depth_map = get_depth_data_new(gaussian,viewpoint_cam,dataset)
        depth_image_path = dataset.source_path + '/lidar_dense/' + viewpoint_cam.image_name + '.png'
        depth_image = Image.open(depth_image_path)
        depth_image_tensor = transforms.ToTensor()(depth_image)

        viewpoint_cam.depth_image = depth_image_tensor
        viewpoint_cam.normal_image = depth_to_normal(depth_image)

        # print(viewpoint_cam.depth_image)

        # try:
        #     # depth_map_data=interpolated_depth_map.unsqueeze(0).repeat(3, 1, 1)
        #     depth_image_path=dataset.source_path+'/lidar/render_depth_image_'+viewpoint_cam.image_name
        #     depth_image = Image.open(depth_image_path)
        #     viewpoint_cam.depth_image = depth_image
        #     print(viewpoint_cam.depth_image)
        #     # save_pic(depth_map_data,dataset.source_path+'/lidar/'+viewpoint_cam.image_name)
        #     # 释放未使用的内存
        #     # torch.cuda.empty_cache()
        # except:
        #     print(viewpoint_cam.image_name)

    return True


def use_fusion_depth(gaussian, viewpoint_stack, dataset):
    for viewpoint_cam in tqdm(viewpoint_stack, desc="Processing viewpoints"):
        depth_image_path = dataset.source_path + '/fusion/' + viewpoint_cam.image_name + '.JPG'
        depth_image = Image.open(depth_image_path)
        depth_image_tensor = transforms.ToTensor()(depth_image)
        viewpoint_cam.depth_image = depth_image_tensor.repeat(3, 1, 1)
        viewpoint_cam.normal_image = depth_to_normal(depth_image)

        # print(viewpoint_cam.depth_image)

        # try:
        #     # depth_map_data=interpolated_depth_map.unsqueeze(0).repeat(3, 1, 1)
        #     depth_image_path=dataset.source_path+'/lidar/render_depth_image_'+viewpoint_cam.image_name
        #     depth_image = Image.open(depth_image_path)
        #     viewpoint_cam.depth_image = depth_image
        #     print(viewpoint_cam.depth_image)
        #     # save_pic(depth_map_data,dataset.source_path+'/lidar/'+viewpoint_cam.image_name)
        #     # 释放未使用的内存
        #     # torch.cuda.empty_cache()
        # except:
        #     print(viewpoint_cam.image_name)

    return True


def fit_the_depth(gaussian, viewpoint_stack):
    def get_data(viewpoint_cam):
        points_3d = gaussian._xyz
        height = viewpoint_cam.image_height  # 示例图像高度
        width = viewpoint_cam.image_width  # 示例图像宽度
        depth_image = viewpoint_cam.depth_image
        image_size = (height, width)
        fov_x = viewpoint_cam.FoVx
        fov_y = viewpoint_cam.FoVy
        focal_length_x = width / (2 * math.tan(fov_x / 2))
        focal_length_y = height / (2 * math.tan(fov_y / 2))
        cam_center = (width / 2, height / 2)  # 示例相机光心位置
        # 提取相机光心坐标
        cx, cy = cam_center
        R = viewpoint_cam.R  # 示例旋转矩阵
        T = viewpoint_cam.T  # 示例平移向量

        # 确保输入数据类型为 Tensor 并在同一设备上
        device = points_3d.device
        R = torch.tensor(R, dtype=torch.float32).to(device)
        T = torch.tensor(T, dtype=torch.float32).to(device)

        # 将三维点变换到相机坐标系
        points_cam = torch.matmul(points_3d, R.T) + T

        # 投影到图像平面
        x = (points_cam[:, 0] * focal_length_x / points_cam[:, 2]) + cx
        y = (points_cam[:, 1] * focal_length_y / points_cam[:, 2]) + cy

        # 转换为整数索引
        ix = x.round().long()
        iy = y.round().long()

        # 创建有效点掩码
        valid_mask = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)

        ix = ix[valid_mask].cpu().tolist()
        iy = iy[valid_mask].cpu().tolist()
        sparse_depth = points_cam[:, 2][valid_mask].cpu().tolist()
        iz_de = depth_image[0, :, :, ].cpu().todolist()
        monocular_depth = [iz_de[y][x] for x, y in zip(ix, iy)]

        return sparse_depth, monocular_depth, iz_de

    def fit_and_transform(sparse_depth, monocular_depth, depth_image_cpu, viewpoint_cam):
        # 使用线性回归拟合转换关系
        reg = LinearRegression().fit(np.array(monocular_depth).reshape(-1, 1), sparse_depth)
        confidence = reg.score(np.array(monocular_depth).reshape(-1, 1), sparse_depth)

        # 使用拟合的模型进行转换
        # Assuming depth_image_cpu is a numpy array
        # Convert the list to a NumPy array
        depth_image_cpu = np.array(depth_image_cpu)
        shape = depth_image_cpu.shape
        # Now you can use NumPy's indexing syntax
        depth_image_reshaped = depth_image_cpu.flatten().reshape(-1, 1)

        # Predict using the reshaped data
        converted_data = reg.predict(depth_image_reshaped)
        # Convert the 2D list to a NumPy array
        depth_image_array = np.array(converted_data).reshape(shape)
        viewpoint_cam.depth_image = torch.from_numpy(depth_image_array).cuda().unsqueeze(0).repeat(3, 1, 1)

        return True

    for viewpoint_cam in tqdm(viewpoint_stack, desc="Processing viewpoints"):
        sparse_depth, monocular_depth, depth_image_cpu = get_data(viewpoint_cam)
        try:
            fit_and_transform(sparse_depth, monocular_depth, depth_image_cpu, viewpoint_cam)
        except:
            print(viewpoint_cam.name)


def mean_radii(gaussian, viewpoint_stack):
    def get_radii(viewpoint_cam):
        points_3d = gaussian._xyz
        height = viewpoint_cam.image_height  # 示例图像高度
        width = viewpoint_cam.image_width  # 示例图像宽度
        image_size = (height, width)
        fov_x = viewpoint_cam.FoVx
        fov_y = viewpoint_cam.FoVy
        focal_length_x = width / (2 * math.tan(fov_x / 2))
        focal_length_y = height / (2 * math.tan(fov_y / 2))
        cam_center = (width / 2, height / 2)  # 示例相机光心位置
        R = viewpoint_cam.R  # 示例旋转矩阵
        T = viewpoint_cam.T  # 示例平移向量

        # 确保输入数据类型为 Tensor 并在同一设备上
        device = points_3d.device
        R = torch.tensor(R, dtype=torch.float32).to(device)
        T = torch.tensor(T, dtype=torch.float32).to(device)

        # 提取相机光心坐标
        cx, cy = cam_center

        # 初始化深度图和点数量统计图
        height, width = image_size

        # 将三维点变换到相机坐标系
        points_cam = torch.matmul(points_3d, R.T) + T
        x = (points_cam[:, 0] * focal_length_x / points_cam[:, 2]) + cx
        y = (points_cam[:, 1] * focal_length_y / points_cam[:, 2]) + cy

        # 计算平均深度 Z
        Z_avg = points_cam[:, 2].mean()

        # 假设焦距为 f（这里取焦距的平均）
        f_avg = (focal_length_x + focal_length_y) / 2

        # 计算需要的球体半径 R
        R = Z_avg / f_avg

        # 计算深度的范围
        max_val = torch.max(points_cam[:, 2])
        min_val = torch.min(points_cam[:, 2])

        return R, torch.tensor([max_val, min_val])

    mean_radii = []
    depth_ranges = torch.empty((len(viewpoint_stack), 2))
    i = 0

    for viewpoint_cam in viewpoint_stack:
        R, depth_range = get_radii(viewpoint_cam)
        mean_radii.append(R)
        depth_ranges[i] = depth_range
        i += 1

    extent = [gaussian._xyz[:, 0].min(), gaussian._xyz[:, 0].max(), gaussian._xyz[:, 1].min(),
              gaussian._xyz[:, 1].max(), gaussian._xyz[:, 2].min(), gaussian._xyz[:, 2].max()]
    return max(mean_radii), depth_ranges, extent


import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def visualize_points(xyzs, combined_points):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot points before processing
    axs[0].scatter(xyzs[:, 0], xyzs[:, 1], c='b', label='Original Points')
    axs[0].set_title('Before Processing')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].legend()

    # Plot points after processing
    axs[1].scatter(combined_points[:, 0], combined_points[:, 1], c='r', label='Combined Points')
    axs[1].set_title('After Processing')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def Generate_random_test_data():
    # Generate random XYZ coordinates
    num_points = 1000  # Number of points
    xyzs = np.random.uniform(low=0, high=100,
                             size=(num_points, 3))  # Generating random XYZ coordinates between 0 and 100

    # Generate random RGB colors
    rgbs = np.random.randint(low=0, high=256, size=(num_points, 3))  # Generating random RGB colors

    # Generate random errors
    errors = np.random.uniform(low=0, high=1, size=(num_points, 1))  # Generating random errors between 0 and 1
    return xyzs, rgbs, errors


def process_weakly_textured_areas(xyzs, rgbs, errors):
    """
    Identify sparsely populated areas in a 3D point cloud and generate new points to improve density.

    Args:
        xyzs (numpy.ndarray): An (N, 3) array of 3D points.
        rgbs (numpy.ndarray): An (N, 3) array of RGB colors corresponding to the points.
        errors (numpy.ndarray): An (N, 1) array of error values corresponding to the points.

    Returns:
        combined_points (numpy.ndarray): The updated array of 3D points including new points.
        combined_colors (numpy.ndarray): The updated array of RGB colors including colors for new points.
        combined_errors (numpy.ndarray): The updated array of error values including errors for new points.
    """
    # Perform DBSCAN clustering to remove outliers
    dbscan = DBSCAN(eps=0.03, min_samples=10)
    labels = dbscan.fit_predict(xyzs)

    # Remove outliers (label == -1)
    mask = labels != -1
    xyzs = xyzs[mask]
    rgbs = rgbs[mask]
    errors = errors[mask]

    x, y, z = xyzs.T  # Extract x, y, z coordinates

    # Create a KD-tree for quick nearest-neighbor lookup
    tree = cKDTree(xyzs[:, :2])

    # Define the rectangle bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Define the grid resolution
    grid_size = 0.02
    x_bins = np.arange(x_min, x_max, grid_size)
    y_bins = np.arange(y_min, y_max, grid_size)

    # Calculate the histogram of points
    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Find cells with a density less than the threshold
    threshold = 1  # Less than 1 point per cell
    sparse_x, sparse_y = np.where(histogram < threshold)

    # Generate new points in sparse areas
    random_x = np.random.uniform(size=len(sparse_x))
    random_y = np.random.uniform(size=len(sparse_y))

    new_points_x = x_edges[sparse_x] + random_x * grid_size
    new_points_y = y_edges[sparse_y] + random_y * grid_size

    # Find the nearest neighbors in the original point cloud to get z-values and other attributes
    idx = tree.query(list(zip(new_points_x, new_points_y)))[1]

    # Combine the new points with the original points
    new_points_z = z[idx]
    combined_points = np.vstack((xyzs, np.column_stack((new_points_x, new_points_y, new_points_z))))
    combined_colors = np.vstack((rgbs, rgbs[idx]))
    combined_errors = np.vstack((errors, errors[idx]))
    visualize_points(xyzs, combined_points)

    return combined_points, combined_colors, combined_errors


def depth_to_normal(depth_image):
    try:
        depth_map = np.array(depth_image).astype(np.float32)  # 转换为 NumPy 数组并确保为浮点数

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

        return normal_image
    except Exception as e:
        print(f"Error processing depth image: {e}")


def load_depth_image(image_path):
    def remove_last_two_parts(file_path):
        path = Path(file_path)
        return str(path.parents[1])  # 获取倒数第二个父目录

    def get_depth_image(data_path):
        has_depth = 'depth' in os.listdir(data_path)
        has_lidar = 'lidar' in os.listdir(data_path)

        if has_depth and has_lidar:

            depth_image_path = image_path.replace('images', 'depth')
            depth_image = Image.open(depth_image_path)

        elif has_depth:
            depth_image_path = image_path.replace('images', 'depth')
            depth_image = Image.open(depth_image_path)

        elif has_lidar:
            depth_image_path = image_path.replace('images', 'depth')
            depth_image = Image.open(depth_image_path)
        else:
            depth_image = None

        return depth_image

    data_path = remove_last_two_parts(image_path)
    depth_image = get_depth_image(data_path)
    normal_image=depth_to_normal(depth_image)
    # normal_image.show()

    transform = transforms.ToTensor()  # ToTensor() 将 [0, 255] 范围的像素值转换为 [0.0, 1.0] 浮点数范围
    normal_tensor = transform(normal_image)
    return depth_image
