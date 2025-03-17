import torch
import math
import logging

def depth_from_points_backup(gaussian, viewpoint_stack,depth_range_tensor,data_extent):
    def project_points_to_image(viewpoint_cam,depth_range):
        """
        将三维点投影到二维图像平面并生成深度图和点数量统计图。

        参数：
        - viewpoint_cam: 包含相机参数和图像大小的对象

        返回：
        - num_adopted: 被采用的点数量
        - num_not_adopted: 未被采用的点数量
        """
        # 确保输入数据类型为 Tensor 并在同一设备上
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


        device = points_3d.device
        R = torch.tensor(R, dtype=torch.float32).to(device)
        T = torch.tensor(T, dtype=torch.float32).to(device)

        # 提取相机光心坐标
        cx, cy = cam_center

        # 初始化深度图
        # depth_map = torch.full(image_size, float('inf'), device=device)
        depth_map = viewpoint_cam.depth_image[0]

        # 分块处理三维点云
        chunk_size = 100000  # 每块处理的点数量
        num_points = points_3d.shape[0]
        num_chunks = (num_points + chunk_size - 1) // chunk_size  # 计算块的数量

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_points)

            chunk_points = points_3d[start:end]
            points_cam = torch.matmul(chunk_points, R.T) + T

            # 投影到图像平面
            x = (points_cam[:, 0] * focal_length_x / points_cam[:, 2]) + cx
            y = (points_cam[:, 1] * focal_length_y / points_cam[:, 2]) + cy


            # 转换为整数索引
            ix = x.round().long()
            iy = y.round().long()

            # 创建有效点掩码
            valid_mask = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height) & (points_cam[:, 2]> depth_range[0]) & (points_cam[:, 2]< depth_range[1])

            ix = ix[valid_mask]
            iy = iy[valid_mask]
            depth = points_cam[:, 2][valid_mask]

            # 获取原始索引
            original_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()

            # 将 ix, iy 转换为二维张量的索引
            indices = iy * width + ix

            # 更新深度图
            depth_map_flat = depth_map.view(-1)
            existing_depths = depth_map_flat.index_select(0, indices)
            new_depths = torch.min(existing_depths, depth)

            update_mask = new_depths < existing_depths

            depth_map_flat.index_put_((indices[update_mask],), new_depths[update_mask], accumulate=False)

            # 释放未使用的内存
            torch.cuda.empty_cache()

        # 获取最终的深度图
        final_depths = depth_map.view(-1).index_select(0, indices)

        adopted_mask = final_depths == depth
        not_adopted_indices = original_indices[~adopted_mask]

        return not_adopted_indices

    all_not_adopted_indices = torch.tensor([], dtype=torch.long, device=gaussian._xyz.device)
    all_indices = torch.arange(gaussian._xyz.shape[0], device=gaussian._xyz.device)

    i=0
    for viewpoint_cam in viewpoint_stack:
        not_adopted_indices = project_points_to_image(viewpoint_cam,depth_range_tensor[i])
        all_not_adopted_indices = torch.cat((all_not_adopted_indices, not_adopted_indices))
        

    all_not_adopted_indices = torch.unique(all_not_adopted_indices)
    # never_adopted_indices = torch.tensor(list(set(all_indices.cpu().numpy()) -
    #                                             set(all_not_adopted_indices.cpu().numpy())),
    #                                         dtype=torch.long, device=gaussian._xyz.device)

    mask = torch.zeros(gaussian._xyz.shape[0], dtype=torch.bool, device=gaussian._xyz.device)
    mask[all_not_adopted_indices] = True

    # gaussian.prune_points(mask)
    gaussian._opacity[mask] = 0

    num_removed = mask.sum().item()
    print(f"Removed {num_removed} points out of {gaussian._xyz.shape[0]}")

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # data_numpy = gaussian._xyz.cpu().numpy()
    # # 使用DBSCAN进行聚类
    # dbscan = DBSCAN(eps=0.2, min_samples=5)
    # labels = dbscan.fit_predict(data_numpy)
    #
    # # 创建一个布尔数组，异常点被标记为True
    # outlier_mask = labels == -1
    #
    # # gaussian.prune_points(outlier_mask)
    # gaussian._opacity[outlier_mask] = 0

    #
    # out_of_range_mask = ((gaussian._xyz[:,0]>data_extent[0])&(gaussian._xyz[:,0]<data_extent[1])
    #                         &(gaussian._xyz[:,1]>data_extent[2])&(gaussian._xyz[:,1]<data_extent[3])
    #                         &(gaussian._xyz[:,2]>data_extent[4])&(gaussian._xyz[:,2]<data_extent[5]))

    # gaussian._opacity[~out_of_range_mask] = 0
    # 释放未使用的内存
    torch.cuda.empty_cache()

def depth_from_points(gaussian, viewpoint_stack):
    """
    筛选出在所有给定视图中都没有出现的高斯椭球，并将其不透明度设置为0。

    参数:
    gaussian: 包含点云数据的对象，必须有 _xyz 和 _opacity 属性
    viewpoint_stack: 包含多个视点相机参数的列表

    返回:
    None，直接修改 gaussian 对象的 _opacity 属性
    """
    
    def project_points_to_image(viewpoint_cam):
        """
        将三维点投影到二维图像平面并识别被采用的点。

        参数:
        viewpoint_cam: 包含相机参数和图像大小的对象

        返回:
        adopted_indices: 被该视图采用的点的索引
        """
        points_3d = gaussian._xyz
        height, width = viewpoint_cam.image_height, viewpoint_cam.image_width
        focal_length_x = width / (2 * math.tan(viewpoint_cam.FoVx / 2))
        focal_length_y = height / (2 * math.tan(viewpoint_cam.FoVy / 2))
        cx, cy = width / 2, height / 2

        device = points_3d.device
        R = torch.tensor(viewpoint_cam.R, dtype=torch.float32, device=device)
        T = torch.tensor(viewpoint_cam.T, dtype=torch.float32, device=device)

        depth_map = viewpoint_cam.depth_image*0.9

        chunk_size = 100000
        num_points = points_3d.shape[0]
        adopted_indices = []

        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            chunk_points = points_3d[start:end]
            
            points_cam = torch.matmul(chunk_points, R.T) + T
            x = (points_cam[:, 0] * focal_length_x / points_cam[:, 2]) + cx
            y = (points_cam[:, 1] * focal_length_y / points_cam[:, 2]) + cy

            valid_mask = (
                (x >= 0) & (x < width) & 
                (y >= 0) & (y < height) & 
                (points_cam[:, 2] > depth_map.min()) & 
                (points_cam[:, 2] < depth_map.max())
            )

            ix = x[valid_mask].round().long()
            iy = y[valid_mask].round().long()
            depth = points_cam[:, 2][valid_mask]

            # 获取原始索引
            original_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()

            indices = iy * width + ix
            # 更新深度图
            depth_map_flat = depth_map.view(-1)
            existing_depths = depth_map_flat.index_select(0, indices)
            new_depths = torch.min(existing_depths, depth)

            update_mask = new_depths < existing_depths

            depth_map_flat.index_put_((indices[update_mask],), new_depths[update_mask], accumulate=False)

            adopted_indices.extend(start + original_indices[update_mask])

            # 释放未使用的内存
            torch.cuda.empty_cache()

        return torch.tensor(adopted_indices, device=device, dtype=torch.long)

    if not viewpoint_stack:
        raise ValueError("viewpoint_stack cannot be empty")
    if not hasattr(gaussian, '_xyz') or not hasattr(gaussian, '_opacity'):
        raise AttributeError("gaussian object must have _xyz and _opacity attributes")

    all_adopted_points = torch.tensor([], dtype=torch.long, device=gaussian._xyz.device)
    all_indices = torch.arange(gaussian._xyz.shape[0], device=gaussian._xyz.device)

    for viewpoint_cam in viewpoint_stack:
        adopted_indices = project_points_to_image(viewpoint_cam)
        all_adopted_points = torch.cat((all_adopted_points, adopted_indices))

    all_adopted_points = torch.unique(all_adopted_points)
    never_adopted_mask = ~torch.isin(all_indices, all_adopted_points)

    gaussian._opacity[never_adopted_mask] = 0

    num_removed = never_adopted_mask.sum().item()
    logging.info(f"Removed {num_removed} points out of {gaussian._xyz.shape[0]}")

    torch.cuda.empty_cache()

def remove_low_contrib_gaussian(gaussians, viewpoint_stack,render,pipe, bg,contribrange):
    # unique_tensor = torch.unique(contrib_map)
    #
    # print(unique_tensor.size())
    result_tensor = None

    for viewpoint_cam in viewpoint_stack:
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        contrib_map = render_pkg["contrib_map"]

        if result_tensor is None:
            result_tensor = contrib_map.clone()  # 第一次复制 contrib_map
        else:
            result_tensor += contrib_map  # 后续累加

    # 计算5%分位数
    result_tensor = result_tensor.float()
    threshold = torch.quantile(result_tensor, contribrange)

    # 创建掩码：标记所有小于等于阈值的元素
    mask = result_tensor <= threshold

    return mask


