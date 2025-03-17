import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def initialize_voxels(points):
    """
    初始化点云的体素结构，并标记存在2个以上点的体素
    :param points: tensor(n, 3) 形状的点云数据
    :param voxel_size: 体素的大小
    :return: 一个包含体素索引及其标记的字典
    """
    extent = [points[:, 0].min(), points[:, 0].max(),
              points[:, 1].min(), points[:, 1].max(),
              points[:, 2].min(), points[:, 2].max()]

    # 计算每个维度的长度
    lengths = [extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4]]

    # 找到最长边
    longest_edge = max(lengths)

    # 取最长边的1/10
    voxel_size = longest_edge / 10

    # 确定点云的边界
    min_coords = torch.min(points, dim=0)[0]

    # 将点云数据映射到体素索引
    voxel_indices = torch.floor((points - min_coords) / voxel_size).int()

    # 找到每个体素索引及其出现次数
    unique_voxel_indices, counts = torch.unique(voxel_indices, return_counts=True, dim=0)

    # 标记存在2个以上点的体素为1，其余为0
    voxel_labels = {tuple(voxel_idx.tolist()): 1 if count >= 2 else 0
                    for voxel_idx, count in zip(unique_voxel_indices, counts)}

    return voxel_labels, min_coords, voxel_size

def plot_voxels_and_points(voxel_labels, min_coords, voxel_size, points):
    """
    可视化体素和三维点云，并为标记为1和0的体素分别绘制不同颜色的立方体边框
    :param voxel_labels: 体素索引及其标记的字典
    :param min_coords: 最小坐标，用于绘图对齐
    :param voxel_size: 体素的大小
    :param points: 三维点云数据
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维点云
    ax.scatter(points[:, 0].numpy(), points[:, 1].numpy(), points[:, 2].numpy(), color='green', s=1, label='Points')

    # 遍历体素并绘制边框
    for voxel_idx, label in voxel_labels.items():
        # 获取体素中心坐标
        voxel_center = np.array(voxel_idx) * voxel_size + min_coords.numpy()

        # 计算体素的八个顶点
        r = voxel_size / 2
        x = [voxel_center[0] - r, voxel_center[0] + r]
        y = [voxel_center[1] - r, voxel_center[1] + r]
        z = [voxel_center[2] - r, voxel_center[2] + r]

        # 顶点坐标
        verts = [[x[0], y[0], z[0]], [x[0], y[1], z[0]], [x[1], y[1], z[0]], [x[1], y[0], z[0]],
                 [x[0], y[0], z[1]], [x[0], y[1], z[1]], [x[1], y[1], z[1]], [x[1], y[0], z[1]]]

        # 定义体素的12条边线
        edges = [
            [verts[0], verts[1]], [verts[1], verts[2]], [verts[2], verts[3]], [verts[3], verts[0]],  # 底面
            [verts[4], verts[5]], [verts[5], verts[6]], [verts[6], verts[7]], [verts[7], verts[4]],  # 顶面
            [verts[0], verts[4]], [verts[1], verts[5]], [verts[2], verts[6]], [verts[3], verts[7]]  # 侧面
        ]

        # 根据体素类型选择边框颜色，1 为红色，0 为绿色
        color = 'red' if label == 1 else 'green'

        # 绘制每条边
        for edge in edges:
            ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], [edge[0][2], edge[1][2]], color=color,
                    linewidth=1)

    # 设置坐标轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.legend()
    plt.show()

def generate_random_pyramid_points(num_points):
    """
    生成随机的四棱锥点云
    :param num_points: 生成的点数
    :return: 随机的四棱锥点云 tensor(num_points, 3)
    """
    # 四棱锥的底部为正方形，顶点位于 (0, 0, height)
    height = 2.0  # 锥顶的高度
    base_size = 2.0  # 正方形底面的边长

    # 随机生成在正方形底面上的点
    base_points = torch.rand(num_points, 2) * base_size - base_size / 2

    # 随机生成 z 轴坐标，使点从底面逐渐靠近锥顶
    z_coords = torch.rand(num_points)  # z 坐标范围 [0, 1]

    # 使用 z 坐标将点向锥顶 (0, 0, height) 方向收缩
    x_coords = base_points[:, 0] * (1 - z_coords)
    y_coords = base_points[:, 1] * (1 - z_coords)
    z_coords = z_coords * height  # 拉伸 z 坐标

    # 合并 x, y, z 坐标，得到三维点云
    points = torch.stack([x_coords, y_coords, z_coords], dim=1)

    return points

def compare_points_with_voxels(new_points, initial_voxel_labels, min_coords, voxel_size):
    """
    统计新点云中位于标记为1的初始体素中的点的比例（在CUDA上执行）
    :param new_points: 新的点云数据 tensor(n, 3)
    :param initial_voxel_labels: 初始体素索引的字典
    :param min_coords: 初始体素的最小坐标
    :param voxel_size: 体素的大小
    :return: 新点云中位于标记为1的初始体素中的比例
    """
    # 将标记为1的初始体素转换为张量（CUDA 上执行）
    label1_voxel_list = torch.tensor([list(key) for key, label in initial_voxel_labels.items() if label == 1],
                                     device=new_points.device)

    # 将新的点云映射到体素索引
    new_voxel_indices = torch.floor((new_points - min_coords) / voxel_size).int()

    # 比较新点云中的体素索引是否存在于标记为1的体素中
    if label1_voxel_list.numel() == 0:
        # 如果没有标记为1的体素，直接返回比例为 0
        return 1.0

    # 找出新点云中位于标记为1的体素中的点
    new_voxel_indices = new_voxel_indices.unsqueeze(1)  # (n, 1, 3)
    label1_voxel_list = label1_voxel_list.unsqueeze(0)  # (1, m, 3)

    # 对每个点与所有体素标签进行逐元素比较
    matches = (new_voxel_indices == label1_voxel_list).all(dim=2)  # (n, m)

    # 统计新点云中位于标记为1的体素中的点数
    count_in_label1_voxels = matches.any(dim=1).sum().item()

    # 计算比例
    proportion = count_in_label1_voxels / new_points.shape[0]
    return 1 - proportion


def get_mask_for_points_outside_voxels(gaussian, initial_voxel_labels, min_coords, voxel_size):
    """
    挑选出所有不在标记为1的体素中的点，并返回一个布尔掩码（在CUDA上执行）

    :param gaussian: 新的点云数据集
    :param initial_voxel_labels: 初始体素索引的字典
    :param min_coords: 初始体素的最小坐标
    :param voxel_size: 体素的大小
    :return: 不在标记为1的体素中的点的布尔掩码 tensor(n,)
    """
    # 将标记为1的初始体素转换为张量（CUDA 上执行）
    new_points=gaussian._xyz
    label1_voxel_list = torch.tensor([list(key) for key, label in initial_voxel_labels.items() if label == 1],
                                     device=new_points.device)

    # 将新的点云映射到体素索引
    new_voxel_indices = torch.floor((new_points - min_coords) / voxel_size).int()

    # 比较新点云中的体素索引是否存在于标记为1的体素中
    if label1_voxel_list.numel() == 0:
        # 如果没有标记为1的体素，直接返回一个全 True 的掩码（即所有点都不在体素中）
        return torch.ones(new_points.shape[0], dtype=torch.bool, device=new_points.device)

    # 找出新点云中位于标记为1的体素中的点
    new_voxel_indices = new_voxel_indices.unsqueeze(1)  # (n, 1, 3)
    label1_voxel_list = label1_voxel_list.unsqueeze(0)  # (1, m, 3)

    # 对每个点与所有体素标签进行逐元素比较
    matches = (new_voxel_indices == label1_voxel_list).all(dim=2)  # (n, m)

    # 生成布尔掩码，表示不在标记为1的体素中的点
    mask_outside_voxels = ~matches.any(dim=1)  # (n,)

    gaussian.prune_points(mask_outside_voxels)

    # return mask_outside_voxels

def update_voxel_labels(points, voxel_labels, min_coords, voxel_size):
    """
    根据点的数量更新体素标签：
    - 如果体素的 label 为 2 或者没有 label，且点数大于 5，则更新为 0
    - 如果 label 为 0 且点数大于 10，则更新为 1

    :param points: 新的点云数据 tensor(n, 3)
    :param voxel_labels: 当前的体素索引及其标签的字典
    :param min_coords: 最小坐标，用于体素索引的计算
    :param voxel_size: 体素的大小
    :return: 更新后的 voxel_labels
    """
    # 将点云数据映射到体素索引
    voxel_indices = torch.floor((points - min_coords) / voxel_size).int()

    # 找到每个体素索引及其出现次数
    unique_voxel_indices, counts = torch.unique(voxel_indices, return_counts=True, dim=0)

    # 遍历每个唯一的体素索引，并根据点的数量和现有标签来更新标签
    for voxel_idx, count in zip(unique_voxel_indices, counts):
        voxel_idx_tuple = tuple(voxel_idx.tolist())  # 将 tensor 转换为元组，便于索引

        # 如果体素的标签为2或者该体素没有label，且点数大于2，设置为0
        if voxel_idx_tuple not in voxel_labels or voxel_labels.get(voxel_idx_tuple) == 2:
            if count > 1:
                voxel_labels[voxel_idx_tuple] = 1

    return voxel_labels

def test():
    # 示例使用
    num_points = 100000
    points = generate_random_pyramid_points(num_points)

    voxel_size = 0.2
    voxel_labels, min_coords, voxel_size = initialize_voxels(points, voxel_size)

    # 新的点云
    new_points = generate_random_pyramid_points(10000)

    # 计算新点云中位于初始体素中的点的比例
    proportion_in_voxels = compare_points_with_voxels(new_points, voxel_labels, min_coords, voxel_size)

    print(f"新点云中位于初始体素中的点的比例: {proportion_in_voxels:.2%}")

    # 可视化体素和点云
    plot_voxels_and_points(voxel_labels, min_coords, voxel_size, points)

if __name__ == '__main__':
    # test()
    print('')
