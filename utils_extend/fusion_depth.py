import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from PIL import Image
import os


def apply_gaussian_smoothing(image, sigma=2):
    """
    对图像应用高斯滤波进行平滑处理。

    参数:
    - image: 输入的图像（可以是NumPy数组或者PIL图像）。
    - sigma: 高斯滤波的标准差，数值越大，图像越模糊。

    返回:
    - 平滑后的图像。
    """
    # 如果图像是PIL图像，先将其转换为NumPy数组
    if isinstance(image, Image.Image):
        image = np.array(image)

    # 如果图像是彩色图像，逐通道进行高斯滤波
    if image.ndim == 3:  # RGB图像
        smoothed_image = np.zeros_like(image)
        for i in range(3):  # 对每个通道应用高斯滤波
            smoothed_image[:, :, i] = gaussian_filter(image[:, :, i], sigma=sigma)
    else:  # 灰度图像
        smoothed_image = gaussian_filter(image, sigma=sigma)

    return smoothed_image


def stretch_depth_image(image, percentile=1):
    """
    处理深度图像，将前1%和后1%像素值裁剪掉后，将图像拉伸到0~255范围。

    Args:
    - image (np.ndarray): 输入的深度图像
    - percentile (float): 用于控制裁剪的百分比，默认是前1%和后1%裁剪

    Returns:
    - stretched_image (np.ndarray): 处理后的图像，值范围为0~255
    """
    # 将图像展平为一维
    flattened = image.flatten()

    # 计算前1%和后1%的阈值
    lower_bound = np.percentile(flattened, percentile)
    upper_bound = np.percentile(flattened, 100 - percentile)

    # 裁剪图像，使值限定在lower_bound和upper_bound之间
    clipped_image = np.clip(image, lower_bound, upper_bound)

    # 拉伸图像，将值映射到0到255之间
    stretched_image = 255 * (clipped_image - lower_bound) / (upper_bound - lower_bound)

    # 转换为uint8格式
    stretched_image = stretched_image.astype(np.uint8)

    return stretched_image


# 展示三角剖分结果的函数
def show_delaunay_triangulation(points_2d, triangles):
    plt.figure(figsize=(8, 8))
    plt.triplot(points_2d[:, 0], points_2d[:, 1], triangles, color='blue')
    plt.scatter(points_2d[:, 0], points_2d[:, 1], color='red', marker='o', s=0.01)
    plt.title('Delaunay Triangulation')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.show()


def show_image(depth_map, title):
    plt.imshow(depth_map, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()


def interpolate_depth_for_triangles(tri_coords, tri_depths, depth_map):
    for i in range(len(tri_coords)):
        # 获取当前三角形的顶点坐标和深度
        coords = tri_coords[i]  # 形状为 (3, 2)
        depths = tri_depths[i]  # 形状为 (3,)

        # 计算三角形的最小边界框
        min_x = int(max(0, np.floor(np.min(coords[:, 0]))))
        max_x = int(min(depth_map.shape[1] - 1, np.ceil(np.max(coords[:, 0]))))
        min_y = int(max(0, np.floor(np.min(coords[:, 1]))))
        max_y = int(min(depth_map.shape[0] - 1, np.ceil(np.max(coords[:, 1]))))

        # 使用np.meshgrid生成边界框内所有像素点的坐标网格
        x_grid, y_grid = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        num_pixels = x_grid.size

        # 将网格中的像素点展开为二维数组，形状为 (N, 3)，其中N是像素数量
        P = np.vstack((x_grid.ravel(), y_grid.ravel(), np.ones(num_pixels))).T  # 形状为 (N, 3)

        # 构建矩阵A，用于计算重心坐标
        A = np.array([[coords[0, 0], coords[1, 0], coords[2, 0]],
                      [coords[0, 1], coords[1, 1], coords[2, 1]],
                      [1, 1, 1]], dtype=np.float32)

        try:
            # 使用矩阵运算一次性计算所有像素点的重心坐标
            bary_coords = np.linalg.solve(A, P.T).T  # 形状为 (N, 3)
        except np.linalg.LinAlgError:
            continue  # 如果A矩阵不可逆，跳过该三角形

        # 判断重心坐标是否在0到1之间，过滤掉无效的像素
        valid_mask = np.all((bary_coords >= 0) & (bary_coords <= 1), axis=1)
        valid_pixels = np.where(valid_mask)[0]

        if len(valid_pixels) == 0:
            continue  # 如果没有有效像素，跳过该三角形

        # 取出有效的重心坐标和对应的像素坐标
        valid_bary_coords = bary_coords[valid_pixels]
        valid_x = x_grid.ravel()[valid_pixels]
        valid_y = y_grid.ravel()[valid_pixels]

        # 计算插值深度
        interpolated_depth = valid_bary_coords @ depths  # 矢量化的深度插值

        # 将插值后的深度值写入深度图
        depth_map[valid_y, valid_x] = interpolated_depth


def partition_and_process_triangles(tri_coords, tri_depths, depth_map, num_partitions):
    # 获取三角形的总数
    total_triangles = len(tri_coords)
    partition_size = total_triangles // num_partitions

    # 分割三角形集并逐一处理
    for i in tqdm(range(num_partitions), desc="处理三角形分区"):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i != num_partitions - 1 else total_triangles

        # 获取当前分区的三角形顶点坐标和深度
        current_tri_coords = tri_coords[start_idx:end_idx]
        current_tri_depths = tri_depths[start_idx:end_idx]

        # 在全局深度图上处理当前分区的三角形
        interpolate_depth_for_triangles(current_tri_coords, current_tri_depths, depth_map)


def erode_image_foreground(lidar_depth, n_iterations=2):
    # 创建一个副本用于处理
    processed_depth = lidar_depth.copy()

    # 定义一个卷积核用于检查3*3窗口
    kernel = np.ones((3, 3), np.uint8)

    for _ in range(n_iterations):
        # 创建一个标记矩阵，标记那些中心点的3*3窗口中有255的点
        dilation = cv2.dilate((processed_depth == 255).astype(np.uint8), kernel)

        # 将这些点的值变为255
        processed_depth[(dilation == 1) & (processed_depth != 255)] = 255

    return processed_depth


def process(lidar_path, mono_path, result_path):
    # 阶段1：数据加载和预处理
    print("阶段1：数据加载和预处理")
    lidar_depth = cv2.imread(lidar_path, cv2.IMREAD_GRAYSCALE)
    mono_depth = cv2.imread(mono_path, cv2.IMREAD_GRAYSCALE)

    if lidar_depth is None or mono_depth is None:
        raise FileNotFoundError("无法加载深度图，请检查文件路径。")

    show_image(lidar_depth, 'lidar_depth')

    # 阶段2：提取有效的雷达点
    print("阶段2：提取有效的雷达点")
    erode_image = erode_image_foreground(lidar_depth)
    show_image(erode_image, 'erode_image')

    valid_mask = erode_image != 255
    u_coords, v_coords = np.where(valid_mask)
    z_values = erode_image[valid_mask].astype(np.float32)
    points_2d = np.vstack((v_coords, u_coords)).T  # 形状为 (N, 2)
    depth_values = z_values

    # 阶段3：构建Delaunay三角剖分
    print("阶段3：构建Delaunay三角剖分")
    tri = Delaunay(points_2d)
    triangles = tri.simplices
    show_delaunay_triangulation(points_2d, triangles)
    # 阶段4：在三角形内进行深度插值（矢量化优化）
    print("阶段4：在三角形内进行深度插值（矢量化优化）")
    interpolated_depth = np.full_like(lidar_depth, fill_value=255, dtype=np.float32)

    # 将三角形顶点的坐标和深度值整理为数组
    tri_coords = points_2d[triangles]  # 形状为 (M, 3, 2)
    tri_depths = depth_values[triangles]  # 形状为 (M, 3)

    # 设置分区数目
    num_partitions = 100  # 可根据实际情况调整

    # 分区处理三角形并在全局深度图上进行插值
    partition_and_process_triangles(tri_coords, tri_depths, interpolated_depth, num_partitions)

    show_image(interpolated_depth, 'delaunay interpolated depth')
    # 阶段5：利用单目深度图进行数据补全
    print("阶段5：利用单目深度图进行数据补全")
    missing_mask = interpolated_depth == 255
    interpolated_depth[missing_mask] = mono_depth[missing_mask].astype(np.float32)

    # 阶段6：融合深度图并输出结果
    print("阶段6：融合深度图并输出结果")
    weight_lidar = 0.5
    weight_mono = 0.5

    fused_depth = np.where(
        erode_image != 255,
        erode_image.astype(np.float32),
        interpolated_depth
    )

    # 如果需要进一步融合单目深度，可以使用加权平均
    fused_depth = weight_lidar * stretch_depth_image(fused_depth, percentile=1) + weight_mono * stretch_depth_image(
        mono_depth.astype(np.float32), percentile=1)

    fused_depth_normalized = cv2.normalize(fused_depth, None, 0, 255, cv2.NORM_MINMAX)
    fused_depth_uint8 = fused_depth_normalized.astype(np.uint8)

    show_image(fused_depth_uint8, 'fused_depth')

    # 阶段7 高斯平滑

    gaussian_smoothing_image = apply_gaussian_smoothing(fused_depth_uint8)
    show_image(gaussian_smoothing_image, 'gaussian_smoothing_image')
    cv2.imwrite(result_path, gaussian_smoothing_image)


def find_associated_images_by_keyword(folder1, folder2):
    """
    根据 folder1 中图片的名称去除后缀作为关键词，在 folder2 中进行检索，寻找关联图片。

    参数:
    - folder1: 第一个文件夹的路径。
    - folder2: 第二个文件夹的路径。

    返回:
    - 关联的文件名对 (文件夹1中的文件, 文件夹2中的关联文件)
    """
    # 获取两个文件夹中的文件名列表
    files1 = sorted([f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))])
    files2 = sorted([f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))])

    # 用于存储关联文件名对的列表
    associated_files = []

    # 遍历folder1中的文件
    for file1 in files1:
        # 去除file1的后缀，作为关键词
        keyword = os.path.splitext(file1)[0]

        # 在folder2中搜索包含关键词的文件
        for file2 in files2:
            if keyword in file2:  # 如果file2包含关键词
                associated_files.append((file1, file2))
                break  # 找到后跳出循环，避免重复匹配

    return associated_files


if __name__ == '__main__':
    # 文件夹路径
    mono_folder = 'data/depth'
    lidar_folder = 'data/lidar'

    # 获取关联的文件对
    associated_images = find_associated_images_by_keyword(mono_folder, lidar_folder)
    # 输出关联的图片文件名对
    for file1, file2 in associated_images:
        print(f"Associated pair: {file1} <--> {file2}")
        mono_path = mono_folder + '/' + file1
        lidar_path = lidar_folder + '/' + file2
        result_path = 'data/result' + '/' + file1
        print(lidar_path, mono_path, result_path)
        process(lidar_path, mono_path, result_path)



