import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# 设置日志配置，日志将同时输出到控制台和日志文件
def setup_logging(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def normalize_image(image, p5, p95):
    """
    根据给定的第5百分位和第95百分位对图像进行归一化，裁剪到[0,1]范围内。

    参数:
        image (np.ndarray): 要归一化的图像。
        p5 (float): 第5百分位数值。
        p95 (float): 第95百分位数值。

    返回:
        normalized_image (np.ndarray): 归一化后的图像。
    """
    if p95 == p5:
        logging.warning("p95 等于 p5。返回全零图像以避免除以零。")
        return np.zeros_like(image, dtype="float32")

    normalized = (image.astype("float32") - p5) / (p95 - p5)
    normalized_clipped = np.clip(normalized, 0, 1)
    return normalized_clipped

def calculate_normalized_rmse(image1, image2):
    """
    计算归一化后图像之间的RMSE。

    归一化基于image1的第5和第95百分位。

    参数:
        image1 (np.ndarray): 基准深度图（灰度图）。
        image2 (np.ndarray): 需要比较的深度图（灰度图）。

    返回:
        rmse (float): 归一化后的 RMSE 值。
        error (np.ndarray): 绝对误差图。
    """
    # 计算第5和第95百分位
    p5 = np.percentile(image1, 1)
    p95 = np.percentile(image1, 99)

    # 归一化图像
    img1_normalized = normalize_image(image1, p5, p95)
    img2_normalized = normalize_image(image2, p5, p95)

    # 计算RMSE
    error = img1_normalized - img2_normalized
    squared_error = np.square(error)
    mse = np.mean(squared_error)
    rmse = np.sqrt(mse)
    return rmse, np.abs(error)

def plot_error_image(error, image_name, save_path, title_suffix=''):
    """
    绘制误差图，并保存（使用红色渐变颜色图），去除图例边框和刻度。

    被归一化后的误差范围在 [0, 1]，根据需要可以调整。

    参数:
        error (np.ndarray): 绝对误差图。
        image_name (str): 图像名称，用于标题和保存文件名。
        save_path (str): 保存误差图的路径。
        title_suffix (str): 标题后缀，用于区分不同的比较集。
    """
    plt.figure(figsize=(6, 6))

    # 显示误差图
    img = plt.imshow(error, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)

    # # 添加颜色条
    # cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
    # cbar.set_label('归一化 RMSE 误差', fontsize=10)
    #
    # # 移除颜色条的边框
    # cbar.outline.set_visible(False)
    #
    # # 移除颜色条的刻度线和标签
    # cbar.ax.tick_params(size=0, labelsize=8)
    # cbar.ax.yaxis.set_ticks_position('none')  # 如果颜色条是水平的，可以使用 'x' 替代 'y'
    #
    # # 设置标题
    # plt.title(f'{image_name} 的归一化 RMSE 误差图 {title_suffix}', fontsize=12)

    # 隐藏坐标轴
    plt.axis('off')

    # 保存图像
    save_filename = f"{os.path.splitext(image_name)[0]}_normalized_error.png"
    plt.savefig(os.path.join(save_path, save_filename), bbox_inches='tight', dpi=300,pad_inches=0)
    plt.close()

def process_images(ground_truth_folder, comparison_folder, save_path):
    """
    计算同名图片的归一化RMSE（基于第5和第95百分位）并保存误差图。

    参数:
        ground_truth_folder (str): 基准深度图的文件夹路径。
        comparison_folder (str): 需要比较的深度图文件夹路径。
        save_path (str): 保存误差图的路径。

    返回:
        rmse_results (list of tuples): 每张图片的 (image_name, rmse)。
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 获取支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    ground_truth_images = {f for f in os.listdir(ground_truth_folder) if f.lower().endswith(supported_formats)}
    comparison_images = {f for f in os.listdir(comparison_folder) if f.lower().endswith(supported_formats)}

    common_images = ground_truth_images.intersection(comparison_images)

    rmse_results = []
    for image_name in common_images:
        gt_path = os.path.join(ground_truth_folder, image_name)
        cmp_path = os.path.join(comparison_folder, image_name)

        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        cmp_image = cv2.imread(cmp_path, cv2.IMREAD_GRAYSCALE)

        if gt_image is None or cmp_image is None:
            logging.warning(f"跳过 {image_name}：无法加载一张或两张图像。")
            continue

        if gt_image.shape != cmp_image.shape:
            logging.warning(f"跳过 {image_name}：图像尺寸不匹配。")
            continue

        rmse, error = calculate_normalized_rmse(gt_image, cmp_image)
        rmse_results.append((image_name, rmse))

        # 生成误差图
        plot_error_image(error, image_name, save_path)

    return rmse_results

def subtract_error_images(folder1_errors, folder2_errors, difference_save_path):
    """
    对两种方法的误差图进行相减，生成误差差异图。

    参数:
        folder1_errors (str): 第一个比较文件夹的误差图路径。
        folder2_errors (str): 第二个比较文件夹的误差图路径。
        difference_save_path (str): 保存误差差异图的路径。
    """
    if not os.path.exists(folder1_errors):
        logging.error(f"误差图文件夹1不存在: {folder1_errors}")
        return
    if not os.path.exists(folder2_errors):
        logging.error(f"误差图文件夹2不存在: {folder2_errors}")
        return

    if not os.path.exists(difference_save_path):
        os.makedirs(difference_save_path)

    # 获取所有误差图文件
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    error_images1 = {f for f in os.listdir(folder1_errors) if f.lower().endswith(supported_formats)}
    error_images2 = {f for f in os.listdir(folder2_errors) if f.lower().endswith(supported_formats)}

    common_error_images = error_images1.intersection(error_images2)

    for image_name in common_error_images:
        img1_path = os.path.join(folder1_errors, image_name)
        img2_path = os.path.join(folder2_errors, image_name)

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            logging.warning(f"跳过 {image_name} 的相减：无法加载一张或两张误差图。")
            continue

        if img1.shape != img2.shape:
            logging.warning(f"跳过 {image_name} 的相减：误差图尺寸不匹配。")
            continue

        # 将图像转换为浮点型
        img1_float = img1.astype("float32") / 255.0  # 假设原始误差图归一化到 [0,1] 并保存为 8-bit
        img2_float = img2.astype("float32") / 255.0

        # 相减
        difference = img1_float - img2_float

        # 可视化差异
        plt.figure(figsize=(6, 6))
        # 使用分歧色彩图，例如 'bwr'，以显示正负差异
        img = plt.imshow(difference, cmap='bwr', interpolation='nearest', vmin=-1, vmax=1)

        # # 添加颜色条
        # cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
        # cbar.set_label('归一化 RMSE 误差差异', fontsize=10)
        #
        # # 移除颜色条的边框
        # cbar.outline.set_visible(False)
        #
        # # 移除颜色条的刻度线和标签
        # cbar.ax.tick_params(size=0, labelsize=8)
        # cbar.ax.yaxis.set_ticks_position('none')  # 如果颜色条是水平的，可以使用 'x' 替代 'y'
        #
        # # 设置标题
        base_image_name = os.path.splitext(image_name)[0].replace('_normalized_error', '')
        # plt.title(f'{base_image_name} 的 RMSE 误差差异图', fontsize=12)

        # 隐藏坐标轴
        plt.axis('off')

        # 保存差异图
        save_filename = f"{base_image_name}_rmse_difference.png"
        plt.savefig(os.path.join(difference_save_path, save_filename), bbox_inches='tight', dpi=300,pad_inches=0)
        plt.close()

        logging.info(f"已保存 RMSE 误差差异图: {save_filename}")

def subtract_error_images_multiple_sets(error_folders_pairs, difference_save_base_path):
    """
    对多个误差图文件夹对进行相减，生成误差差异图。

    参数:
        error_folders_pairs (list of tuples): 每个元组包含两个误差图文件夹路径 (folder1, folder2)。
        difference_save_base_path (str): 保存所有误差差异图的基础路径。
    """
    for idx, (folder1, folder2) in enumerate(error_folders_pairs, start=1):
        difference_save_path = os.path.join(difference_save_base_path, f'difference_set{idx}')
        logging.info(f"正在处理比较集{idx}的误差图相减：{folder1} 与 {folder2}")
        subtract_error_images(folder1, folder2, difference_save_path)

def compare_rmse_sets(rmse_results_set1, rmse_results_set2):
    """
    比较两个 RMSE 结果集，筛选出 Set1 优于 Set2 的图像，并按差距排序。

    参数:
        rmse_results_set1 (list of tuples): Set1 的 RMSE 结果，格式为 (image_name, rmse)。
        rmse_results_set2 (list of tuples): Set2 的 RMSE 结果，格式为 (image_name, rmse)。

    返回:
        sorted_better_images (list of tuples): 按差距排序的 Set1 优于 Set2 的图像，格式为 (image_name, rmse_set1, rmse_set2, difference)。
    """
    # 将结果转换为字典以便快速查找
    rmse_dict_set1 = dict(rmse_results_set1)
    rmse_dict_set2 = dict(rmse_results_set2)

    better_images = []
    for image_name, rmse1 in rmse_dict_set1.items():
        rmse2 = rmse_dict_set2.get(image_name)
        if rmse2 is not None and rmse1 < rmse2:
            difference = rmse2 - rmse1
            better_images.append((image_name, rmse1, rmse2, difference))

    # 按差距从大到小排序
    sorted_better_images = sorted(better_images, key=lambda x: x[3], reverse=True)
    return sorted_better_images

def main():
    # 定义日志文件路径
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"depth_comparison_log_{current_time}.log"
    setup_logging(log_file_path)

    # 定义文件夹路径
    base_depth_folder = '../output/biosp_lidar/train/ours_1/renders_depth'  # 基准深度图文件夹

    # 定义多个比较文件夹集
    comparison_sets = [
        {
            'comparison_folder1': '../output/Task1_it30000_cF_dN_dtM_dw0.2_vF/train/ours_30000/renders_depth',  # 第一个比较文件夹路径
            'comparison_folder2': '../output/Task1_it15000_cF_dMV_dtL_dw0.2_vF/train/ours_15000/renders_depth',  # 第二个比较文件夹路径
            'label': 'Set1'  # 比较集标签
        },
        {
            # 第二个比较文件夹路径
            'comparison_folder1': '../output/Task1_it30000_cF_dMV_dtF_dw0.2_vF/train/ours_30000/renders_depth',
            'comparison_folder2': '../output/Task1_it30000_cF_dMV_dtM_dw0.2_vF/train/ours_30000/renders_depth',  # 第一个比较文件夹路径
            'label': 'Set2'  # 比较集标签
        }
    ]

    # 定义输出的基础路径
    output_base_path = '../output/depth_comparison_output'
    output_errors_base = os.path.join(output_base_path, 'errors')
    difference_save_base_path = os.path.join(output_base_path, 'differences')

    # 确保输出路径存在
    for path in [output_errors_base, difference_save_base_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # 存储每个比较集的 RMSE 结果
    rmse_results_sets = []

    # 处理每个比较集
    for comparison_set in comparison_sets:
        label = comparison_set['label']
        comparison_folder1 = comparison_set['comparison_folder1']
        comparison_folder2 = comparison_set['comparison_folder2']

        logging.info(f"正在处理比较集: {label}")

        # 定义每个比较集的误差图保存路径
        save_errors1 = os.path.join(output_errors_base, f'comparison1_{label}_errors')
        save_errors2 = os.path.join(output_errors_base, f'comparison2_{label}_errors')

        # 处理第一个比较文件夹
        logging.info(f"处理比较集 {label} 的第一个比较文件夹: {comparison_folder1}")
        rmse_results1 = process_images(base_depth_folder, comparison_folder1, save_errors1)

        # 输出每张图片的RMSE结果
        for image_name, rmse in rmse_results1:
            logging.info(f"[{label} 比较1] 图片: {image_name}, RMSE: {rmse}")

        # 计算并输出平均RMSE
        if rmse_results1:
            average_rmse1 = sum(rmse for _, rmse in rmse_results1) / len(rmse_results1)
            logging.info(f"[{label} 比较1] 平均 RMSE: {average_rmse1}")
        else:
            logging.info(f"[{label} 比较1] 未处理任何图片。")

        # 处理第二个比较文件夹
        logging.info(f"处理比较集 {label} 的第二个比较文件夹: {comparison_folder2}")
        rmse_results2 = process_images(base_depth_folder, comparison_folder2, save_errors2)

        # 输出每张图片的RMSE结果
        for image_name, rmse in rmse_results2:
            logging.info(f"[{label} 比较2] 图片: {image_name}, RMSE: {rmse}")

        # 计算并输出平均RMSE
        if rmse_results2:
            average_rmse2 = sum(rmse for _, rmse in rmse_results2) / len(rmse_results2)
            logging.info(f"[{label} 比较2] 平均 RMSE: {average_rmse2}")
        else:
            logging.info(f"[{label} 比较2] 未处理任何图片。")

        # 存储 RMSE 结果
        rmse_results_sets.append({
            'label': label,
            'rmse_set1': rmse_results1,
            'rmse_set2': rmse_results2
        })

    # 比较每个比较集的两个比较文件夹，筛选出比较1优于比较2的图像，并按差距排序
    for rmse_set in rmse_results_sets:
        label = rmse_set['label']
        rmse_set1 = rmse_set['rmse_set1']
        rmse_set2 = rmse_set['rmse_set2']

        better_images = compare_rmse_sets(rmse_set1, rmse_set2)

        if better_images:
            logging.info(f"比较集 {label} 中，比较1 优于 比较2 的图像名单（按 RMSE 差距排序）：")
            logging.info(f"{'图像名称':<10} {'比较1 RMSE':<15} {'比较2 RMSE':<15} {'RMSE 差距':<15}")
            for img in better_images:
                image_name, rmse1, rmse2, diff = img
                logging.info(f"{image_name:<10} {rmse1:<15.6f} {rmse2:<15.6f} {diff:<15.6f}")
        else:
            logging.info(f"比较集 {label} 中，没有图像显示比较1 优于 比较2。")

    # 生成误差差异图
    error_folders_pairs = []
    for rmse_set in rmse_results_sets:
        label = rmse_set['label']
        folder1_errors = os.path.join(output_errors_base, f'comparison1_{label}_errors')
        folder2_errors = os.path.join(output_errors_base, f'comparison2_{label}_errors')
        error_folders_pairs.append((folder1_errors, folder2_errors))

    logging.info("开始生成所有比较集的 RMSE 误差差异图")
    subtract_error_images_multiple_sets(error_folders_pairs, difference_save_base_path)

    logging.info("所有处理完成。")

if __name__ == "__main__":
    main()
