import cv2
import numpy as np

# 读取 PNG 图像
def read_png_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # 保持图像的原始通道
    if image is None:
        raise ValueError(f"无法读取文件: {file_path}")
    return image

# 保存 PNG 图像
def save_png_image(file_path, image):
    cv2.imwrite(file_path, image)
    print(f"图像已保存为: {file_path}")

# 高斯滤波平滑
def smooth_depth_gaussian(depth_map, kernel_size=5, sigma=1):
    smoothed_depth = cv2.GaussianBlur(depth_map, (kernel_size, kernel_size), sigma)
    return smoothed_depth

# 双边滤波平滑
def smooth_depth_bilateral(depth_map, diameter=9, sigma_color=75, sigma_space=75):
    smoothed_depth = cv2.bilateralFilter(depth_map.astype(np.float32), diameter, sigma_color, sigma_space)
    return smoothed_depth

# 中值滤波平滑
def smooth_depth_median(depth_map, kernel_size=5):
    smoothed_depth = cv2.medianBlur(depth_map.astype(np.float32), kernel_size)
    return smoothed_depth

# 主函数示例
if __name__ == "__main__":
    input_path = "00000.png"  # 输入深度图文件路径
    output_path_gaussian = "depth_map_gaussian.png"  # 高斯滤波保存路径
    output_path_bilateral = "depth_map_bilateral.png"  # 双边滤波保存路径
    output_path_median = "depth_map_median.png"  # 中值滤波保存路径

    # 读取深度图
    depth_map = read_png_image(input_path)

    # 应用平滑处理
    smoothed_gaussian = smooth_depth_gaussian(depth_map)
    smoothed_bilateral = smooth_depth_bilateral(depth_map)
    smoothed_median = smooth_depth_median(depth_map)

    # 保存处理后的深度图
    save_png_image(output_path_gaussian, smoothed_gaussian)
    save_png_image(output_path_bilateral, smoothed_bilateral)
    save_png_image(output_path_median, smoothed_median)
