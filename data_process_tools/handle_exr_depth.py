import OpenEXR
import Imath
import numpy as np
import cv2

def read_exr_depth(file_path):
    # 打开 EXR 文件
    exr_file = OpenEXR.InputFile(file_path)

    # 获取图像分辨率
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 定义像素类型
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    # 读取深度通道
    depth_channel = exr_file.channel('Y', pixel_type)

    # 将深度数据转换为 NumPy 数组
    depth_array = np.frombuffer(depth_channel, dtype=np.float32)
    depth_array = depth_array.reshape((height, width))

    return depth_array


def preprocess_depth_map(depth_map):
    # 将无效值（NaN 和 Inf）替换为 0
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    # 确保数据中没有 float32 的最大值
    max_float32 = np.finfo(np.float32).max
    depth_map[depth_map == max_float32] = -1  # 替换 float32 最大值为 0

    # 计算最大有效深度值（去除无效值后的最大值）
    max_valid_value = np.max(depth_map)

    # 剔除异常值（限制在 [0, max_valid_value] 的范围内）
    depth_map = np.clip(depth_map, 0, max_valid_value)

    return depth_map, max_valid_value


def depth_to_24bit_color(depth_map, max_valid_value):
    # 归一化深度值到 [0, 1]，使用最大有效深度值
    normalized_depth = depth_map / max_valid_value

    # 将深度值映射到 24 位伪彩色（8 位红、8 位绿、8 位蓝）
    color_map = (normalized_depth * 255).astype(np.uint8)
    red_channel = color_map
    green_channel = np.zeros_like(color_map)
    blue_channel = 255 - color_map

    # 合并通道
    color_image = cv2.merge((blue_channel, green_channel, red_channel))
    return color_image


def save_depth_as_png(color_image, output_path):
    # 保存为 PNG 文件
    cv2.imwrite(output_path, color_image)
    print(f"24 位深度图保存为 PNG 文件: {output_path}")


# 示例用法
depth_map = read_exr_depth('TEST_3.JPG.depth.exr')

# 打印原始深度数据信息
print("原始深度图形状:", depth_map.shape)
print("原始深度值范围:", depth_map.min(), "-", depth_map.max())

# 预处理深度数据
depth_map, max_valid_value = preprocess_depth_map(depth_map)

# 打印处理后的深度数据信息
print("处理后深度值范围:", depth_map.min(), "-", depth_map.max())
print("有效深度的最大值:", max_valid_value)

# 转换为 24 位伪彩色图像
color_image = depth_to_24bit_color(depth_map, max_valid_value)

# 保存为 PNG 文件
save_depth_as_png(color_image, 'depth/00002.png')
