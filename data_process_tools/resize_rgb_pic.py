import os
from PIL import Image, ExifTags
import numpy as np


def process_depth_image(depth_array):
    """
    对深度图像对应的数组进行反序处理。
    这里假设 depth_array 是一个二维的深度图，不包含 RGB 信息。
    """
    min_depth = np.min(depth_array)
    max_depth = np.max(depth_array)

    # 归一化并反序
    reversed_depth = (max_depth - depth_array) / (max_depth - min_depth)

    # 转回 [0, 255] 范围，方便保存为8位图像
    reversed_depth_uint8 = np.uint8(reversed_depth * 255)

    return reversed_depth_uint8


def resize_image(input_path, output_path, target_size=(1600, 1066)):
    """
    使用 Pillow 打开图像，读取并保留 EXIF，然后缩放并保存。
    保持 RGB 色彩不变。
    """
    with Image.open(input_path) as img:
        # 获取 EXIF 信息
        exif_data = img.info.get('exif')

        # 如果是 RGB 图像，保持颜色不变
        if img.mode == 'RGB':
            # 直接调整大小
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        elif img.mode in ('L', 'I', 'F'):  # 处理灰度或深度图
            # 将图像转换为 NumPy 数组
            np_img = np.array(img, dtype=np.float32)

            # 处理深度图像
            processed_depth = process_depth_image(np_img)

            # 转回 Pillow 图像
            resized_img = Image.fromarray(processed_depth)

            # 如果原图是单通道，保持模式不变
            resized_img = resized_img.convert(img.mode)
        else:
            # 其他模式直接调整大小
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

        # 保存图像，并保留 EXIF 信息
        if exif_data:
            resized_img.save(output_path, exif=exif_data)
        else:
            resized_img.save(output_path)


def process_images_in_folder(input_folder, output_folder, target_size=(1600, 1066)):
    """
    遍历输入文件夹中的所有图像，调整大小后保存到输出文件夹，
    保留 EXIF 信息，并保持 RGB 色彩不变。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_exts = (".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG")
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_exts):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                resize_image(input_path, output_path, target_size=target_size)
                print(f"保存处理后的文件: {output_path}")
            except Exception as e:
                print(f"无法处理文件 {input_path} ，错误原因：{e}")
        else:
            print(f"跳过非图像文件: {filename}")

# 示例调用
if __name__ == "__main__":
    input_folder = "../data/Bispo_g5_select/images"        # 替换为实际的输入文件夹路径
    output_folder = "../data/Bispo_g5_select/images_resize"  # 替换为实际的输出文件夹路径
    process_images_in_folder(input_folder, output_folder)
