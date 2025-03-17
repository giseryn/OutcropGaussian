from PIL import Image


def resize_image(input_path, output_path, target_size=(1600, 1066)):
    """
    将图片调整为指定比例尺寸。

    参数:
        input_path (str): 输入图片路径。
        output_path (str): 输出图片路径。
        target_size (tuple): 目标尺寸 (宽, 高)，默认为 (1600, 1066)。
    """
    # 打开图片
    with Image.open(input_path) as img:
        # 调整尺寸
        resized_img = img.resize(target_size, Image.ANTIALIAS)
        # 保存图片
        resized_img.save(output_path)
        print(f"图片已调整为 {target_size}，保存到 {output_path}")


# 示例用法
input_image_path = "depth/00002.png"  # 输入图片路径
output_image_path = "depth/00002.png"  # 输出图片路径
resize_image(input_image_path, output_image_path, target_size=(1600, 1066))