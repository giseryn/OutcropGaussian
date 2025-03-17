from PIL import Image
import numpy as np

def compute_normals_from_depth(depth_map, scale=1.0):
    """
    计算深度图的法线图。
    :param depth_map: 深度图的 NumPy 数组。
    :param scale: 深度图的比例因子，用于控制法线计算的敏感度。
    :return: 法线图的 NumPy 数组。
    """
    # 计算梯度
    dz_dx = np.gradient(depth_map, axis=1) * scale
    dz_dy = np.gradient(depth_map, axis=0) * scale

    # 初始化法线向量数组
    height, width = depth_map.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)

    # 计算法线
    normals[..., 0] = -dz_dx  # x 分量
    normals[..., 1] = -dz_dy  # y 分量
    normals[..., 2] = 1.0     # z 分量

    # 归一化
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + 1e-8)

    # 转换到颜色空间 [0, 255]
    normals = (normals + 1.0) / 2.0 * 255
    normals = normals.astype(np.uint8)
    return normals

# 读取深度图
depth_image_path = "depth/00002.png"  # 替换为深度图文件路径
depth_image = Image.open(depth_image_path).convert("L")  # 灰度图
depth_map = np.array(depth_image, dtype=np.float32)

# 生成法线图
normals_map = compute_normals_from_depth(depth_map, scale=10)

# 保存法线图
normals_image = Image.fromarray(normals_map)
normals_image.save("normal/00002.png")
normals_image.show()
