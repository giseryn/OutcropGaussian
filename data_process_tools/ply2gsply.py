from plyfile import PlyData
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库

def parse_ply_to_custom_format(ply_file):
    # 读取 PLY 文件
    ply_data = PlyData.read(ply_file)

    # 提取点数据
    vertex_data = ply_data['vertex']

    # 获取 XYZ 坐标
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    num_points = len(x)

    with open(txt_file, 'w') as f:
        # 写入头部信息
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {num_points}, mean track length: {0}\n")

        # 写入每个点的数据
        for i in tqdm(range(num_points), desc="Processing Points"):
            if i%10==0:
                point_id = i + 1
                x_val, y_val, z_val = x[i], y[i], z[i]

                line = f"{point_id} {x_val} {y_val} {z_val} {0} {0} {0} {0} {0}\n"
                f.write(line)


# 使用你的 PLY 文件路径
# ply_file = "data/courtyard_clean/lidar/lidar.ply"
# txt_file = "data/courtyard_clean/lidar/out.txt"

ply_file = "../data/Bispo_g5_select/lidar/Tile_3.ply"
txt_file = "../data/Bispo_g5_select/lidar/Points3D.txt"
parse_ply_to_custom_format(ply_file)
