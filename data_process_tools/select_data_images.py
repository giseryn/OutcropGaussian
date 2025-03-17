import os
import exifread
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import KDTree
import shutil

def dms_to_decimal(dms, ref):
    """
    将 DMS（度、分、秒）和方向（N/S/E/W）转换为十进制坐标。

    参数:
        dms (list): 包含 exifread.utils.Ratio 对象的列表，例如 [Ratio(30,1), Ratio(15,1), Ratio(1234,100)]
        ref (str): 方向参考，例如 'N', 'S', 'E', 'W'

    返回:
        float: 十进制坐标
    """
    deg = float(dms[0].num) / float(dms[0].den)
    minu = float(dms[1].num) / float(dms[1].den)
    sec = float(dms[2].num) / float(dms[2].den)

    decimal = deg + minu / 60.0 + sec / 3600.0
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_gps_data(image_path):
    """
    从图像的 EXIF 信息中提取纬度、经度和海拔高度。

    参数:
        image_path (str): 图像文件的路径

    返回:
        tuple: (纬度, 经度, 海拔) 或 (None, None, None) 如果信息缺失
    """
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)

    # 检查是否包含必要的 GPS 标签
    if ('GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags):
        gps_latitude = tags['GPS GPSLatitude']
        gps_longitude = tags['GPS GPSLongitude']
        gps_lat_ref = tags.get('GPS GPSLatitudeRef', None)
        gps_lon_ref = tags.get('GPS GPSLongitudeRef', None)

        # 提取纬度和经度
        lat_ref = str(gps_lat_ref.values) if gps_lat_ref else 'N'
        lon_ref = str(gps_lon_ref.values) if gps_lon_ref else 'E'

        lat = dms_to_decimal(gps_latitude.values, lat_ref)
        lon = dms_to_decimal(gps_longitude.values, lon_ref)

        # 提取海拔高度（如果存在）
        if 'GPS GPSAltitude' in tags:
            gps_altitude = tags['GPS GPSAltitude']
            gps_altitude_ref = tags.get('GPS GPSAltitudeRef', None)
            altitude = float(gps_altitude.values[0].num) / float(gps_altitude.values[0].den)
            # 根据 GPSAltitudeRef 判断海拔的正负（0=海平面以上, 1=海平面以下）
            if gps_altitude_ref:
                altitude_ref = gps_altitude_ref.values[0]
                if altitude_ref == 1:
                    altitude = -altitude
        else:
            altitude = None  # 海拔信息不可用

        return lat, lon, altitude
    else:
        return None, None, None

def fit_trend_line(longitudes, latitudes):
    """
    拟合线性回归趋势线。

    参数:
        longitudes (numpy.ndarray): 经度数组
        latitudes (numpy.ndarray): 纬度数组

    返回:
        tuple: (斜率, 截距)
    """
    slope, intercept = np.polyfit(longitudes, latitudes, 1)
    return slope, intercept

def assign_groups(t_p, num_groups=9):
    """
    根据投影参数 t_p 将数据分配到不同的组中。

    参数:
        t_p (numpy.ndarray): 投影参数数组
        num_groups (int): 组的数量，默认为9

    返回:
        numpy.ndarray: 组别索引数组
    """
    t_min, t_max = t_p.min(), t_p.max()
    t_bins = np.linspace(t_min, t_max, num_groups + 1)  # 10 个边界点
    group_indices = np.digitize(t_p, t_bins)
    group_indices = np.clip(group_indices, 1, num_groups)  # 将组别限制在 [1, num_groups]
    return group_indices

def plot_groupings(longitudes, latitudes, group_indices, slope, intercept):
    """
    绘制分组情况的图表。

    参数:
        longitudes (numpy.ndarray): 经度数组
        latitudes (numpy.ndarray): 纬度数组
        group_indices (numpy.ndarray): 组别索引数组
        slope (float): 趋势线斜率
        intercept (float): 趋势线截距
    """
    fig = plt.figure(figsize=(16, 10))

    # (a) 分组散点图
    ax1 = fig.add_subplot(2, 2, 1)
    sc = ax1.scatter(
        longitudes, latitudes,
        c=group_indices,
        cmap='viridis',
        alpha=0.7,
        edgecolor='k'
    )
    # 绘制趋势线
    x_line = np.linspace(longitudes.min(), longitudes.max(), 200)
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'r--', label='Trend line')

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("Scatter Plot (9 Groups by Strips Perpendicular to Trend)")
    ax1.legend(loc='best')

    # 分组颜色条
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label("Group Index (1~9)")

    # (b) 经度分布直方图
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(longitudes, bins=30, color='orange', alpha=0.7)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Count")
    ax2.set_title("Longitude Distribution")

    # (c) 纬度分布直方图
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(latitudes, bins=30, color='green', alpha=0.7)
    ax3.set_xlabel("Latitude")
    ax3.set_ylabel("Count")
    ax3.set_title("Latitude Distribution")

    # (d) 2D Hexbin 图
    ax4 = fig.add_subplot(2, 2, 4)
    hb = ax4.hexbin(
        longitudes, latitudes,
        gridsize=30,
        cmap='inferno',
        extent=[longitudes.min(), longitudes.max(),
                latitudes.min(), latitudes.max()]
    )
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")
    ax4.set_title("2D Hexbin")
    plt.colorbar(hb, ax=ax4, label='count')

    plt.tight_layout()
    plt.show()

def plot_group5_distribution(longitudes, altitudes, filenames):
    """
    绘制编号为5的图像在经度和海拔上的分布，并在图中标注文件名。

    参数:
        longitudes (numpy.ndarray): 经度数组
        altitudes (numpy.ndarray): 海拔高度数组
        filenames (list): 文件名列表
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(longitudes, altitudes, c='blue', alpha=0.7, edgecolor='k', label='Group 5 Images')

    # 添加文件名标签
    for i, fname in enumerate(filenames):
        plt.annotate(fname, (longitudes[i], altitudes[i]),
                     textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

    plt.xlabel("Longitude")
    plt.ylabel("Altitude (meters)")
    plt.title("Distribution of Images in Group 5 (Longitude vs Altitude)")
    plt.legend()
    plt.grid(True)
    plt.show()

def save_filenames(filenames, output_path):
    """
    将文件名列表保存到指定的文本文件中。

    参数:
        filenames (list): 文件名列表
        output_path (str): 输出文件的路径
    """
    with open(output_path, "w") as f:
        for fname in filenames:
            f.write(f"{fname}\n")
    print(f"\nImage filenames have been saved to '{output_path}'.")

def copy_images(filenames, source_folder, destination_folder):
    """
    将指定的图片文件从源文件夹复制到目标文件夹。

    参数:
        filenames (list): 要复制的图片文件名列表
        source_folder (str): 源文件夹路径
        destination_folder (str): 目标文件夹路径
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")

    for fname in filenames:
        src_path = os.path.join(source_folder, fname)
        dst_path = os.path.join(destination_folder, fname)
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Copied '{fname}' to '{destination_folder}'")
        except Exception as e:
            print(f"Failed to copy '{fname}': {e}")

def main(folder_path, destination_folder):
    """
    主函数，执行所有数据处理和可视化步骤。

    参数:
        folder_path (str): 包含图像的文件夹路径。
        destination_folder (str): 第五组图像的目标文件夹路径。
    """
    # 1) 收集所有图像文件
    extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    file_list = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and f.lower().endswith(extensions)
    ]

    latitudes = []
    longitudes = []
    altitudes = []
    filenames = []

    # 2) 从每个文件中提取 GPS 数据
    for filename in file_list:
        image_path = os.path.join(folder_path, filename)
        lat, lon, alt = get_gps_data(image_path)
        if lat is not None and lon is not None and alt is not None:
            latitudes.append(lat)
            longitudes.append(lon)
            altitudes.append(alt)
            filenames.append(filename)

    if not latitudes:
        print("No valid GPS (including altitude) info found in this folder.")
        return

    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    altitudes = np.array(altitudes)

    # 3) 拟合线性回归（趋势线）
    slope, intercept = fit_trend_line(longitudes, latitudes)

    # 4) 计算每个点在趋势线上的投影参数 t_p
    m = slope
    b = intercept
    x_p = longitudes
    y_p = latitudes
    t_p = (x_p + m * y_p - m * b) / (1 + m**2)

    # 5) 将 [t_min, t_max] 划分为 9 个等分段，并分配组别
    group_indices = assign_groups(t_p, num_groups=9)

    # 6) 可视化分组情况
    plot_groupings(longitudes, latitudes, group_indices, slope, intercept)

    # 7) 筛选出编号为5的图像
    group_5_indices = np.where(group_indices == 5)[0]

    if group_5_indices.size == 0:
        print("No images found in Group 5.")
        return

    group_5_lons = longitudes[group_5_indices]
    group_5_alts = altitudes[group_5_indices]
    group_5_filenames = [filenames[i] for i in group_5_indices]

    # 8) 绘制 Group 5 的经度 vs 海拔分布，并标注文件名
    plot_group5_distribution(group_5_lons, group_5_alts, group_5_filenames)

    # 9) 可选：将 Group 5 的图像文件名保存到文本文件
    output_file = "group5_images_filenames.txt"
    save_filenames(group_5_filenames, output_file)

    # 10) 复制 Group 5 的图像到新文件夹
    copy_images(group_5_filenames, folder_path, destination_folder)

    # 可选：打印 Group 5 的图像文件名
    print(f"\nImages in Group 5 (Total: {group_5_indices.size}):")
    for fname in group_5_filenames:
        print(f" - {fname}")

if __name__ == "__main__":
    # 替换为你的图像文件夹路径
    source_folder = r"../data/Bispo/images"
    # 定义第五组图像的目标文件夹路径
    destination_folder = r"../data/Bispo_select/images"

    main(source_folder, destination_folder)