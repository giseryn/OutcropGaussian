import json
import os
import shutil


# 读取 JSON 文件
with open('cameras.json', 'r') as file:
    data = json.load(file)

# 忽略前三行并生成字典，将 id 补全为五位，id-3 作为键，img_name 作为值
processed_index_dict = {
    f"{entry['id'] - 3:05}": entry['img_name']  # 使用 f-string 补全为五位
    for entry in data[3:]  # 忽略前三行
}

# 输出结果
print(processed_index_dict)

# 文件夹路径
source_folder = "images_lidar"  # 原始图片文件夹
destination_folder = "rename_lidar"  # 目标文件夹

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 遍历字典，寻找匹配的文件并重命名复制
for key, value in processed_index_dict.items():
    source_file = os.path.join(source_folder, f"{key}.png")  # 假设图片扩展名是 .jpg
    destination_file = os.path.join(destination_folder, f"{value}.png")

    # 检查文件是否存在
    if os.path.exists(source_file):
        shutil.copy(source_file, destination_file)  # 复制并重命名
        print(f"Copied and renamed: {source_file} -> {destination_file}")
    else:
        print(f"File not found: {source_file}")

print("Processing completed.")