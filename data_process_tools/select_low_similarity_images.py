import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import shutil

#---------------------------
# 配置参数
#---------------------------
image_folder = "../data/Bispo/images_resize"  # 替换为你的图片文件夹路径
selected_output_folder = "../data/Bispo/images_resize_select"   # 替换为输出文件夹路径（不存在则自动创建）
distance_threshold = 0.9  # 相似度（余弦相似度）阈值，越小说明要求越严格，可根据需要调优
# 注：因为我们用的是余弦相似度，1表示完全相同，0表示不相关，越低越不相似。
# 因此这里的阈值越小，要求不相似的程度越高(比如0.3即要求与已选图片集的相似度小于0.3才能选入)。

# 如果你想选取固定数量的图片，也可以设定num_selected
num_selected = 64

#---------------------------
# 加载模型并修改为特征提取模式
#---------------------------
model = resnet50(pretrained=True)
# 移除最后的分类层，只保留特征层（ResNet50的avgpool后为2048维特征）
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 预处理transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#---------------------------
# 读取文件夹中的图片路径
#---------------------------
img_paths = glob.glob(os.path.join(image_folder, "*"))
img_paths = [p for p in img_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

#---------------------------
# 定义特征提取函数
#---------------------------
def extract_feature(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img)  # 输出shape: [1, 2048, 1, 1]
    return feat.squeeze().cpu().numpy()  # -> [2048,]

#---------------------------
# 批量提取特征并展示进度
#---------------------------
features = []
print("Extracting features:")
for p in tqdm(img_paths, desc="Processing images", unit="image"):
    f = extract_feature(p)
    features.append(f)
features = np.stack(features)  # shape: [N, 2048]

#---------------------------
# 计算余弦相似度矩阵并展示进度
#---------------------------
# 对于较多的图片，直接一次性计算相似度矩阵可能会比较占内存。
# 一般N张图片会产生N*N的矩阵。如果图片数很多，请小心内存占用。
print("Calculating similarity matrix...")
sim_matrix = cosine_similarity(features, features)  # NxN矩阵

#---------------------------
# 贪心选择不相似图片并展示进度
#---------------------------
print("Selecting non-similar images...")
selected_indices = []
for i in tqdm(range(len(img_paths)), desc="Selecting", unit="image"):
    if len(selected_indices) == 0:
        selected_indices.append(i)
    else:
        # 找当前图片与已选图片的最大相似度
        max_sim = sim_matrix[i, selected_indices].max()
        if max_sim < distance_threshold:
            selected_indices.append(i)

    if len(selected_indices) >= num_selected:
        break

print("Selected {} images from {} total.".format(len(selected_indices), len(img_paths)))
selected_paths = [img_paths[i] for i in selected_indices]

#---------------------------
# 将选出的图片复制到指定文件夹
#---------------------------
os.makedirs(selected_output_folder, exist_ok=True)
print("Copying selected images to:", selected_output_folder)
for sp in tqdm(selected_paths, desc="Copying", unit="image"):
    shutil.copy(sp, selected_output_folder)

print("Done!")
