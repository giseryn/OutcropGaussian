import cv2
import numpy as np
import torch
import torchvision.transforms as T


###########################
# MiDaS 推理函数
###########################
def run_midas_inference(input_img, midas_model, midas_transform, device='cpu'):
    """
    输入:
        input_img: numpy array (H, W, 3) in BGR 或 RGB
        midas_model: MiDaS 模型
        midas_transform: MiDaS 转换器
        device: 运行设备 (cpu 或 cuda)
    输出:
        depth_map: numpy array, 与输入图像大小相同的深度图
    """
    # 若为灰度图, 转成3通道
    if len(input_img.shape) == 2:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    # BGR 转 RGB
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # 转为 PyTorch tensor
    input_batch = midas_transform(input_img).to(device)

    # 推理
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=input_img.shape[:2],  # 插值到输入图像大小
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    # 深度值归一化到 [0, 1] (可选)
    depth_min, depth_max = prediction.min(), prediction.max()
    depth_map = (prediction - depth_min) / (depth_max - depth_min + 1e-8)

    return depth_map


###########################
# 重复纹理检测
###########################
def detect_top10_repetitive_blocks(img, block_size=64):
    """
    检测重复纹理最强的10个块
    输入:
        img: 原图 (BGR 或 灰度)
        block_size: 块大小
    输出:
        List[(x, y, w, h)]
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    H, W = gray.shape
    candidates = []

    for y in range(0, H, block_size):
        if y + block_size > H:
            break
        for x in range(0, W, block_size):
            if x + block_size > W:
                break

            # 当前块
            patch = gray[y:y+block_size, x:x+block_size]
            patch_fft = np.fft.fft2(patch)
            mag_spectrum = np.fft.fftshift(np.abs(patch_fft))

            center_region = mag_spectrum[block_size//2 - 2:block_size//2 + 2,
                                         block_size//2 - 2:block_size//2 + 2]
            total_energy = mag_spectrum.sum()
            center_energy = center_region.sum()
            ratio = center_energy / total_energy if total_energy > 1e-6 else 1.0

            candidates.append((ratio, x, y, block_size, block_size))

    candidates.sort(key=lambda x: x[0])
    return [(c[1], c[2], c[3], c[4]) for c in candidates[:10]]


###########################
# 主流程
###########################
def main():
    # 1. 读取图像
    img_path = "00023.png"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("图像读取失败")
        return

    # 2. 加载 MiDaS 模型
    model_type = "DPT_Hybrid"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas_model = torch.hub.load("isl-org/MiDaS", model_type)
    midas_model.to(device)
    midas_model.eval()
    midas_transform = torch.hub.load("isl-org/MiDaS", "transforms").dpt_transform

    # 3. 对全图进行全局深度估计
    global_depth = run_midas_inference(img, midas_model, midas_transform, device)

    # 4. 检测重复纹理最强的10个块
    block_size = 64
    blocks = detect_top10_repetitive_blocks(img, block_size=block_size)

    # 5. 创建最终融合后的深度图 (初始化为全局深度)
    fused_depth = global_depth.copy()

    # 6. 对每个块进行局部深度估计，并融合到全局深度图
    for (x, y, w, h) in blocks:
        crop_img = img[y:y+h, x:x+w]

        # 局部深度估计
        local_depth = run_midas_inference(crop_img, midas_model, midas_transform, device)

        # 简单融合策略: 覆盖局部区域到全局深度图
        fused_depth[y:y+h, x:x+w] = local_depth

    # 7. 保存或显示结果
    # 深度图归一化到 [0, 255] 便于可视化
    global_vis = (global_depth * 255).astype(np.uint8)
    fused_vis = (fused_depth * 255).astype(np.uint8)

    cv2.imwrite("global_depth.png", global_vis)
    cv2.imwrite("fused_depth.png", fused_vis)
    print("深度图已保存为 global_depth.png 和 fused_depth.png")

    # 可显示结果
    # cv2.imshow("Global Depth", global_vis)
    # cv2.imshow("Fused Depth", fused_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
