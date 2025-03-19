import json
import os
import subprocess
from datetime import datetime

from third_party_libs import depth_init, load_depth_model,depth_init_with_local_fusion

# 定义 JSON 配置，非必要参数设置为 False
config = {
    "source_path": "data/church",  # 数据源路径
    "model_path": "output/church_0725",  # 模型保存路径
    "iterations": 30000,  # 训练迭代次数
    "test_iterations": False,  # 测试迭代次数，默认值: "7000 30000"
    "save_iterations": False,  # 保存模型的迭代次数，默认值: "7000 30000 <iterations>"
    "checkpoint_iterations": 29000,  # 检查点迭代次数，默认值: None
    "start_checkpoint": False,  # 起始检查点，默认值: None
    "eval": False,  # 是否进行评估，默认值: False
    "resolution": False,  # 输出分辨率，默认值: None
    "data_device": "cuda",  # 设备类型，默认值: "cuda"
    "white_background": False,  # 是否使用白色背景，默认值: False
    "images": False,  # 图片目录，默认值: "images"
    "sh_degree": False,  # 球谐函数的度数，默认值: 3
    "convert_SHs_python": False,  # 是否使用 Python 转换 SH，默认值: False
    "convert_cov3D_python": False,  # 是否使用 Python 转换 3D 体积，默认值: False
    "debug": False,  # 是否启用调试模式，默认值: False
    "debug_from": False,  # 从哪个迭代开始调试，默认值: None
    "ip": False,  # 服务器 IP 地址，默认值: "127.0.0.1"
    "port": False,  # 服务器端口，默认值: 6009
    "quiet": False,  # 是否静默运行，默认值: False
    "feature_lr": False,  # 特征学习率，默认值: 0.0025
    "opacity_lr": False,  # 不透明度学习率，默认值: 0.05
    "scaling_lr": False,  # 缩放学习率，默认值: 0.005
    "rotation_lr": False,  # 旋转学习率，默认值: 0.001
    "position_lr_max_steps": False,  # 位置学习率最大步数，默认值: 30000
    "position_lr_init": False,  # 初始位置学习率，默认值: 0.00016
    "position_lr_final": False,  # 最终位置学习率，默认值: 0.0000016
    "position_lr_delay_mult": False,  # 位置学习率延迟乘数，默认值: 0.01
    "densify_from_iter": False,  # 从哪个迭代开始稠密化，默认值: 500
    "densify_until_iter": False,  # 直到哪个迭代稠密化，默认值: 15000
    "densify_grad_threshold": False,  # 稠密化梯度阈值，默认值: 0.0002
    "densification_interval": False,  # 稠密化间隔，默认值: 100
    "opacity_reset_interval": False,  # 不透明度重置间隔，默认值: 3000
    "lambda_dssim": False,  # DSSIM 权重，默认值: 0.2
    "percent_dense": False  # 稠密化百分比，默认值: 0.01
}


def render_run_by_json(model_path):
    # 生成命令行字符串
    # command = ["python", "render.py", "-m", model_path, "--skip_train"]
    command = ["python", "render.py", "-m", model_path]
    command_str = ' '.join(command)
    print('render command:', command_str)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("命令执行失败:\n", e.stderr)


def metrics_run_by_json(model_path):
    # 生成命令行字符串
    command = ["python", "metrics.py", "-m", model_path]
    command_str = ' '.join(command)
    print('metrics command:', command_str)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("命令执行失败:\n", e.stderr)


def train_run_by_json(config):
    # 生成命令行字符串
    command = ["python", "train.py"]

    # 遍历配置生成参数
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        elif value is not None:
            command.append(f"--{key}")
            command.append(f"{value}")
    command.append('--eval')

    command_str = ' '.join(command)
    print('train command:', command_str)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("命令执行失败:\n", e.stderr)

    # 获取模型路径
    model_path = config["model_path"]

    # 确保目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # 保存 config 为 JSON 文件
    json_file_path = os.path.join(model_path, "config.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(config, json_file, indent=4)

    print(f"配置已保存到: {json_file_path}")


def ablation_experiments():
    # 定义 JSON 配置，非必要参数设置为 False
    configs = [
        # # task1: 测试不同类型深度图类型对深度几何约束的影响
        # {
        #     "task": "Task1",
        #     "contribution": False,  # True, False
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',  # 'Multi-view', 'Single-view', 'NONE'
        #     "depth_type": 'mono',  # mono, lidar, fusion
        #     "depth_weight": 0.2,  # 0~1
        #     "voxel": False,  # True, False
        #     "voxel_range": 150,  # 50 ~ 1000
        # },
        # {
        #     "task": "Task1",
        #     "contribution": False,
        #     "contribution_range": 0.05,
        #     "depth": 'Multi-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task1",
        #     "contribution": False,
        #     "contribution_range": 0.05,
        #     "depth": 'Multi-view',
        #     "depth_type": 'lidar',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        {
            "task": "Task1",
            "contribution": False,
            "contribution_range": 0.05,
            "depth": 'Multi-view',
            "depth_type": 'fusion',
            "depth_weight": 0.2,
            "voxel": False,
            "voxel_range": 150,
        },
        #
        # # task2: 测试渐进式引入深度的影响
        # {
        #     "task": "Task2",
        #     "contribution": False,  # True, False
        #     "contribution_range": 0,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task2",
        #     "contribution": False,
        #     "contribution_range": 0,
        #     "depth": 'Multi-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task2",
        #     "contribution": False,
        #     "contribution_range": 0,
        #     "depth": 'Single-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task2",
        #     "contribution": False,
        #     "contribution_range": 0,
        #     "depth": 'Single-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.1,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task2",
        #     "contribution": False,
        #     "contribution_range": 0,
        #     "depth": 'Multi-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.1,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task2",
        #     "contribution": False,
        #     "contribution_range": 0,
        #     "depth": 'Multi-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.3,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task2",
        #     "contribution": False,
        #     "contribution_range": 0,
        #     "depth": 'Multi-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.4,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task2",
        #     "contribution": False,
        #     "contribution_range": 0,
        #     "depth": 'Single-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.3,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task2",
        #     "contribution": False,
        #     "contribution_range": 0,
        #     "depth": 'Single-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.4,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        #
        # # task3: 测试贡献度参数的影响
        # {
        #     "task": "Task3",
        #     "contribution": False,
        #     "contribution_range": 0,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task3",
        #     "contribution": True,
        #     "contribution_range": 0.02,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task3",
        #     "contribution": True,
        #     "contribution_range": 0.04,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task3",
        #     "contribution": True,
        #     "contribution_range": 0.06,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task3",
        #     "contribution": True,
        #     "contribution_range": 0.08,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task3",
        #     "contribution": True,
        #     "contribution_range": 0.1,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task3",
        #     "contribution": True,
        #     "contribution_range": 0.12,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task3",
        #     "contribution": True,
        #     "contribution_range": 0.14,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task3",
        #     "contribution": True,
        #     "contribution_range": 0.16,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task3",
        #     "contribution": True,
        #     "contribution_range": 0.18,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task3",
        #     "contribution": True,
        #     "contribution_range": 0.2,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },

        # task4: voxel range效果
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 50,
        # },
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 100,
        # },
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 200,
        # },
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 250,
        # },
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 300,
        # },
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 350,
        # },
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 400,
        # },
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 600,
        # },
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 800,
        # },
        # {
        #     "task": "Task4",
        #     "contribution": True,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 1000,
        # },

        # task5: 消融实验
        # {
        #     "task": "Task5",
        #     "contribution": False,
        #     "contribution_range": 0.05,
        #     "depth": 'NONE',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task5",
        #     "contribution": False,
        #     "contribution_range": 0.05,
        #     "depth": 'Multi-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task5",
        #     "contribution": True,
        #     "contribution_range": 0.02,
        #     "depth": 'Multi-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": False,
        #     "voxel_range": 150,
        # },
        # {
        #     "task": "Task5",
        #     "contribution": True,
        #     "contribution_range": 0.02,
        #     "depth": 'Multi-view',
        #     "depth_type": 'mono',
        #     "depth_weight": 0.2,
        #     "voxel": True,
        #     "voxel_range": 150,
        # },
    ]

    # 参数简写映射
    depth_abbr_map = {
        'Multi-view': 'MV',
        'Single-view': 'SV',
        'NONE': 'N'
    }

    depth_type_abbr_map = {
        'mono': 'M',
        'lidar': 'L',
        'fusion': 'F'
    }

    for experiment_config in configs:
        # 设置固定参数
        experiment_config["iterations"] = 20000
        experiment_config["test_iterations"] = False
        experiment_config["save_iterations"] = False
        experiment_config["white_background"] = True

        # 提取参数
        task = experiment_config.get("task", "Task")
        depth = experiment_config["depth"]
        voxel = experiment_config["voxel"]
        iterations = experiment_config["iterations"]
        contribution = experiment_config["contribution"]
        contribution_range = experiment_config["contribution_range"]
        voxel_range = experiment_config["voxel_range"]
        depth_type = experiment_config["depth_type"]
        depth_weight = experiment_config["depth_weight"]

        # 获取简写
        depth_abbr = depth_abbr_map.get(depth, depth)
        depth_type_abbr = depth_type_abbr_map.get(depth_type, depth_type)
        voxel_abbr = 'T' if voxel else 'F'

        # 构建模型路径
        model_path = f"output/{task}_it{iterations}"

        # 贡献度
        if contribution:
            model_path += f"_c{contribution_range:.2f}"
        else:
            model_path += f"_cF"

        # 深度
        model_path += f"_d{depth_abbr}"
        model_path += f"_dt{depth_type_abbr}"
        model_path += f"_dw{depth_weight:.1f}"

        # 体素化
        model_path += f"_v{voxel_abbr}"
        if voxel:
            model_path += f"{voxel_range}"

        model_path += "/"  # 确保路径以斜杠结尾

        # 更新模型路径
        experiment_config["source_path"] =  f'data/{data_dir}'
        experiment_config["model_path"] = model_path

        # 打印模型路径
        print(f"Running experiment with model path: {model_path}")

        # 运行训练、渲染和评估
        train_run_by_json(experiment_config)
        render_run_by_json(experiment_config["model_path"])
        metrics_run_by_json(experiment_config["model_path"])


if __name__ == '__main__':


    data_dir = 'data_path'
    data_path = f'data/{data_dir}'
    depth_path = f'data/{data_dir}/depth'

    # if not os.path.exists(depth_path):
    #     depth_anything, DEVICE = load_depth_model('vitb')
    #     depth_init(data_path + '/images', depth_path, depth_anything, DEVICE)
    #     # depth_init_with_local_fusion(data_path + '/images', depth_path, depth_anything, DEVICE)
    # else:
    #     print(f"Directory already exists: {depth_path}")
    # depth_anything, DEVICE = load_depth_model('vitb')
    # depth_init(data_path + '/images', depth_path, depth_anything, DEVICE)
    # depth_anything, DEVICE = load_depth_model('vitb')
    # depth_init_with_local_fusion(data_path + '/images', depth_path, depth_anything, DEVICE)

    ablation_experiments()

    # 'wandb sync wandb/offline-run-*-*'
