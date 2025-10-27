"""
DiffMS Modal Inference
使用Modal云平台运行DiffMS分子结构预测
"""

import modal
from pathlib import Path

# 创建Modal App
app = modal.App("diffms-inference")

# 定义容器镜像 - 包含DiffMS运行所需的所有依赖
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        # 系统依赖
        ["git", "wget"]
    )
    .pip_install(
        # PyTorch相关 (CUDA 11.8)
        "torch==2.0.1",
        "torchvision==0.15.2",
        # PyTorch Lightning
        "pytorch-lightning==2.0.0",
        # 化学信息学
        "rdkit==2023.3.2",
        # 数据处理
        "pandas==2.0.3",
        "numpy==1.24.3",
        # 配置管理
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        # 图神经网络
        "torch-geometric==2.3.1",
        "torch-scatter==2.1.1",
        "torch-sparse==0.6.17",
        # 其他依赖
        "tqdm==4.65.0",
        "h5py==3.9.0",
        "networkx==3.1",
    )
)

# 创建持久化Volume用于存储数据和模型
# 数据Volume - 存储预处理的质谱数据
data_volume = modal.Volume.from_name("diffms-data", create_if_missing=True)
data_path = Path("/data")

# 模型Volume - 存储预训练模型checkpoint
model_volume = modal.Volume.from_name("diffms-models", create_if_missing=True)
model_path = Path("/models")

# 输出Volume - 存储推理结果
output_volume = modal.Volume.from_name("diffms-outputs", create_if_missing=True)
output_path = Path("/outputs")

# MSG统计信息Volume
stats_volume = modal.Volume.from_name("diffms-msg-stats", create_if_missing=True)
stats_path = Path("/msg_stats")

# 时间常量
MINUTES = 60
HOURS = 60 * MINUTES

