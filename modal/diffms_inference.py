"""
DiffMS Modal Inference
使用Modal云平台运行DiffMS分子结构预测
"""

import modal
from pathlib import Path

# 创建Modal App
app = modal.App("diffms-inference")

# DiffMS源代码路径（本地）
DIFFMS_SRC_PATH = Path("/Users/aylin/yaolab_projects/diffms_yaolab/DiffMS/src")

# 定义容器镜像 - 包含DiffMS运行所需的所有依赖
# 分阶段安装，避免依赖顺序问题
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        # 系统依赖
        "git",
        "wget",
        # X11和图形库（RDKit绘图需要）
        "libxrender1",
        "libxext6",
        "libsm6",
        "libice6",
        "libx11-6",
        "libglib2.0-0"
    )
    # 第一步：安装PyTorch
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
    )
    # 第二步：安装依赖PyTorch的图神经网络库  
    .pip_install(
        "torch-scatter==2.1.1",
        "torch-sparse==0.6.17",
    )
    # 第三步：安装torch-geometric和其他依赖
    .pip_install(
        "torch-geometric==2.3.1",
        "pytorch-lightning==2.0.0",
        "rdkit==2023.3.2",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "tqdm==4.65.0",
        "h5py==3.9.0",
        "networkx==3.1",
        "wandb",
        "matplotlib",
        "seaborn",
    )
    # 第四步：添加DiffMS源代码到容器
    .add_local_dir(
        str(DIFFMS_SRC_PATH),
        "/root/src"
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


@app.function(
    image=image,
    volumes={
        str(data_path): data_volume,
        str(model_path): model_volume,
        str(output_path): output_volume,
        str(stats_path): stats_volume,
    },
    gpu="A100",  # 或 "H100", "T4", "A10G"
    timeout=4 * HOURS,
)
def run_inference(max_count: int = None):
    """
    在Modal云端运行DiffMS推理
    
    Args:
        max_count: 限制处理的数据点数量（用于测试），None表示处理所有数据
    """
    import sys
    import os
    import logging
    from pathlib import Path
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger
    from omegaconf import DictConfig, OmegaConf
    
    # 添加DiffMS源代码到Python路径
    diffms_src = Path("/root/src")
    sys.path.insert(0, str(diffms_src))
    os.chdir(str(diffms_src))
    
    # 导入DiffMS模块
    from src import utils
    from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
    from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
    from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
    from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
    from src.analysis.visualization import MolecularVisualization
    from src.datasets import spec2mol_dataset
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 禁用RDKit警告
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    logger.info("=" * 80)
    logger.info("开始 DiffMS 推理 on Modal")
    logger.info("=" * 80)
    
    # 显示配置
    logger.info(f"数据路径: {data_path}")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"统计路径: {stats_path}")
    logger.info(f"GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
    logger.info(f"处理数据量: {'全部' if max_count is None else max_count}")
    
    # 定义路径
    checkpoint_path = model_path / "diffms_msg.ckpt"
    processed_data_dir = data_path / "processed_data"
    
    logger.info(f"加载checkpoint: {checkpoint_path}")
    
    # 1. 加载checkpoint和配置
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    if 'hyper_parameters' in checkpoint:
        cfg = checkpoint['hyper_parameters'].get('cfg', None)
    else:
        raise ValueError("Checkpoint中未找到配置信息")
    
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    
    # 2. 修改配置为推理模式
    cfg.dataset.name = 'custom_data'
    cfg.dataset.datadir = str(processed_data_dir)
    cfg.dataset.split_file = str(processed_data_dir / 'split.tsv')
    cfg.dataset.labels_file = str(processed_data_dir / 'labels.tsv')
    cfg.dataset.spec_folder = str(processed_data_dir / 'spec_files')
    cfg.dataset.subform_folder = str(processed_data_dir / 'subformulae' / 'default_subformulae')
    cfg.dataset.stats_dir = str(stats_path)
    
    if max_count is not None:
        cfg.dataset.max_count = max_count
        logger.info(f"限制测试数据量: {max_count}")
    
    cfg.general.test_only = str(checkpoint_path)
    cfg.general.name = 'modal_inference'
    cfg.general.gpus = 1 if torch.cuda.is_available() else 0
    cfg.general.test_samples_to_generate = 100
    cfg.general.wandb = 'disabled'
    
    logger.info("配置已修改为推理模式")
    
    # 3. 创建输出目录
    preds_dir = output_path / "predictions"
    logs_dir = output_path / "logs"
    preds_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. 创建数据模块
    logger.info("创建数据模块...")
    datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)
    
    # 5. 加载数据集信息
    logger.info("加载数据集信息...")
    dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)
    train_smiles = datamodule.train_smiles if hasattr(datamodule, 'train_smiles') else None
    
    # 6. 创建模型
    logger.info("创建模型...")
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
    
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, 
        extra_features=extra_features, 
        domain_features=domain_features
    )
    
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    visualization_tools = MolecularVisualization(
        cfg.dataset.remove_h, 
        dataset_infos=dataset_infos
    )
    
    model = Spec2MolDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        visualization_tools=visualization_tools,
        extra_features=extra_features,
        domain_features=domain_features,
        train_smiles=train_smiles
    )
    
    # 7. 加载权重
    logger.info(f"加载预训练权重...")
    model.load_state_dict(checkpoint['state_dict'])
    
    # 8. 创建Trainer
    logger.info("创建Trainer...")
    csv_logger = CSVLogger(str(logs_dir), name='modal_inference')
    
    trainer = Trainer(
        accelerator='gpu' if cfg.general.gpus > 0 else 'cpu',
        devices=cfg.general.gpus if cfg.general.gpus > 0 else 1,
        logger=csv_logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    # 9. 运行推理
    logger.info("=" * 80)
    logger.info("开始推理...")
    logger.info("=" * 80)
    
    trainer.test(model, datamodule=datamodule)
    
    logger.info("=" * 80)
    logger.info("推理完成！")
    logger.info("=" * 80)
    logger.info(f"结果保存在: {output_path}")
    
    # 确保volume更新被保存
    output_volume.commit()
    
    return {
        "status": "success",
        "max_count": max_count,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "output_path": str(output_path)
    }


@app.local_entrypoint()
def main(max_count: int = None):
    """
    本地入口点 - 从命令行调用
    
    使用方法:
        modal run diffms_inference.py --max-count 10
    """
    result = run_inference.remote(max_count=max_count)
    print("\n" + "=" * 60)
    print("推理完成！")
    print(f"状态: {result['status']}")
    print(f"处理数据量: {result['max_count'] or '全部'}")
    print(f"使用GPU: {result['gpu']}")
    print("=" * 60)
    print(f"\n下载结果: modal volume get diffms-outputs {output_path} ./local_outputs")

