"""
DiffMS Modal Inference
使用Modal云平台运行DiffMS分子结构预测
"""

import modal
from pathlib import Path

# 创建Modal App
app = modal.App("diffms-inference")

# DiffMS路径（本地）
DIFFMS_PATH = Path("/Users/aylin/yaolab_projects/diffms_yaolab/DiffMS")
DIFFMS_SRC_PATH = DIFFMS_PATH / "src"
DIFFMS_CONFIGS_PATH = DIFFMS_PATH / "configs"

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
    # 第四步：添加DiffMS源代码和配置到容器
    .add_local_dir(
        str(DIFFMS_SRC_PATH),
        "/root/src"
    )
    .add_local_dir(
        str(DIFFMS_CONFIGS_PATH),
        "/root/configs"
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
def run_inference(max_count: int = None, data_subdir: str = "processed_data"):
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
    processed_data_dir = data_path / data_subdir  # 支持选择数据子目录
    
    # 检查文件是否存在
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    if not processed_data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {processed_data_dir}")
    
    logger.info(f"✓ Checkpoint文件: {checkpoint_path}")
    logger.info(f"✓ 数据目录: {processed_data_dir}")
    
    # 1. 从YAML加载配置（checkpoint只包含权重，没有配置）
    logger.info("从YAML配置文件加载配置...")
    from hydra import compose, initialize_config_dir
    
    # DiffMS配置目录（容器内路径）
    config_dir = Path("/root/configs")
    logger.info(f"配置目录: {config_dir}")
    
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["dataset=msg"],  # 使用MSG配置作为基础
        )
    
    logger.info("✓ 配置加载成功")
    
    # 2. 修改配置为推理模式
    logger.info("修改配置为推理模式...")
    
    # 允许添加新字段到配置
    OmegaConf.set_struct(cfg, False)
    
    # 检查必要的文件
    required_files = {
        'split.tsv': processed_data_dir / 'split.tsv',
        'labels.tsv': processed_data_dir / 'labels.tsv',
        'spec_folder': processed_data_dir / 'spec_files',
        'subform_folder': processed_data_dir / 'subformulae' / 'default_subformulae',
    }
    
    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"缺少必要文件/目录: {name} -> {path}")
        logger.info(f"  ✓ {name}: {path}")
    
    # 更新配置
    cfg.dataset.name = 'custom_data'
    cfg.dataset.datadir = str(processed_data_dir)
    cfg.dataset.split_file = str(required_files['split.tsv'])
    cfg.dataset.labels_file = str(required_files['labels.tsv'])
    cfg.dataset.spec_folder = str(required_files['spec_folder'])
    cfg.dataset.subform_folder = str(required_files['subform_folder'])
    cfg.dataset.stats_dir = str(stats_path)
    
    # 关键配置：允许空SMILES（推理模式）
    cfg.dataset.allow_none_smiles = True
    
    if max_count is not None:
        cfg.dataset.max_count = max_count
        logger.info(f"  ⚠ 限制测试数据量: {max_count}")
    
    cfg.general.test_only = str(checkpoint_path)
    cfg.general.name = 'modal_inference'
    cfg.general.gpus = 1 if torch.cuda.is_available() else 0
    cfg.general.test_samples_to_generate = 10  # 减少采样数量（测试用10，生产可改为100）
    cfg.general.wandb = 'disabled'
    
    # 使用MSG Large Model配置（与checkpoint匹配）
    cfg.model.encoder_hidden_dim = 512       # Large Model (MSG)
    cfg.model.encoder_magma_modulo = 2048    # Large Model (MSG)
    
    logger.info("✓ 配置修改完成（使用MSG Large Model配置）")
    
    # 3. 创建输出目录
    logger.info("创建输出目录...")
    preds_dir = output_path / "predictions"
    logs_dir = output_path / "logs"
    preds_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建preds目录（用于保存pkl文件）
    preds_pkl_dir = Path("/root/src/preds")
    preds_pkl_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  ✓ 预测输出: {preds_dir}")
    logger.info(f"  ✓ 日志目录: {logs_dir}")
    logger.info(f"  ✓ PKL输出: {preds_pkl_dir}")
    
    # 4. 创建数据模块
    logger.info("步骤 4: 创建数据模块...")
    try:
        datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)
        logger.info("✓ 数据模块创建成功")
    except Exception as e:
        logger.error(f"✗ 数据模块创建失败: {e}")
        raise
    
    # 5. 加载数据集信息
    logger.info("步骤 5: 加载数据集信息...")
    try:
        dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)
        logger.info("✓ 数据集信息加载成功")
    except Exception as e:
        logger.error(f"✗ 数据集信息加载失败: {e}")
        raise
    
    # 6. 创建特征提取器（顺序很重要！）
    logger.info("步骤 6: 创建特征提取器...")
    try:
        # 推理模式：直接从checkpoint/config设置维度，不从数据推导
        # 这些维度必须与训练时的维度一致
        logger.info("推理模式：使用checkpoint中的固定维度")
        
        # MSG数据集的标准维度（从checkpoint推导）
        dataset_infos.input_dims = {
            'X': 16,    # 从checkpoint error可以看到：16
            'E': 5,     # 5种边类型
            'y': 2061   # 从checkpoint error可以看到：2061
        }
        dataset_infos.output_dims = {
            'X': 8,     # 8种原子类型
            'E': 5,     # 5种边类型
            'y': 2048   # 从checkpoint mlp_out_y看到
        }
        
        # 创建特征（但不会被调用，因为我们不重新计算维度）
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        if cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            logger.info("  ✓ 使用额外特征")
        else:
            extra_features = DummyExtraFeatures()
            logger.info("  ✓ 不使用额外特征")
        
        logger.info("✓ 特征提取器创建成功")
    except Exception as e:
        logger.error(f"✗ 特征提取器创建失败: {e}")
        raise
    
    # 7. 创建模型组件
    logger.info("步骤 7: 创建模型组件...")
    try:
        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        visualization_tools = MolecularVisualization(
            cfg.dataset.remove_h, 
            dataset_infos=dataset_infos
        )
        logger.info("✓ 模型组件创建成功")
    except Exception as e:
        logger.error(f"✗ 模型组件创建失败: {e}")
        raise
    
    # 8. 创建模型并加载权重（checkpoint只包含权重）
    logger.info("步骤 8: 创建模型并加载权重...")
    try:
        # 创建模型
        model = Spec2MolDenoisingDiffusion(
            cfg=cfg,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            visualization_tools=visualization_tools,
            extra_features=extra_features,
            domain_features=domain_features
        )
        logger.info("  ✓ 模型创建成功")
        
        # 加载权重
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("  ✓ 从checkpoint['state_dict']加载权重")
        else:
            model.load_state_dict(checkpoint)
            logger.info("  ✓ 直接加载checkpoint作为权重")
        
        logger.info("✓ 模型和权重加载成功")
    except Exception as e:
        logger.error(f"✗ 模型加载失败: {e}")
        raise
    
    # 9. 创建Trainer
    logger.info("步骤 9: 创建Trainer...")
    csv_logger = CSVLogger(str(logs_dir), name='modal_inference')
    
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    
    trainer = Trainer(
        accelerator='gpu' if use_gpu else 'cpu',
        devices=cfg.general.gpus if use_gpu else 1,
        logger=csv_logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    logger.info(f"✓ Trainer创建成功 (设备: {'GPU' if use_gpu else 'CPU'})")
    
    # 10. 运行推理
    logger.info("=" * 80)
    logger.info("步骤 10: 开始推理...")
    logger.info("=" * 80)
    
    try:
        trainer.test(model, datamodule=datamodule)
        logger.info("=" * 80)
        logger.info("✓ 推理完成！")
        logger.info("=" * 80)
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"✗ 推理失败: {e}")
        logger.error("=" * 80)
        raise
    
    # 将pkl文件复制到outputs volume
    logger.info("复制预测结果到outputs volume...")
    import shutil
    for pkl_file in preds_pkl_dir.glob("*.pkl"):
        dest = preds_dir / pkl_file.name
        shutil.copy2(pkl_file, dest)
        logger.info(f"  ✓ 复制: {pkl_file.name}")
    
    logger.info(f"结果保存在: {output_path}")
    logger.info(f"  - 预测结果: {preds_dir}")
    logger.info(f"  - 日志文件: {logs_dir}")
    
    # 确保volume更新被保存
    logger.info("保存volume更新...")
    output_volume.commit()
    logger.info("✓ Volume更新已保存")
    
    return {
        "status": "success",
        "max_count": max_count,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "output_path": str(output_path),
        "predictions_dir": str(preds_dir),
        "logs_dir": str(logs_dir)
    }


@app.local_entrypoint()
def main(max_count: int = None, data_subdir: str = "processed_data"):
    """
    本地入口点 - 从命令行调用
    
    使用方法:
        modal run diffms_inference.py --max-count 10
    """
    result = run_inference.remote(max_count=max_count, data_subdir=data_subdir)
    print("\n" + "=" * 60)
    print("推理完成！")
    print(f"状态: {result['status']}")
    print(f"数据目录: {data_subdir}")
    print(f"处理数据量: {result['max_count'] or '全部'}")
    print(f"使用GPU: {result['gpu']}")
    print("=" * 60)
    print(f"\n下载结果: modal volume get diffms-outputs {output_path} ./local_outputs")

