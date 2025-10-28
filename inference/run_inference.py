#!/usr/bin/env python
"""
推理脚本：使用预训练的DiffMS模型对自定义质谱数据进行分子结构预测

使用方法:
    python run_inference.py --checkpoint_path <预训练模型路径> [--output_dir <输出目录>]

示例:
    python run_inference.py --checkpoint_path /path/to/checkpoint.ckpt --output_dir ./predictions
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import torch


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('inference.log')
        ]
    )
    return logging.getLogger(__name__)


def load_config_from_yaml():
    """
    从YAML配置文件加载配置（因为checkpoint只有权重）
    
    Returns:
        cfg: 配置对象
    """
    from omegaconf import DictConfig, OmegaConf
    import hydra
    from hydra import compose, initialize_config_dir
    
    logger = logging.getLogger(__name__)
    logger.info("从YAML配置文件加载配置...")
    
    # DiffMS配置目录
    config_dir = Path.cwd() / ".." / "configs"  # 相对于src目录
    config_dir = config_dir.resolve()
    
    logger.info(f"配置目录: {config_dir}")
    
    # 使用Hydra加载配置
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # 加载msg数据集配置
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=msg",  # 使用MSG数据集配置作为基础
            ]
        )
    
    logger.info("✓ 配置加载成功")
    return cfg


def modify_config_for_inference(cfg, checkpoint_path: str, max_count: int = None):
    """
    修改配置用于推理
    
    Args:
        cfg: 原始配置
        checkpoint_path: 预训练模型路径
        max_count: 最大测试数据量（用于测试性能）
    
    Returns:
        modified_cfg: 修改后的配置
    """
    from omegaconf import OmegaConf
    
    logger = logging.getLogger(__name__)
    
    # 创建配置副本
    cfg = cfg.copy()
    
    # 允许添加新字段
    OmegaConf.set_struct(cfg, False)
    
    # 修改数据集为自定义数据集
    cfg.dataset.name = 'custom_data'
    cfg.dataset.datadir = '/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data'
    cfg.dataset.split_file = '/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data/split.tsv'
    cfg.dataset.labels_file = '/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data/labels.tsv'
    cfg.dataset.spec_folder = '/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data/spec_files'
    cfg.dataset.subform_folder = '/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data/subformulae/default_subformulae'
    
    # 关键配置：允许空SMILES（推理模式）
    cfg.dataset.allow_none_smiles = True
    
    # 使用MSG的统计信息（绝对路径）
    cfg.dataset.stats_dir = '/Users/aylin/Downloads/msg'
    
    # 限制测试数据量（用于快速测试）
    if max_count is not None:
        cfg.dataset.max_count = max_count
        logger.info(f"限制测试数据量: {max_count}")
    
    # 设置为测试模式
    cfg.general.test_only = checkpoint_path
    cfg.general.name = 'custom_inference'
    cfg.general.gpus = 1 if torch.cuda.is_available() else 0
    
    # 生成样本数量
    cfg.general.test_samples_to_generate = 100  # 可以根据需要调整
    
    # 关闭wandb
    cfg.general.wandb = 'disabled'
    
    # 使用MSG Large Model配置（与checkpoint匹配）
    cfg.model.encoder_hidden_dim = 512       # Large Model (MSG)
    cfg.model.encoder_magma_modulo = 2048    # Large Model (MSG)
    
    logger.info("配置已修改为推理模式（使用MSG Large Model配置）")
    logger.info(f"数据集: {cfg.dataset.name}")
    logger.info(f"数据路径: {cfg.dataset.datadir}")
    logger.info(f"使用GPU: {cfg.general.gpus > 0}")
    
    return cfg


def run_inference(checkpoint_path: str, output_dir: str = "./predictions", max_count: int = None):
    """
    运行推理
    
    Args:
        checkpoint_path: 预训练模型路径
        output_dir: 输出目录
        max_count: 最大测试数据量（用于测试性能），None表示测试全部数据
    """
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("开始推理")
    logger.info("=" * 80)
    
    # 设置Python路径和工作目录
    diffms_dir = Path(__file__).parent.parent / "DiffMS"
    diffms_src_dir = diffms_dir / "src"
    
    # 添加两个路径：DiffMS用于 'from src import'，DiffMS/src用于相对导入
    sys.path.insert(0, str(diffms_src_dir))
    sys.path.insert(0, str(diffms_dir))
    
    # 切换到src目录（配置文件在这里）
    os.chdir(str(diffms_src_dir))
    logger.info(f"工作目录: {os.getcwd()}")
    
    # 导入DiffMS模块（在设置路径后）
    from omegaconf import DictConfig, OmegaConf
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger
    from src import utils
    from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
    from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
    from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
    from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
    from src.analysis.visualization import MolecularVisualization
    from src.datasets import spec2mol_dataset
    
    # 禁用RDKit警告
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载和修改配置
    logger.info("\n步骤 1: 加载配置")
    cfg = load_config_from_yaml()
    cfg = modify_config_for_inference(cfg, checkpoint_path, max_count=max_count)
    
    # 创建数据模块
    logger.info("\n步骤 2: 创建数据模块")
    datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)
    logger.info(f"  Train数据集大小: {len(datamodule.train_dataset)}")
    logger.info(f"  Val数据集大小: {len(datamodule.val_dataset)}")
    logger.info(f"  Test数据集大小: {len(datamodule.test_dataset)}")
    
    dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)
    
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
    else:
        extra_features = DummyExtraFeatures()
    
    # 创建指标和可视化工具
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    visualization_tools = MolecularVisualization(
        cfg.dataset.remove_h,
        dataset_infos=dataset_infos
    )
    
    model_kwargs = {
        'dataset_infos': dataset_infos,
        'train_metrics': train_metrics,
        'visualization_tools': visualization_tools,
        'extra_features': extra_features,
        'domain_features': domain_features
    }
    
    # 创建模型
    logger.info("\n步骤 3: 创建模型")
    model = Spec2MolDenoisingDiffusion(
        cfg=cfg,
        **model_kwargs
    )
    
    # 加载权重
    logger.info(f"加载权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("✓ 从checkpoint['state_dict']加载权重")
    else:
        # 如果checkpoint本身就是state_dict
        model.load_state_dict(checkpoint)
        logger.info("✓ 直接加载checkpoint作为权重")
    
    logger.info("✓ 模型创建和权重加载成功")
    
    # 创建Trainer
    logger.info("\n步骤 4: 创建Trainer并运行推理")
    
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    
    # 创建日志记录器
    csv_logger = CSVLogger(save_dir=str(output_dir), name='inference_logs')
    
    trainer = Trainer(
        accelerator='gpu' if use_gpu else 'cpu',
        devices=cfg.general.gpus if use_gpu else 1,
        logger=csv_logger,
        enable_progress_bar=True,
    )
    
    # 运行测试
    logger.info("开始生成分子结构...")
    results = trainer.test(model, datamodule=datamodule)
    
    logger.info("\n" + "=" * 80)
    logger.info("推理完成！")
    logger.info("=" * 80)
    logger.info(f"\n结果保存在: {output_dir}")
    logger.info(f"预测结果: {output_dir.parent / 'preds'}")
    logger.info(f"日志文件: {output_dir / 'inference_logs'}")
    
    # 打印测试结果
    if results:
        logger.info("\n测试指标:")
        for key, value in results[0].items():
            logger.info(f"  {key}: {value}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DiffMS推理脚本')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='预训练模型checkpoint路径（.ckpt文件）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./predictions',
        help='输出目录，默认为./predictions'
    )
    parser.add_argument(
        '--max_count',
        type=int,
        default=None,
        help='最大测试数据量（用于快速测试），默认为None（测试全部数据）'
    )
    
    args = parser.parse_args()
    
    # 检查checkpoint文件是否存在
    if not Path(args.checkpoint_path).exists():
        print(f"错误: 未找到checkpoint文件: {args.checkpoint_path}")
        sys.exit(1)
    
    # 运行推理
    run_inference(args.checkpoint_path, args.output_dir, max_count=args.max_count)


if __name__ == "__main__":
    main()

