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

# 添加DiffMS源代码路径
diffms_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(diffms_path))

import hydra
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


def load_config(checkpoint_path: str):
    """
    从checkpoint加载配置并修改为推理模式
    
    Args:
        checkpoint_path: 预训练模型路径
    
    Returns:
        cfg: 配置对象
    """
    logger = logging.getLogger(__name__)
    logger.info(f"从checkpoint加载配置: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 从checkpoint中获取配置
    if 'hyper_parameters' in checkpoint:
        cfg = checkpoint['hyper_parameters'].get('cfg', None)
    else:
        raise ValueError("Checkpoint中未找到配置信息")
    
    if cfg is None:
        raise ValueError("无法从checkpoint提取配置")
    
    # 转换为OmegaConf对象（如果还不是的话）
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    
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
    logger = logging.getLogger(__name__)
    
    # 创建配置副本
    cfg = cfg.copy()
    
    # 修改数据集为自定义数据集
    cfg.dataset.name = 'custom_data'
    cfg.dataset.datadir = '/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data'
    cfg.dataset.split_file = '/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data/split.tsv'
    cfg.dataset.labels_file = '/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data/labels.tsv'
    cfg.dataset.spec_folder = '/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data/spec_files'
    cfg.dataset.subform_folder = '/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data/subformulae/default_subformulae'
    
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
    
    logger.info("配置已修改为推理模式")
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
    
    # 禁用RDKit警告
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置工作目录为DiffMS/src
    diffms_src_dir = Path(__file__).parent.parent / "src"
    os.chdir(str(diffms_src_dir))
    logger.info(f"工作目录: {os.getcwd()}")
    
    # 加载和修改配置
    logger.info("\n步骤 1: 加载配置")
    cfg = load_config(checkpoint_path)
    cfg = modify_config_for_inference(cfg, checkpoint_path, max_count=max_count)
    
    # 创建数据模块
    logger.info("\n步骤 2: 创建数据模块")
    datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)
    dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)
    
    # 创建特征
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
    
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features
    )
    
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
    
    # 加载模型
    logger.info("\n步骤 3: 加载预训练模型")
    logger.info(f"Checkpoint路径: {checkpoint_path}")
    
    model = Spec2MolDenoisingDiffusion.load_from_checkpoint(
        checkpoint_path,
        **model_kwargs
    )
    
    # 更新模型配置
    model.cfg = cfg
    
    logger.info("模型加载成功")
    
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

