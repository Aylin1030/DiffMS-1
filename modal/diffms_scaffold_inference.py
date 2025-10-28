
"""
DiffMS Modal Scaffold-Constrained Inference
支持骨架约束的 DiffMS 推理（在 Modal 云平台运行）
"""

import modal
from pathlib import Path

# 创建Modal App
app = modal.App("diffms-scaffold-inference")

# DiffMS路径（本地）
DIFFMS_PATH = Path("/Users/aylin/yaolab_projects/diffms_yaolab/DiffMS")
DIFFMS_SRC_PATH = DIFFMS_PATH / "src"
DIFFMS_CONFIGS_PATH = DIFFMS_PATH / "configs"

# 定义容器镜像 - 包含DiffMS运行所需的所有依赖
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget",
        "libxrender1", "libxext6", "libsm6", "libice6", "libx11-6", "libglib2.0-0"
    )
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
    )
    .pip_install(
        "torch-scatter==2.1.1",
        "torch-sparse==0.6.17",
    )
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
    .add_local_dir(str(DIFFMS_SRC_PATH), "/root/src")
    .add_local_dir(str(DIFFMS_CONFIGS_PATH), "/root/configs")
)

# 创建持久化Volume
data_volume = modal.Volume.from_name("diffms-data", create_if_missing=True)
data_path = Path("/data")

model_volume = modal.Volume.from_name("diffms-models", create_if_missing=True)
model_path = Path("/models")

output_volume = modal.Volume.from_name("diffms-outputs", create_if_missing=True)
output_path = Path("/outputs")

stats_volume = modal.Volume.from_name("diffms-msg-stats", create_if_missing=True)
stats_path = Path("/msg_stats")

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
    gpu="A100",
    timeout=4 * HOURS,
)
def run_scaffold_inference(
    scaffold_smiles: str,
    max_count: int = None,
    data_subdir: str = "processed_data",
    attachment_indices: str = None,
    enforce_scaffold: bool = True,
    use_rerank: bool = True
):
    """
    在Modal云端运行骨架约束的DiffMS推理
    
    Args:
        scaffold_smiles: 骨架的SMILES字符串
        max_count: 限制处理的数据点数量
        data_subdir: 数据子目录名称
        attachment_indices: 锚点索引（逗号分隔），如 "2,5,7"
        enforce_scaffold: 是否强制要求包含骨架
        use_rerank: 是否启用谱重排
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
    
    # 导入骨架约束工具
    from src.inference.scaffold_hooks import smiles_to_mol, formula_of, formula_to_string
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    logger.info("=" * 80)
    logger.info("开始 DiffMS 骨架约束推理 on Modal")
    logger.info("=" * 80)
    
    # 验证和解析骨架
    logger.info(f"\n骨架信息:")
    logger.info(f"  SMILES: {scaffold_smiles}")
    
    try:
        scaffold_mol = smiles_to_mol(scaffold_smiles)
        scaffold_formula = formula_of(scaffold_mol)
        scaffold_formula_str = formula_to_string(scaffold_formula)
        logger.info(f"  分子式: {scaffold_formula_str}")
        logger.info(f"  重原子数: {scaffold_mol.GetNumAtoms()}")
        logger.info(f"  ✓ 骨架验证成功")
    except Exception as e:
        logger.error(f"  ✗ 骨架SMILES无效: {e}")
        raise
    
    # 显示配置
    logger.info(f"\n推理配置:")
    logger.info(f"  数据路径: {data_path}")
    logger.info(f"  模型路径: {model_path}")
    logger.info(f"  输出路径: {output_path}")
    logger.info(f"  数据子目录: {data_subdir}")
    logger.info(f"  GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU型号: {torch.cuda.get_device_name(0)}")
    logger.info(f"  处理数据量: {'全部' if max_count is None else max_count}")
    logger.info(f"  锚点索引: {attachment_indices or '未指定（允许所有位置）'}")
    logger.info(f"  强制骨架: {enforce_scaffold}")
    logger.info(f"  启用重排: {use_rerank}")
    
    # 定义路径
    checkpoint_path = model_path / "diffms_msg.ckpt"
    processed_data_dir = data_path / data_subdir
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    if not processed_data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {processed_data_dir}")
    
    logger.info(f"\n✓ Checkpoint文件: {checkpoint_path}")
    logger.info(f"✓ 数据目录: {processed_data_dir}")
    
    # 1. 从YAML加载配置
    logger.info("\n步骤 1: 加载配置...")
    from hydra import compose, initialize_config_dir
    
    config_dir = Path("/root/configs")
    
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["dataset=msg"],
        )
    
    logger.info("✓ 配置加载成功")
    
    # 2. 修改配置为骨架约束推理模式
    logger.info("\n步骤 2: 配置骨架约束参数...")
    
    OmegaConf.set_struct(cfg, False)
    
    # 检查必要文件
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
    
    # 更新数据集配置
    cfg.dataset.name = 'custom_data'
    cfg.dataset.datadir = str(processed_data_dir)
    cfg.dataset.split_file = str(required_files['split.tsv'])
    cfg.dataset.labels_file = str(required_files['labels.tsv'])
    cfg.dataset.spec_folder = str(required_files['spec_folder'])
    cfg.dataset.subform_folder = str(required_files['subform_folder'])
    cfg.dataset.stats_dir = str(stats_path)
    cfg.dataset.allow_none_smiles = True
    
    if max_count is not None:
        cfg.dataset.max_count = max_count
        logger.info(f"  ⚠ 限制测试数据量: {max_count}")
    
    # 推理模式配置
    cfg.general.test_only = True
    cfg.general.name = 'scaffold_inference'
    cfg.general.gpus = 1 if torch.cuda.is_available() else 0
    cfg.general.test_samples_to_generate = 10
    cfg.general.wandb = 'disabled'
    cfg.general.decoder = None
    cfg.general.encoder = None
    
    # ===== 骨架约束配置（关键！）=====
    cfg.general.scaffold_smiles = scaffold_smiles
    cfg.general.attachment_indices = attachment_indices
    cfg.general.enforce_scaffold = enforce_scaffold
    cfg.general.use_rerank = use_rerank
    
    logger.info("\n骨架约束配置:")
    logger.info(f"  scaffold_smiles: {cfg.general.scaffold_smiles}")
    logger.info(f"  attachment_indices: {cfg.general.attachment_indices}")
    logger.info(f"  enforce_scaffold: {cfg.general.enforce_scaffold}")
    logger.info(f"  use_rerank: {cfg.general.use_rerank}")
    
    # MSG Large Model配置
    cfg.model.encoder_hidden_dim = 512
    cfg.model.encoder_magma_modulo = 2048
    
    logger.info("✓ 配置修改完成（使用MSG Large Model + 骨架约束）")
    
    # 3. 读取目标分子式并验证
    logger.info("\n步骤 3: 验证骨架与目标分子式的兼容性...")
    
    import pandas as pd
    labels_df = pd.DataFrame(pd.read_csv(required_files['labels.tsv'], sep='\t'))
    
    logger.info(f"  总共 {len(labels_df)} 个谱图")
    
    # 对每个样本，将目标分子式设置为labels中的formula
    # 注意：这里需要确保 target_formula 在运行时动态设置
    # 我们在配置中先不设置，让模型在test_step中读取
    logger.info("  将在推理时为每个样本动态设置 target_formula")
    
    # 检查骨架是否适用于至少一个样本
    from src.inference.scaffold_hooks import parse_formula, formula_subtract
    
    compatible_count = 0
    for idx, row in labels_df.iterrows():
        target_formula_str = row['formula']
        try:
            target_f = parse_formula(target_formula_str)
            delta_f = formula_subtract(target_f, scaffold_formula)
            compatible_count += 1
            logger.info(f"  ✓ {row['spec']}: {target_formula_str} (ΔF = {formula_to_string(delta_f)})")
        except ValueError as e:
            logger.warning(f"  ✗ {row['spec']}: {target_formula_str} - 骨架过大，跳过")
    
    if compatible_count == 0:
        logger.error("  ✗ 骨架与所有样本的分子式都不兼容！")
        logger.error(f"  骨架分子式: {scaffold_formula_str}")
        logger.error("  建议使用更小的骨架或检查分子式")
        raise ValueError("骨架与所有目标分子式不兼容")
    
    logger.info(f"\n  ✓ {compatible_count}/{len(labels_df)} 个样本与骨架兼容")
    
    # 将labels DataFrame保存到配置中，供后续使用
    # （实际上，我们需要修改 test_step 来读取 formula）
    
    # 4. 创建输出目录
    logger.info("\n步骤 4: 创建输出目录...")
    preds_dir = output_path / "predictions_scaffold"
    logs_dir = output_path / "logs_scaffold"
    smiles_dir = output_path / "smiles_scaffold"
    viz_dir = output_path / "visualizations_scaffold"
    
    preds_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    smiles_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    preds_pkl_dir = Path("/root/src/preds")
    preds_pkl_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  ✓ 预测输出: {preds_dir}")
    logger.info(f"  ✓ SMILES输出: {smiles_dir}")
    logger.info(f"  ✓ 可视化: {viz_dir}")
    logger.info(f"  ✓ 日志目录: {logs_dir}")
    
    # 5-8. 创建数据模块、模型等（与原脚本相同）
    logger.info("\n步骤 5-8: 创建数据模块和模型...")
    
    try:
        datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)
        dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)
        
        # 固定维度（MSG数据集）
        dataset_infos.input_dims = {'X': 16, 'E': 5, 'y': 2061}
        dataset_infos.output_dims = {'X': 8, 'E': 5, 'y': 2048}
        
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        if cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        
        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)
        
        # 创建模型
        model = Spec2MolDenoisingDiffusion(
            cfg=cfg,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            visualization_tools=visualization_tools,
            extra_features=extra_features,
            domain_features=domain_features
        )
        
        # 加载权重
        logger.info(f"  加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict, strict=True)
            logger.info("  ✓ 权重加载成功")
        else:
            model.load_state_dict(checkpoint, strict=True)
            logger.info("  ✓ 权重加载成功")
        
        logger.info("✓ 模型创建和权重加载完成")
        
    except Exception as e:
        logger.error(f"✗ 模型创建失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # 9. 创建Trainer
    logger.info("\n步骤 9: 创建Trainer...")
    csv_logger = CSVLogger(str(logs_dir), name='scaffold_inference')
    
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
    logger.info("\n" + "=" * 80)
    logger.info("步骤 10: 开始骨架约束推理...")
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
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # 11. 后处理
    logger.info("\n" + "=" * 80)
    logger.info("步骤 11: 后处理 - 转换和可视化")
    logger.info("=" * 80)
    
    import shutil
    pkl_files = list(preds_pkl_dir.glob("*.pkl"))
    for pkl_file in pkl_files:
        dest = preds_dir / pkl_file.name
        shutil.copy2(pkl_file, dest)
        logger.info(f"  ✓ 复制PKL: {pkl_file.name}")
    
    # 转换为SMILES
    logger.info("\n11.1 转换为SMILES...")
    
    try:
        from rdkit import Chem
        import pandas as pd
        import pickle
        
        def mol_to_canonical_smiles(mol):
            if mol is None:
                return None
            try:
                Chem.RemoveStereochemistry(mol)
                smiles = Chem.MolToSmiles(mol, canonical=True)
                test_mol = Chem.MolFromSmiles(smiles)
                if test_mol is None:
                    return None
                return smiles
            except Exception:
                return None
        
        # 合并所有pkl文件
        all_predictions = []
        for pkl_file in sorted(pkl_files):
            logger.info(f"  处理: {pkl_file.name}")
            with open(pkl_file, 'rb') as f:
                predictions = pickle.load(f)
                if isinstance(predictions, list):
                    all_predictions.extend(predictions)
        
        logger.info(f"  总共 {len(all_predictions)} 个谱图")
        
        # 转换为SMILES并验证骨架
        top1_data = []
        all_candidates_data = []
        valid_count = 0
        total_count = 0
        scaffold_match_count = 0
        
        for spec_idx, mol_list in enumerate(all_predictions):
            spec_id = f"spec_{spec_idx:04d}"
            
            if not isinstance(mol_list, list):
                mol_list = [mol_list]
            
            valid_smiles = []
            
            for rank, mol in enumerate(mol_list, start=1):
                total_count += 1
                smiles = mol_to_canonical_smiles(mol)
                
                if smiles is not None:
                    valid_count += 1
                    
                    # 验证是否包含骨架
                    contains_scaf = False
                    if mol is not None:
                        try:
                            contains_scaf = mol.HasSubstructMatch(scaffold_mol)
                            if contains_scaf:
                                scaffold_match_count += 1
                        except:
                            pass
                    
                    valid_smiles.append(smiles)
                    all_candidates_data.append({
                        'spec_id': spec_id,
                        'rank': rank,
                        'smiles': smiles,
                        'contains_scaffold': contains_scaf
                    })
            
            # Top-1
            if valid_smiles:
                top1_data.append({'spec_id': spec_id, 'smiles': valid_smiles[0]})
            else:
                top1_data.append({'spec_id': spec_id, 'smiles': ''})
        
        # 保存TSV文件
        top1_df = pd.DataFrame(top1_data)
        top1_file = smiles_dir / 'predictions_top1.tsv'
        top1_df.to_csv(top1_file, sep='\t', index=False)
        logger.info(f"  ✓ Top-1预测: {top1_file.name} ({len(top1_df)} 行)")
        
        all_candidates_df = pd.DataFrame(all_candidates_data)
        all_candidates_file = smiles_dir / 'predictions_all_candidates.tsv'
        all_candidates_df.to_csv(all_candidates_file, sep='\t', index=False)
        logger.info(f"  ✓ 所有候选: {all_candidates_file.name} ({len(all_candidates_df)} 行)")
        
        logger.info(f"\n  统计:")
        logger.info(f"    有效SMILES: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
        logger.info(f"    包含骨架: {scaffold_match_count}/{total_count} ({scaffold_match_count/total_count*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"  ✗ SMILES转换失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 可视化
    logger.info("\n11.2 生成可视化...")
    
    try:
        from rdkit.Chem import Draw
        
        # 生成Top-1对比图
        valid_top1_mols = []
        valid_top1_legends = []
        
        for spec_idx, mol_list in enumerate(all_predictions[:20]):
            if not isinstance(mol_list, list):
                mol_list = [mol_list]
            
            if len(mol_list) > 0 and mol_list[0] is not None:
                smiles = mol_to_canonical_smiles(mol_list[0])
                if smiles:
                    valid_top1_mols.append(mol_list[0])
                    
                    # 标记是否包含骨架
                    has_scaf = ""
                    try:
                        if mol_list[0].HasSubstructMatch(scaffold_mol):
                            has_scaf = " ✓"
                    except:
                        pass
                    
                    valid_top1_legends.append(f"Spec {spec_idx}{has_scaf}")
        
        if valid_top1_mols:
            top1_img = Draw.MolsToGridImage(
                valid_top1_mols,
                molsPerRow=5,
                subImgSize=(250, 250),
                legends=valid_top1_legends,
                returnPNG=False
            )
            top1_file = viz_dir / 'top1_comparison.png'
            top1_img.save(top1_file)
            logger.info(f"  ✓ Top-1对比图: {top1_file.name}")
        
        logger.info("✓ 可视化完成")
        
    except Exception as e:
        logger.error(f"  ✗ 可视化失败: {e}")
    
    # 保存volume
    logger.info("\n保存volume更新...")
    output_volume.commit()
    logger.info("✓ Volume更新已保存")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 骨架约束推理全部完成！")
    logger.info("=" * 80)
    logger.info(f"\n结果保存在: {output_path}")
    logger.info(f"  - PKL文件: {preds_dir}")
    logger.info(f"  - SMILES文件: {smiles_dir}")
    logger.info(f"  - 可视化: {viz_dir}")
    logger.info(f"  - 日志: {logs_dir}")
    
    return {
        "status": "success",
        "scaffold_smiles": scaffold_smiles,
        "scaffold_formula": scaffold_formula_str,
        "max_count": max_count,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_spectra": len(all_predictions),
        "valid_smiles": valid_count,
        "scaffold_matches": scaffold_match_count,
        "total_candidates": total_count,
        "output_dirs": {
            "predictions": str(preds_dir),
            "smiles": str(smiles_dir),
            "visualizations": str(viz_dir),
            "logs": str(logs_dir)
        }
    }


@app.local_entrypoint()
def main(
    scaffold_smiles: str = "CC(=CCCC(C1CCC2(C1(CCC3=C2CCC4C3(CCC(C4(C)C)O)C)C)C)C(=O)O)C",
    max_count: int = 10,
    data_subdir: str = "test_top10",
    attachment_indices: str = None,
    enforce_scaffold: bool = True,
    use_rerank: bool = True
):
    """
    本地入口点 - 从命令行调用
    
    使用方法:
        # 使用默认骨架（三萜类化合物）
        modal run diffms_scaffold_inference.py
        
        # 自定义骨架
        modal run diffms_scaffold_inference.py --scaffold-smiles "c1ccccc1"
        
        # 指定锚点
        modal run diffms_scaffold_inference.py --attachment-indices "2,5,7"
    """
    result = run_scaffold_inference.remote(
        scaffold_smiles=scaffold_smiles,
        max_count=max_count,
        data_subdir=data_subdir,
        attachment_indices=attachment_indices,
        enforce_scaffold=enforce_scaffold,
        use_rerank=use_rerank
    )
    
    print("\n" + "=" * 70)
    print("🎉 骨架约束推理完成！")
    print("=" * 70)
    print(f"状态: {result['status']}")
    print(f"骨架SMILES: {result['scaffold_smiles']}")
    print(f"骨架分子式: {result['scaffold_formula']}")
    print(f"数据目录: {data_subdir}")
    print(f"处理数据量: {result['max_count'] or '全部'}")
    print(f"使用GPU: {result['gpu']}")
    print(f"\n结果统计:")
    print(f"  总谱图数: {result['total_spectra']}")
    print(f"  有效SMILES: {result['valid_smiles']}/{result['total_candidates']}")
    print(f"  包含骨架: {result['scaffold_matches']}/{result['total_candidates']}")
    print("=" * 70)
    print(f"\n📥 下载结果:")
    print(f"  modal volume get diffms-outputs {output_path} ./scaffold_results")
    print()

