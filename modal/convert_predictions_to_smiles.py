"""
修正5: 将pkl预测文件转换为canonical SMILES格式
确保输出的是合法的SMILES字符串，而不是"乱码"
"""
import pickle
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from typing import List, Optional
import pandas as pd

# 禁用RDKit警告
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def mol_to_canonical_smiles(mol: Optional[Chem.Mol]) -> Optional[str]:
    """
    将RDKit Mol对象转换为canonical SMILES（无立体化学）
    
    论文要求：输出结构使用 canonical SMILES (无立体化学) 表示
    """
    if mol is None:
        return None
    
    try:
        # 1. 移除立体化学信息
        Chem.RemoveStereochemistry(mol)
        
        # 2. 转换为canonical SMILES
        smiles = Chem.MolToSmiles(mol, canonical=True)
        
        # 3. 验证SMILES有效性（反向验证）
        test_mol = Chem.MolFromSmiles(smiles)
        if test_mol is None:
            logger.warning(f"生成的SMILES无效: {smiles}")
            return None
        
        return smiles
    except Exception as e:
        logger.warning(f"Mol转SMILES失败: {e}")
        return None


def process_predictions(pkl_files: List[Path], output_dir: Path) -> dict:
    """
    处理pkl预测文件，转换为SMILES格式
    
    Returns:
        统计信息字典
    """
    all_predictions = []
    stats = {
        'total_specs': 0,
        'total_candidates': 0,
        'valid_candidates': 0,
        'invalid_candidates': 0,
        'none_candidates': 0
    }
    
    logger.info(f"处理 {len(pkl_files)} 个pkl文件...")
    
    # 1. 合并所有pkl文件
    for pkl_file in sorted(pkl_files):
        logger.info(f"  读取: {pkl_file.name}")
        with open(pkl_file, 'rb') as f:
            predictions = pickle.load(f)  # List[List[Mol]]
            
            if not isinstance(predictions, list):
                logger.warning(f"  ⚠ 跳过非列表格式: {pkl_file.name}")
                continue
            
            all_predictions.extend(predictions)
            logger.info(f"    包含 {len(predictions)} 个谱图的预测")
    
    stats['total_specs'] = len(all_predictions)
    logger.info(f"\n✓ 总共 {stats['total_specs']} 个谱图")
    
    # 2. 转换为SMILES
    logger.info("\n转换为canonical SMILES...")
    
    # Top-1预测
    top1_data = []
    # 所有候选（Top-K）
    all_candidates_data = []
    
    for spec_idx, mol_list in enumerate(all_predictions):
        spec_id = f"spec_{spec_idx:04d}"
        
        if not isinstance(mol_list, list):
            logger.warning(f"  ⚠ {spec_id}: 预测不是列表格式")
            mol_list = [mol_list]
        
        valid_smiles = []
        
        for rank, mol in enumerate(mol_list, start=1):
            stats['total_candidates'] += 1
            
            if mol is None:
                stats['none_candidates'] += 1
                continue
            
            # 转换为canonical SMILES
            smiles = mol_to_canonical_smiles(mol)
            
            if smiles is None:
                stats['invalid_candidates'] += 1
                continue
            
            stats['valid_candidates'] += 1
            valid_smiles.append(smiles)
            
            # 添加到所有候选
            all_candidates_data.append({
                'spec_id': spec_id,
                'rank': rank,
                'smiles': smiles
            })
        
        # Top-1（第一个有效的SMILES）
        if valid_smiles:
            top1_data.append({
                'spec_id': spec_id,
                'smiles': valid_smiles[0]
            })
        else:
            # 如果没有有效预测，使用空字符串
            top1_data.append({
                'spec_id': spec_id,
                'smiles': ''
            })
            logger.warning(f"  ⚠ {spec_id}: 没有有效预测")
    
    # 3. 保存为TSV
    logger.info("\n保存结果...")
    
    # Top-1预测
    top1_df = pd.DataFrame(top1_data)
    top1_file = output_dir / 'predictions_top1.tsv'
    top1_df.to_csv(top1_file, sep='\t', index=False)
    logger.info(f"  ✓ Top-1预测: {top1_file}")
    logger.info(f"    {len(top1_df)} 行")
    
    # 所有候选
    all_candidates_df = pd.DataFrame(all_candidates_data)
    all_candidates_file = output_dir / 'predictions_all_candidates.tsv'
    all_candidates_df.to_csv(all_candidates_file, sep='\t', index=False)
    logger.info(f"  ✓ 所有候选: {all_candidates_file}")
    logger.info(f"    {len(all_candidates_df)} 行")
    
    # 4. 打印统计
    logger.info("\n" + "=" * 60)
    logger.info("统计信息:")
    logger.info(f"  总谱图数: {stats['total_specs']}")
    logger.info(f"  总候选数: {stats['total_candidates']}")
    logger.info(f"  有效SMILES: {stats['valid_candidates']} ({stats['valid_candidates']/stats['total_candidates']*100:.1f}%)")
    logger.info(f"  无效Mol: {stats['invalid_candidates']} ({stats['invalid_candidates']/stats['total_candidates']*100:.1f}%)")
    logger.info(f"  None对象: {stats['none_candidates']} ({stats['none_candidates']/stats['total_candidates']*100:.1f}%)")
    logger.info("=" * 60)
    
    return stats


def validate_output_format(tsv_file: Path):
    """
    验证输出TSV格式是否正确
    """
    logger.info(f"\n验证输出格式: {tsv_file.name}")
    
    df = pd.read_csv(tsv_file, sep='\t')
    logger.info(f"  行数: {len(df)}")
    logger.info(f"  列: {list(df.columns)}")
    
    # 检查SMILES有效性
    invalid_count = 0
    empty_count = 0
    
    for idx, row in df.iterrows():
        smiles = row['smiles']
        
        if pd.isna(smiles) or smiles == '':
            empty_count += 1
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_count += 1
            logger.warning(f"  ⚠ 第{idx}行SMILES无效: {smiles}")
    
    logger.info(f"  有效SMILES: {len(df) - invalid_count - empty_count}")
    logger.info(f"  空SMILES: {empty_count}")
    logger.info(f"  无效SMILES: {invalid_count}")
    
    if invalid_count == 0:
        logger.info("  ✓ 所有SMILES都有效！")
    else:
        logger.warning(f"  ⚠ 发现 {invalid_count} 个无效SMILES")


def main():
    """主函数"""
    # 本地测试路径
    preds_dir = Path("/Users/aylin/yaolab_projects/diffms_yaolab/modal")
    output_dir = Path("/Users/aylin/yaolab_projects/diffms_yaolab/modal/results_smiles")
    
    # 查找pkl文件
    pkl_files = list(preds_dir.glob("*_pred_*.pkl"))
    
    if not pkl_files:
        logger.error(f"在 {preds_dir} 中未找到pkl文件")
        return
    
    logger.info("=" * 60)
    logger.info("DiffMS预测结果转换: Mol → Canonical SMILES")
    logger.info("=" * 60)
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 处理预测
    stats = process_predictions(pkl_files, output_dir)
    
    # 验证输出
    validate_output_format(output_dir / 'predictions_top1.tsv')
    validate_output_format(output_dir / 'predictions_all_candidates.tsv')
    
    logger.info("\n✓ 完成！")


if __name__ == "__main__":
    main()

