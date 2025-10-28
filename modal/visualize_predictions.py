#!/usr/bin/env python3
"""
DiffMS预测结果可视化工具

功能:
1. 从pkl文件读取预测的Mol对象
2. 转换为SMILES
3. 生成分子结构图
4. 创建网格对比图
5. 导出详细信息

使用方法:
    python visualize_predictions.py
"""

import pickle
from pathlib import Path
from typing import List, Optional
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 禁用RDKit警告
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class MoleculeVisualizer:
    """分子可视化工具类"""
    
    def __init__(self, pkl_files: List[Path]):
        """
        初始化
        
        Args:
            pkl_files: pkl文件路径列表
        """
        self.pkl_files = pkl_files
        self.all_predictions = []
        self.load_predictions()
    
    def load_predictions(self):
        """加载所有pkl文件"""
        logger.info(f"加载 {len(self.pkl_files)} 个pkl文件...")
        
        for pkl_file in sorted(self.pkl_files):
            logger.info(f"  读取: {pkl_file.name}")
            with open(pkl_file, 'rb') as f:
                predictions = pickle.load(f)
                
                if not isinstance(predictions, list):
                    logger.warning(f"  ⚠ 跳过非列表格式: {pkl_file.name}")
                    continue
                
                self.all_predictions.extend(predictions)
                logger.info(f"    包含 {len(predictions)} 个谱图")
        
        logger.info(f"✓ 总共加载 {len(self.all_predictions)} 个谱图的预测")
    
    def mol_to_info(self, mol: Optional[Chem.Mol]) -> dict:
        """提取分子信息"""
        if mol is None:
            return {
                'valid': False,
                'smiles': '',
                'formula': '',
                'mol_weight': 0,
                'num_atoms': 0,
                'num_bonds': 0,
            }
        
        try:
            # 移除立体化学
            Chem.RemoveStereochemistry(mol)
            
            smiles = Chem.MolToSmiles(mol, canonical=True)
            
            # 验证SMILES
            test_mol = Chem.MolFromSmiles(smiles)
            if test_mol is None:
                return {'valid': False, 'smiles': smiles, 'error': 'Invalid SMILES'}
            
            return {
                'valid': True,
                'smiles': smiles,
                'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
                'mol_weight': round(Descriptors.MolWt(mol), 2),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def create_summary_table(self, output_file: Path):
        """创建详细的摘要表格"""
        logger.info("\n创建摘要表格...")
        
        rows = []
        for spec_idx, mol_list in enumerate(self.all_predictions):
            spec_id = f"spec_{spec_idx:04d}"
            
            if not isinstance(mol_list, list):
                mol_list = [mol_list]
            
            for rank, mol in enumerate(mol_list, start=1):
                info = self.mol_to_info(mol)
                info['spec_id'] = spec_id
                info['rank'] = rank
                rows.append(info)
        
        df = pd.DataFrame(rows)
        
        # 重新排列列
        columns = ['spec_id', 'rank', 'valid', 'smiles', 'formula', 
                   'mol_weight', 'num_atoms', 'num_bonds']
        df = df[[c for c in columns if c in df.columns]]
        
        # 保存
        df.to_csv(output_file, sep='\t', index=False)
        logger.info(f"✓ 摘要表格: {output_file}")
        logger.info(f"  总行数: {len(df)}")
        logger.info(f"  有效分子: {df['valid'].sum()}")
        
        return df
    
    def visualize_spectrum_grid(self, spec_idx: int, output_file: Path, 
                                max_mols: int = 10):
        """
        可视化单个谱图的所有候选（网格图）
        
        Args:
            spec_idx: 谱图索引
            output_file: 输出图片路径
            max_mols: 最多显示分子数
        """
        if spec_idx >= len(self.all_predictions):
            logger.error(f"谱图索引 {spec_idx} 超出范围 (最大: {len(self.all_predictions)-1})")
            return
        
        mol_list = self.all_predictions[spec_idx]
        if not isinstance(mol_list, list):
            mol_list = [mol_list]
        
        # 过滤有效分子
        valid_data = []
        for rank, mol in enumerate(mol_list[:max_mols], start=1):
            info = self.mol_to_info(mol)
            if info['valid'] and mol is not None:
                valid_data.append({
                    'mol': mol,
                    'rank': rank,
                    'smiles': info['smiles'],
                    'formula': info['formula']
                })
        
        if not valid_data:
            logger.warning(f"谱图 {spec_idx} 没有有效分子")
            return
        
        # 提取分子和标签
        mols = [d['mol'] for d in valid_data]
        legends = [
            f"Rank {d['rank']}\n{d['formula']}\n{d['smiles'][:30]}..."
            for d in valid_data
        ]
        
        # 绘制网格图
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(300, 300),
            legends=legends,
            returnPNG=False
        )
        
        img.save(output_file)
        logger.info(f"✓ 网格图: {output_file} ({len(mols)} 个分子)")
    
    def visualize_all_spectra(self, output_dir: Path):
        """可视化所有谱图"""
        logger.info(f"\n可视化所有谱图到 {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        for spec_idx in range(len(self.all_predictions)):
            output_file = output_dir / f'spectrum_{spec_idx:04d}_grid.png'
            try:
                self.visualize_spectrum_grid(spec_idx, output_file)
                success_count += 1
            except Exception as e:
                logger.error(f"✗ 谱图 {spec_idx} 可视化失败: {e}")
        
        logger.info(f"✓ 成功生成 {success_count}/{len(self.all_predictions)} 个网格图")
    
    def visualize_top1_comparison(self, output_file: Path):
        """
        可视化所有谱图的Top-1预测（对比图）
        """
        logger.info(f"\n创建Top-1对比图...")
        
        valid_data = []
        for spec_idx, mol_list in enumerate(self.all_predictions):
            if not isinstance(mol_list, list):
                mol_list = [mol_list]
            
            if len(mol_list) > 0:
                mol = mol_list[0]  # Top-1
                info = self.mol_to_info(mol)
                if info['valid'] and mol is not None:
                    valid_data.append({
                        'mol': mol,
                        'spec_idx': spec_idx,
                        'smiles': info['smiles'],
                        'formula': info['formula']
                    })
        
        if not valid_data:
            logger.warning("没有有效的Top-1预测")
            return
        
        # 提取分子和标签
        mols = [d['mol'] for d in valid_data]
        legends = [
            f"Spec {d['spec_idx']}\n{d['formula']}"
            for d in valid_data
        ]
        
        # 绘制网格图
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(250, 250),
            legends=legends,
            returnPNG=False
        )
        
        img.save(output_file)
        logger.info(f"✓ Top-1对比图: {output_file} ({len(mols)} 个分子)")
    
    def generate_statistics(self):
        """生成统计信息"""
        logger.info("\n" + "=" * 60)
        logger.info("统计信息")
        logger.info("=" * 60)
        
        total_specs = len(self.all_predictions)
        total_candidates = 0
        valid_candidates = 0
        invalid_candidates = 0
        
        for mol_list in self.all_predictions:
            if not isinstance(mol_list, list):
                mol_list = [mol_list]
            
            for mol in mol_list:
                total_candidates += 1
                info = self.mol_to_info(mol)
                if info['valid']:
                    valid_candidates += 1
                else:
                    invalid_candidates += 1
        
        logger.info(f"总谱图数: {total_specs}")
        logger.info(f"总候选数: {total_candidates}")
        logger.info(f"有效分子: {valid_candidates} ({valid_candidates/total_candidates*100:.1f}%)")
        logger.info(f"无效分子: {invalid_candidates} ({invalid_candidates/total_candidates*100:.1f}%)")
        logger.info("=" * 60)


def main():
    """主函数"""
    # 配置路径
    predictions_dir = Path("/Users/aylin/yaolab_projects/diffms_yaolab/modal")
    output_dir = Path("/Users/aylin/yaolab_projects/diffms_yaolab/modal/visualizations")
    
    # 查找pkl文件
    pkl_files = list(predictions_dir.glob("*_pred_*.pkl"))
    
    if not pkl_files:
        logger.error(f"在 {predictions_dir} 中未找到pkl文件")
        return
    
    logger.info("=" * 60)
    logger.info("DiffMS预测结果可视化")
    logger.info("=" * 60)
    logger.info(f"找到 {len(pkl_files)} 个pkl文件:")
    for f in pkl_files:
        logger.info(f"  - {f.name}")
    
    # 创建可视化器
    visualizer = MoleculeVisualizer(pkl_files)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成统计信息
    visualizer.generate_statistics()
    
    # 2. 创建摘要表格
    summary_file = output_dir / 'predictions_summary.tsv'
    visualizer.create_summary_table(summary_file)
    
    # 3. Top-1对比图
    top1_file = output_dir / 'top1_comparison.png'
    visualizer.visualize_top1_comparison(top1_file)
    
    # 4. 可视化所有谱图（网格图）
    grid_dir = output_dir / 'spectrum_grids'
    visualizer.visualize_all_spectra(grid_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ 可视化完成！")
    logger.info("=" * 60)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"  - 摘要表格: {summary_file.name}")
    logger.info(f"  - Top-1对比图: {top1_file.name}")
    logger.info(f"  - 谱图网格图: {grid_dir.name}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

