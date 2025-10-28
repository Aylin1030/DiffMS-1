"""
综合验证脚本 - 检查所有关键配置点
根据论文和建议清单验证DiffMS推理设置
"""
import pandas as pd
from pathlib import Path
import torch
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SetupValidator:
    """验证器：检查所有关键配置"""
    
    def __init__(self, data_dir: Path, checkpoint_path: Path):
        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.issues = []
        self.warnings = []
    
    def check_checkpoint_structure(self) -> bool:
        """检查点1: Checkpoint内容结构"""
        logger.info("\n" + "=" * 60)
        logger.info("检查点1: Checkpoint结构")
        logger.info("=" * 60)
        
        if not self.checkpoint_path.exists():
            self.issues.append(f"Checkpoint不存在: {self.checkpoint_path}")
            return False
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # 1.1 顶层keys
        logger.info(f"✓ Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'state_dict' not in checkpoint:
            self.issues.append("Checkpoint缺少'state_dict'")
            return False
        
        state_dict = checkpoint['state_dict']
        
        # 1.2 Decoder权重
        decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]
        logger.info(f"✓ Decoder权重: {len(decoder_keys)} 个")
        
        if len(decoder_keys) == 0:
            self.issues.append("Checkpoint缺少decoder权重")
            return False
        
        # 1.3 Encoder权重
        encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.')]
        logger.info(f"✓ Encoder权重: {len(encoder_keys)} 个")
        
        if len(encoder_keys) == 0:
            self.issues.append("Checkpoint缺少encoder权重")
            return False
        
        # 1.4 关键维度
        expected_dims = {
            'decoder.mlp_in_X.0.weight': (None, 16),  # (out, 16)
            'decoder.mlp_in_E.0.weight': (None, 5),   # (out, 5)
            'decoder.mlp_in_y.0.weight': (None, 2061), # (out, 2061)
            'decoder.mlp_out_X.2.weight': (8, None),  # (8, in)
            'decoder.mlp_out_E.2.weight': (5, None),  # (5, in)
        }
        
        for key, expected_shape in expected_dims.items():
            if key in state_dict:
                actual_shape = state_dict[key].shape
                match = all(
                    exp is None or exp == act 
                    for exp, act in zip(expected_shape, actual_shape)
                )
                status = "✓" if match else "✗"
                logger.info(f"{status} {key}: {actual_shape} (期望: {expected_shape})")
                
                if not match:
                    self.issues.append(f"维度不匹配: {key}")
        
        return len(self.issues) == 0
    
    def check_data_format(self) -> bool:
        """检查点4: Dataset.formula字段验证"""
        logger.info("\n" + "=" * 60)
        logger.info("检查点4: 数据集格式验证")
        logger.info("=" * 60)
        
        # 4.1 检查split.tsv
        split_file = self.data_dir / 'split.tsv'
        if not split_file.exists():
            self.issues.append(f"split.tsv不存在: {split_file}")
            return False
        
        split_df = pd.read_csv(split_file, sep='\t')
        logger.info(f"✓ split.tsv: {len(split_df)} 行")
        logger.info(f"  列: {list(split_df.columns)}")
        
        # 4.2 检查labels.tsv
        labels_file = self.data_dir / 'labels.tsv'
        if not labels_file.exists():
            self.issues.append(f"labels.tsv不存在: {labels_file}")
            return False
        
        labels_df = pd.read_csv(labels_file, sep='\t')
        logger.info(f"✓ labels.tsv: {len(labels_df)} 行")
        logger.info(f"  列: {list(labels_df.columns)}")
        
        # 4.3 验证formula字段
        if 'formula' not in labels_df.columns:
            self.issues.append("labels.tsv缺少'formula'列")
            return False
        
        # 检查formula格式
        import re
        formula_pattern = re.compile(r'^[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*$')
        
        invalid_formulas = []
        missing_formulas = 0
        
        for idx, formula in enumerate(labels_df['formula']):
            if pd.isna(formula) or formula == '':
                missing_formulas += 1
                continue
            
            if not formula_pattern.match(str(formula)):
                invalid_formulas.append((idx, formula))
        
        logger.info(f"  Formula统计:")
        logger.info(f"    总数: {len(labels_df)}")
        logger.info(f"    缺失: {missing_formulas}")
        logger.info(f"    无效格式: {len(invalid_formulas)}")
        
        if invalid_formulas:
            for idx, formula in invalid_formulas[:5]:
                logger.warning(f"    ⚠ 第{idx}行: {formula}")
            if len(invalid_formulas) > 5:
                logger.warning(f"    ... 还有 {len(invalid_formulas)-5} 个")
            
            self.warnings.append(f"发现 {len(invalid_formulas)} 个无效formula格式")
        
        # 4.4 检查SMILES字段（推理模式可以为空）
        if 'smiles' in labels_df.columns:
            empty_smiles = labels_df['smiles'].isna().sum()
            logger.info(f"  SMILES统计:")
            logger.info(f"    非空: {len(labels_df) - empty_smiles}")
            logger.info(f"    空值: {empty_smiles}")
            
            if empty_smiles == len(labels_df):
                logger.info(f"  ℹ 推理模式：所有SMILES为空（符合预期）")
        
        return len(self.issues) == 0
    
    def check_file_structure(self) -> bool:
        """检查点6: 文件路径结构"""
        logger.info("\n" + "=" * 60)
        logger.info("检查点6: 文件路径结构")
        logger.info("=" * 60)
        
        required_paths = {
            'spec_folder': self.data_dir / 'spec_files',
            'subform_folder': self.data_dir / 'subformulae' / 'default_subformulae',
        }
        
        all_exist = True
        for name, path in required_paths.items():
            if path.exists():
                if path.is_dir():
                    count = len(list(path.glob('*')))
                    logger.info(f"✓ {name}: {path} ({count} 文件)")
                else:
                    logger.info(f"✓ {name}: {path}")
            else:
                logger.error(f"✗ {name}不存在: {path}")
                self.issues.append(f"{name}不存在")
                all_exist = False
        
        return all_exist
    
    def check_mol_to_smiles_pipeline(self) -> bool:
        """检查点5: Mol→SMILES转换管道"""
        logger.info("\n" + "=" * 60)
        logger.info("检查点5: Mol→SMILES转换管道")
        logger.info("=" * 60)
        
        # 检查visualization_tools和correct_mol函数
        try:
            import sys
            diffms_src = Path("/Users/aylin/yaolab_projects/diffms_yaolab/DiffMS/src")
            sys.path.insert(0, str(diffms_src))
            
            from rdkit import Chem
            from analysis.rdkit_functions import correct_mol
            
            logger.info("✓ correct_mol函数可导入")
            
            # 测试用例
            test_smiles = "CCO"  # 乙醇
            test_mol = Chem.MolFromSmiles(test_smiles)
            
            editable_mol = Chem.RWMol(test_mol)
            corrected_mol, no_correct = correct_mol(editable_mol)
            
            if corrected_mol is not None:
                result_smiles = Chem.MolToSmiles(corrected_mol, canonical=True)
                logger.info(f"✓ 测试转换: {test_smiles} → {result_smiles}")
            else:
                logger.warning("⚠ correct_mol返回None")
            
            # 验证sample_batch中使用了correct_mol
            model_file = diffms_src / "diffusion_model_spec2mol.py"
            with open(model_file, 'r') as f:
                content = f.read()
                if 'correct_mol' in content:
                    logger.info("✓ diffusion_model_spec2mol.py使用了correct_mol")
                else:
                    self.warnings.append("diffusion_model_spec2mol.py未使用correct_mol")
            
        except Exception as e:
            logger.error(f"✗ Mol→SMILES管道测试失败: {e}")
            self.issues.append("Mol→SMILES管道有问题")
            return False
        
        return True
    
    def generate_report(self) -> Dict:
        """生成验证报告"""
        logger.info("\n" + "=" * 60)
        logger.info("验证报告")
        logger.info("=" * 60)
        
        # 运行所有检查
        checks = {
            '1. Checkpoint结构': self.check_checkpoint_structure(),
            '4. 数据集格式': self.check_data_format(),
            '5. Mol→SMILES管道': self.check_mol_to_smiles_pipeline(),
            '6. 文件路径结构': self.check_file_structure(),
        }
        
        # 总结
        logger.info("\n检查结果:")
        for name, passed in checks.items():
            status = "✓ 通过" if passed else "✗ 失败"
            logger.info(f"  {name}: {status}")
        
        if self.issues:
            logger.error("\n发现的问题:")
            for i, issue in enumerate(self.issues, 1):
                logger.error(f"  {i}. {issue}")
        
        if self.warnings:
            logger.warning("\n警告:")
            for i, warning in enumerate(self.warnings, 1):
                logger.warning(f"  {i}. {warning}")
        
        all_passed = all(checks.values()) and len(self.issues) == 0
        
        if all_passed:
            logger.info("\n" + "=" * 60)
            logger.info("✓ 所有检查通过！可以开始推理。")
            logger.info("=" * 60)
        else:
            logger.error("\n" + "=" * 60)
            logger.error("✗ 验证失败！请修复上述问题后再运行推理。")
            logger.error("=" * 60)
        
        return {
            'passed': all_passed,
            'checks': checks,
            'issues': self.issues,
            'warnings': self.warnings
        }


def main():
    """主函数"""
    # 配置路径
    data_dir = Path("/Users/aylin/yaolab_projects/diffms_yaolab/msg_official_test5/processed_data")
    checkpoint_path = Path("/Users/aylin/Downloads/checkpoints/diffms_msg.ckpt")
    
    logger.info("=" * 60)
    logger.info("DiffMS推理设置验证")
    logger.info("根据论文要求和建议清单检查")
    logger.info("=" * 60)
    
    validator = SetupValidator(data_dir, checkpoint_path)
    report = validator.generate_report()
    
    return report


if __name__ == "__main__":
    report = main()

