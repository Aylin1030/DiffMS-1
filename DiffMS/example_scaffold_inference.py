"""
DiffMS 骨架约束推理示例
演示如何使用骨架约束功能进行分子生成

使用方法:
    cd DiffMS
    python example_scaffold_inference.py
"""
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not found. Some examples will be skipped.")
    RDKIT_AVAILABLE = False

# 导入推理工具（仅导入独立模块，不依赖DiffMS主模块）
try:
    from src.inference.scaffold_hooks import (
        smiles_to_mol, 
        formula_of, 
        formula_subtract,
        contains_scaffold,
        parse_formula,
        formula_to_string
    )
    from src.inference.rerank import (
        rerank_by_spectrum,
        filter_by_scaffold,
        deduplicate_candidates
    )
    INFERENCE_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cannot import inference tools: {e}")
    print("Make sure you are in the DiffMS directory and run: python example_scaffold_inference.py")
    INFERENCE_TOOLS_AVAILABLE = False


def example_1_basic_scaffold():
    """示例1: 基本骨架约束"""
    if not INFERENCE_TOOLS_AVAILABLE or not RDKIT_AVAILABLE:
        print("\n⚠️  示例1需要RDKit和inference tools，跳过...")
        return
        
    print("\n" + "="*60)
    print("示例 1: 基本苯环骨架约束")
    print("="*60)
    
    # 定义骨架和目标
    scaffold_smiles = "c1ccccc1"  # 苯环
    target_formula = "C10H14O"    # 目标：含苯环的C10化合物
    
    # 解析
    scaffold_mol = smiles_to_mol(scaffold_smiles)
    scaffold_f = formula_of(scaffold_mol)
    
    print(f"骨架 SMILES: {scaffold_smiles}")
    print(f"骨架分子式: {scaffold_f}")
    print(f"目标分子式: {target_formula}")
    
    # 计算剩余化学式
    target_f = parse_formula(target_formula)
    remaining_f = formula_subtract(target_f, scaffold_f)
    
    print(f"剩余可用 (ΔF): {remaining_f}")
    print("\n预期结果: 生成包含苯环的分子，如对甲酚、苯丙酮等")
    

def example_2_attachment_points():
    """示例2: 指定锚点"""
    print("\n" + "="*60)
    print("示例 2: 带锚点控制的骨架约束")
    print("="*60)
    
    scaffold_smiles = "c1ccc(cc1)C(=O)N"  # 苯甲酰胺
    target_formula = "C12H14N2O3"
    attachment_indices = [3, 7, 9]  # 只允许在这些位置接枝
    
    print(f"骨架 SMILES: {scaffold_smiles}")
    print(f"目标分子式: {target_formula}")
    print(f"锚点索引: {attachment_indices}")
    
    scaffold_mol = smiles_to_mol(scaffold_smiles)
    scaffold_f = formula_of(scaffold_mol)
    
    from src.inference.scaffold_hooks import parse_formula
    target_f = parse_formula(target_formula)
    remaining_f = formula_subtract(target_f, scaffold_f)
    
    print(f"\n骨架分子式: {scaffold_f}")
    print(f"剩余可用 (ΔF): {remaining_f}")
    print("\n预期结果: 新片段只连接到指定的锚点位置")


def example_3_validate_candidates():
    """示例3: 验证候选分子"""
    print("\n" + "="*60)
    print("示例 3: 验证生成的分子是否包含骨架")
    print("="*60)
    
    scaffold_smiles = "c1ccccc1"  # 苯环
    scaffold_mol = smiles_to_mol(scaffold_smiles)
    
    # 一些测试候选
    test_candidates = [
        "c1ccc(cc1)CC(=O)O",      # 苯乙酸 - 包含苯环 ✓
        "c1ccc(cc1)O",             # 苯酚 - 包含苯环 ✓
        "CC(=O)O",                 # 乙酸 - 不含苯环 ✗
        "c1ccc2c(c1)OCO2",         # 苯并二氧杂环戊烯 - 包含苯环 ✓
    ]
    
    print(f"骨架: {scaffold_smiles}")
    print("\n检查候选分子:")
    
    for smiles in test_candidates:
        mol = Chem.MolFromSmiles(smiles)
        contains = contains_scaffold(mol, scaffold_mol)
        status = "✅" if contains else "❌"
        print(f"  {status} {smiles:30s} - {'包含骨架' if contains else '不含骨架'}")


def example_4_formula_constraints():
    """示例4: 化学式约束详解"""
    print("\n" + "="*60)
    print("示例 4: 化学式约束机制")
    print("="*60)
    
    from src.inference.scaffold_hooks import parse_formula, formula_to_string
    
    # 案例1: 合理的约束
    print("\n案例 1: 合理约束")
    scaffold_f = parse_formula("C7H5N")    # 苯甲腈
    target_f = parse_formula("C10H12N2O")
    remaining_f = formula_subtract(target_f, scaffold_f)
    
    print(f"  骨架: {formula_to_string(scaffold_f)}")
    print(f"  目标: {formula_to_string(target_f)}")
    print(f"  剩余: {formula_to_string(remaining_f)}")
    print(f"  ✅ 可行 - 还能添加 {remaining_f.get('C', 0)} 个C, {remaining_f.get('N', 0)} 个N, {remaining_f.get('O', 0)} 个O")
    
    # 案例2: 不合理的约束
    print("\n案例 2: 不合理约束（会失败）")
    scaffold_f = parse_formula("C10H10O2")  # 大骨架
    target_f = parse_formula("C8H6O")       # 小目标
    
    print(f"  骨架: {formula_to_string(scaffold_f)}")
    print(f"  目标: {formula_to_string(target_f)}")
    
    try:
        remaining_f = formula_subtract(target_f, scaffold_f)
        print(f"  剩余: {formula_to_string(remaining_f)}")
    except ValueError as e:
        print(f"  ❌ 错误: {e}")
        print(f"  原因: 骨架需要的原子数超过了目标分子式")


def example_5_reranking():
    """示例5: 重排功能"""
    print("\n" + "="*60)
    print("示例 5: 基于谱相似度的重排")
    print("="*60)
    
    # 假设有一些候选分子
    candidate_smiles = [
        "c1ccc(cc1)CC(=O)O",
        "c1ccc(cc1)C(=O)O",
        "c1ccc(cc1)CO",
        "c1ccc(cc1)CCO",
    ]
    
    candidates = [Chem.MolFromSmiles(s) for s in candidate_smiles]
    
    print("原始候选分子:")
    for i, smiles in enumerate(candidate_smiles):
        mol = candidates[i]
        mw = Descriptors.ExactMolWt(mol)
        print(f"  {i+1}. {smiles:30s} (MW: {mw:.2f})")
    
    # 模拟质谱数据（实际应从真实质谱获取）
    import numpy as np
    mock_spectrum = np.array([
        [120.0, 100.0],  # m/z, intensity
        [91.0, 80.0],
        [77.0, 60.0],
    ])
    
    print("\n应用重排...")
    
    # 去重
    unique_candidates = deduplicate_candidates(candidates)
    print(f"去重后: {len(unique_candidates)} 个分子")
    
    # 重排（注意：这里使用简化的打分）
    from src.inference.rerank import rerank_by_spectrum
    reranked = rerank_by_spectrum(unique_candidates, mock_spectrum)
    
    print("\n重排后的候选分子:")
    for i, mol in enumerate(reranked):
        smiles = Chem.MolToSmiles(mol)
        mw = Descriptors.ExactMolWt(mol)
        print(f"  {i+1}. {smiles:30s} (MW: {mw:.2f})")


def example_6_complete_workflow():
    """示例6: 完整工作流程"""
    print("\n" + "="*60)
    print("示例 6: 完整骨架约束推理工作流")
    print("="*60)
    
    print("""
    完整流程:
    
    1. 准备输入
       - 质谱数据 (spectrum.ms 文件)
       - 骨架 SMILES
       - 目标分子式
       - (可选) 锚点索引
    
    2. 配置参数
       在 configs/general/general_default.yaml 中设置:
       ```yaml
       scaffold_smiles: "c1ccccc1"
       target_formula: "C10H14O"
       attachment_indices: "2,5"
       enforce_scaffold: True
       use_rerank: True
       ```
    
    3. 运行推理
       ```bash
       python -m src.spec2mol_main \\
           general.test_only=checkpoints/best.ckpt \\
           general.test_samples_to_generate=10
       ```
    
    4. 查看结果
       ```python
       import pickle
       with open('preds/model_rank_0_pred_0.pkl', 'rb') as f:
           predicted_mols = pickle.load(f)
       
       # predicted_mols[i] = 第i个样本的候选分子列表
       # 每个候选是 rdkit.Chem.Mol 对象或 None
       ```
    
    5. 后处理
       - 过滤: 只保留包含骨架的分子
       - 验证: 检查化学式、价态、有效性
       - 可视化: 使用 RDKit 绘制结构
       - 分析: 计算相似度、多样性等指标
    """)


def main():
    """运行所有示例"""
    print("\n" + "🚀 "*15)
    print("DiffMS 骨架约束推理示例集合")
    print("🚀 "*15)
    
    # 运行示例
    example_1_basic_scaffold()
    example_2_attachment_points()
    example_3_validate_candidates()
    example_4_formula_constraints()
    example_5_reranking()
    example_6_complete_workflow()
    
    print("\n" + "="*60)
    print("✅ 所有示例运行完成!")
    print("="*60)
    print("\n详细文档请参考: docs/SCAFFOLD_CONSTRAINED_INFERENCE_20251028.md")
    print("测试脚本: bash test_scaffold_inference.sh")
    print("\n")


if __name__ == "__main__":
    main()

