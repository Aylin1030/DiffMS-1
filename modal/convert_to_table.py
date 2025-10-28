#!/usr/bin/env python3
"""
将Modal推理结果转换为表格格式
"""
import pickle
import pandas as pd
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
import sys

def extract_predictions(pkl_file, output_dir="results"):
    """
    从pkl文件提取预测结果并保存为表格
    
    Args:
        pkl_file: pkl文件路径
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"读取文件: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"样本数量: {len(predictions)}")
    
    # 准备数据
    rows = []
    
    for spectrum_idx, candidates in enumerate(tqdm(predictions, desc="处理样本")):
        if not candidates or len(candidates) == 0:
            # 没有候选分子
            rows.append({
                'spectrum_id': spectrum_idx,
                'rank': 0,
                'smiles': '',
                'num_atoms': 0,
                'valid': False,
                'total_candidates': 0
            })
            continue
        
        # 处理每个候选分子
        valid_candidates = []
        for rank, mol in enumerate(candidates, 1):
            if mol is None:
                continue
            
            try:
                smiles = Chem.MolToSmiles(mol)
                num_atoms = mol.GetNumAtoms()
                valid_candidates.append({
                    'spectrum_id': spectrum_idx,
                    'rank': rank,
                    'smiles': smiles,
                    'num_atoms': num_atoms,
                    'valid': True,
                    'total_candidates': len(candidates)
                })
            except Exception as e:
                print(f"警告: Spectrum {spectrum_idx}, Rank {rank} 转换失败: {e}")
                continue
        
        if valid_candidates:
            rows.extend(valid_candidates)
        else:
            # 所有候选都无效
            rows.append({
                'spectrum_id': spectrum_idx,
                'rank': 0,
                'smiles': '',
                'num_atoms': 0,
                'valid': False,
                'total_candidates': len(candidates)
            })
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    
    # 保存完整结果（所有候选）
    full_output = output_dir / "predictions_all_candidates.tsv"
    df.to_csv(full_output, sep='\t', index=False)
    print(f"\n✓ 保存完整结果: {full_output}")
    print(f"  - 总行数: {len(df)}")
    
    # 创建Top-1结果（每个spectrum只保留最佳预测）
    if 'rank' in df.columns:
        df_top1 = df[df['rank'] == 1].copy()
        top1_output = output_dir / "predictions_top1.tsv"
        df_top1.to_csv(top1_output, sep='\t', index=False)
        print(f"\n✓ 保存Top-1结果: {top1_output}")
        print(f"  - 总样本数: {len(df_top1)}")
        print(f"  - 有效预测: {df_top1['valid'].sum()}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息:")
    print("=" * 60)
    
    total_spectra = df['spectrum_id'].nunique()
    valid_predictions = df[df['valid']].groupby('spectrum_id').size().count()
    
    print(f"总样本数: {total_spectra}")
    print(f"有有效预测的样本: {valid_predictions} ({valid_predictions/total_spectra*100:.1f}%)")
    print(f"平均候选数/样本: {df.groupby('spectrum_id').size().mean():.1f}")
    print(f"有效候选总数: {df['valid'].sum()}")
    
    if 'num_atoms' in df.columns:
        valid_df = df[df['valid']]
        if len(valid_df) > 0:
            print(f"\n原子数统计:")
            print(f"  - 最小: {valid_df['num_atoms'].min()}")
            print(f"  - 最大: {valid_df['num_atoms'].max()}")
            print(f"  - 平均: {valid_df['num_atoms'].mean():.1f}")
    
    # 显示前几条
    print("\n" + "=" * 60)
    print("前10条预测 (Top-1):")
    print("=" * 60)
    if 'rank' in df.columns:
        display_df = df[df['rank'] == 1].head(10)[['spectrum_id', 'smiles', 'num_atoms', 'valid']]
    else:
        display_df = df.head(10)[['spectrum_id', 'smiles', 'num_atoms', 'valid']]
    
    print(display_df.to_string(index=False))
    
    return df

def main():
    pkl_file = Path("modal_inference_rank_0_pred_0.pkl")
    
    if len(sys.argv) > 1:
        pkl_file = Path(sys.argv[1])
    
    if not pkl_file.exists():
        print(f"错误: 文件不存在: {pkl_file}")
        print("\n使用方法:")
        print("  python convert_to_table.py [pkl_file]")
        print("\n示例:")
        print("  python convert_to_table.py modal_inference_rank_0_pred_0.pkl")
        sys.exit(1)
    
    df = extract_predictions(pkl_file)
    
    print("\n✓ 转换完成！")
    print("\n输出文件:")
    print("  - results/predictions_all_candidates.tsv  (所有候选)")
    print("  - results/predictions_top1.tsv            (Top-1预测)")

if __name__ == "__main__":
    main()

