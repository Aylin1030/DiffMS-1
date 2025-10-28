#!/bin/bash
# 测试骨架约束推理功能
# 使用方法: bash test_scaffold_inference.sh

set -e  # 遇到错误立即退出

echo "========================================="
echo "DiffMS 骨架约束推理测试"
echo "========================================="
echo ""

# 配置参数
CHECKPOINT_PATH="checkpoints/best.ckpt"  # 修改为你的checkpoint路径
DATA_DIR="data/msg"                       # 修改为你的数据目录
OUTPUT_DIR="outputs/scaffold_test"

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ 错误: 找不到checkpoint文件: $CHECKPOINT_PATH"
    echo "请修改脚本中的 CHECKPOINT_PATH 变量"
    exit 1
fi

echo "✅ Checkpoint: $CHECKPOINT_PATH"
echo "✅ 输出目录: $OUTPUT_DIR"
echo ""

# 测试1: 简单苯环骨架
echo "测试 1/3: 简单苯环骨架约束"
echo "-------------------------------------------"
python -m src.spec2mol_main \
    general.name="scaffold_test1_benzene" \
    general.test_only="$CHECKPOINT_PATH" \
    general.scaffold_smiles="c1ccccc1" \
    general.target_formula="C10H14O" \
    general.enforce_scaffold=True \
    general.use_rerank=False \
    general.test_samples_to_generate=5 \
    dataset.datadir="$DATA_DIR" || {
        echo "❌ 测试1失败"
        exit 1
    }
echo "✅ 测试1完成"
echo ""

# 测试2: 带锚点的苯甲酰胺
echo "测试 2/3: 苯甲酰胺骨架 + 锚点约束"
echo "-------------------------------------------"
python -m src.spec2mol_main \
    general.name="scaffold_test2_benzamide" \
    general.test_only="$CHECKPOINT_PATH" \
    general.scaffold_smiles="c1ccc(cc1)C(=O)N" \
    general.target_formula="C12H14N2O3" \
    general.attachment_indices="3,7" \
    general.enforce_scaffold=True \
    general.use_rerank=True \
    general.test_samples_to_generate=5 \
    dataset.datadir="$DATA_DIR" || {
        echo "❌ 测试2失败"
        exit 1
    }
echo "✅ 测试2完成"
echo ""

# 测试3: 软约束模式（不强制骨架）
echo "测试 3/3: 软约束模式（优先但不强制骨架）"
echo "-------------------------------------------"
python -m src.spec2mol_main \
    general.name="scaffold_test3_soft" \
    general.test_only="$CHECKPOINT_PATH" \
    general.scaffold_smiles="c1ccccc1" \
    general.target_formula="C15H20N2O2" \
    general.enforce_scaffold=False \
    general.use_rerank=True \
    general.test_samples_to_generate=5 \
    dataset.datadir="$DATA_DIR" || {
        echo "❌ 测试3失败"
        exit 1
    }
echo "✅ 测试3完成"
echo ""

echo "========================================="
echo "✅ 所有测试完成！"
echo "========================================="
echo ""
echo "结果保存在以下目录："
echo "  - outputs/scaffold_test1_benzene/preds/"
echo "  - outputs/scaffold_test2_benzamide/preds/"
echo "  - outputs/scaffold_test3_soft/preds/"
echo ""
echo "查看结果:"
echo "  import pickle"
echo "  with open('outputs/scaffold_test1_benzene/preds/scaffold_test1_benzene_rank_0_pred_0.pkl', 'rb') as f:"
echo "      mols = pickle.load(f)"
echo "      print(f'生成了 {len(mols)} 个样本')"
echo "      print(f'第1个样本有 {len(mols[0])} 个候选分子')"

