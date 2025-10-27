#!/bin/bash
# 快速测试脚本 - 使用前10个数据点测试推理

echo "========================================"
echo "DiffMS 推理测试 - 前10个数据点"
echo "========================================"

# 设置路径
CHECKPOINT="/Users/aylin/Downloads/checkpoints/diffms_msg.ckpt"
OUTPUT_DIR="./predictions_test"
MAX_COUNT=10

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 未找到checkpoint文件: $CHECKPOINT"
    echo "可用的checkpoint文件:"
    ls -lh /Users/aylin/Downloads/checkpoints/*.ckpt
    exit 1
fi

echo "使用模型: $CHECKPOINT"
echo "输出目录: $OUTPUT_DIR"
echo "测试数据量: $MAX_COUNT"
echo ""

# 运行推理
python run_inference.py \
    --checkpoint_path "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --max_count $MAX_COUNT

echo ""
echo "========================================"
echo "测试完成！"
echo "========================================"
echo "检查结果:"
echo "- 输出: $OUTPUT_DIR/"
echo "- 预测: ../preds/"
echo "- 日志: inference.log"
