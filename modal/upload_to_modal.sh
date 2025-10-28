#!/bin/bash
# Modal Volume 上传脚本
# 用于将本地数据上传到Modal volumes

set -e  # 遇到错误立即退出

echo "=================================="
echo "DiffMS Modal Volume 上传脚本"
echo "=================================="

# 定义路径
DATA_DIR="/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data"
MODEL_FILE="/Users/aylin/Downloads/checkpoints/diffms_msg.ckpt"
STATS_DIR="/Users/aylin/Downloads/msg"

# 检查文件是否存在
echo ""
echo "1️⃣  检查本地文件..."

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 数据目录不存在: $DATA_DIR"
    exit 1
fi
echo "✓ 数据目录: $DATA_DIR"

if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ 模型文件不存在: $MODEL_FILE"
    exit 1
fi
SIZE_MB=$(du -m "$MODEL_FILE" | cut -f1)
echo "✓ 模型文件: $MODEL_FILE ($SIZE_MB MB)"

if [ ! -d "$STATS_DIR" ]; then
    echo "❌ 统计目录不存在: $STATS_DIR"
    exit 1
fi
echo "✓ 统计目录: $STATS_DIR"

# 上传数据
echo ""
echo "2️⃣  上传预处理数据到 diffms-data..."
echo "   这可能需要几分钟..."
modal volume put diffms-data "$DATA_DIR" /data/processed_data
echo "✓ 数据上传完成"

# 上传模型
echo ""
echo "3️⃣  上传模型checkpoint到 diffms-models..."
echo "   模型文件较大，这可能需要较长时间..."
modal volume put diffms-models "$MODEL_FILE" /models/diffms_msg.ckpt
echo "✓ 模型上传完成"

# 上传统计信息
echo ""
echo "4️⃣  上传MSG统计信息到 diffms-msg-stats..."
modal volume put diffms-msg-stats "$STATS_DIR" /msg_stats
echo "✓ 统计信息上传完成"

# 验证上传
echo ""
echo "5️⃣  验证上传结果..."
echo ""
echo "📦 diffms-data 内容:"
modal volume ls diffms-data /data

echo ""
echo "📦 diffms-models 内容:"
modal volume ls diffms-models /models

echo ""
echo "📦 diffms-msg-stats 内容:"
modal volume ls diffms-msg-stats /msg_stats | head -10

echo ""
echo "=================================="
echo "✅ 所有文件上传完成！"
echo "=================================="
echo ""
echo "下一步:"
echo "  1. 运行检查脚本验证: modal run check_modal_volumes.py"
echo "  2. 测试推理(10个样本): modal run diffms_inference.py --max-count 10"
echo "  3. 完整推理(478个样本): modal run diffms_inference.py"
echo ""

