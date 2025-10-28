#!/bin/bash
# 上传测试数据到 Modal Volume
# 使用方法: bash upload_test_data.sh

set -e

echo "================================================"
echo "上传 DiffMS 测试数据到 Modal Volume"
echo "================================================"
echo ""

# 配置
LOCAL_DATA_PATH="/Users/aylin/yaolab_projects/madgen_yaolab/msdata/test_top10"
REMOTE_DATA_PATH="/data/test_top10"
VOLUME_NAME="diffms-data"

# 检查本地数据是否存在
if [ ! -d "$LOCAL_DATA_PATH" ]; then
    echo "❌ 错误: 本地数据目录不存在: $LOCAL_DATA_PATH"
    exit 1
fi

echo "✅ 本地数据目录: $LOCAL_DATA_PATH"
echo ""

# 检查必要文件
echo "检查必要文件..."
REQUIRED_FILES=(
    "split.tsv"
    "labels.tsv"
    "spec_files"
    "subformulae"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -e "$LOCAL_DATA_PATH/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file 不存在"
        exit 1
    fi
done

echo ""
echo "所有必要文件都存在！"
echo ""

# 上传到 Modal Volume
echo "开始上传到 Modal Volume: $VOLUME_NAME"
echo "目标路径: $REMOTE_DATA_PATH"
echo ""

modal volume put "$VOLUME_NAME" \
    "$LOCAL_DATA_PATH" \
    "$REMOTE_DATA_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 上传成功！"
else
    echo ""
    echo "❌ 上传失败"
    exit 1
fi

# 验证上传
echo ""
echo "验证上传..."
echo "================================================"
modal volume ls "$VOLUME_NAME" "$REMOTE_DATA_PATH"
echo "================================================"

echo ""
echo "✅ 数据上传完成！"
echo ""
echo "下一步:"
echo "  1. 确保模型已上传: modal volume ls diffms-models /models/"
echo "  2. 运行推理: modal run diffms_scaffold_inference.py"
echo ""

