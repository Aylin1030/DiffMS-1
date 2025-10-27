#!/bin/bash
# Modal Volumes 设置脚本 - 上传数据、模型和统计信息到Modal云端

set -e

echo "=========================================="
echo "DiffMS Modal Volumes 设置"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Modal CLI是否安装
if ! command -v modal &> /dev/null; then
    echo -e "${RED}错误: Modal CLI未安装${NC}"
    echo "请运行: pip install modal"
    exit 1
fi

echo -e "${GREEN}✓ Modal CLI 已安装${NC}"

# 检查Modal认证
if ! modal profile current &> /dev/null; then
    echo -e "${YELLOW}警告: Modal未认证${NC}"
    echo "请运行: modal setup"
    exit 1
fi

WORKSPACE=$(modal profile current)
echo -e "${GREEN}✓ Modal 已认证 (workspace: ${WORKSPACE})${NC}"

# 定义路径
DATA_SOURCE="/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data"
MODEL_SOURCE="/Users/aylin/Downloads/checkpoints/diffms_msg.ckpt"
STATS_SOURCE="/Users/aylin/Downloads/msg"

# 检查源文件是否存在
echo ""
echo "检查源文件..."

if [ ! -d "$DATA_SOURCE" ]; then
    echo -e "${RED}✗ 数据目录不存在: $DATA_SOURCE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 数据目录存在${NC}"

if [ ! -f "$MODEL_SOURCE" ]; then
    echo -e "${RED}✗ 模型文件不存在: $MODEL_SOURCE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 模型文件存在${NC}"

if [ ! -d "$STATS_SOURCE" ]; then
    echo -e "${RED}✗ 统计信息目录不存在: $STATS_SOURCE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 统计信息目录存在${NC}"

# 创建Volumes
echo ""
echo "创建Modal Volumes..."
modal volume create diffms-data || echo "Volume diffms-data 已存在"
modal volume create diffms-models || echo "Volume diffms-models 已存在"
modal volume create diffms-outputs || echo "Volume diffms-outputs 已存在"
modal volume create diffms-msg-stats || echo "Volume diffms-msg-stats 已存在"

# 上传数据
echo ""
echo "=========================================="
echo "上传预处理数据到Modal..."
echo "=========================================="
modal volume put diffms-data "$DATA_SOURCE" /data/processed_data

echo ""
echo "=========================================="
echo "上传模型到Modal..."
echo "=========================================="
modal volume put diffms-models "$MODEL_SOURCE" /models/diffms_msg.ckpt

echo ""
echo "=========================================="
echo "上传MSG统计信息到Modal..."
echo "=========================================="
# 上传所有.txt文件
for file in "$STATS_SOURCE"/*.txt; do
    filename=$(basename "$file")
    echo "上传 $filename..."
    modal volume put diffms-msg-stats "$file" "/msg_stats/$filename"
done

echo ""
echo "=========================================="
echo -e "${GREEN}✓ 所有数据已上传完成！${NC}"
echo "=========================================="

# 验证上传
echo ""
echo "验证上传的文件..."
echo "Data volume:"
modal volume ls diffms-data /data/processed_data

echo ""
echo "Models volume:"
modal volume ls diffms-models /models

echo ""
echo "Stats volume:"
modal volume ls diffms-msg-stats /msg_stats

echo ""
echo -e "${GREEN}设置完成！现在可以运行推理了。${NC}"
echo "运行: modal run diffms_inference.py --max-count 10"

