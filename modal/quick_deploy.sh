#!/bin/bash
# DiffMS Modal快速部署和运行脚本
# 使用方法: ./quick_deploy.sh [test|full]

set -e  # 遇到错误立即退出

echo "============================================================"
echo "DiffMS Modal 快速部署和运行"
echo "============================================================"

# 配置
DATA_SUBDIR="msg_official_test5"
LOCAL_DATA_PATH="../msg_official_test5"
CHECKPOINT_PATH="/Users/aylin/Downloads/checkpoints/diffms_msg.ckpt"

# 运行模式
MODE="${1:-test}"  # 默认test模式

if [ "$MODE" = "test" ]; then
    MAX_COUNT=5
    echo "模式: 测试 (5个谱图)"
elif [ "$MODE" = "full" ]; then
    MAX_COUNT=""
    echo "模式: 完整 (所有谱图)"
else
    echo "错误: 未知模式 '$MODE'"
    echo "使用方法: ./quick_deploy.sh [test|full]"
    exit 1
fi

echo ""
echo "步骤 1: 检查Modal环境"
echo "============================================================"

if ! command -v modal &> /dev/null; then
    echo "✗ Modal未安装"
    echo "请运行: pip install modal"
    exit 1
fi

echo "✓ Modal已安装"

# 检查是否登录
if ! modal token show &> /dev/null; then
    echo "✗ 未登录Modal"
    echo "请运行: modal token new"
    exit 1
fi

echo "✓ Modal已登录"

echo ""
echo "步骤 2: 创建/验证Volumes"
echo "============================================================"

# 创建volumes（如果不存在）
modal volume create diffms-data 2>/dev/null && echo "✓ 创建 diffms-data" || echo "✓ diffms-data 已存在"
modal volume create diffms-models 2>/dev/null && echo "✓ 创建 diffms-models" || echo "✓ diffms-models 已存在"
modal volume create diffms-outputs 2>/dev/null && echo "✓ 创建 diffms-outputs" || echo "✓ diffms-outputs 已存在"
modal volume create diffms-msg-stats 2>/dev/null && echo "✓ 创建 diffms-msg-stats" || echo "✓ diffms-msg-stats 已存在"

echo ""
echo "步骤 3: 上传数据（如果需要）"
echo "============================================================"

read -p "是否上传数据到Modal? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "$LOCAL_DATA_PATH" ]; then
        echo "上传数据: $LOCAL_DATA_PATH → diffms-data/$DATA_SUBDIR"
        modal volume put diffms-data "$LOCAL_DATA_PATH" "$DATA_SUBDIR"
        echo "✓ 数据上传完成"
    else
        echo "✗ 数据目录不存在: $LOCAL_DATA_PATH"
        exit 1
    fi
else
    echo "⊳ 跳过数据上传"
fi

echo ""
read -p "是否上传checkpoint到Modal? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "上传checkpoint: $CHECKPOINT_PATH → diffms-models/diffms_msg.ckpt"
        modal volume put diffms-models "$CHECKPOINT_PATH" diffms_msg.ckpt
        echo "✓ Checkpoint上传完成"
    else
        echo "✗ Checkpoint不存在: $CHECKPOINT_PATH"
        exit 1
    fi
else
    echo "⊳ 跳过checkpoint上传"
fi

echo ""
echo "步骤 4: 验证文件"
echo "============================================================"

echo "数据volume内容:"
modal volume ls diffms-data | head -10

echo ""
echo "模型volume内容:"
modal volume ls diffms-models

echo ""
echo "步骤 5: 运行推理"
echo "============================================================"

if [ -z "$MAX_COUNT" ]; then
    echo "运行命令: modal run diffms_inference.py --data-subdir $DATA_SUBDIR"
    modal run diffms_inference.py --data-subdir "$DATA_SUBDIR"
else
    echo "运行命令: modal run diffms_inference.py --max-count $MAX_COUNT --data-subdir $DATA_SUBDIR"
    modal run diffms_inference.py --max-count "$MAX_COUNT" --data-subdir "$DATA_SUBDIR"
fi

echo ""
echo "步骤 6: 下载结果"
echo "============================================================"

RESULTS_DIR="./modal_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "下载到: $RESULTS_DIR"
modal volume get diffms-outputs /outputs "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "✓ 完成！"
echo "============================================================"
echo ""
echo "结果保存在: $RESULTS_DIR"
echo ""
echo "文件结构:"
echo "  $RESULTS_DIR/"
echo "  ├── predictions/          # PKL文件（原始输出）"
echo "  ├── smiles/              # SMILES字符串（TSV格式）"
echo "  │   ├── predictions_top1.tsv"
echo "  │   └── predictions_all_candidates.tsv"
echo "  ├── visualizations/      # 可视化图片"
echo "  │   ├── predictions_summary.tsv"
echo "  │   ├── top1_comparison.png"
echo "  │   └── spectrum_grids/"
echo "  └── logs/                # 运行日志"
echo ""
echo "查看结果:"
echo "  cat $RESULTS_DIR/smiles/predictions_top1.tsv"
echo "  open $RESULTS_DIR/visualizations/top1_comparison.png"
echo ""
echo "============================================================"

