#!/bin/bash
# 检查Modal推理状态并下载结果

cd /Users/aylin/yaolab_projects/diffms_yaolab/modal

echo "🔍 检查Modal推理任务状态..."
echo ""

# 获取最新的app状态
modal app list | head -10

echo ""
echo "📊 查看日志输出文件："
if [ -f "inference_output.log" ]; then
    echo "最后50行日志:"
    tail -50 inference_output.log
else
    echo "日志文件尚未生成"
fi

echo ""
echo "=================================="
echo "如果任务已完成，按回车下载结果..."
read

echo ""
echo "📥 下载推理结果..."
modal volume get diffms-outputs /outputs ./local_outputs

echo ""
echo "✅ 下载完成！"
echo ""
echo "📁 结果位置:"
echo "  - predictions: ./local_outputs/predictions/"
echo "  - logs: ./local_outputs/logs/"
echo ""
echo "查看结果文件:"
ls -lh ./local_outputs/predictions/ 2>/dev/null || echo "  (predictions目录为空或不存在)"
ls -lh ./local_outputs/logs/ 2>/dev/null || echo "  (logs目录为空或不存在)"

