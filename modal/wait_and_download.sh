#!/bin/bash
# 等待Modal推理完成并下载结果

echo "⏳ 等待Modal推理任务完成..."
echo "提示: 首次运行需要构建Docker镜像（约10-15分钟）"
echo ""

# 等待后台进程完成
wait

echo ""
echo "✅ 推理任务完成！"
echo ""
echo "📥 下载推理结果..."

# 下载结果到本地
modal volume get diffms-outputs /outputs ./local_outputs

echo ""
echo "🎉 完成！结果已下载到:"
echo "   ./local_outputs/"
echo ""
echo "查看预测结果:"
ls -lh ./local_outputs/predictions/ 2>/dev/null || echo "   predictions/ 目录将包含预测的分子结构"
echo ""
echo "查看日志:"
ls -lh ./local_outputs/logs/ 2>/dev/null || echo "   logs/ 目录将包含推理日志"

