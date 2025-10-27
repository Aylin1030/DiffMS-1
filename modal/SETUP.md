# Modal 环境配置指南

## 前置要求

1. **安装Modal CLI**
   ```bash
   pip install modal
   ```

2. **Modal账号认证**
   ```bash
   modal setup
   ```
   按照提示完成浏览器认证

3. **检查认证状态**
   ```bash
   modal config check
   ```

## 配置步骤

### 第一步：上传数据到Modal Volumes

运行自动化脚本：

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
bash setup_volumes.sh
```

这个脚本会：
- ✅ 创建4个Modal Volumes
- ✅ 上传预处理的质谱数据
- ✅ 上传预训练模型
- ✅ 上传MSG统计信息
- ✅ 验证上传完成

### 第二步：验证数据

查看已上传的数据：

```bash
# 查看数据volume
modal volume ls diffms-data /data

# 查看模型volume
modal volume ls diffms-models /models

# 查看统计volume
modal volume ls diffms-msg-stats /msg_stats

# 查看输出volume（初始为空）
modal volume ls diffms-outputs /outputs
```

### 第三步：运行测试推理

```bash
# 测试10个数据点
modal run diffms_inference.py --max-count 10
```

### 第四步：下载结果

```bash
# 下载所有输出
modal volume get diffms-outputs /outputs ./local_outputs

# 或者只下载预测文件
modal volume get diffms-outputs /outputs/predictions ./local_predictions
```

## 完整推理

处理所有数据：

```bash
modal run diffms_inference.py
```

## 监控和调试

### 查看日志

Modal会实时显示运行日志，包括：
- GPU使用情况
- 数据加载进度
- 模型推理进度
- 错误和警告信息

### 查看Volume使用情况

```bash
modal volume list
```

### 清理Volume（如需重新上传）

```bash
modal volume delete diffms-data
modal volume delete diffms-models
modal volume delete diffms-outputs
modal volume delete diffms-msg-stats
```

## 常见问题

### Q: 如何选择GPU类型？

在`diffms_inference.py`中修改：
```python
@app.function(
    gpu="A100",  # 可选: "A100", "H100", "T4", "A10G"
    ...
)
```

推荐：
- **测试**: T4 (便宜)
- **生产**: A100 (高性能)
- **最快**: H100 (最贵)

### Q: 上传大文件超时怎么办？

增加timeout或分批上传：
```bash
# 分批上传数据文件
for file in processed_data/spec_files/*.ms; do
    modal volume put diffms-data "$file" /data/spec_files/
done
```

### Q: 如何更新数据或模型？

直接重新上传即可覆盖：
```bash
modal volume put diffms-models /path/to/new_model.ckpt /models/diffms_msg.ckpt
```

## 成本优化

1. **使用测试模式** (`--max-count`) 先验证
2. **选择合适的GPU** (T4 vs A100)
3. **及时删除不用的Volume**
4. **批量处理** 而不是多次小量处理

## 下一步

完成配置后，查看 `README.md` 了解使用说明。

