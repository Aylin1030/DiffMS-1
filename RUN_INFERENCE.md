# DiffMS 推理运行指南

## 在Modal云端运行推理

### 前提条件

1. 确保已上传数据和模型到Modal volumes:
```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
./upload_to_modal.sh
```

2. 确保Modal已配置并登录:
```bash
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
```

### 运行推理

#### 使用完整数据集

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_inference.py
```

#### 使用测试数据（前10个样本）

```bash
modal run diffms_inference.py --data-subdir test_top10
```

#### 限制样本数量

```bash
# 只处理前5个样本
modal run diffms_inference.py --max-count 5
```

### 下载结果

```bash
# 下载所有结果到本地
modal volume get diffms-outputs /outputs ./local_outputs
```

### 查看结果

结果会保存在Modal的`diffms-outputs` volume中：
- `/outputs/predictions/` - 原始pkl预测文件
- `/outputs/logs/` - 训练日志

## 参数说明

- `--max-count`: 限制处理的样本数量（用于快速测试）
- `--data-subdir`: 指定数据子目录，默认为`processed_data`
  - `processed_data`: 完整数据集
  - `test_top10`: 前10个测试样本

## 配置说明

推理配置在`diffms_inference.py`中：
- GPU: A100 (可改为H100/T4/A10G)
- 采样数量: 10个候选分子/样本
- 超时时间: 4小时

## 本地转换结果为表格

下载结果后，可以在本地转换为表格格式：

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
python convert_to_table.py
```

这会生成：
- `results/predictions_top1.tsv` - 每个样本的最佳预测
- `results/predictions_all_candidates.tsv` - 所有候选分子

## 故障排查

### 问题1: 数据未找到

错误: `FileNotFoundError: 数据目录不存在`

解决: 重新上传数据
```bash
./upload_to_modal.sh
```

### 问题2: GPU超时

错误: `TimeoutError`

解决: 减少样本数量或增加超时时间
```bash
# 减少样本
modal run diffms_inference.py --max-count 5

# 或修改diffms_inference.py中的timeout参数
```

### 问题3: 内存不足

错误: `OutOfMemoryError`

解决: 使用更大的GPU或减少batch size
- 修改`diffms_inference.py`中的GPU类型为"H100"
- 或减少`test_samples_to_generate`

## 性能优化

- **快速测试**: 使用`--max-count 5 --data-subdir test_top10`
- **完整运行**: 移除所有限制参数
- **提高质量**: 增加`test_samples_to_generate`（默认10，可改为100）

## 分子式约束

本次修复确保生成的分子符合输入的分子式约束：
- 所有10个候选分子都应该符合输入的分子式
- 候选分子是同分异构体（相同元素组成，不同结构）

详见: `docs/FORMULA_CONSTRAINT_FIX_20251028.md`

## 联系支持

如遇问题，请检查:
1. Modal日志: `modal run diffms_inference.py`的输出
2. 代码文档: `docs/`目录下的相关文档
