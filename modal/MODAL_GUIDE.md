# DiffMS Modal 云端推理指南

## 📋 快速开始

### 1. 上传数据到Modal Volume

```bash
# 上传预处理数据
modal volume put diffms-data /Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data /data/processed_data

# 上传模型checkpoint
modal volume put diffms-models /Users/aylin/Downloads/checkpoints/diffms_msg.ckpt /models/diffms_msg.ckpt

# 上传MSG统计信息
modal volume put diffms-msg-stats /Users/aylin/Downloads/msg /msg_stats
```

### 2. 运行推理

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal

# 测试运行（10个样本）
modal run diffms_inference.py --max-count 10

# 完整推理（478个样本）
modal run diffms_inference.py
```

### 3. 下载结果

```bash
# 下载预测结果
modal volume get diffms-outputs /outputs ./modal_outputs

# 查看日志
modal volume get diffms-outputs /outputs/logs ./modal_logs
```

---

## 🔧 关键修复说明

### 问题1: 模型加载方式
**问题**: 手动load_state_dict可能不正确加载所有组件  
**修复**: 使用`load_from_checkpoint()`方法

### 问题2: 特征提取器创建顺序
**问题**: extra_features在domain_features之前创建  
**修复**: 先创建domain_features，再创建extra_features

### 问题3: train_smiles处理
**问题**: 没有正确处理空的train_smiles  
**修复**: 添加None检查和hasattr判断

### 问题4: 缺少文件检查
**问题**: 运行时才发现缺少必要文件  
**修复**: 启动时检查所有必要文件和目录

---

## 📁 必需的Volume结构

### diffms-data
```
/data/
└── processed_data/
    ├── labels.tsv
    ├── split.tsv
    ├── spec_files/
    │   └── SPEC_*.ms (478个文件)
    └── subformulae/default_subformulae/
        └── SPEC_*.json (478个文件)
```

### diffms-models
```
/models/
└── diffms_msg.ckpt
```

### diffms-msg-stats
```
/msg_stats/
├── train_smiles.txt
├── train_atom_types.pt
├── train_bond_types.pt
└── ... (其他统计文件)
```

### diffms-outputs（自动创建）
```
/outputs/
├── predictions/
└── logs/
    └── modal_inference/
```

---

## 🐛 常见问题排查

### 问题1: FileNotFoundError
**症状**: `Checkpoint文件不存在` 或 `数据目录不存在`

**检查步骤**:
```bash
# 查看volume内容
modal volume ls diffms-data
modal volume ls diffms-models
modal volume ls diffms-msg-stats

# 确认文件路径
modal volume get diffms-models /models/diffms_msg.ckpt - | head -c 100
```

**解决方案**: 确保上传时路径正确，volume内路径应该是：
- `/data/processed_data/` (不是 `/data/`)
- `/models/diffms_msg.ckpt` (不是 `/models/checkpoints/`)

### 问题2: 数据集为空或加载失败
**症状**: `数据模块创建失败` 或 `找不到训练数据`

**原因**: `split.tsv`中所有数据都标记为`test`，但DataModule需要创建train/val/test三个集合

**检查**:
```bash
# 检查split.tsv格式
modal volume get diffms-data /data/processed_data/split.tsv - | head -5
```

**解决方案**: 这是正常的！模型会创建空的train/val集合，只使用test集合进行推理。

### 问题3: Checkpoint配置缺失
**症状**: `Checkpoint中未找到配置信息 (hyper_parameters)`

**原因**: checkpoint文件损坏或格式不正确

**检查**:
```python
import torch
ckpt = torch.load('diffms_msg.ckpt', map_location='cpu')
print(ckpt.keys())  # 应该包含 'hyper_parameters' 和 'state_dict'
```

**解决方案**: 重新下载或使用正确的checkpoint文件

### 问题4: GPU不可用
**症状**: `GPU可用: False` 但期望使用GPU

**解决方案**: 检查Modal函数装饰器中的GPU配置：
```python
@app.function(
    gpu="A100",  # 或 "H100", "T4", "A10G"
    ...
)
```

### 问题5: 内存不足 (OOM)
**症状**: CUDA out of memory 错误

**解决方案**:
1. 使用`--max-count`限制批次大小
2. 修改`cfg.dataset.eval_batch_size`（在脚本中添加）
3. 升级到更大显存的GPU（A100 → H100）

---

## 💡 性能优化建议

### 1. 分批处理
```bash
# 分10批处理
for i in {0..9}; do
    modal run diffms_inference.py --max-count 50 --offset $((i*50))
done
```

### 2. 使用更快的GPU
```python
@app.function(
    gpu="H100",  # 比A100快约2-3倍
    ...
)
```

### 3. 调整批次大小
在脚本中添加：
```python
cfg.dataset.eval_batch_size = 32  # 默认可能是8或16
```

---

## 📊 监控和日志

### 查看实时日志
```bash
modal app logs diffms-inference
```

### 查看Volume使用情况
```bash
modal volume ls diffms-data
modal volume ls diffms-models
modal volume ls diffms-outputs
```

### 清理旧的输出
```bash
# 谨慎使用！会删除所有输出
modal volume rm diffms-outputs /outputs/predictions/*
```

---

## 🔄 与本地版本的对比

| 特性 | Modal版本 | 本地版本 |
|------|-----------|----------|
| 运行环境 | 云端容器 | 本地机器 |
| GPU | A100/H100等 | 本地GPU/CPU |
| 存储 | Volume持久化 | 本地文件系统 |
| 依赖管理 | 镜像自动安装 | 需手动配置 |
| 路径 | 容器内路径（/data等） | 绝对路径 |
| 模型加载 | load_from_checkpoint | load_from_checkpoint |
| 适用场景 | 大规模批量推理 | 快速测试调试 |

---

## 📝 修改历史

### 2025-10-28 修复
1. ✅ 修复模型加载方式（使用load_from_checkpoint）
2. ✅ 修复特征提取器创建顺序
3. ✅ 添加文件存在性检查
4. ✅ 改进错误处理和日志输出
5. ✅ 修复train_smiles为None的问题
6. ✅ 添加详细的步骤日志
7. ✅ 改进GPU检测逻辑

### 下一步优化
- [ ] 支持断点续传
- [ ] 添加结果自动下载
- [ ] 支持多个checkpoint批量测试
- [ ] 添加性能指标收集

---

## 🆘 获取帮助

如果遇到问题：

1. **查看详细日志**: `modal app logs diffms-inference`
2. **检查Volume**: `modal volume ls <volume-name>`
3. **验证文件**: 下载到本地检查格式
4. **测试本地版本**: 先在本地测试是否能正常运行
5. **检查网络**: Modal需要稳定的网络连接

---

## 📄 相关文件

- `diffms_inference.py` - Modal推理脚本
- `../inference/run_inference.py` - 本地推理脚本（用于对比）
- `../inference/README.md` - 本地推理说明
- `/Users/aylin/yaolab_projects/madgen_yaolab/msdata/README.md` - 数据格式说明

