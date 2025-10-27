# DiffMS Modal 云端推理

使用Modal云平台运行DiffMS分子结构预测，无需本地GPU。

## 环境配置

### 镜像依赖

- Python 3.10
- PyTorch 2.0.1 (CUDA 11.8)
- PyTorch Lightning 2.0.0
- RDKit 2023.3.2
- Torch Geometric 2.3.1
- Hydra + OmegaConf

### Volumes (持久化存储)

| Volume名称 | 挂载路径 | 用途 |
|-----------|---------|------|
| `diffms-data` | `/data` | 预处理的质谱数据 |
| `diffms-models` | `/models` | 预训练模型checkpoint |
| `diffms-outputs` | `/outputs` | 推理结果输出 |
| `diffms-msg-stats` | `/msg_stats` | MSG统计信息 |

## 使用步骤

### 1. 安装Modal

```bash
pip install modal
modal setup
```

### 2. 上传数据和模型

```bash
# 上传预处理数据
modal volume put diffms-data /Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data /data

# 上传模型
modal volume put diffms-models /Users/aylin/Downloads/checkpoints/diffms_msg.ckpt /models/diffms_msg.ckpt

# 上传MSG统计信息
modal volume put diffms-msg-stats /Users/aylin/Downloads/msg/*.txt /msg_stats/
```

### 3. 运行推理

```bash
# 测试推理（10个数据点）
modal run diffms_inference.py --max-count 10

# 完整推理（所有数据）
modal run diffms_inference.py
```

### 4. 下载结果

```bash
# 下载推理结果
modal volume get diffms-outputs /outputs ./local_outputs
```

## 文件说明

- `diffms_inference.py` - Modal推理应用主文件
- `README.md` - 本文档

## 优势

✅ **无需本地GPU** - 在云端使用高性能GPU  
✅ **按需付费** - 只在运行时计费  
✅ **自动扩展** - 可并行处理多个任务  
✅ **持久化存储** - 数据和结果永久保存  

## 成本估算

- GPU类型: A100 (40GB) 或 H100
- 预计时间: 
  - 10个数据点: ~5-10分钟
  - 478个数据点: ~2-4小时

## 注意事项

1. 首次运行会下载所有依赖，需要一些时间
2. 数据上传到Volume后会持久化，无需重复上传
3. 模型权重也是持久化的

