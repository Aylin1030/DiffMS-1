# DiffMS 推理目录

## 快速开始

### 测试推理（10个数据点）

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/inference
bash test_inference.sh
```

### 完整推理（478个数据点）

```bash
python run_inference.py \
    --checkpoint_path /Users/aylin/Downloads/checkpoints/diffms_msg.ckpt \
    --output_dir ./predictions
```

## 参数说明

- `--checkpoint_path`: 预训练模型路径（必需）
- `--output_dir`: 输出目录（默认: ./predictions）
- `--max_count`: 限制测试数据量（默认: None，测试全部）

## 输出结构

```
inference/
├── predictions_test/          # 测试输出（max_count=10）
│   └── inference_logs/
├── predictions/               # 完整推理输出
│   └── inference_logs/
└── inference.log             # 推理日志
```

预测的分子结构保存在：`DiffMS/preds/custom_inference/`

## 数据配置

- **数据源**: `/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data/`
- **数据格式**: MSG测试集格式
- **统计信息**: `/Users/aylin/Downloads/msg/`
- **数据量**: 478个质谱

## 关键特性

✅ **测试集模式**: 无需SMILES，`output_tbl: null`  
✅ **GPU自动检测**: 有GPU自动使用  
✅ **限制数据量**: 支持`--max_count`快速测试  
✅ **DiffMS标准输出**: SMILES, InChI等完整格式

