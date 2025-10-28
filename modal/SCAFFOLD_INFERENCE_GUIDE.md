# DiffMS 骨架约束推理 - Modal 部署指南

**日期**: 2024-10-28  
**用途**: 在 Modal 云平台运行骨架约束的 DiffMS 推理

---

## 🎯 概述

本指南帮助你使用 Modal 云平台运行骨架约束的质谱分子结构推理。

### 骨架信息

- **SMILES**: `CC(=CCCC(C1CCC2(C1(CCC3=C2CCC4C3(CCC(C4(C)C)O)C)C)C)C(=O)O)C`
- **类型**: 三萜类化合物骨架
- **分子式**: C30H48O3（骨架本身）
- **重原子数**: 33 个

---

## 📋 准备工作

### 1. 安装 Modal CLI

```bash
pip install modal
modal setup  # 首次使用需要登录
```

### 2. 上传测试数据到 Modal Volume

```bash
# 进入 modal 目录
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal

# 上传测试数据（前10个样本）
modal volume put diffms-data \
    /Users/aylin/yaolab_projects/madgen_yaolab/msdata/test_top10 \
    /data/test_top10
```

验证上传：

```bash
modal volume ls diffms-data /data/test_top10
```

应该看到：
```
spec_files/
subformulae/
split.tsv
labels.tsv
```

### 3. 确保模型 checkpoint 已上传

```bash
# 检查模型是否存在
modal volume ls diffms-models /models/

# 如果没有，上传模型
modal volume put diffms-models \
    /path/to/your/diffms_msg.ckpt \
    /models/diffms_msg.ckpt
```

---

## 🚀 运行推理

### 方法 1: 使用默认骨架（三萜类化合物）

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal

modal run diffms_scaffold_inference.py
```

**默认参数**:
- `scaffold_smiles`: 三萜类化合物（上面的长SMILES）
- `max_count`: 10（处理前10个样本）
- `data_subdir`: "test_top10"
- `enforce_scaffold`: True
- `use_rerank`: True

### 方法 2: 自定义参数

```bash
# 示例：使用苯环骨架
modal run diffms_scaffold_inference.py \
    --scaffold-smiles "c1ccccc1" \
    --max-count 5

# 示例：指定锚点位置
modal run diffms_scaffold_inference.py \
    --attachment-indices "2,5,7,10"
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--scaffold-smiles` | str | 三萜骨架 | 骨架的SMILES字符串 |
| `--max-count` | int | 10 | 处理的最大样本数 |
| `--data-subdir` | str | "test_top10" | 数据子目录 |
| `--attachment-indices` | str | None | 锚点索引（逗号分隔） |
| `--enforce-scaffold` | bool | True | 是否强制包含骨架 |
| `--use-rerank` | bool | True | 是否启用谱重排 |

---

## 📊 查看结果

### 1. 运行完成后，查看输出

脚本会输出：

```
🎉 骨架约束推理完成！
==================================================
状态: success
骨架SMILES: CC(=CCCC(...))C
骨架分子式: C30H48O3
数据目录: test_top10
处理数据量: 10
使用GPU: NVIDIA A100-SXM4-40GB

结果统计:
  总谱图数: 10
  有效SMILES: 95/100
  包含骨架: 87/100 (87.0%)
==================================================
```

### 2. 下载结果到本地

```bash
# 下载所有结果
modal volume get diffms-outputs /outputs/predictions_scaffold ./scaffold_results

# 下载 SMILES 文件
modal volume get diffms-outputs /outputs/smiles_scaffold ./smiles_results

# 下载可视化图片
modal volume get diffms-outputs /outputs/visualizations_scaffold ./viz_results
```

### 3. 查看结果文件

下载后的目录结构：

```
scaffold_results/
├── scaffold_inference_rank_0_pred_0.pkl  # 预测分子（RDKit Mol对象）
├── scaffold_inference_rank_0_pred_1.pkl
└── ...

smiles_results/
├── predictions_top1.tsv                   # Top-1 SMILES
└── predictions_all_candidates.tsv         # 所有候选 SMILES

viz_results/
└── top1_comparison.png                    # Top-1 分子对比图
```

### 4. 读取 SMILES 文件

```python
import pandas as pd

# Top-1 预测
top1_df = pd.read_csv('smiles_results/predictions_top1.tsv', sep='\t')
print(top1_df.head())

# 所有候选（包含骨架标记）
all_df = pd.read_csv('smiles_results/predictions_all_candidates.tsv', sep='\t')
print(all_df[all_df['contains_scaffold'] == True])  # 只看包含骨架的
```

---

## 🔧 重要说明

### 骨架与分子式的兼容性

骨架约束要求：**目标分子式 >= 骨架分子式**

测试数据中的分子式：
```
SPEC_4922  : C30H48O3  ✅ 与骨架相同（边界情况）
SPEC_6652  : C33H52O5  ✅ 大于骨架
SPEC_4838  : C36H58O8  ✅ 大于骨架
SPEC_5680  : C31H48O3  ✅ 大于骨架
...
```

**注意**：
- 骨架本身是 C30H48O3
- 如果目标分子式小于骨架，该样本会跳过骨架约束
- 如果所有样本都不兼容，推理会失败

### 性能估算

- **单个样本耗时**: ~10-15秒（A100 GPU）
- **10个样本总耗时**: ~2-3分钟
- **100个样本**: ~20-30分钟

### 成本估算（Modal）

- **GPU**: A100 @ $1.10/hour
- **10个样本**: ~$0.05
- **100个样本**: ~$0.50

---

## 🐛 故障排查

### 问题 1: "骨架与所有目标分子式都不兼容"

**原因**: 骨架太大，超过了所有测试样本的分子式

**解决**:
```bash
# 使用更小的骨架，例如苯环
modal run diffms_scaffold_inference.py \
    --scaffold-smiles "c1ccccc1"
```

### 问题 2: "Checkpoint文件不存在"

**原因**: 模型未上传到 Modal volume

**解决**:
```bash
# 上传模型
modal volume put diffms-models \
    /path/to/diffms_msg.ckpt \
    /models/diffms_msg.ckpt
```

### 问题 3: "数据目录不存在"

**原因**: 测试数据未上传

**解决**:
```bash
modal volume put diffms-data \
    /Users/aylin/yaolab_projects/madgen_yaolab/msdata/test_top10 \
    /data/test_top10
```

### 问题 4: 推理失败，但日志不清楚

**查看详细日志**:
```bash
modal run diffms_scaffold_inference.py 2>&1 | tee inference.log
```

---

## 📝 高级用法

### 1. 批量测试不同骨架

创建脚本 `test_multiple_scaffolds.sh`:

```bash
#!/bin/bash

# 苯环
modal run diffms_scaffold_inference.py \
    --scaffold-smiles "c1ccccc1" \
    --max-count 5

# 环己烷
modal run diffms_scaffold_inference.py \
    --scaffold-smiles "C1CCCCC1" \
    --max-count 5

# 萘
modal run diffms_scaffold_inference.py \
    --scaffold-smiles "c1ccc2ccccc2c1" \
    --max-count 5
```

### 2. 只处理特定样本

修改 `test_top10/split.tsv`，只保留需要的样本：

```tsv
name	split
SPEC_4922	test
SPEC_6652	test
```

### 3. 不强制骨架（软约束）

```bash
modal run diffms_scaffold_inference.py \
    --scaffold-smiles "c1ccccc1" \
    --enforce-scaffold False
```

这样会优先生成包含骨架的分子，但如果质谱不匹配也允许其他候选。

---

## 📚 相关文档

- **骨架约束原理**: `/Users/aylin/yaolab_projects/diffms_yaolab/docs/SCAFFOLD_CONSTRAINED_INFERENCE_20251028.md`
- **补丁说明**: `/Users/aylin/yaolab_projects/diffms_yaolab/README_SCAFFOLD_PATCH.md`
- **实现总结**: `/Users/aylin/yaolab_projects/diffms_yaolab/IMPLEMENTATION_SUMMARY_20251028.md`

---

## ✅ 完整运行清单

1. [ ] 安装 Modal CLI (`modal setup`)
2. [ ] 上传测试数据到 `diffms-data` volume
3. [ ] 确认模型存在于 `diffms-models` volume
4. [ ] 运行推理 (`modal run diffms_scaffold_inference.py`)
5. [ ] 等待完成（~2-3分钟 for 10 samples）
6. [ ] 下载结果 (`modal volume get ...`)
7. [ ] 查看 SMILES 和可视化

---

**维护者**: Yao Lab  
**最后更新**: 2024-10-28  
**状态**: ✅ 测试通过

