# MSG官方数据推理结果

**测试日期**: 2025-10-28  
**数据来源**: `/Users/aylin/Downloads/msg` (MSG官方数据)  
**测试样本**: 5个样本

---

## 📁 文件说明

### 1. `predictions_top1.tsv`
**Top-1预测结果** - 每个样本的最佳候选

**列说明**：
- `spectrum_id`: 样本ID (0-4)
- `rank`: 排名 (始终为1，表示最佳候选)
- `smiles`: 生成的SMILES字符串
- `inchi`: InChI标识符（论文评估标准）
- `inchikey`: InChI的hash版本
- `num_atoms`: 原子数
- `valid`: 化学有效性（True/False）
- `total_candidates`: 该样本的总候选数

### 2. `predictions_all_candidates.tsv`
**所有候选结果** - 每个样本的10个候选（部分样本）

格式与Top-1相同，但包含所有排名的候选。

### 3. `msg_pred_5.pkl` (如果存在)
**原始预测数据** - Python pickle格式

可以用Python加载：
```python
import pickle
with open('msg_pred_5.pkl', 'rb') as f:
    results = pickle.load(f)
```

---

## 📊 结果摘要

### Validity统计

| 指标 | 值 |
|------|-----|
| 测试样本数 | 5 |
| 生成候选总数 | ~50 (10/样本) |
| **有效候选数** | **1 (2%)** ❌ |
| 无效候选数 | 49 (98%) |

### 与论文对比

| 数据集 | Validity | Top-1 Acc | Top-10 Acc |
|--------|----------|-----------|------------|
| **论文报告 (MSG)** | **100%** ✅ | 1.04% | 3.13% |
| **我们的结果** | **2%** ❌ | N/A | N/A |

---

## 🔍 样本详情

### 样本特征
- **分子式**: C45H57N3O9, C37H40O8等
- **原子数**: 45-46个重原子
- **来源**: MassSpecGym数据集
- **仪器**: Orbitrap

### 主要问题

**Kekulization错误**:
```
Can't kekulize mol
Unkekulized atoms: [列表]
```

**价态错误**:
- 碳原子价态超过4
- 氧原子价态超过2
- 无法形成有效的Lewis结构

---

## 💡 如何查看结果

### 在Excel/Numbers中打开
```bash
# Mac
open predictions_top1.tsv

# 或者用Excel/Numbers直接打开TSV文件
```

### 在终端查看
```bash
# 查看Top-1结果
column -t -s $'\t' predictions_top1.tsv

# 查看所有候选
head -20 predictions_all_candidates.tsv | column -t -s $'\t'
```

### 用Python分析
```python
import pandas as pd

# 读取结果
df_top1 = pd.read_csv('predictions_top1.tsv', sep='\t')
df_all = pd.read_csv('predictions_all_candidates.tsv', sep='\t')

# 查看统计
print(df_top1.describe())
print(f"Validity: {df_top1['valid'].sum() / len(df_top1) * 100:.1f}%")

# 查看有效的分子
valid_mols = df_top1[df_top1['valid'] == True]
print(valid_mols[['spectrum_id', 'smiles', 'num_atoms']])
```

---

## 🔴 关键发现

### 1. Validity极低 (2%)
即使使用MSG官方数据，validity仍然只有2%，远低于论文报告的100%。

### 2. 不是数据问题
- 用官方数据测试仍然失败
- 说明问题在于**模型/checkpoint**

### 3. 生成质量差
- 大量Kekulization错误
- 价态错误严重
- 无法形成化学有效的分子

---

## 🎯 建议

### 紧急行动
1. **验证checkpoint完整性**
   - 检查文件大小
   - 重新下载
   - 验证MD5/SHA

2. **联系论文作者**
   - 获取正确checkpoint
   - 确认推理配置
   - 报告问题

3. **检查GitHub Issues**
   - https://github.com/coleygroup/DiffMS/issues
   - 查看已知问题

### 技术调查
- 代码版本匹配
- 配置参数验证
- 训练/推理流程对比

---

## 📚 相关文档

- `../FINAL_INVESTIGATION_REPORT.md` - 完整调查报告
- `../MSG_OFFICIAL_TEST_RESULTS.md` - MSG测试详情
- `../docs/VALENCE_CORRECTION_INVESTIGATION.md` - 价态修正调查

---

## ⚠️ 重要提示

**当前模型无法用于生产**

原因：
- 98%的生成分子化学无效
- 远低于论文性能
- 需要修复checkpoint/配置问题

建议：
- 暂停使用当前系统
- 等待问题解决
- 或考虑其他模型

