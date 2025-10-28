# DiffMS Modal云端部署和运行指南

**日期**: 2024-10-28  
**版本**: 2.0 (集成后处理)

---

## 🎯 新增功能

### ✅ 自动化完整流程

现在一次运行即可完成：

1. **推理**: 生成分子图 → pkl文件
2. **转换**: pkl → canonical SMILES (TSV)
3. **可视化**: 生成分子结构图 (PNG)

**输出结构**:
```
/outputs/
├── predictions/              # PKL文件
│   └── modal_inference_rank_0_pred_0.pkl
├── smiles/                   # SMILES字符串（TSV）
│   ├── predictions_top1.tsv
│   └── predictions_all_candidates.tsv
├── visualizations/           # 可视化图片
│   ├── predictions_summary.tsv
│   ├── top1_comparison.png
│   └── spectrum_grids/
│       ├── spectrum_0000_grid.png
│       ├── spectrum_0001_grid.png
│       └── ...
└── logs/                     # 日志
    └── modal_inference/
```

---

## 📋 部署步骤

### 步骤1: 准备数据

确保数据目录包含所有必要文件：

```bash
msg_official_test5/
├── split.tsv              # 谱图ID和split信息
├── labels.tsv             # formula, smiles, inchikey
└── spec_files/            # .ms谱图文件
    ├── MassSpecGymID0000201.ms
    ├── MassSpecGymID0000202.ms
    └── ...
```

**验证数据**:
```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
python validate_setup.py
```

### 步骤2: 上传数据到Modal Volume

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab

# 上传测试数据
modal volume put diffms-data msg_official_test5 msg_official_test5

# 上传checkpoint（如果还没上传）
modal volume put diffms-models /Users/aylin/Downloads/checkpoints/diffms_msg.ckpt diffms_msg.ckpt

# 上传MSG统计文件（如果有）
modal volume put diffms-msg-stats /path/to/msg_stats msg_stats
```

**验证上传**:
```bash
# 查看数据volume
modal volume ls diffms-data

# 查看模型volume
modal volume ls diffms-models
```

### 步骤3: 运行推理（一键完成所有）

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal

# 测试运行（5个谱图）
modal run diffms_inference.py --max-count 5 --data-subdir msg_official_test5

# 完整运行（所有谱图）
modal run diffms_inference.py --data-subdir msg_official_test5
```

**运行流程**:
```
步骤 1-9:  初始化和模型加载
步骤 10:   运行推理 → 生成pkl文件
步骤 11.1: 转换为SMILES → 生成TSV文件
步骤 11.2: 生成可视化 → 生成PNG文件
```

### 步骤4: 下载结果

```bash
# 下载所有结果
modal volume get diffms-outputs /outputs ./modal_results

# 或分别下载
modal volume get diffms-outputs /outputs/smiles ./modal_results/smiles
modal volume get diffms-outputs /outputs/visualizations ./modal_results/visualizations
```

---

## 📊 输出文件说明

### 1. SMILES文件 (TSV格式)

**predictions_top1.tsv**:
```tsv
spec_id         smiles
spec_0000      CCO
spec_0001      CC(C)O
spec_0002      
```

**predictions_all_candidates.tsv**:
```tsv
spec_id         rank    smiles
spec_0000      1       CCO
spec_0000      2       CC(O)C
spec_0001      1       CC(C)O
```

### 2. 可视化文件

**predictions_summary.tsv**:
```tsv
spec_id  rank  valid  smiles
spec_0000  1   True   CCO
spec_0000  2   False  
```

**top1_comparison.png**: 
- 所有谱图Top-1预测的网格对比图
- 最多显示20个分子

**spectrum_grids/**:
- 每个谱图的所有候选（最多10个）
- 文件名：`spectrum_0000_grid.png`

### 3. PKL文件 (原始输出)

```python
# 读取pkl文件
import pickle
with open('modal_inference_rank_0_pred_0.pkl', 'rb') as f:
    predictions = pickle.load(f)

# 结构: List[List[Mol对象]]
# predictions[spec_idx][rank] → rdkit.Chem.Mol
```

---

## 🔍 监控和调试

### 查看运行日志

```bash
# 实时查看日志（运行时）
modal run diffms_inference.py --max-count 5

# 日志会实时显示：
# - 数据加载
# - 模型初始化
# - 推理进度
# - 转换统计
# - 可视化生成
```

### 检查输出统计

运行完成后会显示：
```
步骤 11: 后处理 - 转换和可视化
================================================================================
11.1 转换为SMILES...
  处理: modal_inference_rank_0_pred_0.pkl
  总共 5 个谱图
  ✓ Top-1预测: predictions_top1.tsv (5 行)
  ✓ 所有候选: predictions_all_candidates.tsv (XX 行)
  统计: XX/50 有效SMILES (XX.X%)

11.2 生成可视化图片...
  ✓ 摘要表格: predictions_summary.tsv
  ✓ Top-1对比图: top1_comparison.png (X 个分子)
  ✓ 网格图: X 个文件

✓ 后处理完成！
================================================================================
```

### 验证结果

```bash
# 下载后验证SMILES有效性
cd modal_results/smiles

python -c "
import pandas as pd
from rdkit import Chem

df = pd.read_csv('predictions_top1.tsv', sep='\t')
invalid = 0
for idx, row in df.iterrows():
    if row['smiles'] and row['smiles'] != '':
        if Chem.MolFromSmiles(row['smiles']) is None:
            invalid += 1
            print(f'Invalid: {row[\"smiles\"]}')

print(f'Total: {len(df)}, Invalid: {invalid}')
"
```

---

## ⚙️ 高级配置

### 调整采样数量

编辑 `diffms_inference.py:226`:
```python
cfg.general.test_samples_to_generate = 10  # 改为100用于生产
```

### 更换GPU类型

编辑 `diffms_inference.py:100`:
```python
gpu="A100"  # 可选: "H100", "T4", "A10G"
```

### 调整超时时间

编辑 `diffms_inference.py:101`:
```python
timeout=4 * HOURS  # 根据数据量调整
```

---

## 🚨 常见问题

### 问题1: Volume不存在

```bash
# 创建volumes
modal volume create diffms-data
modal volume create diffms-models
modal volume create diffms-outputs
modal volume create diffms-msg-stats
```

### 问题2: 数据文件缺失

**错误**: `FileNotFoundError: 缺少必要文件/目录: spec_folder`

**解决**:
```bash
# 确保数据目录结构正确
ls msg_official_test5/
# 应该包含: split.tsv, labels.tsv, spec_files/

# 重新上传
modal volume put diffms-data msg_official_test5 msg_official_test5
```

### 问题3: Checkpoint加载失败

**错误**: `RuntimeError: Error(s) in loading state_dict`

**解决**:
```bash
# 验证checkpoint
python modal/debug_checkpoint.py

# 重新上传checkpoint
modal volume put diffms-models /path/to/diffms_msg.ckpt diffms_msg.ckpt
```

### 问题4: 可视化生成失败

**错误**: `✗ 可视化生成失败`

**原因**: 没有有效的分子

**解决**: 检查模型输出，可能需要调整模型参数或数据

---

## 📈 性能建议

### 1. GPU选择

| GPU类型 | 内存 | 速度 | 适用场景 |
|---------|------|------|----------|
| T4 | 16GB | 慢 | 小数据测试 |
| A10G | 24GB | 中 | 中等数据 |
| A100 | 40GB | 快 | 大规模推理 |
| H100 | 80GB | 最快 | 超大数据 |

### 2. 批次大小

根据GPU内存调整数据加载批次大小（在config中）

### 3. 采样数量

- 测试: `test_samples_to_generate = 10`
- 生产: `test_samples_to_generate = 100`

---

## 🎯 完整示例

### 端到端运行示例

```bash
# 1. 准备环境
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal

# 2. 验证数据
python validate_setup.py

# 3. 上传数据（首次）
modal volume put diffms-data msg_official_test5 msg_official_test5

# 4. 运行推理（自动完成转换和可视化）
modal run diffms_inference.py \
    --max-count 5 \
    --data-subdir msg_official_test5

# 5. 下载结果
modal volume get diffms-outputs /outputs ./modal_results

# 6. 查看结果
ls -R modal_results/
# modal_results/
# ├── predictions/          # PKL文件
# ├── smiles/              # TSV文件
# ├── visualizations/      # PNG文件
# └── logs/

# 7. 验证SMILES
cat modal_results/smiles/predictions_top1.tsv

# 8. 查看图片
open modal_results/visualizations/top1_comparison.png
```

---

## 📚 相关文档

- **快速参考**: `QUICK_FIX_REFERENCE.md`
- **完整流程**: `COMPLETE_WORKFLOW_SUMMARY.md`
- **可视化指南**: `VISUALIZATION_GUIDE.md`
- **图结构说明**: `docs/GRAPH_TO_MOLECULE_PIPELINE.md`

---

## ✅ 检查清单

部署前检查：

- [ ] Modal账号已设置
- [ ] Volumes已创建
- [ ] 数据已上传
- [ ] Checkpoint已上传
- [ ] 数据格式验证通过

运行前检查：

- [ ] `diffms_inference.py` 已更新
- [ ] GPU类型已选择
- [ ] 采样数量已设置
- [ ] 数据子目录路径正确

运行后检查：

- [ ] PKL文件已生成
- [ ] TSV文件已生成
- [ ] PNG图片已生成
- [ ] SMILES有效性验证通过

---

## 🎉 总结

现在你只需要一个命令：

```bash
modal run diffms_inference.py --data-subdir msg_official_test5
```

就能完成：
1. ✅ 推理
2. ✅ 转换为SMILES
3. ✅ 生成可视化图片

所有结果自动保存到Modal Volume！

---

**更新日期**: 2024-10-28  
**版本**: 2.0  
**状态**: ✅ 生产就绪

