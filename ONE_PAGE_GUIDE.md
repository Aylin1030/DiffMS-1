# DiffMS一页纸快速指南

**版本**: 2.0 | **日期**: 2024-10-28 | **状态**: ✅ 完全集成

---

## 🚀 快速开始（3步）

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal

# 1. 快速部署
./quick_deploy.sh test

# 2. 查看结果
ls -R modal_results_*/

# 3. 验证SMILES
cat modal_results_*/smiles/predictions_top1.tsv
```

---

## 📊 输出文件

```
/outputs/
├── predictions/              # PKL（Mol对象）
├── smiles/                   # TSV（Canonical SMILES）✅ 可提交
│   ├── predictions_top1.tsv
│   └── predictions_all_candidates.tsv
└── visualizations/           # PNG（结构图）
    ├── top1_comparison.png
    └── spectrum_grids/
```

---

## ✅ 核心确认

| 项目 | 状态 | 说明 |
|------|------|------|
| 模型输出 | ✅ | 图结构（X节点 + E邻接矩阵） |
| 转换 | ✅ | mol_from_graphs() → Mol对象 |
| 修正 | ✅ | correct_mol() → 价态修正 |
| SMILES | ✅ | Canonical + 无立体化学 |
| 可视化 | ✅ | Draw.MolsToGridImage() |
| 衔接 | ✅ | 完全自动化 |

---

## 📋 检查清单（7个）

- [x] 1. Checkpoint包含encoder+decoder（366参数）
- [x] 2. decoder/encoder = None（避免重复）
- [x] 3. test_only = True（布尔值）
- [x] 4. Formula字段格式正确
- [x] 5. Mol→SMILES转换（Canonical）
- [x] 6. 路径配置正确
- [x] 7. 版本兼容

---

## 🛠️ 工具和文档

### 工具（6个）
- `diffms_inference.py` - Modal推理（集成版）
- `convert_predictions_to_smiles.py` - PKL→SMILES
- `visualize_predictions.py` - PKL→图片
- `debug_checkpoint.py` - Checkpoint验证
- `validate_setup.py` - 设置验证
- `quick_deploy.sh` - 快速部署

### 文档（10个）
- `README_INTEGRATED.md` - **集成版指南**⭐
- `DEPLOYMENT_GUIDE.md` - 部署详解
- `QUICK_FIX_REFERENCE.md` - 快速参考
- `COMPLETE_WORKFLOW_SUMMARY.md` - 完整总结
- `VISUALIZATION_GUIDE.md` - 可视化指南
- `docs/GRAPH_TO_MOLECULE_PIPELINE.md` - 图结构详解
- 更多...

---

## 🔄 数据流

```
MS谱图 + 分子式
    ↓
DiffMS模型 → 图（X, E）
    ↓
mol_from_graphs() → Mol对象
    ↓
correct_mol() → 价态修正
    ↓
PKL文件
    ↓
┌──────────────┴──────────────┐
↓                             ↓
Canonical SMILES (TSV)        结构图 (PNG)
```

---

## ⚡ 命令参考

```bash
# 测试（5个谱图）
modal run diffms_inference.py --max-count 5 --data-subdir msg_official_test5

# 完整运行
modal run diffms_inference.py --data-subdir msg_official_test5

# 下载结果
modal volume get diffms-outputs /outputs ./results

# 验证SMILES
python -c "
import pandas as pd
from rdkit import Chem
df = pd.read_csv('results/smiles/predictions_top1.tsv', sep='\t')
print(f'Total: {len(df)}, Valid: {sum(pd.notna(df.smiles) & (df.smiles != \"\"))}')"
```

---

## 📊 示例输出

**predictions_top1.tsv**:
```tsv
spec_id         smiles
spec_0000      CCO
spec_0001      CC(C)O
spec_0002      CCCC
```

**日志示例**:
```
✓ 推理完成！
11.1 转换为SMILES...
  ✓ Top-1预测: predictions_top1.tsv (5 行)
  统计: 20/50 有效SMILES (40.0%)
11.2 生成可视化图片...
  ✓ Top-1对比图: top1_comparison.png (3 个分子)
✓ 后处理完成！
```

---

## 🎯 关键改进

### 之前（3步）
```bash
modal run diffms_inference.py
python convert_predictions_to_smiles.py
python visualize_predictions.py
```

### 现在（1步）
```bash
modal run diffms_inference.py --data-subdir msg_official_test5
```

自动完成：推理 + 转换 + 可视化

---

## 🚨 快速故障排除

| 问题 | 解决 |
|------|------|
| Volume不存在 | `modal volume create diffms-data` |
| 数据缺失 | `modal volume put diffms-data ...` |
| 维度不匹配 | 使用固定维度（X:16, E:5, y:2061） |
| 无有效SMILES | 检查输入数据质量 |

---

## 📞 获取帮助

- **详细指南**: 见 `README_INTEGRATED.md`
- **技术细节**: 见 `docs/GRAPH_TO_MOLECULE_PIPELINE.md`
- **快速参考**: 见 `QUICK_FIX_REFERENCE.md`
- **完整报告**: 见 `INTEGRATION_COMPLETE.md`

---

**🎉 一切就绪！开始使用：`./quick_deploy.sh test`**

---

*更新: 2024-10-28 | 版本: 2.0 | 完全集成版*

