# DiffMS推理完整指南

**更新日期**: 2024-10-28  
**状态**: ✅ 已完成所有检查和修正

---

## 🎯 快速开始

### 1行命令运行完整流程

```bash
# 运行推理 + 转换 + 可视化
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal && \
modal run diffms_inference.py --data-subdir msg_official_test5 && \
python convert_predictions_to_smiles.py && \
python visualize_predictions.py
```

**输出**:
- `results_smiles/predictions_top1.tsv` - SMILES字符串
- `visualizations/` - 分子结构图

---

## 📚 文档导航

### 🔥 必读文档

1. **[QUICK_FIX_REFERENCE.md](QUICK_FIX_REFERENCE.md)** - 快速参考卡片
   - 7个检查点修正
   - 常见错误解决
   - 核心要点

2. **[COMPLETE_WORKFLOW_SUMMARY.md](COMPLETE_WORKFLOW_SUMMARY.md)** - 完整工作流程
   - 使用指南
   - 工具清单
   - 验证清单

### 📖 详细文档

3. **[docs/INFERENCE_CHECKLIST_FIXES_20251028.md](docs/INFERENCE_CHECKLIST_FIXES_20251028.md)**
   - 7个检查点详细说明
   - 配置修正代码
   - 论文要求对照

4. **[docs/GRAPH_TO_MOLECULE_PIPELINE.md](docs/GRAPH_TO_MOLECULE_PIPELINE.md)**
   - 图结构详细说明
   - 转换流程代码
   - 完整示例

5. **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)**
   - 可视化完整指南
   - 工具使用说明
   - 高级选项

6. **[FINAL_CHECKLIST_SUMMARY.md](FINAL_CHECKLIST_SUMMARY.md)**
   - 修正总结
   - 验证状态
   - 准备清单

---

## ✅ 核心确认

### 1. 模型输出 = 图结构 ✅

```python
# DiffMS生成离散分子图
X: Tensor  # 节点类型 [batch, n]  - 原子 (C, N, O, F, P, S, Cl, Br)
E: Tensor  # 邻接矩阵 [batch, n, n] - 键 (无, 单, 双, 三, 芳香)
```

### 2. 图 → RDKit Mol转换 ✅

```python
# visualization.py: mol_from_graphs()
图结构 → RWMol对象 → 添加原子和键 → Mol对象
```

### 3. 价态修正 ✅

```python
# diffusion_model_spec2mol.py: correct_mol()
Mol对象 → 检查价态 → 调整氢原子 → 修正键阶 → 修正后的Mol
```

### 4. 输出格式 ✅

- **pkl文件**: `List[List[Mol对象]]` - 中间结果
- **TSV文件**: `spec_id\tsmiles` - 最终结果（Canonical SMILES）

---

## 🛠️ 工具使用

### 推理工具

```bash
# Modal云端推理
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_inference.py \
    --data-subdir msg_official_test5 \
    --max-count 5
```

**输出**: `modal_inference_rank_0_pred_0.pkl`

### 转换工具

```bash
# pkl → SMILES (TSV)
python convert_predictions_to_smiles.py
```

**输出**:
- `results_smiles/predictions_top1.tsv`
- `results_smiles/predictions_all_candidates.tsv`

### 可视化工具

```bash
# pkl → 结构图 (PNG)
python visualize_predictions.py
```

**输出**:
```
visualizations/
├── predictions_summary.tsv        # 详细信息
├── top1_comparison.png           # Top-1对比
└── spectrum_grids/               # 网格图
    ├── spectrum_0000_grid.png
    └── ...
```

### 验证工具

```bash
# Checkpoint验证
python debug_checkpoint.py

# 完整设置验证
python validate_setup.py
```

---

## 📋 修正清单

根据建议完成的7个检查点：

| # | 检查点 | 状态 | 文件位置 |
|---|--------|------|----------|
| 1 | Checkpoint结构 | ✅ | `debug_checkpoint.py` |
| 2 | decoder/encoder配置 | ✅ | `diffms_inference.py:229-232` |
| 3 | test_only配置 | ✅ | `diffms_inference.py:222-223` |
| 4 | formula字段 | ✅ | `labels.tsv` 验证 |
| 5 | Mol→SMILES转换 | ✅ | `convert_predictions_to_smiles.py` |
| 6 | 路径配置 | ✅ | `diffms_inference.py:119-122` |
| 7 | 版本兼容 | ✅ | `diffms_inference.py:34-58` |

**详细说明**: 见 [INFERENCE_CHECKLIST_FIXES_20251028.md](docs/INFERENCE_CHECKLIST_FIXES_20251028.md)

---

## 🎨 可视化示例

### 查看pkl文件

```python
import pickle
from rdkit import Chem

# 读取pkl
with open('modal_inference_rank_0_pred_0.pkl', 'rb') as f:
    predictions = pickle.load(f)

# 第一个谱图的第一个候选
mol = predictions[0][0]

# 转换为SMILES
smiles = Chem.MolToSmiles(mol, canonical=True)
print(f"SMILES: {smiles}")

# 分子信息
print(f"分子式: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
print(f"原子数: {mol.GetNumAtoms()}")
print(f"键数: {mol.GetNumBonds()}")
```

### 绘制分子结构

```python
from rdkit.Chem import Draw

# 单个分子
img = Draw.MolToImage(mol, size=(400, 400))
img.save('molecule.png')

# 网格对比
candidates = predictions[0][:10]
valid_mols = [m for m in candidates if m is not None]

img = Draw.MolsToGridImage(
    valid_mols,
    molsPerRow=5,
    subImgSize=(300, 300),
    legends=[f"Rank {i+1}" for i in range(len(valid_mols))]
)
img.save('candidates_grid.png')
```

---

## ⚠️ 关键注意事项

### 1. pkl文件不是最终输出！

```python
# ✗ 错误：直接使用pkl
predictions = pickle.load(open('pred.pkl', 'rb'))
# 这是Mol对象，不是SMILES字符串！

# ✓ 正确：转换为SMILES
python convert_predictions_to_smiles.py
# 生成TSV文件，包含SMILES字符串
```

### 2. 必须是Canonical SMILES

```python
# 论文要求：
# - Canonical格式
# - 无立体化学

Chem.RemoveStereochemistry(mol)
smiles = Chem.MolToSmiles(mol, canonical=True)
```

### 3. 验证所有输出

```python
# 确保所有SMILES都是有效字符串
for smiles in output_smiles:
    assert isinstance(smiles, str)
    assert Chem.MolFromSmiles(smiles) is not None
```

---

## 📊 数据流

```
MS谱图 + 分子式
    ↓
DiffMS模型
    ↓
图结构 (X, E)
    ↓
mol_from_graphs()
    ↓
RDKit Mol对象
    ↓
correct_mol()
    ↓
pkl文件
    ↓
┌───────┴────────┐
↓                ↓
SMILES (TSV)    结构图 (PNG)
```

---

## 🔍 故障排除

### 问题1: "乱码"输出

**原因**: 直接使用pkl文件  
**解决**: 运行 `convert_predictions_to_smiles.py`

### 问题2: 维度不匹配

**原因**: 配置维度与checkpoint不一致  
**解决**: 使用固定维度 (X:16, E:5, y:2061)

### 问题3: Formula字段缺失

**原因**: labels.tsv格式不正确  
**解决**: 确保包含formula列，格式如 `C45H57N3O9`

---

## 📁 项目结构

```
diffms_yaolab/
├── DiffMS/                          # 源代码
│   └── src/
│       ├── diffusion_model_spec2mol.py  # 扩散模型
│       └── analysis/
│           ├── visualization.py         # 图→Mol转换
│           └── rdkit_functions.py       # 价态修正
│
├── modal/                           # 推理脚本
│   ├── diffms_inference.py         # Modal推理
│   ├── convert_predictions_to_smiles.py  # pkl→SMILES
│   ├── visualize_predictions.py    # pkl→图片
│   ├── debug_checkpoint.py         # Checkpoint验证
│   └── validate_setup.py           # 完整验证
│
├── docs/                            # 文档
│   ├── INFERENCE_CHECKLIST_FIXES_20251028.md
│   └── GRAPH_TO_MOLECULE_PIPELINE.md
│
├── QUICK_FIX_REFERENCE.md          # 快速参考
├── COMPLETE_WORKFLOW_SUMMARY.md    # 完整总结
├── VISUALIZATION_GUIDE.md          # 可视化指南
├── FINAL_CHECKLIST_SUMMARY.md      # 最终清单
└── README_INFERENCE.md             # 本文档
```

---

## 🎓 论文要求对照

| 论文要求 | 实现 | 验证 |
|---------|------|------|
| 输入: Spectra + Formula | ✅ | labels.tsv |
| 输出: Canonical SMILES | ✅ | `Chem.MolToSmiles(canonical=True)` |
| 无立体化学 | ✅ | `RemoveStereochemistry()` |
| 价态修正 | ✅ | `correct_mol()` |
| 图结构 | ✅ | X (节点) + E (边) |
| RDKit转换 | ✅ | `mol_from_graphs()` |

---

## 💡 使用建议

1. **首次使用**: 先读 [QUICK_FIX_REFERENCE.md](QUICK_FIX_REFERENCE.md)
2. **深入理解**: 读 [GRAPH_TO_MOLECULE_PIPELINE.md](docs/GRAPH_TO_MOLECULE_PIPELINE.md)
3. **可视化**: 读 [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)
4. **完整流程**: 读 [COMPLETE_WORKFLOW_SUMMARY.md](COMPLETE_WORKFLOW_SUMMARY.md)

---

## 📞 支持

- **Issues**: GitHub Issues
- **文档**: 见上方文档导航
- **示例**: `modal/` 目录下的脚本

---

## ✅ 状态总结

- ✅ **所有检查点已完成**
- ✅ **图结构已确认**
- ✅ **工具链已就绪**
- ✅ **文档已完善**
- ✅ **可以开始生产环境推理**

---

**最后更新**: 2024-10-28  
**版本**: 1.0  
**状态**: ✅ 生产就绪

🎉 **一切准备就绪！开始推理吧！**

