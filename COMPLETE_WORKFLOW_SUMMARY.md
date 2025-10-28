# DiffMS完整工作流程总结

**更新日期**: 2024-10-28  
**状态**: ✅ 所有检查点已完成并验证

---

## 🎯 核心确认

### ✅ 1. 模型输出是图结构

```python
# DiffMS模型生成的是离散分子图
输出 = {
    'X': Tensor([batch, n]),        # 节点类型 (0-7: C,N,O,F,P,S,Cl,Br)
    'E': Tensor([batch, n, n])      # 邻接矩阵 (0-4: 无,单,双,三,芳香)
}
```

**代码位置**: `diffusion_model_spec2mol.py:664-710 (sample_batch方法)`

### ✅ 2. 图→RDKit Mol转换

```python
# visualization.py:16-59
def mol_from_graphs(node_list, adjacency_matrix):
    """
    图结构 → RDKit Mol对象
    
    步骤:
    1. 创建RWMol对象
    2. 添加原子（使用atom_decoder映射）
    3. 添加键（使用bond_type映射）
    4. 转换为不可编辑的Mol对象
    """
    mol = Chem.RWMol()
    # ... 添加原子和键 ...
    return mol.GetMol()
```

### ✅ 3. 价态修正

```python
# diffusion_model_spec2mol.py:692-708
from analysis.rdkit_functions import correct_mol

if mol is not None:
    editable_mol = Chem.RWMol(mol)
    corrected_mol, no_correct = correct_mol(editable_mol)
    if corrected_mol is not None:
        mol = corrected_mol
```

### ✅ 4. 输出格式

**pkl文件** (中间结果):
```python
# List[List[Mol对象]]
# [谱图索引][候选排名] → rdkit.Chem.Mol
predicted_mols[0][0]  # 第一个谱图的第一个候选
```

**TSV文件** (最终结果):
```tsv
spec_id         smiles
spec_0000      CCO
spec_0001      CC(C)O
```

---

## 📋 完整工作流程

### Phase 1: 修正配置（已完成 ✅）

根据建议清单完成的7个检查点：

| 检查点 | 修正内容 | 文件 |
|--------|----------|------|
| 1. Checkpoint结构 | ✅ 验证包含encoder和decoder | `debug_checkpoint.py` |
| 2. decoder/encoder配置 | ✅ 设为None避免重复加载 | `diffms_inference.py:229-232` |
| 3. test_only配置 | ✅ 改为布尔值True | `diffms_inference.py:222-223` |
| 4. formula字段 | ✅ 验证格式正确 | `labels.tsv` |
| 5. Mol→SMILES转换 | ✅ Canonical+无立体化学 | `convert_predictions_to_smiles.py` |
| 6. 路径配置 | ✅ 工作目录正确 | `diffms_inference.py:119-122` |
| 7. 版本兼容 | ✅ 依赖版本匹配 | `diffms_inference.py:34-58` |

**详细文档**:
- `docs/INFERENCE_CHECKLIST_FIXES_20251028.md`
- `QUICK_FIX_REFERENCE.md`
- `FINAL_CHECKLIST_SUMMARY.md`

### Phase 2: 图结构确认（已完成 ✅）

**确认项**:
1. ✅ 模型生成**分子图**（节点+边）
2. ✅ 通过`mol_from_graphs`转换为**Mol对象**
3. ✅ 使用`correct_mol`进行**价态修正**
4. ✅ 保存为**pkl文件**

**详细文档**:
- `docs/GRAPH_TO_MOLECULE_PIPELINE.md`
- `VISUALIZATION_GUIDE.md`

### Phase 3: 可视化工具（已完成 ✅）

创建的工具：

1. **convert_predictions_to_smiles.py**
   - pkl → SMILES字符串（TSV）
   - Canonical格式，无立体化学
   - 符合论文要求

2. **visualize_predictions.py**
   - pkl → 分子结构图（PNG）
   - 网格对比图
   - 详细信息表格

3. **debug_checkpoint.py**
   - Checkpoint结构验证
   - 维度检查

4. **validate_setup.py**
   - 完整配置验证
   - 数据格式检查

---

## 🚀 使用指南

### 步骤1: 运行推理

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal

# Modal云端推理
modal run diffms_inference.py \
    --data-subdir msg_official_test5 \
    --max-count 5

# 等待完成，会生成pkl文件
# 输出: modal_inference_rank_0_pred_0.pkl
```

**输出**: pkl文件（包含Mol对象）

### 步骤2: 转换为SMILES（关键！）

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal

# 将pkl文件转换为TSV格式的SMILES
python convert_predictions_to_smiles.py
```

**输出**:
- `results_smiles/predictions_top1.tsv` - Top-1预测
- `results_smiles/predictions_all_candidates.tsv` - 所有候选

**格式**:
```tsv
spec_id         smiles
spec_0000      CCO
spec_0001      CC(C)O
```

### 步骤3: 生成可视化图片

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal

# 生成分子结构图
python visualize_predictions.py
```

**输出**:
```
visualizations/
├── predictions_summary.tsv     # 详细信息
├── top1_comparison.png         # Top-1对比
└── spectrum_grids/             # 每个谱图的网格图
    ├── spectrum_0000_grid.png
    ├── spectrum_0001_grid.png
    └── ...
```

### 步骤4: 验证结果

```python
import pandas as pd
from rdkit import Chem

# 读取SMILES文件
df = pd.read_csv('results_smiles/predictions_top1.tsv', sep='\t')

# 验证每个SMILES
for idx, row in df.iterrows():
    if not pd.isna(row['smiles']) and row['smiles'] != '':
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            print(f"✗ 无效SMILES (第{idx}行): {row['smiles']}")
        else:
            print(f"✓ 有效SMILES: {row['smiles']}")
```

---

## 🔍 深入理解：从图到分子

### 图结构示例

```python
# 输入：乙醇分子的图表示
nodes = [0, 0, 2]           # [C, C, O]

adjacency_matrix = [
    [0, 1, 0],              # C-C 单键
    [1, 0, 1],              # C-O 单键
    [0, 1, 0]
]
```

### 转换过程

```python
# 1. 创建Mol对象
mol = Chem.RWMol()

# 2. 添加原子
mol.AddAtom(Chem.Atom('C'))  # 原子0
mol.AddAtom(Chem.Atom('C'))  # 原子1
mol.AddAtom(Chem.Atom('O'))  # 原子2

# 3. 添加键
mol.AddBond(0, 1, Chem.rdchem.BondType.SINGLE)  # C-C
mol.AddBond(1, 2, Chem.rdchem.BondType.SINGLE)  # C-O

# 4. 转换为不可编辑
mol = mol.GetMol()

# 5. 转换为SMILES
smiles = Chem.MolToSmiles(mol)  # "CCO"
```

### 验证图结构

```python
# 查看原子信息
for atom in mol.GetAtoms():
    print(f"原子 {atom.GetIdx()}: {atom.GetSymbol()}")

# 查看键信息
for bond in mol.GetBonds():
    begin = bond.GetBeginAtom().GetSymbol()
    end = bond.GetEndAtom().GetSymbol()
    bond_type = bond.GetBondType()
    print(f"键: {begin}-{end} ({bond_type})")
```

**输出**:
```
原子 0: C
原子 1: C
原子 2: O
键: C-C (SINGLE)
键: C-O (SINGLE)
```

---

## 📊 数据流图

```
┌──────────────────────────────────────┐
│ 1. 质谱输入                           │
│    - MS数据 (.ms文件)                │
│    - 分子式 (labels.tsv)             │
└──────────┬───────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│ 2. DiffMS模型推理                    │
│    - Encoder: MS → 特征向量         │
│    - Diffusion: 生成分子图          │
└──────────┬───────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│ 3. 图表示 (Graph)                    │
│    X: [batch, n] 节点类型           │
│    E: [batch, n, n] 邻接矩阵        │
└──────────┬───────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│ 4. mol_from_graphs()                 │
│    图 → RDKit Mol对象               │
└──────────┬───────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│ 5. correct_mol()                     │
│    价态修正                          │
└──────────┬───────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│ 6. pkl文件                           │
│    List[List[Mol对象]]              │
└──────────┬───────────────────────────┘
           ↓
     ┌─────┴──────┐
     ↓            ↓
┌─────────┐  ┌─────────┐
│ SMILES  │  │ 结构图  │
│ (TSV)   │  │ (PNG)   │
└─────────┘  └─────────┘
```

---

## 🛠️ 工具清单

### 验证工具
```bash
# Checkpoint验证
python modal/debug_checkpoint.py

# 完整设置验证
python modal/validate_setup.py
```

### 转换工具
```bash
# pkl → SMILES (TSV)
python modal/convert_predictions_to_smiles.py

# pkl → 结构图 (PNG)
python modal/visualize_predictions.py
```

### 查看pkl文件
```python
import pickle
from rdkit import Chem

# 读取
with open('pred.pkl', 'rb') as f:
    predictions = pickle.load(f)

# 查看结构
print(f"类型: {type(predictions)}")
print(f"谱图数: {len(predictions)}")
print(f"第一个谱图的候选数: {len(predictions[0])}")

# 获取Mol对象
mol = predictions[0][0]
print(f"SMILES: {Chem.MolToSmiles(mol)}")
```

---

## 📚 文档索引

### 核心文档
1. **INFERENCE_CHECKLIST_FIXES_20251028.md**
   - 7个检查点的详细修正
   - 配置修正说明
   - 验证结果

2. **GRAPH_TO_MOLECULE_PIPELINE.md**
   - 图结构详细说明
   - 转换流程代码
   - 可视化示例

3. **VISUALIZATION_GUIDE.md**
   - 完整可视化指南
   - 工具使用说明
   - 示例代码

### 快速参考
1. **QUICK_FIX_REFERENCE.md**
   - 快速修正清单
   - 常见错误
   - 核心要点

2. **FINAL_CHECKLIST_SUMMARY.md**
   - 所有检查点总结
   - 修正内容对照
   - 验证清单

3. **COMPLETE_WORKFLOW_SUMMARY.md** (本文档)
   - 完整工作流程
   - 使用指南
   - 工具清单

---

## ✅ 最终验证清单

### 配置修正 ✅
- [x] Checkpoint包含encoder和decoder权重
- [x] `cfg.general.test_only = True` (布尔值)
- [x] `cfg.general.decoder = None`
- [x] `cfg.general.encoder = None`
- [x] Formula字段格式正确
- [x] 路径配置正确
- [x] 版本兼容

### 图结构确认 ✅
- [x] 模型输出是图结构（X + E）
- [x] `mol_from_graphs`转换正确
- [x] `correct_mol`价态修正
- [x] pkl文件包含Mol对象

### 可视化工具 ✅
- [x] SMILES转换工具
- [x] 结构图生成工具
- [x] 验证工具
- [x] 详细文档

### 输出格式 ✅
- [x] TSV格式（spec_id, smiles）
- [x] SMILES是字符串
- [x] Canonical格式
- [x] 无立体化学
- [x] 可被RDKit解析

---

## 🎯 总结

### 核心成果

1. **✅ 完成7个检查点的修正**
   - Checkpoint配置正确
   - 维度匹配
   - 加载验证

2. **✅ 确认图结构转换流程**
   - 图 → Mol对象
   - 价态修正
   - pkl格式

3. **✅ 创建完整工具链**
   - SMILES转换
   - 结构可视化
   - 验证脚本

4. **✅ 提供详细文档**
   - 使用指南
   - 代码示例
   - 故障排除

### 关键要点

1. **pkl文件不是最终输出**
   - 必须转换为SMILES字符串
   - 使用 `convert_predictions_to_smiles.py`

2. **输出必须是canonical SMILES**
   - 移除立体化学
   - Canonical格式
   - 可被RDKit验证

3. **完整的可视化工具链**
   - pkl → TSV (SMILES)
   - pkl → PNG (结构图)
   - 详细信息表格

---

**状态**: ✅ 所有工作已完成  
**可以开始**: ✅ 生产环境推理  
**工具就绪**: ✅ 完整的工具链和文档

🎉 **准备就绪！**

