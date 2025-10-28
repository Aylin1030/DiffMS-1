# DiffMS预测结果可视化指南

**更新日期**: 2024-10-28  
**目标**: 完整的图结构 → 分子对象 → 可视化流程说明

---

## 🎯 核心确认

### ✅ 模型输出是图结构

DiffMS模型生成的是**离散的分子图**，包含：

```python
# 输出格式（来自sample_batch方法）
X: torch.Tensor          # 节点类型（原子类型）
    shape: [batch_size, max_nodes]
    values: 0-7 对应 [C, N, O, F, P, S, Cl, Br]
    
E: torch.Tensor          # 边类型（键类型）
    shape: [batch_size, max_nodes, max_nodes]  
    values: 0-4 对应 [无键, 单键, 双键, 三键, 芳香键]
```

**代码位置**: `DiffMS/src/diffusion_model_spec2mol.py:664-710`

---

## 🔄 完整转换流程

### 步骤1: 扩散模型生成图

```python
# diffusion_model_spec2mol.py: sample_batch()
def sample_batch(self, data):
    # ... 扩散采样过程 ...
    
    # 返回采样的图结构
    X: Tensor  # 节点（原子）
    E: Tensor  # 边（键）
    
    return sampled_graph
```

### 步骤2: 图 → RDKit Mol对象

```python
# analysis/visualization.py: mol_from_graphs()
def mol_from_graphs(self, node_list, adjacency_matrix):
    """
    将图结构转换为RDKit分子对象
    
    流程:
    1. 创建空的RWMol对象
    2. 根据node_list添加原子（使用atom_decoder映射）
    3. 根据adjacency_matrix添加键
    4. 转换为不可编辑的Mol对象
    """
    mol = Chem.RWMol()
    
    # 添加原子
    for node_type in node_list:
        atom = Chem.Atom(atom_decoder[node_type])
        mol.AddAtom(atom)
    
    # 添加键
    for i, j in edges:
        bond_type = bond_decoder[adjacency_matrix[i, j]]
        mol.AddBond(i, j, bond_type)
    
    return mol.GetMol()
```

### 步骤3: 价态修正

```python
# diffusion_model_spec2mol.py:692-708
from analysis.rdkit_functions import correct_mol

if mol is not None:
    editable_mol = Chem.RWMol(mol)
    corrected_mol, no_correct = correct_mol(editable_mol)
    if corrected_mol is not None:
        mol = corrected_mol
```

### 步骤4: 保存为pkl文件

```python
# diffusion_model_spec2mol.py:424-426
with open(f"preds/{self.name}_rank_{rank}_pred_{i}.pkl", "wb") as f:
    pickle.dump(predicted_mols, f)

# predicted_mols的结构：
# List[List[Mol]] - [谱图索引][候选排名] -> Mol对象
```

---

## 🛠️ 可视化工具

### 工具1: convert_predictions_to_smiles.py

**功能**: pkl文件 → SMILES字符串（TSV格式）

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
python convert_predictions_to_smiles.py
```

**输出**:
- `results_smiles/predictions_top1.tsv` - Top-1预测
- `results_smiles/predictions_all_candidates.tsv` - 所有候选

**输出格式**:
```tsv
spec_id         smiles
spec_0000      CCO
spec_0001      CC(C)O
```

### 工具2: visualize_predictions.py

**功能**: pkl文件 → 分子结构图（PNG格式）

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
python visualize_predictions.py
```

**输出**:
```
visualizations/
├── predictions_summary.tsv         # 详细信息表格
├── top1_comparison.png            # Top-1候选对比图
└── spectrum_grids/                # 每个谱图的网格图
    ├── spectrum_0000_grid.png
    ├── spectrum_0001_grid.png
    └── ...
```

**摘要表格格式**:
```tsv
spec_id  rank  valid  smiles              formula   mol_weight  num_atoms  num_bonds
spec_0000  1   True   CCCC(C)CCc1cccc...  C37O7     556.4       44         50
spec_0000  2   False  ...                                                  
```

---

## 📊 可视化示例

### 1. 查看单个分子

```python
import pickle
from rdkit import Chem
from rdkit.Chem import Draw

# 读取pkl文件
with open('modal_inference_rank_0_pred_0.pkl', 'rb') as f:
    predictions = pickle.load(f)

# 获取第一个谱图的第一个候选
mol = predictions[0][0]

# 显示分子信息
print(f"SMILES: {Chem.MolToSmiles(mol)}")
print(f"分子式: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
print(f"分子量: {Chem.Descriptors.MolWt(mol):.2f}")

# 绘制2D结构
img = Draw.MolToImage(mol, size=(400, 400))
img.save('molecule.png')
```

### 2. 对比多个候选

```python
from rdkit.Chem import Draw

# 获取一个谱图的所有候选
candidates = predictions[0][:10]  # 前10个
valid_mols = [mol for mol in candidates if mol is not None]

# 绘制网格图
img = Draw.MolsToGridImage(
    valid_mols,
    molsPerRow=5,
    subImgSize=(300, 300),
    legends=[f"Rank {i+1}" for i in range(len(valid_mols))]
)

img.save('candidates_grid.png')
```

### 3. 查看图结构信息

```python
mol = predictions[0][0]

print("=== 原子信息 ===")
for atom in mol.GetAtoms():
    print(f"原子 {atom.GetIdx()}: {atom.GetSymbol()} "
          f"(度: {atom.GetDegree()}, 价态: {atom.GetTotalValence()})")

print("\n=== 键信息 ===")
for bond in mol.GetBonds():
    begin = bond.GetBeginAtom()
    end = bond.GetEndAtom()
    print(f"键 {bond.GetIdx()}: "
          f"{begin.GetSymbol()}({begin.GetIdx()})-"
          f"{end.GetSymbol()}({end.GetIdx()}) "
          f"[{bond.GetBondType()}]")
```

**输出示例**:
```
=== 原子信息 ===
原子 0: C (度: 4, 价态: 4)
原子 1: C (度: 4, 价态: 4)
原子 2: O (度: 2, 价态: 2)

=== 键信息 ===
键 0: C(0)-C(1) [SINGLE]
键 1: C(1)-O(2) [SINGLE]
```

---

## 🎨 高级可视化选项

### 选项1: 添加原子编号

```python
from rdkit.Chem import Draw

mol = predictions[0][0]

# 显示原子索引
for atom in mol.GetAtoms():
    atom.SetProp('atomLabel', str(atom.GetIdx()))

img = Draw.MolToImage(mol, size=(400, 400))
img.save('molecule_with_indices.png')
```

### 选项2: 高亮特定原子/键

```python
from rdkit.Chem import Draw

mol = predictions[0][0]

# 高亮特定原子（例如氧原子）
highlight_atoms = [atom.GetIdx() for atom in mol.GetAtoms() 
                   if atom.GetSymbol() == 'O']

img = Draw.MolToImage(
    mol, 
    size=(400, 400),
    highlightAtoms=highlight_atoms
)
img.save('molecule_highlight_oxygen.png')
```

### 选项3: 3D构象

```python
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

mol = predictions[0][0]

# 生成3D构象
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)

# 保存3D SDF文件
writer = Chem.SDWriter('molecule_3d.sdf')
writer.write(mol)
writer.close()

# 可以用PyMOL、Avogadro等工具打开SDF文件
```

---

## 📋 完整工作流程

### 推理 + 可视化完整流程

```bash
# 步骤1: 运行推理（生成pkl文件）
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_inference.py --data-subdir msg_official_test5

# 步骤2: 转换为SMILES（生成TSV文件）
python convert_predictions_to_smiles.py

# 步骤3: 生成可视化图片
python visualize_predictions.py

# 步骤4: 查看结果
# - TSV文件: results_smiles/predictions_top1.tsv
# - 图片: visualizations/top1_comparison.png
# - 网格图: visualizations/spectrum_grids/
```

---

## 🔍 验证结果

### 检查Mol对象有效性

```python
import pickle
from rdkit import Chem

with open('pred.pkl', 'rb') as f:
    predictions = pickle.load(f)

print("=== 有效性检查 ===")
for spec_idx, mol_list in enumerate(predictions):
    valid_count = 0
    for mol in mol_list:
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol)
                # 反向验证
                test_mol = Chem.MolFromSmiles(smiles)
                if test_mol is not None:
                    valid_count += 1
            except:
                pass
    
    print(f"谱图 {spec_idx}: {valid_count}/{len(mol_list)} 个有效分子")
```

### 检查图结构完整性

```python
mol = predictions[0][0]

# 检查价态
from rdkit.Chem import Descriptors

print("=== 价态检查 ===")
for atom in mol.GetAtoms():
    valence = atom.GetTotalValence()
    expected = atom.GetValence()
    status = "✓" if valence == expected else "✗"
    print(f"{status} 原子 {atom.GetIdx()} ({atom.GetSymbol()}): "
          f"价态={valence}, 期望={expected}")

# 检查连通性
print(f"\n分子片段数: {len(Chem.GetMolFrags(mol))}")
if len(Chem.GetMolFrags(mol)) > 1:
    print("⚠ 警告：分子不连通！")
```

---

## 📊 数据流总结图

```
┌─────────────────────────────────────────┐
│ 输入: Mass Spectra + Formula            │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ DiffMS扩散模型                           │
│ - Encoder: MS → 特征                    │
│ - Diffusion: 生成分子图                 │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 图表示 (Graph)                          │
│ - X: [batch, n] 节点类型               │
│ - E: [batch, n, n] 邻接矩阵            │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ mol_from_graphs() 转换                  │
│ - 创建RWMol对象                         │
│ - 添加原子和键                          │
│ - 转换为Mol对象                         │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ correct_mol() 价态修正                  │
│ - 检查价态                              │
│ - 调整氢原子                            │
│ - 修正键阶                              │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ pkl文件: List[List[Mol]]               │
│ - [谱图索引][候选排名] → Mol对象       │
└────────────┬────────────────────────────┘
             ↓
     ┌───────┴────────┐
     ↓                ↓
┌──────────┐   ┌──────────────┐
│ SMILES   │   │ 2D结构图     │
│ (TSV)    │   │ (PNG)        │
└──────────┘   └──────────────┘
```

---

## 📁 相关文件

### 核心代码
- `DiffMS/src/diffusion_model_spec2mol.py` - 扩散模型（图生成）
- `DiffMS/src/analysis/visualization.py` - 图→Mol转换
- `DiffMS/src/analysis/rdkit_functions.py` - 价态修正

### 工具脚本
- `modal/convert_predictions_to_smiles.py` - pkl → SMILES
- `modal/visualize_predictions.py` - pkl → 图片
- `modal/debug_checkpoint.py` - Checkpoint验证

### 文档
- `docs/GRAPH_TO_MOLECULE_PIPELINE.md` - 详细流程说明
- `docs/INFERENCE_CHECKLIST_FIXES_20251028.md` - 检查清单
- `QUICK_FIX_REFERENCE.md` - 快速参考

---

## ✅ 核心确认总结

| 确认项 | 状态 | 说明 |
|--------|------|------|
| 模型输出是图结构 | ✅ | X (节点) + E (邻接矩阵) |
| 图→Mol转换 | ✅ | `mol_from_graphs()` 函数 |
| 价态修正 | ✅ | `correct_mol()` 函数 |
| pkl文件格式 | ✅ | List[List[Mol对象]] |
| SMILES转换 | ✅ | `Chem.MolToSmiles()` |
| 2D结构可视化 | ✅ | `Draw.MolToImage()` |
| 网格对比图 | ✅ | `Draw.MolsToGridImage()` |
| 图结构验证 | ✅ | 原子/键信息检查 |

---

**结论**: 
1. ✅ DiffMS输出**离散分子图**（节点+边）
2. ✅ 通过`mol_from_graphs`转换为**RDKit Mol对象**
3. ✅ 使用`correct_mol`进行**价态修正**
4. ✅ 保存为**pkl文件**（包含Mol对象）
5. ✅ 提供**完整的可视化工具链**

**所有确认完成！** 🎉

