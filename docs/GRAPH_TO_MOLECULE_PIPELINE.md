# DiffMS: 图生成到分子可视化完整流程

**日期**: 2024-10-28  
**目标**: 详细说明从扩散模型生成的图结构到最终可视化分子的完整管道

---

## 📊 完整流程概览

```
质谱输入 (MS) + 分子式 (Formula)
    ↓
Encoder: 编码为特征向量
    ↓
Diffusion Model: 生成分子图
    ↓
图表示: [节点类型, 邻接矩阵]
    ↓
RDKit转换: 图 → Mol对象
    ↓
价态修正: correct_mol()
    ↓
最终输出: SMILES / 可视化
```

---

## 1️⃣ 模型输出：分子图 (Graph)

### 图结构定义

模型生成的是**离散的分子图**，包含：

```python
# 来自 sample_batch() 的输出
X: torch.Tensor  # 节点类型 (原子类型)
    shape: [batch_size, max_nodes]
    values: 0-7 (8种原子类型)
    
E: torch.Tensor  # 边类型 (键类型)  
    shape: [batch_size, max_nodes, max_nodes]
    values: 0-4 (5种键类型)
```

### 原子类型映射

**代码位置**: `diffusion_model_spec2mol.py:684-687`

```python
mols = []
for nodes, adj_mat in zip(X, E):
    mol = self.visualization_tools.mol_from_graphs(nodes, adj_mat)
```

**数据集信息** (dataset_infos):
```python
atom_decoder = {
    0: 'C',   # 碳
    1: 'N',   # 氮
    2: 'O',   # 氧
    3: 'F',   # 氟
    4: 'P',   # 磷
    5: 'S',   # 硫
    6: 'Cl',  # 氯
    7: 'Br',  # 溴
}
```

### 键类型映射

**代码位置**: `analysis/visualization.py:42-51`

```python
bond_type_mapping = {
    0: None,           # 无键
    1: SINGLE,         # 单键
    2: DOUBLE,         # 双键
    3: TRIPLE,         # 三键
    4: AROMATIC,       # 芳香键
}
```

---

## 2️⃣ 图 → RDKit Mol对象转换

### 转换函数

**代码位置**: `analysis/visualization.py:16-59`

```python
def mol_from_graphs(self, node_list, adjacency_matrix):
    """
    将图结构转换为RDKit分子对象
    
    参数:
        node_list: 节点类型列表 (长度 n)
        adjacency_matrix: 邻接矩阵 (n × n)
    
    返回:
        rdkit.Chem.Mol: 分子对象
    """
    atom_decoder = self.dataset_infos.atom_decoder
    
    # 1. 创建空的可编辑分子对象
    mol = Chem.RWMol()
    
    # 2. 添加原子
    node_to_idx = {}
    for i in range(len(node_list)):
        if node_list[i] == -1:  # 跳过填充节点
            continue
        a = Chem.Atom(atom_decoder[int(node_list[i])])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx
    
    # 3. 添加键
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            if iy <= ix:  # 只遍历上三角矩阵
                continue
            
            # 映射键类型
            if bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
            elif bond == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
            elif bond == 4:
                bond_type = Chem.rdchem.BondType.AROMATIC
            else:
                continue  # 无键
            
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
    
    # 4. 转换为不可编辑的Mol对象
    try:
        mol = mol.GetMol()
    except rdkit.Chem.KekulizeException:
        print("Can't kekulize molecule")
        mol = None
    
    return mol
```

### 转换示例

**输入（图）**:
```python
nodes = [0, 0, 2]           # [C, C, O]
adj_matrix = [
    [0, 1, 0],              # C-C (单键)
    [1, 0, 1],              # C-O (单键)  
    [0, 1, 0]
]
```

**输出（Mol对象）**:
```python
mol = Chem.MolFromSmiles('CCO')  # 乙醇
```

---

## 3️⃣ 价态修正 (Valence Correction)

### 修正函数

**代码位置**: `diffusion_model_spec2mol.py:692-708`

```python
# 关键修复：应用价态修正（与论文一致）
if mol is not None:
    from rdkit import Chem
    from analysis.rdkit_functions import correct_mol
    try:
        # 转换为RWMol（可编辑）
        editable_mol = Chem.RWMol(mol)
        corrected_mol, no_correct = correct_mol(editable_mol)
        if corrected_mol is not None:
            mol = corrected_mol
        # 如果correct_mol返回None，保留原分子
    except Exception as e:
        # 修正失败，保留原分子
        import logging
        logging.debug(f"Molecule correction failed: {e}")
```

### correct_mol 函数详解

**代码位置**: `analysis/rdkit_functions.py`

该函数执行以下修正：
1. **价态检查**: 检查每个原子的价态是否合法
2. **氢原子调整**: 自动添加或移除隐式氢
3. **键阶修正**: 调整键的类型以满足价态要求
4. **芳香性处理**: 正确处理芳香环

---

## 4️⃣ 输出格式

### 4.1 pkl文件（中间结果）

**保存位置**: `preds/{model_name}_rank_{rank}_pred_{batch_id}.pkl`

**内容**:
```python
predicted_mols = [
    [mol1_candidate1, mol1_candidate2, ..., mol1_candidate10],  # 谱图1的10个候选
    [mol2_candidate1, mol2_candidate2, ..., mol2_candidate10],  # 谱图2的10个候选
    ...
]

# 每个mol是rdkit.Chem.Mol对象
type(predicted_mols[0][0])  # <class 'rdkit.Chem.rdchem.Mol'>
```

**读取示例**:
```python
import pickle
from rdkit import Chem

with open('modal_inference_rank_0_pred_0.pkl', 'rb') as f:
    predictions = pickle.load(f)

# 获取第一个谱图的第一个候选
mol = predictions[0][0]

# 转换为SMILES
if mol is not None:
    smiles = Chem.MolToSmiles(mol)
    print(f"SMILES: {smiles}")
    
    # 获取分子信息
    print(f"原子数: {mol.GetNumAtoms()}")
    print(f"键数: {mol.GetNumBonds()}")
    print(f"分子式: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
```

### 4.2 TSV文件（最终结果）

**生成方式**: 使用 `convert_predictions_to_smiles.py`

```bash
python modal/convert_predictions_to_smiles.py
```

**输出文件**:

**predictions_top1.tsv**:
```tsv
spec_id                 smiles
spec_0000              CCO
spec_0001              CC(C)O
spec_0002              CCCC
```

**predictions_all_candidates.tsv**:
```tsv
spec_id                 rank    smiles
spec_0000              1       CCO
spec_0000              2       CC(O)C
spec_0000              3       C(C)O
spec_0001              1       CC(C)O
```

---

## 5️⃣ 可视化方法

### 方法1: 使用RDKit绘制2D结构

```python
from rdkit import Chem
from rdkit.Chem import Draw
import pickle

# 读取pkl文件
with open('modal_inference_rank_0_pred_0.pkl', 'rb') as f:
    predictions = pickle.load(f)

# 可视化第一个谱图的所有候选
mols = predictions[0]
valid_mols = [mol for mol in mols if mol is not None]

# 绘制网格图
img = Draw.MolsToGridImage(
    valid_mols[:10],  # 最多10个
    molsPerRow=5,
    subImgSize=(200, 200),
    legends=[f"Rank {i+1}" for i in range(len(valid_mols[:10]))]
)

# 保存图片
img.save('molecules_grid.png')
```

**输出**: `molecules_grid.png` - 网格排列的分子结构图

### 方法2: 使用自带的可视化工具

```python
from src.analysis.visualization import MolecularVisualization

# 创建可视化工具
vis = MolecularVisualization(
    remove_h=True,
    dataset_infos=dataset_infos
)

# 可视化分子
vis.visualize(
    path='output_images',
    molecules=molecules,  # [(nodes, adj_matrix), ...]
    num_molecules_to_visualize=10
)
```

**输出**: `output_images/molecule_0.png`, `molecule_1.png`, ...

### 方法3: 使用我们创建的可视化工具

**代码位置**: `visualization/`

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/visualization

# 生成所有分子图片
python generate_images.py

# 启动Web查看器
python app.py
```

**访问**: http://localhost:8501

**功能**:
- 查看所有预测的分子结构
- 对比不同rank的候选
- 显示SMILES、分子式等信息

---

## 6️⃣ 完整示例代码

### 从pkl到可视化的完整流程

```python
import pickle
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import pandas as pd
from pathlib import Path

class MoleculeVisualizer:
    """从pkl文件到完整可视化的工具类"""
    
    def __init__(self, pkl_file: Path):
        self.pkl_file = pkl_file
        with open(pkl_file, 'rb') as f:
            self.predictions = pickle.load(f)
    
    def extract_smiles(self, canonical=True, remove_stereo=True):
        """提取所有SMILES"""
        results = []
        
        for spec_idx, mol_list in enumerate(self.predictions):
            for rank, mol in enumerate(mol_list, start=1):
                if mol is None:
                    continue
                
                try:
                    if remove_stereo:
                        Chem.RemoveStereochemistry(mol)
                    
                    smiles = Chem.MolToSmiles(mol, canonical=canonical)
                    
                    results.append({
                        'spec_id': f'spec_{spec_idx:04d}',
                        'rank': rank,
                        'smiles': smiles,
                        'mol': mol
                    })
                except Exception as e:
                    print(f"Error for spec {spec_idx}, rank {rank}: {e}")
        
        return pd.DataFrame(results)
    
    def save_tsv(self, output_dir: Path):
        """保存为TSV格式"""
        df = self.extract_smiles()
        
        # Top-1预测
        top1 = df[df['rank'] == 1][['spec_id', 'smiles']]
        top1.to_csv(output_dir / 'predictions_top1.tsv', sep='\t', index=False)
        
        # 所有候选
        all_candidates = df[['spec_id', 'rank', 'smiles']]
        all_candidates.to_csv(output_dir / 'predictions_all_candidates.tsv', sep='\t', index=False)
        
        print(f"✓ 保存到 {output_dir}")
    
    def visualize_grid(self, spec_idx: int, output_file: Path, max_mols: int = 10):
        """可视化单个谱图的所有候选"""
        if spec_idx >= len(self.predictions):
            raise ValueError(f"spec_idx {spec_idx} 超出范围")
        
        mols = self.predictions[spec_idx]
        valid_mols = [mol for mol in mols[:max_mols] if mol is not None]
        
        if not valid_mols:
            print(f"谱图 {spec_idx} 没有有效分子")
            return
        
        # 生成SMILES标签
        legends = []
        for i, mol in enumerate(valid_mols, 1):
            smiles = Chem.MolToSmiles(mol, canonical=True)
            legends.append(f"Rank {i}\n{smiles[:30]}...")
        
        # 绘制网格
        img = Draw.MolsToGridImage(
            valid_mols,
            molsPerRow=5,
            subImgSize=(300, 300),
            legends=legends
        )
        
        img.save(output_file)
        print(f"✓ 保存到 {output_file}")
    
    def visualize_all(self, output_dir: Path):
        """可视化所有谱图"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for spec_idx in range(len(self.predictions)):
            output_file = output_dir / f'spectrum_{spec_idx:04d}.png'
            try:
                self.visualize_grid(spec_idx, output_file)
            except Exception as e:
                print(f"Error visualizing spectrum {spec_idx}: {e}")

# 使用示例
if __name__ == "__main__":
    # 1. 读取pkl文件
    pkl_file = Path('modal/modal_inference_rank_0_pred_0.pkl')
    visualizer = MoleculeVisualizer(pkl_file)
    
    # 2. 保存为TSV
    visualizer.save_tsv(Path('results_smiles'))
    
    # 3. 可视化单个谱图
    visualizer.visualize_grid(
        spec_idx=0,
        output_file=Path('spectrum_0_candidates.png')
    )
    
    # 4. 可视化所有谱图
    visualizer.visualize_all(Path('molecule_images'))
```

---

## 7️⃣ 数据流总结

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 扩散模型输出                                          │
├─────────────────────────────────────────────────────────────┤
│ X: [batch, n] - 节点类型 (0-7: C,N,O,F,P,S,Cl,Br)          │
│ E: [batch, n, n] - 邻接矩阵 (0-4: 无,单,双,三,芳香)         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 图 → RDKit Mol                                       │
├─────────────────────────────────────────────────────────────┤
│ visualization_tools.mol_from_graphs(nodes, adj_mat)          │
│   → Chem.RWMol() 创建                                        │
│   → AddAtom() 添加原子                                       │
│   → AddBond() 添加键                                         │
│   → GetMol() 转换为Mol                                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 价态修正                                             │
├─────────────────────────────────────────────────────────────┤
│ correct_mol(editable_mol)                                    │
│   → 检查价态                                                 │
│   → 调整氢原子                                               │
│   → 修正键阶                                                 │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 保存pkl文件                                          │
├─────────────────────────────────────────────────────────────┤
│ pickle.dump(predicted_mols, file)                            │
│ 内容: List[List[Chem.Mol]] - [谱图][候选][Mol对象]          │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: 转换为SMILES (可读格式)                              │
├─────────────────────────────────────────────────────────────┤
│ convert_predictions_to_smiles.py                             │
│   → Chem.RemoveStereochemistry()                             │
│   → Chem.MolToSmiles(mol, canonical=True)                    │
│   → 保存为TSV文件                                            │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: 可视化                                               │
├─────────────────────────────────────────────────────────────┤
│ Draw.MolToImage(mol) - 2D结构图                              │
│ Draw.MolsToGridImage() - 网格图                              │
│ Streamlit Web App - 交互式查看                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 验证图结构的正确性

### 检查节点和边

```python
import pickle
from rdkit import Chem

with open('pred.pkl', 'rb') as f:
    predictions = pickle.load(f)

mol = predictions[0][0]  # 第一个谱图的第一个候选

if mol is not None:
    print("=== 分子信息 ===")
    print(f"SMILES: {Chem.MolToSmiles(mol)}")
    print(f"分子式: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    print(f"分子量: {Chem.rdMolDescriptors.CalcExactMolWt(mol):.2f}")
    
    print("\n=== 原子信息 ===")
    for atom in mol.GetAtoms():
        print(f"原子 {atom.GetIdx()}: {atom.GetSymbol()} "
              f"(价态: {atom.GetTotalValence()})")
    
    print("\n=== 键信息 ===")
    for bond in mol.GetBonds():
        print(f"键 {bond.GetIdx()}: "
              f"{bond.GetBeginAtom().GetSymbol()}-"
              f"{bond.GetEndAtom().GetSymbol()} "
              f"({bond.GetBondType()})")
```

---

## 📚 参考文档

- RDKit文档: https://www.rdkit.org/docs/
- 图神经网络: PyTorch Geometric
- 扩散模型: DDPM for discrete data

---

**总结**:
1. ✓ 模型输出是**离散图结构**（节点类型+邻接矩阵）
2. ✓ 通过`mol_from_graphs`转换为**RDKit Mol对象**
3. ✓ 使用`correct_mol`进行**价态修正**
4. ✓ 保存为**pkl文件**（Mol对象列表）
5. ✓ 转换为**SMILES字符串**（TSV文件）
6. ✓ 使用**RDKit Draw**或**Streamlit**可视化

**完整工具链已就绪！** 🎉

