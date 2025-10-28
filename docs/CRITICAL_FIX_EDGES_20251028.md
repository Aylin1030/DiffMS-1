# 🔧 关键修复：边（Bonds）冻结

**日期**: 2025-10-28  
**问题**: 骨架100%失败  
**根本原因**: 只冻结了节点（原子类型），未冻结边（键）

---

## ❌ 问题分析

### 调试日志显示

✅ **节点类型（X）完全正确**:
```
[DEBUG] After diffusion loop, verifying X:
  Node 0: C (expected: C) ✓
  Node 1: C (expected: C) ✓
  Node 2: C (expected: C) ✓
```

❌ **但生成的分子完全错误**:
```
[DEBUG] Generated mol: [H]C1=CCCC(CC([H])C)CC2C=c(C)c2O[PH](=O)OCC2C3=CC2...
[DEBUG] Scaffold: CC(C)=CCCC(C(=O)O)C1CCC2(C)C3=C(CCC12C)C1(C)CCC(O)C(C)(C)C1CC3...
WARNING - Generated molecule does not contain scaffold. Discarding.
```

---

## 🔍 根本原因

**分子 = 原子 + 键**

我们之前只保证了：
- ✅ 前33个节点的类型是正确的（C, C, C, ..., O）
- ❌ 但这些节点之间的连接（键）是随机生成的

结果：
- 有正确的原子（33个C和3个O）
- 但连接方式完全错误
- → 完全不同的分子！

### 类比

就像拼图：
- ❌ **之前**：拥有正确的33块拼图，但随机拼接 → 完全错误的图案
- ✅ **现在**：拥有正确的33块拼图，并按正确方式拼接 → 正确的图案

---

## ✅ 修复方案

### 1. 初始化时冻结边（第907-968行）

**之前**:
```python
# 只初始化节点类型X
X[:, local_idx, atom_type_idx] = 1
```

**现在**:
```python
# A. 初始化节点类型X
X[:, local_idx, atom_type_idx] = 1

# B. 初始化边类型E（新增）
for bond in scaffold_mol.GetBonds():
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    
    # 获取键类型（单键、双键等）
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        edge_type_idx = 0
    elif bond_type == Chem.BondType.DOUBLE:
        edge_type_idx = 1
    # ... 其他类型
    
    # 设置边（对称）
    E[:, i, j, edge_type_idx] = 1
    E[:, j, i, edge_type_idx] = 1
```

---

### 2. 采样时冻结边（第1156-1211行）

**之前** (HOOK 3):
```python
# 只冻结节点概率
prob_X[:, local_idx, atom_type_idx] = 1
```

**现在** (HOOK 3):
```python
# A. 冻结节点概率
prob_X[:, local_idx, atom_type_idx] = 1

# B. 冻结边概率（新增）
for bond in scaffold_mol.GetBonds():
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    
    # 获取键类型
    edge_type_idx = get_bond_type_idx(bond)
    
    # 冻结边概率（对称）
    prob_E[:, i, j, :] = 0
    prob_E[:, i, j, edge_type_idx] = 1
    prob_E[:, j, i, :] = 0
    prob_E[:, j, i, edge_type_idx] = 1
```

---

### 3. 调试验证边

添加了边的检查（第1002-1031行）:

```python
# 检查前几个键是否保持
for bond in scaffold_mol.GetBonds():
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    
    edge_types = E[0, i, j, :]
    predicted_edge = torch.argmax(edge_types).item()
    expected_edge = get_bond_type_idx(bond)
    
    if predicted_edge != expected_edge:
        logging.warning(f"  Edge {i}-{j}: type {predicted_edge} != {expected_edge} ✗")
```

---

## 📊 键类型映射

DiffMS模型的边类型编码：

| RDKit 键类型 | 索引 | 说明 |
|-------------|------|------|
| `SINGLE` | 0 | 单键 (C-C) |
| `DOUBLE` | 1 | 双键 (C=C) |
| `TRIPLE` | 2 | 三键 (C≡C) |
| `AROMATIC` | 3 | 芳香键 (苯环) |

---

## 🔄 完整的骨架冻结流程

### 初始化阶段
```
1. 加载骨架 SMILES
2. 设置节点类型 X[0:33] = scaffold atoms
3. 设置边类型 E[i,j] = scaffold bonds ← 新增
4. 验证初始化正确性
```

### 扩散反演阶段（每一步）
```
for t in [500, 499, ..., 1]:
    1. 模型预测 logits
    2. 计算 prob_X, prob_E
    3. HOOK 3A: 冻结 prob_X (节点)
    4. HOOK 3B: 冻结 prob_E (边) ← 新增
    5. 采样下一状态
    6. 更新 X, E ← 关键！
```

### 后处理阶段
```
1. 将 (X, E) 转换为 RDKit 分子
2. VF2 子图同构检查
3. 验证骨架是否完整
```

---

## 🎯 预期改进

### 之前
- 节点类型正确：✅
- 边类型正确：❌
- 骨架匹配率：0%

### 现在
- 节点类型正确：✅
- 边类型正确：✅
- 骨架匹配率：预期 >80%

---

## 🚀 下一步

运行新版本：
```bash
modal run diffms_scaffold_inference.py
```

**关注新的调试日志**：
```
[DEBUG] Initializing scaffold atoms and bonds:
  Node 0: set to C (idx=0)
  Bond 0-1: SINGLE (idx=0)
  Bond 1-2: DOUBLE (idx=1)  ← 新增！
  Total: 33 atoms, 34 bonds initialized  ← 新增！

[DEBUG] Step 400: Checking scaffold preservation...
  Edge 0-1: type 0 = 0 ✓  ← 新增！边也保持正确
  Edge 1-2: type 1 = 1 ✓

[DEBUG] Generated mol: CC(C)=CCCC(C(=O)O)C1CCC2...  ← 应该与骨架相似
✅ Generated molecule CONTAINS scaffold!  ← 成功！
```

---

## 📝 技术细节

### 边矩阵的对称性

由于分子图是无向的，边矩阵必须对称：
```python
E[i, j, bond_type] == E[j, i, bond_type]
```

因此冻结时需要设置两个方向：
```python
E[:, i, j, edge_type_idx] = 1
E[:, j, i, edge_type_idx] = 1  # 对称
```

### 为什么之前没有发现

之前的调试只检查了X（节点），而默认假设E（边）是正确的。但实际上：
- 模型在每一步都会重新预测E
- 如果不冻结E，骨架的键会被随机修改
- 最终导致完全不同的分子

---

## ✨ 关键洞察

**分子图 = (V, E)**
- V (Vertices) = 节点/原子
- E (Edges) = 边/键

**两者缺一不可！**

只冻结V而不冻结E，就像：
- 有正确的字母，但顺序错误 → "HELLO" vs "OLLEH"
- 有正确的音符，但节奏错误 → 不同的旋律

---

**这次修复应该能显著提高骨架匹配率！** 🎉

