# 🔍 转换问题调试

**问题**: X和E都完美保持骨架，但转换为分子后骨架丢失

---

## 📊 问题现象

### ✅ 第1-3步：完美
```
[DEBUG] Verifying edge initialization:
  Edge 0-1: type 0 (expected: 0) ✓  ← 初始化正确
  Edge 1-2: type 1 (expected: 1) ✓

[HOOK 3] Frozen 33 atoms, 36 bonds  ← HOOK 3执行

[DEBUG] Step 400: Checking scaffold preservation...
  ✓ All 33 atoms match  ← 采样中保持
  ✓ All 36 edges match

[DEBUG] After diffusion loop, final verification:
  Node 0: C (expected: C) ✓  ← 最终状态正确
  Edge 0-1: type 0 (expected: 0) ✓
  Edge 1-2: type 1 (expected: 1) ✓
```

### ❌ 第4步：转换失败
```
[DEBUG] Generated mol: C.Cc1cCCCC2C3=CC45CC2fc3...  ← 完全不同！
[DEBUG] Scaffold: CC(C)=CCCC(C(=O)O)C1CCC2...
WARNING - Generated molecule does not contain scaffold
```

---

## 🔬 已添加的新调试

### 1. 转换前的X和E检查（第1134-1153行）

```python
[DEBUG] Converting graph #0 to molecule:
  nodes.shape = torch.Size([40, 8])      ← 40个节点，8种原子类型
  adj_mat.shape = torch.Size([40, 40, 5]) ← 40x40邻接矩阵，5种边类型
  First 10 node types: ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
  Edge type counts: SINGLE=35, DOUBLE=1, TRIPLE=0, AROMATIC=0, NO_EDGE=745
```

### 2. mol_from_graphs 之后（第1157-1166行）

```python
[DEBUG] After mol_from_graphs (before valence correction):
  Mol has 38 atoms  ← 看看是多少个
  Mol SMILES: C.Cc1cCCCC2...  ← 转换后的SMILES
```

### 3. 价态校正之后（第1178-1183行）

```python
[DEBUG] After valence correction:
  Corrected has 38 atoms
  Corrected SMILES: ...
```

---

## 🎯 可能的原因

### 假设 1: `mol_from_graphs` 不理解我们的edge格式

**问题**: 
- 我们设置 `E[i, j, edge_type] = 1`
- 但 `mol_from_graphs` 可能期待不同的格式

**测试方法**: 查看转换前后的边数量
- 输入: 36个bonds (SINGLE=35, DOUBLE=1)
- 输出: 如果完全不同，说明格式不对

---

### 假设 2: 节点顺序不一致

**问题**:
- 我们假设骨架原子0 → 节点0
- 但 `mol_from_graphs` 可能重新排序节点

**测试方法**: 检查转换后的前10个原子类型
- 如果全是C，但连接不对 → 顺序问题

---

### 假设 3: 价态校正破坏了骨架

**问题**:
- `correct_mol` 可能修改键或原子
- 破坏了骨架结构

**测试方法**: 比较校正前后
- 校正前包含骨架，校正后不包含 → 是这个问题

---

### 假设 4: node_mask 问题

**问题**:
- `sampled_s.mask(node_mask, collapse=True)` 可能改变节点顺序
- 把骨架节点移到了错误的位置

**测试方法**: 在mask之前和之后检查
- mask前骨架在0-32
- mask后骨架可能移位

---

## 🚀 下一步运行

```bash
modal run /Users/aylin/yaolab_projects/diffms_yaolab/modal/diffms_scaffold_inference.py
```

### 关注新日志

1. **转换前**:
```
[DEBUG] Converting graph #0 to molecule:
  First 10 node types: [?, ?, ...]  ← 应该是 ['C', 'C', 'C', ...]
  Edge type counts: SINGLE=?, DOUBLE=?  ← 应该是 35, 1
```

2. **转换后（价态校正前）**:
```
[DEBUG] After mol_from_graphs (before valence correction):
  Mol has ? atoms  ← 应该是33或更多
  Mol SMILES: ?  ← 看看是什么
```

3. **价态校正后**:
```
[DEBUG] After valence correction:
  Corrected has ? atoms
  Corrected SMILES: ?  ← 是否改变了？
```

---

## 🔧 可能的修复

### 如果是假设1（格式问题）

需要检查 `mol_from_graphs` 的实现，确保它理解我们的edge格式。

### 如果是假设2（顺序问题）

需要使用子图匹配来找到骨架在生成图中的正确位置，而不是假设位置0-32。

### 如果是假设3（价态校正问题）

需要在价态校正时保护骨架部分，或者禁用校正。

### 如果是假设4（mask问题）

需要在mask之前保存骨架信息，mask之后重新映射。

---

## 📝 关键问题

运行后请告诉我：

1. **转换前的边统计是什么？**
   - `Edge type counts: SINGLE=?, DOUBLE=?`
   - 应该是 35 和 1

2. **mol_from_graphs 之后有多少原子？**
   - `Mol has ? atoms`
   - 应该是 33 或更多

3. **转换后的SMILES前20个字符是什么？**
   - 与骨架SMILES `CC(C)=CCCC(C(=O)O)...` 比较

4. **价态校正改变了什么？**
   - 校正前后原子数和SMILES是否变化

---

**这些信息将帮助我们精确定位是哪一步出了问题！** 🎯

