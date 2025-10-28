# 🚀 转换调试 - 立即运行

**目标**: 找出为什么X和E正确，但转换后的分子不包含骨架

---

## ⚡ 运行命令

```bash
modal run /Users/aylin/yaolab_projects/diffms_yaolab/modal/diffms_scaffold_inference.py
```

---

## 📊 预期看到的新日志

### 1. 转换前（X和E的状态）

```
[DEBUG] Converting graph #0 to molecule:
  nodes.shape = torch.Size([?, 8])
  adj_mat.shape = torch.Size([?, ?, 5])
  First 10 node types: ['C', 'C', 'C', ...]  ← 应该全是C
  Edge type counts: SINGLE=?, DOUBLE=?, ...  ← 关键！
```

**应该是**:
- nodes.shape = `[33~40, 8]`（33是骨架，可能有额外原子）
- First 10 node types = `['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']`（至少前10个是C）
- Edge counts: `SINGLE=35, DOUBLE=1`（骨架有36个键）

---

### 2. mol_from_graphs 之后（转换结果）

```
[DEBUG] After mol_from_graphs (before valence correction):
  Mol has ? atoms
  Mol SMILES: ?
```

**应该是**:
- Mol has `33~40` atoms（至少包含骨架的33个）
- Mol SMILES 开头类似 `CC(C)=CCCC...`

---

### 3. 价态校正之后

```
[DEBUG] After valence correction:
  Corrected has ? atoms
  Corrected SMILES: ?
```

**检查**:
- 原子数是否改变？
- SMILES是否改变？

---

## 🎯 4种诊断结果

### 情况 A: 边统计就不对

```
Edge type counts: SINGLE=5, DOUBLE=3, ...  ← 完全不对！
```

**原因**: `mol_from_graphs` 不理解我们的edge格式，或者mask操作破坏了边

**解决**: 检查边的格式和mask逻辑

---

### 情况 B: 节点类型不对

```
First 10 node types: ['O', 'N', 'C', 'O', ...]  ← 顺序混乱
```

**原因**: mask或collapse改变了节点顺序

**解决**: 在mask前后检查节点顺序

---

### 情况 C: mol_from_graphs 转换错误

```
After mol_from_graphs:
  Mol has 38 atoms  ← 对
  Mol SMILES: C.CCCC1OC...  ← 但SMILES完全不对
```

**原因**: `mol_from_graphs` 的转换逻辑有问题

**解决**: 需要看 `mol_from_graphs` 的实现

---

### 情况 D: 价态校正破坏骨架

```
After mol_from_graphs:
  Mol SMILES: CC(C)=CCCC...  ← 对！包含骨架

After valence correction:
  Corrected SMILES: C.Cc1c...  ← 完全变了！
```

**原因**: `correct_mol` 修改了结构

**解决**: 禁用价态校正或保护骨架部分

---

## 📋 运行检查清单

运行后，记录以下数据：

- [ ] 转换前边统计: `SINGLE=?, DOUBLE=?, NO_EDGE=?`
- [ ] 转换前节点类型（前10个）: `[?, ?, ...]`
- [ ] mol_from_graphs后原子数: `?`
- [ ] mol_from_graphs后SMILES（前20字符）: `?`
- [ ] 价态校正后原子数: `?`
- [ ] 价态校正后SMILES（前20字符）: `?`
- [ ] 最终骨架检查结果: `包含 / 不包含`

---

## 🔍 快速分析

填入您看到的数据：

```
转换前:
  边: SINGLE=__, DOUBLE=__, NO_EDGE=__
  节点: [__, __, __, ...]

转换后(校正前):
  原子: __
  SMILES: __________

转换后(校正后):
  原子: __
  SMILES: __________
  
骨架检查: __________
```

---

**现在运行并填写上面的数据！** 🎯

