# 🔍 发现根本问题：mask操作丢失双键

**时间**: 2025-10-28  
**状态**: 🔴 双键在mask后丢失

---

## 💥 关键发现

从日志第787行看到：
```
Edge type counts: SINGLE=703, DOUBLE=0, TRIPLE=0, AROMATIC=0, NO_EDGE=0
```

### ❌ 错误的边统计

**应该是**:
- SINGLE = 35 (骨架的35个单键)
- DOUBLE = 1 (骨架的1个双键：C=C)
- NO_EDGE = 大量 (非连接的节点对)

**实际是**:
- SINGLE = 703 (所有节点对！)
- DOUBLE = 0 ❌ **双键丢失**
- NO_EDGE = 0 ❌ **全部变成了键**

---

## 🔬 问题定位

### 1. 初始化时双键是对的

```
Bond 1-2: DOUBLE (idx=1)  ← 第741行，初始化正确
```

### 2. 采样过程中也保持了

```
[HOOK 3] Frozen 33 atoms, 36 bonds  ← 第752行，HOOK 3执行
[DEBUG] Step 400: All 36 edges match  ← 第755行，中间检查OK
```

### 3. 最终验证还是对的

```
Edge 1-2: type 1 (expected: 1) ✓  ← 第779行，最后验证OK
```

### 4. mask之后丢失

```python
# 第1142行：关键操作
sampled_s = sampled_s.mask(node_mask, collapse=True)
X, E, y = sampled_s.X, sampled_s.E, data.y
```

**之后**:
```
Edge type counts: SINGLE=703, DOUBLE=0  ← 双键消失！
```

---

## 💡 根本原因假设

### 假设：`mask(collapse=True)` 的 bug

`collapse=True` 可能：
1. 把 E 从 `[batch, n, n, edge_types]` 压缩成 `[n, n]`
2. 压缩时使用了错误的策略（如：只保留type 0，或者所有非NO_EDGE都变成SINGLE）
3. 丢失了双键信息

---

## 🔧 已添加的调试

### 1. mask前检查（第1127-1139行）

```python
[DEBUG] Before mask:
  X.shape = ?
  E.shape = ?
  First few edges: ['0-1:type0', '1-2:type1', ...]  ← 应该看到type1(双键)
```

### 2. mask后检查（第1145-1164行）

```python
[DEBUG] After mask (collapse=True):
  X.shape = ?
  E.shape = ?
  E is 2D/3D?
  Edges after mask: [...]  ← 看双键还在不在
```

### 3. 修复了统计bug（第1184-1194行）

之前的统计代码假设adj_mat是3D的，现在兼容2D和3D：

```python
if len(adj_mat.shape) == 2:  # 2D: [n, n]
    edge_type = adj_mat[i, j].item()  # 直接取值
elif len(adj_mat.shape) == 3:  # 3D: [n, n, edge_types]
    edge_type = torch.argmax(adj_mat[i, j, :]).item()
```

---

## 🎯 预期看到的新日志

### 情况 A: mask前就错了（不太可能）

```
[DEBUG] Before mask:
  First few edges: ['0-1:type0', '1-2:type0', ...]  ← 都是type0，双键已经丢失
```

→ 说明问题在采样循环中，而不是mask

---

### 情况 B: mask导致的（最可能）

```
[DEBUG] Before mask:
  E.shape = torch.Size([1, 38, 38, 5])
  First few edges: ['0-1:type0', '1-2:type1', ...]  ← type1存在！

[DEBUG] After mask:
  E.shape = torch.Size([38, 38])  ← 变成2D
  E is 2D - checking first few values:
    E[0,1] = 0  ← SINGLE
    E[1,2] = 0  ← 应该是1(DOUBLE)，但变成了0！
```

→ **mask操作丢失了双键**

---

### 情况 C: 只是统计bug（希望）

```
[DEBUG] After mask:
  E.shape = torch.Size([38, 38])
  E[0,1] = 0
  E[1,2] = 1  ← 双键还在！
  ...
Edge type counts: SINGLE=35, DOUBLE=1, NO_EDGE=665  ← 统计正确
```

→ 之前只是统计代码的bug，实际边是对的

---

## 🚀 下一步

1. **运行新版本**:
```bash
modal run /Users/aylin/yaolab_projects/diffms_yaolab/modal/diffms_scaffold_inference.py
```

2. **查看新日志**，重点关注：
   - `[DEBUG] Before mask: First few edges`
   - `[DEBUG] After mask: E is 2D/3D?`
   - `Edge type counts` (修复后的统计)

3. **根据结果决定**：
   - 如果是情况A → 检查采样循环
   - 如果是情况B → **绕过或修复mask**
   - 如果是情况C → 问题已解决！

---

## 📝 可能的修复方案

### 如果mask确实丢失双键

#### 方案1: 不用collapse

```python
# 当前（第1142行）
sampled_s = sampled_s.mask(node_mask, collapse=True)

# 改为
sampled_s = sampled_s.mask(node_mask, collapse=False)
# 然后手动处理batch维度
```

#### 方案2: mask后手动恢复双键

```python
sampled_s = sampled_s.mask(node_mask, collapse=True)
X, E, y = sampled_s.X, sampled_s.E, data.y

# 恢复骨架的边
for bond in scaffold_mol.GetBonds():
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    bond_type_idx = ...
    E[i, j] = bond_type_idx
    E[j, i] = bond_type_idx
```

#### 方案3: 完全绕过mask

```python
# 不用mask，直接从X和E构建分子
# （需要处理batch维度和padding）
```

---

**关键**: 这次调试会告诉我们双键到底是在mask时丢失的，还是根本没被正确设置！

