# 🎯 关键修复：每步绕过Mask

**时间**: 2025-10-28  
**问题**: NO_EDGE设置成功但被mask破坏  
**解决**: 在每一步采样时都绕过mask操作

---

## 💥 发现的问题

### 现象

```
Edge type counts: SINGLE=174, DOUBLE=35, NO_EDGE=492  ← NO_EDGE设置成功
Mol SMILES: F123=c456...  ← 但分子完全错误（应该是C和O，却有F）
```

NO_EDGE确实被设置了，但生成的分子完全错误！

### 根本原因

在`sample_p_zs_given_zt_with_scaffold`函数中（第1523-1532行）：

```python
# 1. POST-SAMPLING HOOK设置NO_EDGE (第1475-1518行)
E_s[0:33, 0:33] = NO_EDGE  # 设置NO_EDGE ✅
E_s[36个真实边] = 真实类型  # 设置真实边 ✅

# 2. 创建PlaceHolder
out_discrete = utils.PlaceHolder(X=X_s, E=E_s, ...)

# 3. 返回时调用mask（问题！）
return out_discrete.mask(node_mask, collapse=True)  ❌
```

**`mask(collapse=True)`操作会破坏我们精心设置的边信息！**

---

## 🔍 问题分析

### 流程

```
每一步 t → t-1:
  1. HOOK 3: 设置NO_EDGE + 骨架边 (概率层面)
  2. 采样: sample_discrete_features(prob_X, prob_E)
  3. POST-SAMPLING: 再次设置NO_EDGE + 骨架边 (采样后)
  4. mask(collapse=True)  ← 破坏了第3步的设置！❌
  5. 返回到主循环
```

### 为什么会破坏

`mask(collapse=True)`会：
- 重新整理节点和边
- 可能改变边的索引
- 可能改变边的类型
- **我们之前发现它会把DOUBLE变成SINGLE或TRIPLE**

### 证据

**第一次运行（14:58）**：
- 设置NO_EDGE前没有绕过per-step mask
- 结果：edge counts完全混乱

**第二次运行（15:05）**：
- 设置了NO_EDGE（HOOK 3 + POST-SAMPLING）
- 但每步仍调用mask
- 结果：NO_EDGE=492出现了，但分子仍然错误

---

## 🔧 解决方案

### 修改点：第1526-1532行

```python
# CRITICAL: Bypass mask when scaffold is enforced
if scaffold_mol is not None and scaffold_indices is not None:
    # Don't apply mask - return raw tensors
    return out_one_hot.type_as(y_t), out_discrete.type_as(y_t)
else:
    # Original path: apply mask
    return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)
```

### 原理

**绕过mask的原因**：
1. 我们在HOOK 3中已经正确设置了prob_X和prob_E
2. 采样后的X_s和E_s已经是正确的
3. POST-SAMPLING HOOK又确保了骨架部分正确
4. **不需要mask来"整理"，mask只会破坏**

**为什么可以绕过**：
- 原始模型用mask是为了处理不同大小的图
- 但我们的骨架约束场景下，图的大小是固定的（target formula决定）
- 而且我们精心设置了每个位置的值
- mask反而是多余且有害的

---

## 📊 修复层级

### 第1层：HOOK 3（概率层面）

```python
# 第1414-1451行
# 设置NO_EDGE for scaffold internal
prob_E[0:33, 0:33] = NO_EDGE
# 然后设置真实边
prob_E[36个边] = 真实类型
```

### 第2层：POST-SAMPLING（采样后强制替换）

```python
# 第1475-1518行
# 设置NO_EDGE for scaffold internal
E_s[0:33, 0:33] = NO_EDGE
# 然后设置真实边
E_s[36个边] = 真实类型
```

### 第3层：绕过mask（新增！）

```python
# 第1526-1532行
# 不调用mask，直接返回
if scaffold_mol is not None:
    return out_one_hot.type_as(y_t), out_discrete.type_as(y_t)
```

---

## ✅ 完整的保护机制

### 每一步（t → t-1）

```
1. 模型预测 → pred_X, pred_E

2. HOOK 1: 化学式掩码
   pred_X[non_scaffold] = apply_formula_mask(...)

3. 计算后验概率 → prob_X, prob_E

4. HOOK 3A: 冻结原子
   prob_X[0:33] = one_hot([C,C,C,...])

5. HOOK 3B: 清空骨架区域
   prob_E[0:33, 0:33] = NO_EDGE

6. HOOK 3C: 设置真实边
   prob_E[36个边] = 真实类型

7. 采样
   X_s, E_s = sample(prob_X, prob_E)

8. POST-SAMPLING HOOK:
   E_s[0:33, 0:33] = NO_EDGE
   E_s[36个边] = 真实类型

9. 绕过mask（新增！）
   直接返回 (X_s, E_s)，不调用mask

10. 主循环更新
    X, E = X_s, E_s
```

---

## 🎯 预期效果

运行后应该看到：

```bash
Edge type counts: SINGLE=35, DOUBLE=1, NO_EDGE=492  ✅
First 10 node types: ['C', 'C', 'C', 'C', 'C', 'O', 'C', ...]  ✅

Generated: CC(C)=CCCC(C(=O)O)C1CCC2(C)C3=C(CCC12C)C1(C)CCC(O)C(C)(C)C1CC3...
Scaffold:  CC(C)=CCCC(C(=O)O)C1CCC2(C)C3=C(CCC12C)C1(C)CCC(O)C(C)(C)C1CC3...
✅ Generated molecule CONTAINS scaffold!
```

---

## 📝 修改的文件

- **`DiffMS/src/diffusion_model_spec2mol.py`**:
  - 第1414-1421行：HOOK 3B - 先设置NO_EDGE
  - 第1475-1483行：POST-SAMPLING - 先设置NO_EDGE
  - **第1526-1532行：绕过每步的mask（关键！）**

---

## 💡 核心洞察

**以前的错误理解**：
- mask是必须的，用于整理图结构 ❌

**现在的正确理解**：
- mask对普通生成是有用的 ✅
- 但对骨架约束生成，mask是有害的 ❌
- 因为我们已经精心设置了每个位置的值
- mask会破坏这些精心设置的值

**三层防护**：
1. HOOK 3：概率层面防护
2. POST-SAMPLING：采样后防护
3. **绕过mask：最终防护（新增）**

---

## 🚀 立即测试

```bash
modal run /Users/aylin/yaolab_projects/diffms_yaolab/modal/diffms_scaffold_inference.py
```

**这次真的应该能成功了！** 🎉

