# 🔍 边冻结调试

**目标**: 查看边在采样过程中是否被正确保持

---

## 🚀 立即运行

```bash
modal run /Users/aylin/yaolab_projects/diffms_yaolab/modal/diffms_scaffold_inference.py
```

---

## 📊 新增的调试日志

### 1. 初始化阶段

**之前** (只有这些):
```
Total: 33 atoms, 36 bonds initialized
[DEBUG] Verifying scaffold initialization:
  Node 0: C (expected: C) ✓
```

**现在** (新增边验证):
```
Total: 33 atoms, 36 bonds initialized
[DEBUG] Verifying edge initialization:        ← 新增
  Edge 0-1: type 0 (expected: 0) ✓           ← 新增
  Edge 1-2: type 1 (expected: 1) ✓           ← 新增
  Edge 2-3: type 0 (expected: 0) ✓           ← 新增
[DEBUG] Verifying atom initialization:
  Node 0: C (expected: C) ✓
```

---

### 2. 采样过程（每100步）

**现在会显示**:
```
[HOOK 3] Frozen 33 atoms, 36 bonds at t=0.800  ← 新增：HOOK 3执行信息

[DEBUG] Step 400: Checking scaffold preservation...
  ✓ All 33 atoms match                          ← 新增：原子统计
  ✓ All 36 edges match                          ← 新增：边统计
```

**或者如果有问题**:
```
[HOOK 3] Frozen 33 atoms, 36 bonds at t=0.800

[DEBUG] Step 400: Checking scaffold preservation...
  ✗ 5/33 atoms mismatch!                        ← 警告
  ✗ Edge 0-1: type 2 != 0                       ← 具体哪个边错了
  ✗ Edge 1-2: type 0 != 1
  ✗ 10/36 edges mismatch!                       ← 警告
```

---

### 3. 最终验证

**现在会显示**:
```
[DEBUG] After diffusion loop, final verification:
  Node 0: C (expected: C) ✓
  Node 1: C (expected: C) ✓
  Edge 0-1: type 0 (expected: 0) ✓             ← 新增：边验证
  Edge 1-2: type 1 (expected: 1) ✓
  Edge 2-3: type 0 (expected: 0) ✓
```

**或者如果有问题**:
```
[DEBUG] After diffusion loop, final verification:
  Node 0: C (expected: C) ✓
  Edge 0-1: type 2 (expected: 0) ✗             ← 边错了
  Edge 1-2: type 0 (expected: 1) ✗
[CRITICAL] Scaffold not preserved! Atoms: 0 mismatch, Edges: 2 mismatch  ← 严重警告
```

---

## 🎯 诊断场景

### 场景 1: 初始化就错了

**日志**:
```
[DEBUG] Verifying edge initialization:
  Edge 0-1: type 4 (expected: 0) ✗  ← 初始化时边就错了
```

**原因**: 边类型映射有问题，或者E张量的维度/格式不对

**解决**: 检查 `E.shape` 和键类型映射

---

### 场景 2: HOOK 3没有执行

**日志**:
```
[DEBUG] Step 400: Checking scaffold preservation...
  ✓ All 33 atoms match
  ✗ 36/36 edges mismatch!  ← 所有边都错了
```

**没有看到**:
```
[HOOK 3] Frozen ... bonds  ← 这个日志不存在
```

**原因**: HOOK 3 的边冻结代码没有执行

**解决**: 检查 `scaffold_mol` 和 `scaffold_indices` 是否正确传递

---

### 场景 3: 边在采样中丢失

**日志**:
```
[DEBUG] Verifying edge initialization:
  Edge 0-1: type 0 (expected: 0) ✓  ← 初始化正确

[HOOK 3] Frozen 33 atoms, 36 bonds at t=0.800  ← HOOK 3执行了

[DEBUG] Step 400: Checking scaffold preservation...
  ✗ 20/36 edges mismatch!  ← 但边还是错了！
```

**原因**: HOOK 3 冻结了 `prob_E`，但可能：
1. `prob_E` 在冻结后被归一化破坏
2. 边的采样逻辑有问题
3. E 没有被正确更新

**解决**: 检查 `prob_E` 归一化和 E 的更新逻辑

---

### 场景 4: 转换时丢失

**日志**:
```
[DEBUG] After diffusion loop, final verification:
  Edge 0-1: type 0 (expected: 0) ✓  ← E是对的
  Edge 1-2: type 1 (expected: 1) ✓

[DEBUG] Generated mol: C.CCCC1OC(=O)C=CC...  ← 但生成的分子不对
WARNING - Generated molecule does not contain scaffold.
```

**原因**: `mol_from_graphs(X, E)` 转换有问题

**解决**: 检查 `visualization_tools.mol_from_graphs` 的实现

---

## 📋 检查清单

运行后，观察日志并回答：

- [ ] **初始化**: 边是否正确初始化？
  - 看 `[DEBUG] Verifying edge initialization:`
  - 前3个边应该都是 ✓

- [ ] **HOOK 3**: 是否执行了边冻结？
  - 看 `[HOOK 3] Frozen ... bonds`
  - 应该显示36 bonds

- [ ] **采样过程**: 边是否被保持？
  - 看 `[DEBUG] Step 400: ...`
  - 应该显示 `✓ All 36 edges match`

- [ ] **最终状态**: E 是否正确？
  - 看 `[DEBUG] After diffusion loop:`
  - 前5个边应该都是 ✓

- [ ] **转换**: 分子是否包含骨架？
  - 看 `Generated molecule does not contain scaffold`
  - 应该看到一些成功的

---

## 🔧 可能的修复

### 如果边初始化就错了

```python
# 检查edge_type_idx的计算
logging.info(f"Edge {i}-{j}: bond_type={bond_type}, edge_idx={edge_type_idx}")
logging.info(f"E[0,{i},{j},:] before = {E[0,i,j,:]}")
E[:, i, j, :] = 0
E[:, i, j, edge_type_idx] = 1
logging.info(f"E[0,{i},{j},:] after = {E[0,i,j,:]}")
```

### 如果HOOK 3没执行

```python
# 在HOOK 3开始处添加
logging.info(f"[HOOK 3 DEBUG] scaffold_mol={scaffold_mol is not None}, scaffold_indices={scaffold_indices}")
logging.info(f"[HOOK 3 DEBUG] prob_E.shape={prob_E.shape}")
```

### 如果边在采样中丢失

```python
# 在HOOK 3冻结边之后添加
logging.info(f"[HOOK 3 DEBUG] prob_E[0,0,1,:] after freeze = {prob_E[0,0,1,:]}")
# 应该看到类似 [1, 0, 0, 0, 0] 的 one-hot
```

---

**现在运行并查看新的调试日志！**

