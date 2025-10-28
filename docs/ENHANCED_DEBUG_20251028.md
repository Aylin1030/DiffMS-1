# 🔍 增强调试 - 边冻结验证

**时间**: 2025-10-28  
**目标**: 精确诊断骨架约束失败的原因

---

## 📝 已添加的调试

### 1. 初始化验证（第959-997行）

**检查**: 边是否正确初始化

**输出**:
```python
[DEBUG] Verifying edge initialization:
  Edge 0-1: type 0 (expected: 0) ✓
  Edge 1-2: type 1 (expected: 1) ✓
  Edge 2-3: type 0 (expected: 0) ✓
```

---

### 2. HOOK 3 执行确认（第1364行）

**检查**: 边冻结是否执行

**输出**:
```python
[HOOK 3] Frozen 33 atoms, 36 bonds at t=0.800
```

---

### 3. 采样过程统计（第1020-1074行）

**检查**: 采样中边是否保持

**输出（成功）**:
```python
[DEBUG] Step 400: Checking scaffold preservation...
  ✓ All 33 atoms match
  ✓ All 36 edges match
```

**输出（失败）**:
```python
[DEBUG] Step 400: Checking scaffold preservation...
  ✗ 5/33 atoms mismatch!
  ✗ Edge 0-1: type 2 != 0
  ✗ 10/36 edges mismatch!
```

---

### 4. 最终状态详细验证（第1077-1125行）

**检查**: 扩散结束时X和E的状态

**输出**:
```python
[DEBUG] After diffusion loop, final verification:
  Node 0: C (expected: C) ✓
  Node 1: C (expected: C) ✓
  Edge 0-1: type 0 (expected: 0) ✓
  Edge 1-2: type 1 (expected: 1) ✓
  Edge 2-3: type 0 (expected: 0) ✓
```

**如果有问题**:
```python
[CRITICAL] Scaffold not preserved! Atoms: 0 mismatch, Edges: 5 mismatch
```

---

## 🚀 运行

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_scaffold_inference.py
```

---

## 🎯 4种可能的失败模式

### 模式 1: 初始化失败 ❌

**日志特征**:
```
[DEBUG] Verifying edge initialization:
  Edge 0-1: type 4 (expected: 0) ✗
```

**原因**: 边类型映射错误或E张量格式问题

---

### 模式 2: HOOK 3 未执行 ❌

**日志特征**:
```
[DEBUG] Step 400: ...
  ✗ 36/36 edges mismatch!
```
**没有看到**: `[HOOK 3] Frozen ... bonds`

**原因**: 条件判断问题，`scaffold_mol` 或 `scaffold_indices` 为空

---

### 模式 3: 采样中丢失 ❌

**日志特征**:
```
[DEBUG] Verifying edge initialization:
  Edge 0-1: type 0 (expected: 0) ✓  ← 初始化OK

[HOOK 3] Frozen 33 atoms, 36 bonds ...  ← HOOK 3 OK

[DEBUG] Step 400: ...
  ✗ 20/36 edges mismatch!  ← 但采样中丢失了
```

**原因**: 
- `prob_E` 冻结后被破坏
- E 更新有问题
- 边采样逻辑错误

---

### 模式 4: 转换时丢失 ❌

**日志特征**:
```
[DEBUG] After diffusion loop:
  Edge 0-1: type 0 (expected: 0) ✓  ← E正确

[DEBUG] Generated mol: C.CCCC1OC...  ← 分子错误
WARNING - does not contain scaffold
```

**原因**: `mol_from_graphs(X, E)` 转换有问题

---

## 📊 预期的成功日志

```
========================================
初始化
========================================
[DEBUG] Initializing scaffold atoms and bonds:
  Node 0: set to C (idx=0)
  ...
  Bond 0-1: SINGLE (idx=0)
  Bond 1-2: DOUBLE (idx=1)
  ...
  Total: 33 atoms, 36 bonds initialized

[DEBUG] Verifying edge initialization:
  Edge 0-1: type 0 (expected: 0) ✓
  Edge 1-2: type 1 (expected: 1) ✓
  Edge 2-3: type 0 (expected: 0) ✓

[DEBUG] Verifying atom initialization:
  Node 0: C (expected: C) ✓
  ...

========================================
采样过程
========================================
[HOOK 3] Frozen 33 atoms, 36 bonds at t=0.800

[DEBUG] Step 400: Checking scaffold preservation...
  ✓ All 33 atoms match
  ✓ All 36 edges match

[DEBUG] Step 300: ...
  ✓ All 33 atoms match
  ✓ All 36 edges match

========================================
最终验证
========================================
[DEBUG] After diffusion loop, final verification:
  Node 0: C (expected: C) ✓
  Node 1: C (expected: C) ✓
  Node 2: C (expected: C) ✓
  Edge 0-1: type 0 (expected: 0) ✓
  Edge 1-2: type 1 (expected: 1) ✓
  Edge 2-3: type 0 (expected: 0) ✓
  Edge 3-4: type 0 (expected: 0) ✓
  Edge 4-5: type 0 (expected: 0) ✓

[DEBUG] Generated mol: CC(C)=CCCC(C(=O)O)C1CCC2...
[DEBUG] Scaffold: CC(C)=CCCC(C(=O)O)C1CCC2...
✅ Generated molecule CONTAINS scaffold!
```

---

## ⚡ 下一步

1. **运行** Modal 脚本
2. **观察** 新的调试日志
3. **识别** 失败模式（1-4中的哪一种）
4. **报告** 日志中的关键信息

---

**关键问题**：

- [ ] 初始化时边是否正确？
- [ ] HOOK 3 是否执行？
- [ ] 采样时边是否保持？
- [ ] 最终 E 是否正确？
- [ ] 转换是否有问题？

**现在运行并告诉我看到了什么！** 🎯

