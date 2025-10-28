# 🔍 立即调试

## 问题

**骨架冻结100%失败**，所有分子都不包含骨架。

## 已添加的调试

✅ 已在代码中添加了详细的调试日志：

1. **初始化检查** (第909-930行)
2. **采样过程检查** (第952-962行) 
3. **最终状态检查** (第965-972行)
4. **生成分子信息** (第996-1010行)
5. **HOOK 3 执行** (第1156-1178行)

---

## 立即运行

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_scaffold_inference.py 2>&1 | tee debug.log
```

---

## 观察日志

### 关键日志点

1. **初始化**（应该在采样开始前）:
```
[DEBUG] Initializing scaffold atoms in X:
[DEBUG] Verifying scaffold initialization:
```

2. **采样过程**（每100步）:
```
[DEBUG] Step 400: Checking scaffold preservation...
```

3. **最终状态**（采样结束后）:
```
[DEBUG] After diffusion loop, verifying X:
```

4. **生成分子**（每个样本）:
```
[DEBUG] Generated mol: ...
[DEBUG] Scaffold: ...
```

---

## 三种可能的失败模式

### 模式 1: 初始化就错误

**日志**:
```
[DEBUG] Verifying scaffold initialization:
  Node 0: O (expected: C) ✗
```

**原因**: X 初始化有问题，骨架原子没有被正确设置

---

### 模式 2: 采样过程中丢失

**日志**:
```
[DEBUG] Verifying scaffold initialization:
  Node 0: C (expected: C) ✓  ← 初始化正确
  
[DEBUG] Step 400: Checking scaffold preservation...
  Node 0: O != C ✗  ← 中途丢失！
```

**原因**: HOOK 3 没有生效，或 X 更新有问题

---

### 模式 3: 转换时丢失

**日志**:
```
[DEBUG] After diffusion loop, verifying X:
  Node 0: C (expected: C) ✓  ← X 是正确的
  
[DEBUG] Generated mol: CC(C)O...  ← 分子不包含骨架！
WARNING - Generated molecule does not contain scaffold
```

**原因**: `mol_from_graphs` 转换有问题，或 `contains_scaffold` 检查过严

---

## 快速分析

运行后，执行：

```bash
# 检查初始化
grep "Verifying scaffold initialization" debug.log -A 5

# 检查最终状态  
grep "After diffusion loop" debug.log -A 5

# 检查生成的分子
grep "Generated mol:" debug.log

# 统计失败
grep "does not contain scaffold" debug.log | wc -l
```

---

## 预期成功的日志

```
[DEBUG] Initializing scaffold atoms in X:
  Node 0: set to C (idx=0)
  Node 1: set to C (idx=0)
  Node 2: set to C (idx=0)
  
[DEBUG] Verifying scaffold initialization:
  Node 0: C (expected: C) ✓
  Node 1: C (expected: C) ✓
  Node 2: set to C (expected: C) ✓
  
[DEBUG] After diffusion loop, verifying X:
  Node 0: C (expected: C) ✓
  Node 1: C (expected: C) ✓
  Node 2: C (expected: C) ✓
  
[DEBUG] Generated mol: CC(=CCCC(C1CCC2(C1(CCC3=C2CCC4C3(CCC(C4(C)C)O)C)C)C)C(=O)O)C...
[DEBUG] Scaffold: CC(=CCCC(C1CCC2(C1(CCC3=C2CCC4C3(CCC(C4(C)C)O)C)C)C)C(=O)O)C
[DEBUG] Generated has 35 atoms, scaffold has 33 atoms
✅ Generated molecule CONTAINS scaffold!
```

---

## 现在运行！

```bash
modal run diffms_scaffold_inference.py
```

**观察第一个样本的调试日志，找出失败模式！**

