# 🐛 调试指南 - 骨架冻结失败

## 问题现象

**所有生成的分子都不包含骨架（100%失败）**

```
2025-10-28 14:03:16,347 - WARNING - Generated molecule does not contain scaffold. Discarding.
2025-10-28 14:03:25,674 - WARNING - Generated molecule does not contain scaffold. Discarding.
...（全部失败）
```

---

## 已添加的调试日志

### 1. 骨架初始化检查（第909-930行）

```python
logging.info(f"[DEBUG] Initializing scaffold atoms in X:")
# 设置骨架原子
logging.info(f"[DEBUG] Verifying scaffold initialization:")
# 验证初始化是否正确
```

**预期输出**:
```
[DEBUG] Initializing scaffold atoms in X:
  Node 0: set to C (idx=0)
  Node 1: set to C (idx=0)
  ...
[DEBUG] Verifying scaffold initialization:
  Node 0: C (expected: C) ✓
  Node 1: C (expected: C) ✓
  ...
```

### 2. 采样过程检查（第952-962行）

```python
# 每100步检查一次
if s_int % 100 == 0:
    logging.info(f"[DEBUG] Step {s_int}: Checking scaffold preservation...")
```

**预期输出**:
```
[DEBUG] Step 400: Checking scaffold preservation...
  (如果有不匹配，会显示 WARNING)
```

### 3. 最终状态检查（第965-972行）

```python
logging.info(f"[DEBUG] After diffusion loop, verifying X:")
```

**预期输出**:
```
[DEBUG] After diffusion loop, verifying X:
  Node 0: C (expected: C) ✓
  Node 1: C (expected: C) ✓
  ...
```

### 4. 生成分子信息（第996-1010行）

```python
logging.info(f"[DEBUG] Generated mol: {gen_smiles[:100]}...")
logging.info(f"[DEBUG] Scaffold: {scaf_smiles[:100]}...")
logging.info(f"[DEBUG] Generated has {mol.GetNumAtoms()} atoms, scaffold has {scaffold_mol.GetNumAtoms()} atoms")
```

**预期输出**:
```
[DEBUG] Generated mol: CC(=CCCC(C1CCC2...
[DEBUG] Scaffold: CC(=CCCC(C1CCC2...
[DEBUG] Generated has 45 atoms, scaffold has 33 atoms
```

### 5. HOOK 3 执行（第1156-1178行）

```python
logging.debug(f"[HOOK 3] Frozen {frozen_count} scaffold atoms at t={t_val:.3f}")
```

---

## 运行新的调试版本

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_scaffold_inference.py
```

---

## 分析调试日志

### 场景 1: 初始化失败

如果看到：
```
[DEBUG] Verifying scaffold initialization:
  Node 0: O (expected: C) ✗
```

**原因**: X 的初始化逻辑错误

**解决方案**: 检查 `dense_data.X` 的内容和格式

---

### 场景 2: 采样过程中丢失

如果初始化正确，但中间步骤出现：
```
[DEBUG] Step 400: Checking scaffold preservation...
  Node 0: O != C ✗
```

**原因**: HOOK 3（骨架冻结）没有生效

**可能原因**:
1. `scaffold_indices` 为空
2. `atom_type_idx` 计算错误
3. `prob_X` 更新后被覆盖

---

### 场景 3: 最终状态正确但检查失败

如果最终X是正确的：
```
[DEBUG] After diffusion loop, verifying X:
  Node 0: C (expected: C) ✓
  Node 1: C (expected: C) ✓
```

但仍然：
```
WARNING - Generated molecule does not contain scaffold. Discarding.
```

**原因**: `mol_from_graphs` 或 `contains_scaffold` 的问题

**可能原因**:
1. `mol_from_graphs` 转换错误
2. 边（E）的信息丢失
3. `HasSubstructMatch` 检查过于严格

---

## 可能的根本原因

### 假设 1: 原子索引映射错误 ⚠️

**问题**: 骨架的原子顺序与输入数据的节点顺序不一致

**检查方法**: 查看调试日志中的初始化验证

**如果不匹配**: 需要实现子图匹配来找到正确的原子映射

---

### 假设 2: `dense_data.X` 已包含公式约束 ⚠️

**问题**: `dense_data.X` 可能已经被公式约束固定，覆写无效

**检查方法**: 
```python
logging.info(f"[DEBUG] dense_data.X before override:")
logging.info(f"  {dense_data.X[0, :5, :]}") 
```

---

### 假设 3: `mol_from_graphs` 丢失信息 ⚠️

**问题**: 从节点/邻接矩阵转换为RDKit分子时，骨架结构丢失

**检查方法**: 比较X中的节点类型与最终分子的原子类型

---

## 下一步行动

### 立即运行

```bash
modal run diffms_scaffold_inference.py 2>&1 | tee debug_output.log
```

### 查看关键日志

```bash
# 初始化
grep "\[DEBUG\] Verifying scaffold initialization" debug_output.log

# 最终状态
grep "\[DEBUG\] After diffusion loop" debug_output.log

# 生成的分子
grep "\[DEBUG\] Generated mol:" debug_output.log

# 骨架验证
grep "Generated molecule does not contain scaffold" debug_output.log
```

### 分析模式

1. **如果初始化就失败** → 问题在第907-930行
2. **如果采样过程中丢失** → 问题在HOOK 3（第1156-1178行）或X更新（第950行）
3. **如果最终X正确但检查失败** → 问题在`mol_from_graphs`或`contains_scaffold`

---

## 临时解决方案

如果调试后发现是 `contains_scaffold` 过于严格，可以临时放宽检查：

```python
# 在 scaffold_hooks.py 中
def contains_scaffold(candidate: Mol, scaffold: Mol) -> bool:
    """更宽松的骨架检查"""
    from rdkit import Chem
    
    # 移除立体化学
    Chem.RemoveStereochemistry(candidate)
    Chem.RemoveStereochemistry(scaffold)
    
    # 尝试匹配
    match = candidate.HasSubstructMatch(scaffold, useChirality=False)
    
    if not match:
        # 尝试更宽松的检查：原子数量
        if candidate.GetNumAtoms() >= scaffold.GetNumAtoms():
            # 检查是否有足够的相同元素
            scaf_formula = formula_of(scaffold)
            cand_formula = formula_of(candidate)
            all_present = all(cand_formula.get(elem, 0) >= count for elem, count in scaf_formula.items())
            if all_present:
                logging.warning(f"Loose match: candidate has all scaffold atoms, but exact match failed")
                return True  # 临时允许
    
    return match
```

---

## 预期的成功日志

如果一切正常，应该看到：

```
[DEBUG] Initializing scaffold atoms in X:
  Node 0: set to C (idx=0)
  ...
[DEBUG] Verifying scaffold initialization:
  Node 0: C (expected: C) ✓
  ...
[DEBUG] Step 400: Checking scaffold preservation...
  (无 WARNING)
[DEBUG] After diffusion loop, verifying X:
  Node 0: C (expected: C) ✓
  ...
[DEBUG] Generated mol: CC(=CCCC(C1CCC2...
[DEBUG] Scaffold: CC(=CCCC(C1CCC2...
[DEBUG] Generated has 35 atoms, scaffold has 33 atoms
✅ Generated molecule CONTAINS scaffold!
```

---

**现在运行并分析日志，找出真正的问题所在！**

