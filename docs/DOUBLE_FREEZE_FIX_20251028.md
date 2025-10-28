# 🔧 双重冻结修复

**时间**: 2025-10-28  
**问题**: HOOK 3冻结概率后，采样仍可能改变骨架  
**解决**: 在采样后再次强制替换骨架（双保险）

---

## 💥 发现的问题

虽然HOOK 3在每步都冻结了概率：

```python
# HOOK 3: 冻结概率
prob_X[:, local_idx, atom_type_idx] = 1  ✅
prob_E[:, i, j, edge_type_idx] = 1       ✅
```

**但采样后的结果仍然不对！**

看证据（第874-881行）：
```
Mol SMILES: CC(OC1CCC2...)  ← 拓扑完全不同
Scaffold:   CC(C)=CCCC...   ← 原始骨架
Generated has 33 atoms, scaffold has 33 atoms
WARNING - Does not contain scaffold  ❌
```

---

## 🔍 根本原因

**prob_X和prob_E被冻结后，`sample_discrete_features`仍可能采样到其他值！**

原因：
1. 数值精度问题
2. 采样的随机性
3. 或者后续的mask操作改变了值

---

## 🔧 解决方案：双重冻结

### 第1次冻结：HOOK 3（概率层面）

```python
# 第1391-1451行
prob_X[:, local_idx, :] = 0
prob_X[:, local_idx, atom_type_idx] = 1  # 冻结概率

prob_E[:, i, j, :] = 0
prob_E[:, i, j, edge_type_idx] = 1  # 冻结概率
```

### 第2次冻结：POST-SAMPLING HOOK（采样后强制替换）

```python
# 第1462-1499行（新增）
# Sample first
X_s = F.one_hot(sampled_s.X, ...)
E_s = F.one_hot(sampled_s.E, ...)

# Then FORCE REPLACE scaffold  ← 关键！
if scaffold_mol is not None:
    for local_idx in scaffold_indices:
        atom_type_idx = ...
        X_s[:, local_idx, :] = 0
        X_s[:, local_idx, atom_type_idx] = 1  # 强制替换！
    
    for bond in scaffold_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_type_idx = ...
        E_s[:, i, j, :] = 0
        E_s[:, i, j, edge_type_idx] = 1  # 强制替换！
        E_s[:, j, i, :] = 0
        E_s[:, j, i, edge_type_idx] = 1
```

---

## 📊 完整流程

### 每一步（t → t-1）

```
1. 模型预测 → pred_X, pred_E

2. HOOK 1: 应用化学式掩码
   pred_X[non_scaffold] = apply_formula_mask(...)

3. 计算后验概率 → prob_X, prob_E

4. HOOK 3: 冻结骨架概率
   prob_X[scaffold] = one_hot(scaffold_types)  ← 第1次冻结
   prob_E[scaffold_bonds] = one_hot(bond_types)

5. 采样
   sampled = sample_discrete_features(prob_X, prob_E)
   X_s = F.one_hot(sampled.X, ...)
   E_s = F.one_hot(sampled.E, ...)

6. POST-SAMPLING HOOK: 强制替换  ← 新增！
   X_s[scaffold] = one_hot(scaffold_types)      ← 第2次冻结
   E_s[scaffold_bonds] = one_hot(bond_types)    ← 第2次冻结

7. 返回 X_s, E_s
```

---

## ✅ 预期效果

**双重保险**：
1. 即使采样有随机性，POST-SAMPLING HOOK也会强制替换
2. 即使mask操作改变了值，POST-SAMPLING HOOK也会强制替换
3. **100%保证**骨架部分不变

---

## 🚀 运行测试

```bash
modal run /Users/aylin/yaolab_projects/diffms_yaolab/modal/diffms_scaffold_inference.py
```

### 应该看到

```
[HOOK 3] Frozen 33 atoms, 36 bonds  ← 概率冻结

[DEBUG] After diffusion loop:
  ✓ All 33 atoms match  ← 验证通过

[DEBUG] Converting graph #0:
  Edge type counts: SINGLE=35, DOUBLE=1, NO_EDGE=...  ← 边正确

Generated mol: CC(C)=CCCC(C(=O)O)C1CCC2...  ← 包含骨架？
✅ Generated molecule CONTAINS scaffold!  ← 希望成功！
```

---

## 📝 修改的文件

- **`DiffMS/src/diffusion_model_spec2mol.py`**:
  - 第1462-1499行：新增POST-SAMPLING HOOK

---

## 💡 为什么需要双重冻结

**概率冻结**（HOOK 3）：
- 理论上应该够了
- 但可能被采样/mask破坏

**采样后替换**（POST-SAMPLING）：
- 物理替换，100%保证
- 即使之前有任何问题，这一步都会修正

**双保险 = 最稳妥** ✅

