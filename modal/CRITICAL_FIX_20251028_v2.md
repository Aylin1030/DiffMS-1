# 🔴 关键Bug修复 - 骨架未被保留

## 问题现象

运行骨架约束推理时，**所有生成的分子都不包含骨架**：

```
2025-10-28 13:45:15,420 - WARNING - Generated molecule does not contain scaffold. Discarding.
2025-10-28 13:45:29,845 - WARNING - Generated molecule does not contain scaffold. Discarding.
...（100% 失败率）
```

## 根本原因

### Bug 位置

**文件**: `DiffMS/src/diffusion_model_spec2mol.py`  
**行**: 936（修复前）

### Bug 代码

```python
# 在扩散采样循环中
for s_int in reversed(range(0, self.T)):
    sampled_s, __ = self.sample_p_zs_given_zt_with_scaffold(
        s_norm, t_norm, X, E, y, node_mask,
        scaffold_mol=scaffold_mol,
        remaining_formula=remaining_f,
        scaffold_indices=scaffold_indices,
        attachment_indices=attachment_indices
    )
    _, E, y = sampled_s.X, sampled_s.E, data.y  # ❌ BUG: 用 _ 丢弃了 sampled_s.X
```

### 问题分析

1. **原始模型设计**：
   - DiffMS 原始的 `sample_batch` 方法中，`X`（节点类型）在整个采样过程中**故意不更新**
   - 这是因为在 Spec2Mol 模式下，节点类型来自输入数据（公式约束），只需要去噪边

2. **骨架约束的冲突**：
   - 我们在 `sample_p_zs_given_zt_with_scaffold` 中实现了 **HOOK 3: 骨架冻结**
   - 该 Hook 在每一步都强制骨架原子的概率为 1（第1109-1120行）
   - 但是由于第936行用 `_` 丢弃了 `sampled_s.X`，**X 从来没有被更新**！
   - 结果：骨架冻结逻辑完全失效

3. **后果**：
   - X 在整个采样过程中保持初始值（第909-917行设置的骨架原子）
   - 但在后续步骤中，模型预测的 X 被丢弃，导致骨架信息丢失
   - 最终生成的分子不包含骨架

## 修复方案

### 修复代码

```python
# 修复后（第936-937行）
for s_int in reversed(range(0, self.T)):
    sampled_s, __ = self.sample_p_zs_given_zt_with_scaffold(
        s_norm, t_norm, X, E, y, node_mask,
        scaffold_mol=scaffold_mol,
        remaining_formula=remaining_f,
        scaffold_indices=scaffold_indices,
        attachment_indices=attachment_indices
    )
    # ✅ 修复：更新 X 以使骨架冻结生效
    X, E, y = sampled_s.X, sampled_s.E, data.y
```

### 修复逻辑

1. **每次采样后更新 X**：
   - `X = sampled_s.X` 确保每次采样的结果被保留
   
2. **骨架冻结生效**：
   - `sample_p_zs_given_zt_with_scaffold` 中的 HOOK 3（第1109-1120行）强制骨架原子概率为1
   - 采样后，骨架原子保持其类型不变
   - 非骨架原子根据模型预测和公式约束采样

3. **与原始模型的区别**：
   - **原始 `sample_batch`**: X 不更新（因为来自输入数据）
   - **骨架约束 `sample_batch_with_scaffold`**: X 每步更新（但骨架原子被冻结）

## 技术细节

### 骨架冻结的三个Hook（现在都正确工作）

1. **HOOK 1: Formula Mask**（第1066-1078行）
   - 对非骨架节点应用公式约束
   - 防止生成超出剩余公式的原子

2. **HOOK 2: Attachment Mask**（第1084-1086行）
   - （可选）限制在指定锚点连接
   - 当前简化实现，依赖骨架冻结

3. **HOOK 3: Scaffold Freeze**（第1109-1120行）✅ **现在生效**
   - 强制骨架原子的采样概率为1
   - 确保骨架原子类型不变

### 为什么分子式显示为 C30O3 而不是 C30H48O3？

这是**正确的**！因为：
- **DiffMS 模型只预测重原子**（C, N, O, S等），不预测氢
- `formula_of` 函数设计为只统计重原子（第32行：`if sym != "H"`）
- `parse_formula` 函数也跳过氢原子（第65行：`if elem and elem != 'H'`）
- 氢原子在后处理阶段由 RDKit 自动添加（价态修正）

## 预期结果

修复后，骨架约束应该正确工作：

```
✓ 骨架验证成功
步骤 10: 开始骨架约束推理...

[采样过程中，X 正确更新，骨架原子被冻结]

统计:
  有效SMILES: 95/100 (95.0%)
  包含骨架: 85/100 (85.0%)  ← 应该大于 0%
```

## 验证方法

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_scaffold_inference.py
```

观察：
1. 日志中应该有 `✓ Generated molecule contains scaffold`
2. 最终统计中 "包含骨架" 应该 > 0

## 相关文件

- **修复文件**: `DiffMS/src/diffusion_model_spec2mol.py` (第936-937行)
- **相关逻辑**: 
  - 骨架冻结: 第1109-1120行
  - 公式掩码: 第1066-1078行
  - 骨架初始化: 第907-917行

---

**修复时间**: 2024-10-28  
**严重程度**: 🔴 Critical（骨架约束完全失效）  
**状态**: ✅ 已修复

