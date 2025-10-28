# 🔬 骨架冻结机制验证

## 骨架冻结的三个阶段

### 阶段 1: 初始化（第907-917行）

```python
if enforce_scaffold and scaffold_size <= X.shape[1]:
    # Overwrite scaffold atoms in X
    for local_idx in range(scaffold_size):
        if local_idx >= X.shape[1]:
            break
        atom = scaffold_mol.GetAtomWithIdx(local_idx)
        atom_symbol = atom.GetSymbol()
        if atom_symbol in self.atom_decoder:
            atom_type_idx = self.atom_decoder.index(atom_symbol)
            X[:, local_idx, :] = 0  # 清零
            X[:, local_idx, atom_type_idx] = 1  # 设置为骨架原子类型
```

**作用**: 在扩散开始前，将 X 的前 N 个节点（N = 骨架大小）设置为骨架的原子类型。

**示例**: 
- 骨架: `C-C-C` (3个碳原子)
- 初始化后: `X[:, 0, C_idx] = 1`, `X[:, 1, C_idx] = 1`, `X[:, 2, C_idx] = 1`

---

### 阶段 2: 采样循环中的冻结（第1109-1120行）

```python
# === HOOK 3: Freeze scaffold atoms in prob_X ===
if scaffold_mol is not None and scaffold_indices is not None:
    for local_idx in scaffold_indices:
        if local_idx >= scaffold_mol.GetNumAtoms() or local_idx >= n:
            continue
        atom = scaffold_mol.GetAtomWithIdx(local_idx)
        atom_symbol = atom.GetSymbol()
        if atom_symbol in self.atom_decoder:
            atom_type_idx = self.atom_decoder.index(atom_symbol)
            # Force scaffold atoms to stay fixed
            prob_X[:, local_idx, :] = 0
            prob_X[:, local_idx, atom_type_idx] = 1
```

**作用**: 在每一步采样前，强制骨架原子的采样概率为 100%（保持原子类型）。

**关键**: 这确保了即使模型预测其他原子类型，采样时也会选择骨架的原子类型。

---

### 阶段 3: 更新 X（第937行，已修复）

```python
# 修复前
_, E, y = sampled_s.X, sampled_s.E, data.y  # ❌ X 被丢弃

# 修复后
X, E, y = sampled_s.X, sampled_s.E, data.y  # ✅ X 被更新
```

**作用**: 将采样后的 X（包含冻结的骨架原子）更新回循环变量。

**关键**: 没有这一步，阶段 2 的冻结完全失效！

---

## 完整工作流程

```
初始化
  ↓
X = dense_data.X (来自输入数据)
  ↓
阶段 1: 覆写骨架原子
  X[:, 0:scaffold_size, :] ← scaffold atoms
  ↓
扩散步骤 T → T-1
  ↓
模型预测 pred.X
  ↓
计算概率 prob_X = softmax(pred.X + 扩散先验)
  ↓
阶段 2: 冻结骨架原子概率
  prob_X[:, 0:scaffold_size, :] ← one-hot(scaffold atoms)
  ↓
采样 sampled_s.X ~ Categorical(prob_X)
  结果: sampled_s.X[:, 0:scaffold_size] = scaffold atoms
  ↓
阶段 3: 更新 X
  X ← sampled_s.X
  ↓
重复 T 步
  ↓
最终 X 包含：
  - 前 scaffold_size 个节点：骨架原子（冻结）
  - 后续节点：生成的原子
  ↓
转换为分子
  mol = mol_from_graphs(X, E)
  ↓
验证骨架
  contains_scaffold(mol, scaffold_mol) → True ✅
```

---

## 验证检查点

### ✅ 检查点 1: 初始化是否正确？

**验证代码**（可添加到第918行后）:
```python
# 调试：检查初始化后的 X
if enforce_scaffold:
    for local_idx in range(scaffold_size):
        atom_types = X[0, local_idx, :]  # 第一个 batch
        predicted_type = torch.argmax(atom_types).item()
        expected_symbol = scaffold_mol.GetAtomWithIdx(local_idx).GetSymbol()
        expected_idx = self.atom_decoder.index(expected_symbol)
        assert predicted_type == expected_idx, f"Node {local_idx}: got {predicted_type}, expected {expected_idx}"
    logging.info("✅ Scaffold atoms initialized correctly")
```

### ✅ 检查点 2: 采样概率是否被冻结？

**验证代码**（可添加到第1120行后）:
```python
# 调试：检查概率是否被正确冻结
if scaffold_mol is not None:
    for local_idx in scaffold_indices[:3]:  # 检查前3个
        probs = prob_X[0, local_idx, :]  # 第一个 batch
        max_prob_idx = torch.argmax(probs).item()
        expected_symbol = scaffold_mol.GetAtomWithIdx(local_idx).GetSymbol()
        expected_idx = self.atom_decoder.index(expected_symbol)
        assert max_prob_idx == expected_idx and probs[max_prob_idx] > 0.99, \
            f"Scaffold atom {local_idx} not frozen! prob={probs[max_prob_idx]}"
    logging.debug(f"✅ Scaffold probabilities frozen at step {s_int}")
```

### ✅ 检查点 3: X 是否被更新？

**验证代码**（可添加到第938行后）:
```python
# 调试：确认 X 被更新
if enforce_scaffold:
    for local_idx in range(min(3, scaffold_size)):
        atom_types = X[0, local_idx, :]
        predicted_type = torch.argmax(atom_types).item()
        expected_symbol = scaffold_mol.GetAtomWithIdx(local_idx).GetSymbol()
        expected_idx = self.atom_decoder.index(expected_symbol)
        assert predicted_type == expected_idx, \
            f"Step {s_int}: Node {local_idx} lost scaffold! got {predicted_type}, expected {expected_idx}"
    if s_int % 10 == 0:
        logging.debug(f"✅ Step {s_int}: X updated, scaffold preserved")
```

### ✅ 检查点 4: 最终分子是否包含骨架?

**当前实现**（第960-963行）:
```python
if enforce_scaffold and mol is not None:
    if not scaffold_hooks.contains_scaffold(mol, scaffold_mol):
        logging.warning("Generated molecule does not contain scaffold. Discarding.")
        mol = None
```

**增强验证**:
```python
if enforce_scaffold and mol is not None:
    if scaffold_hooks.contains_scaffold(mol, scaffold_mol):
        logging.info(f"✅ Generated molecule contains scaffold")
    else:
        logging.warning(f"❌ Generated molecule does NOT contain scaffold")
        logging.warning(f"  Generated SMILES: {Chem.MolToSmiles(mol)}")
        logging.warning(f"  Scaffold SMILES: {Chem.MolToSmiles(scaffold_mol)}")
        mol = None
```

---

## 常见失败原因与解决方案

### 失败原因 1: X 未更新（已修复 ✅）

**症状**: 所有分子都不包含骨架

**原因**: 第936行使用 `_` 丢弃了 `sampled_s.X`

**修复**: 改为 `X, E, y = sampled_s.X, sampled_s.E, data.y`

---

### 失败原因 2: 原子索引不匹配

**症状**: 部分分子包含骨架，部分不包含

**原因**: 
- 骨架 SMILES 的原子顺序与输入数据不一致
- 假设前 N 个节点对应骨架可能错误

**诊断**:
```python
# 在第905行后添加
logging.info(f"Scaffold size: {scaffold_size}")
logging.info(f"Graph size: {X.shape[1]}")
logging.info(f"Scaffold atoms: {[a.GetSymbol() for a in scaffold_mol.GetAtoms()]}")
logging.info(f"Graph node types (first batch): {[self.atom_decoder[torch.argmax(X[0, i, :]).item()] for i in range(scaffold_size)]}")
```

**解决方案**: 如果不匹配，需要实现子图匹配来找到正确的原子映射。

---

### 失败原因 3: `contains_scaffold` 检查过于严格

**症状**: 生成的分子确实包含骨架原子，但检查失败

**原因**: 
- RDKit 的 `HasSubstructMatch` 可能对立体化学、芳香性等敏感
- 骨架在价态修正后发生了变化

**诊断**:
```python
# 修改 scaffold_hooks.contains_scaffold
def contains_scaffold(candidate: Mol, scaffold: Mol) -> bool:
    # 移除立体化学信息
    Chem.RemoveStereochemistry(candidate)
    Chem.RemoveStereochemistry(scaffold)
    
    # 尝试匹配
    match = candidate.HasSubstructMatch(scaffold, useChirality=False)
    
    # 调试
    if not match:
        logging.debug(f"Substructure match failed")
        logging.debug(f"  Candidate: {Chem.MolToSmiles(candidate)}")
        logging.debug(f"  Scaffold: {Chem.MolToSmiles(scaffold)}")
    
    return match
```

---

## 推荐的调试策略

### 策略 1: 添加详细日志

在关键位置添加 `logging.info`，输出：
1. 骨架大小和原子类型
2. 每10步的 X 状态
3. 最终分子的 SMILES 和骨架匹配结果

### 策略 2: 可视化中间状态

保存每一步的 X，生成动画：
```python
# 在循环中
if s_int % 10 == 0:
    intermediate_mol = mol_from_graphs(X[0], E[0])
    Chem.MolToFile(intermediate_mol, f"debug_step_{s_int}.png")
```

### 策略 3: 单步测试

只运行 1 个扩散步骤，检查骨架是否保持：
```python
# 临时修改
for s_int in reversed(range(self.T - 1, self.T)):  # 只运行最后一步
    ...
```

---

## 预期结果

修复后（第937行：X 更新），应该看到：

```
2025-10-28 XX:XX:XX - INFO - Scaffold formula: C30O3
2025-10-28 XX:XX:XX - INFO - Target formula: C30O3
2025-10-28 XX:XX:XX - INFO - Remaining formula (ΔF): C0O0
[扩散采样 500 步]
2025-10-28 XX:XX:XX - INFO - ✅ Generated molecule contains scaffold
[价态修正]
2025-10-28 XX:XX:XX - INFO - ✅ Scaffold verification passed
```

**成功率预期**: 70-90% 的生成分子应该包含骨架

---

## 最终验证命令

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_scaffold_inference.py
```

观察日志中：
1. ✅ "Generated molecule contains scaffold" 应该出现
2. ❌ "Generated molecule does NOT contain scaffold" 应该少于30%
3. 最终统计："包含骨架: XX/100 (>70%)"

---

**状态**: 关键修复已完成（第937行），骨架冻结机制应该正常工作 ✅

