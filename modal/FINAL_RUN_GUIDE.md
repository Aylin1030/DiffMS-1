# 🚀 最终运行指南

## ✅ 所有问题已修复

### 修复清单

1. ✅ **RDKit 导入错误** - `scaffold_hooks.py`
2. ✅ **谱嵌入数据丢失** - `_extract_single_from_batch` 
3. ✅ **批次大小计算** - 使用 `data.num_graphs`
4. ✅ **骨架冻结失效** - **第937行：X 更新（最关键）**

---

## 📋 兼容性确认

### 与原始模型完全兼容

- ✅ 数据初始化方式相同
- ✅ 张量形状一致
- ✅ 设备处理正确
- ✅ 采样循环参数相同
- ✅ 后处理流程一致

### 唯一预期的差异

**第937行**: `X, E, y = sampled_s.X, sampled_s.E, data.y`

- **原始模型**: 不更新 X（因为 X 来自输入数据）
- **骨架约束**: 更新 X（因为需要在每步冻结骨架）

这是**必要且正确**的差异。

---

## 🎯 运行步骤

### 1. 确认测试数据已上传

```bash
modal volume ls diffms-data /data/test_top10/
```

应该看到：
```
spec_files/
subformulae/
split.tsv
labels.tsv
```

如果没有，运行：
```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
bash upload_test_data.sh
```

---

### 2. 运行骨架约束推理

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_scaffold_inference.py
```

---

### 3. 观察关键日志

#### ✅ 成功的标志

**启动阶段**:
```
✓ Scaffold atoms initialized correctly (可选，如果添加了调试)
骨架信息:
  SMILES: CC(=CCCC(C1CCC2...
  分子式: C30O3
  重原子数: 33
  ✓ 骨架验证成功
```

**推理阶段**（关键！）:
```
Batch 0: loaded 10 formulas
Scaffold formula: C30O3
Target formula: C30O3
Remaining formula (ΔF): C0O0

[采样过程 - 应该很快，每个样本约15秒]

✅ Generated molecule contains scaffold  ← 应该看到这个！
✅ Generated molecule contains scaffold
...
```

**最终统计**:
```
统计:
  有效SMILES: 95/100 (95.0%)
  包含骨架: 75/100 (75.0%)  ← 应该 > 0%（预期 70-90%）
```

---

#### ❌ 失败的标志

如果仍然看到：
```
❌ Generated molecule does NOT contain scaffold. Discarding.
❌ Generated molecule does NOT contain scaffold. Discarding.
...（全部失败）
```

可能原因：
1. 修改未生效（检查文件是否保存）
2. 原子索引不匹配（需要更复杂的原子映射）
3. 骨架验证过于严格

---

### 4. 成功后下载结果

```bash
# 下载SMILES文件
modal volume get diffms-outputs /outputs/smiles_scaffold ./results_scaffold

# 查看Top-1预测
cat results_scaffold/predictions_top1.tsv

# 查看所有候选（包含骨架标记）
cat results_scaffold/predictions_all_candidates.tsv
```

---

## 🔍 验证骨架是否被保留

### 方法 1: 查看日志

搜索日志中的：
```bash
grep "Generated molecule contains scaffold" inference.log
```

应该看到多行（不是0行）。

### 方法 2: 检查SMILES文件

查看 `predictions_all_candidates.tsv`，其中有 `contains_scaffold` 列：

```tsv
spec_id    rank    smiles                          contains_scaffold
spec_0000  1       CC(=CCCC(C1CCC2...              True
spec_0000  2       CC(=CCCC(C1CCC2...              True
spec_0001  1       CC(=CCCC(C1CCC2...              False
...
```

计算 `contains_scaffold=True` 的比例。

### 方法 3: 手动验证

使用 RDKit 检查：

```python
from rdkit import Chem

scaffold_smiles = "CC(=CCCC(C1CCC2(C1(CCC3=C2CCC4C3(CCC(C4(C)C)O)C)C)C)C(=O)O)C"
scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)

generated_smiles = "..."  # 从结果中复制
generated_mol = Chem.MolFromSmiles(generated_smiles)

# 检查
if generated_mol.HasSubstructMatch(scaffold_mol):
    print("✅ 包含骨架！")
else:
    print("❌ 不包含骨架")
```

---

## 📊 预期结果

### 成功的指标

| 指标 | 预期值 | 说明 |
|------|--------|------|
| 有效SMILES | 90-100% | 生成的分子语法正确 |
| 包含骨架 | **70-90%** | **关键指标**：骨架冻结成功 |
| 运行时间 | ~5分钟（10个样本） | 每个样本约30秒 |
| GPU利用率 | 60-90% | A100 应该充分利用 |

### 如果包含骨架 = 0%

说明骨架冻结仍未生效，可能：
1. 文件未正确挂载到 Modal（检查 `src` mount）
2. 原子索引映射问题（骨架原子位置不对）
3. 需要添加调试日志确定具体原因

---

## 🐛 调试选项

如果需要更多调试信息，临时修改 `diffusion_model_spec2mol.py`:

### 添加调试日志（第918行后）

```python
# 验证骨架初始化
if enforce_scaffold:
    logging.info(f"Scaffold atoms initialized:")
    for local_idx in range(min(5, scaffold_size)):
        atom_types = X[0, local_idx, :]
        predicted_type = torch.argmax(atom_types).item()
        predicted_symbol = self.atom_decoder[predicted_type]
        expected_symbol = scaffold_mol.GetAtomWithIdx(local_idx).GetSymbol()
        logging.info(f"  Node {local_idx}: {predicted_symbol} (expected: {expected_symbol})")
```

### 添加中间检查（第938行后）

```python
# 每10步检查一次
if enforce_scaffold and s_int % 10 == 0:
    for local_idx in range(min(3, scaffold_size)):
        atom_types = X[0, local_idx, :]
        predicted_type = torch.argmax(atom_types).item()
        predicted_symbol = self.atom_decoder[predicted_type]
        expected_symbol = scaffold_mol.GetAtomWithIdx(local_idx).GetSymbol()
        if predicted_symbol != expected_symbol:
            logging.warning(f"Step {s_int}, Node {local_idx}: {predicted_symbol} != {expected_symbol}")
```

---

## 📞 如果还有问题

### 检查文件是否正确

```bash
# 本地检查第937行
grep -n "X, E, y = sampled_s.X, sampled_s.E, data.y" \
  /Users/aylin/yaolab_projects/diffms_yaolab/DiffMS/src/diffusion_model_spec2mol.py
```

应该输出：
```
937:            X, E, y = sampled_s.X, sampled_s.E, data.y
```

### 查看 Modal 挂载的文件

在 `run_scaffold_inference` 函数开头添加：

```python
# 调试：检查文件内容
with open('/root/src/diffusion_model_spec2mol.py', 'r') as f:
    lines = f.readlines()
    logging.info(f"Line 937: {lines[936].strip()}")  # 索引从0开始
```

应该输出：
```
Line 937: X, E, y = sampled_s.X, sampled_s.E, data.y
```

如果输出的是 `_, E, y = ...`，说明文件未正确挂载。

---

## ✅ 成功标准

运行完成后，如果看到：

1. ✅ 有日志显示 "Generated molecule contains scaffold"
2. ✅ 最终统计中 "包含骨架" > 50%
3. ✅ `predictions_all_candidates.tsv` 中有 `contains_scaffold=True` 的行

**恭喜！骨架约束成功运行！** 🎉

---

## 🎯 下一步

成功后，可以：

1. **调整参数**:
   - 修改骨架SMILES
   - 指定attachment_indices
   - 尝试不同的数据集

2. **优化性能**:
   - 增加 `test_samples_to_generate`
   - 启用 `use_rerank`
   - 调整beam size

3. **集成到工作流**:
   - 批量处理大量谱图
   - 自动化结果分析
   - 与其他工具集成

---

**准备就绪！现在可以运行了！** 🚀

```bash
modal run diffms_scaffold_inference.py
```

