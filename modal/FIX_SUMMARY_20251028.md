# 🔧 修复总结 - 2024-10-28

## 问题

运行 `modal run diffms_scaffold_inference.py` 时出现以下错误：

```
TypeError: expected Tensor as element 0 in argument 0, but got NoneType
at: y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
```

## 根本原因

当从批次中提取单个样本进行递归处理时，**谱嵌入数据 `y` 没有被正确提取和传递**。

### 具体问题

1. **RDKit 导入错误**（已修复）
   - `from rdkit.Chem import rdMolOps` ❌
   - 改为直接使用 `from rdkit import Chem` ✅

2. **谱嵌入数据丢失**（已修复）
   - `_extract_single_from_batch` 方法只提取了图数据，没有提取 `y` 字段
   - 当递归调用 `sample_batch_with_scaffold` 时，`single_data.y` 为 `None`
   - 导致在 `forward` 方法中 `noisy_data['y_t']` 为 `None`

3. **批次大小计算错误**（已修复）
   - `len(data)` 对于 `Batch` 对象返回的是节点数，不是批次大小
   - 改为使用 `data.num_graphs` 或 `data.batch.max().item() + 1`

## 修复内容

### 1. 修复 `scaffold_hooks.py` 导入

**文件**: `DiffMS/src/inference/scaffold_hooks.py`

```python
# 修改前
from rdkit.Chem import rdFMCS, rdMolOps  # ❌

# 修改后
from rdkit import Chem  # ✅
from rdkit.Chem.rdchem import Mol
```

### 2. 修复 `_extract_single_from_batch` 方法

**文件**: `DiffMS/src/diffusion_model_spec2mol.py` (第1115-1134行)

```python
def _extract_single_from_batch(self, batch_data, idx: int):
    """从 batch 中提取单个样本，保留所有字段包括 y（谱嵌入）"""
    from torch_geometric.data import Batch, Data
    
    if isinstance(batch_data, dict) and 'graph' in batch_data:
        single_graph = batch_data['graph'].get_example(idx)
        single_batch = Batch.from_data_list([single_graph])
        # 提取对应的 y（谱嵌入）
        if hasattr(batch_data['graph'], 'y') and batch_data['graph'].y is not None:
            single_batch.y = batch_data['graph'].y[idx:idx+1]
        return {'graph': single_batch}
    else:
        # 直接是 Batch 对象
        single_graph = batch_data.get_example(idx)
        single_batch = Batch.from_data_list([single_graph])
        # ✅ 关键修复：保留 y（谱嵌入数据）
        if hasattr(batch_data, 'y') and batch_data.y is not None:
            single_batch.y = batch_data.y[idx:idx+1].clone()
        return single_batch
```

### 3. 修复批次大小计算

**文件**: `DiffMS/src/diffusion_model_spec2mol.py` (第828-840行)

```python
if isinstance(target_formula, list):
    # 正确计算批次大小：使用 num_graphs 或 batch 属性
    if hasattr(data, 'num_graphs'):
        batch_size = data.num_graphs  # ✅
    elif hasattr(data, 'batch'):
        batch_size = data.batch.max().item() + 1  # ✅
    else:
        batch_size = 1
    
    if len(target_formula) != batch_size:
        raise ValueError(f"Formula list length != batch size")
```

### 4. 添加保护性检查

**文件**: `DiffMS/src/diffusion_model_spec2mol.py`

- 在提取单个样本后，检查 `y` 是否存在（第842-849行）
- 在单个formula模式下，确保 `data.y` 不为 `None`（第882-887行）
- 添加详细的错误日志，方便调试

## 修复后的预期行为

1. ✅ 正确导入 RDKit 模块
2. ✅ 正确提取并传递谱嵌入数据 `y`
3. ✅ 正确计算批次大小
4. ✅ 清晰的错误信息（如果仍有问题）

## 现在可以运行

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_scaffold_inference.py
```

## 预期输出

```
开始 DiffMS 骨架约束推理 on Modal
骨架信息:
  SMILES: CC(=CCCC(C1CCC2(C1(CCC3=C2CCC4C3(CCC(C4(C)C)O)C)C)C)C(=O)O)C
  分子式: C30H48O3
  重原子数: 33
  ✓ 骨架验证成功

步骤 3: 验证骨架与目标分子式的兼容性...
  ✓ SPEC_4922: C30H48O3 (ΔF = {})
  ✓ SPEC_6652: C33H52O5 (ΔF = C3H4O2)
  ... (更多)

  ✓ 10/10 个样本与骨架兼容

步骤 10: 开始骨架约束推理...
Batch 0: loaded 10 formulas
[开始采样...]

✅ 推理完成！
```

## 技术细节

### 为什么 `y` 丢失了？

在 PyTorch Geometric 中，`Batch.from_data_list([single_graph])` 只会保留 `Data` 对象内部的属性（如 `x`, `edge_index`, `edge_attr`）。

**谱嵌入 `y` 是批次级别的属性**（每个样本一个），而不是图级别的属性，因此需要**手动从原始 batch 中提取并赋值**。

### 为什么需要 `.clone()`？

使用 `.clone()` 确保：
1. 创建独立的张量副本
2. 避免梯度计算错误
3. 防止多个样本共享同一内存

## 下一步

如果仍有错误：
1. 查看错误日志，特别是"Sample X: Extracted data missing y"
2. 检查 `data.y` 的形状和内容
3. 确认数据加载器正确设置了谱嵌入

---

**修复时间**: 2024-10-28  
**修复文件**: 
- `DiffMS/src/inference/scaffold_hooks.py`
- `DiffMS/src/diffusion_model_spec2mol.py`

**状态**: ✅ 已完成

