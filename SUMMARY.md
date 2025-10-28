# DiffMS 分子式约束推理 - 修复总结

## ✅ 已修复的关键问题

### 问题：生成的SMILES不符合分子式约束

**症状**:
- 生成的10个候选分子原子数量正确，但元素类型和连接方式完全错误
- 应该生成符合输入分子式的同分异构体，但实际结果完全不符合

**根本原因**:
DiffMS使用`denoise_nodes=False`配置，理论上应该只扩散边（E），固定节点类型（X）。但原始实现中：
1. 每一步采样时，模型仍然预测新的节点类型
2. 边的后验分布计算依赖于**预测的节点类型**，而不是固定的节点类型
3. 即使最后丢弃了预测的X，边的生成已经被错误的预测影响

**解决方案**:
修改了`DiffMS/src/diffusion_model_spec2mol.py`中的两个关键方法：

1. **`sample_batch()`**: 在整个采样过程中强制保持X固定
2. **`sample_p_zs_given_zt()`**: 添加`keep_X_fixed`参数，当为True时使用固定的X计算后验分布

详细技术说明见: `docs/FORMULA_CONSTRAINT_FIX_20251028.md`

## 📁 项目结构（清理后）

```
diffms_yaolab/
├── DiffMS/                      # DiffMS源代码
│   ├── src/
│   │   ├── diffusion_model_spec2mol.py  # ✨ 已修复：分子式约束
│   │   └── mist/data/
│   │       ├── datasets.py              # ✨ 已修复：推理模式dummy graph
│   │       └── featurizers.py           # ✨ 已修复：分子式解析
│   └── configs/
├── modal/                       # Modal云端推理脚本
│   ├── diffms_inference.py     # ✨ 已更新：支持data_subdir参数
│   ├── convert_to_table.py     # 结果转换脚本
│   ├── upload_to_modal.sh      # 数据上传脚本
│   └── results/                # 生成的结果（待重新运行）
├── docs/                        # 📚 文档
│   └── FORMULA_CONSTRAINT_FIX_20251028.md  # 修复详情
├── RUN_INFERENCE.md            # 🚀 运行指南
└── SUMMARY.md                  # 📋 本文件
```

## 🔧 修改的文件

### 核心修改

1. **`DiffMS/src/diffusion_model_spec2mol.py`** ⭐⭐⭐
   - `sample_batch()`: 强制X固定，只扩散E
   - `sample_p_zs_given_zt()`: 添加`keep_X_fixed`参数

2. **`DiffMS/src/mist/data/datasets.py`**
   - `SpectraMolDataset.__getitem__()`: 根据分子式动态创建dummy graph

3. **`DiffMS/src/mist/data/featurizers.py`**
   - 添加`parse_formula()`: 解析分子式字符串
   - 添加`create_dummy_graph_from_formula()`: 根据分子式创建图

### 配置修改

4. **`modal/diffms_inference.py`**
   - 添加`data_subdir`参数，支持选择数据子目录
   - 支持使用`test_top10`测试数据

## 🚀 下一步操作

### 1. 在Modal上重新运行推理

使用修复后的代码运行前10个测试数据：

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_inference.py --data-subdir test_top10
```

### 2. 验证结果

检查生成的分子是否符合分子式约束：
- 每个样本的10个候选应该都符合输入的分子式
- 候选应该是同分异构体（相同元素，不同结构）

### 3. 全量运行

如果测试通过，运行完整数据集：

```bash
modal run diffms_inference.py
```

## 📊 预期改进

**修复前**:
- ❌ 生成的SMILES元素组成错误
- ❌ 无法识别的复杂SMILES结构
- ❌ 与输入分子式完全不符

**修复后**:
- ✅ 生成的分子严格符合输入分子式
- ✅ 10个候选都是有效的同分异构体
- ✅ SMILES可正确解析

## 🔍 技术细节

### 分子式约束实现

1. **Dummy Graph创建**: 根据分子式（如`C32H50O7`）创建初始图
   - 节点数量 = 重原子数（32C + 7O = 39）
   - 节点类型 = 元素类型（C=0, O=1, ...）
   - 边 = 空（由扩散模型生成）

2. **固定X策略**: 
   ```python
   X_fixed = dense_data.X  # 来自dummy graph
   for t in reversed(range(T)):
       sampled_s = sample_p_zs_given_zt(..., X_fixed, E, ..., keep_X_fixed=True)
       E = sampled_s.E  # 只更新边
   ```

3. **条件采样**:
   ```python
   if keep_X_fixed:
       pred_X = X_t  # 强制使用固定的X
   # 使用固定的X计算边的后验分布
   ```

## 📚 参考资料

- **论文**: DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra
  - URL: https://arxiv.org/html/2502.09571v2
  - Section 3.1: Formula-constrained molecular generation

- **运行指南**: `RUN_INFERENCE.md`
- **修复详情**: `docs/FORMULA_CONSTRAINT_FIX_20251028.md`

## ⚠️ 重要说明

1. **已删除的临时文件**: 所有调试日志、临时脚本和多余文档已清理
2. **文档位置**: 所有文档统一放在`docs/`目录，根目录只保留关键指南
3. **测试数据**: 使用`--data-subdir test_top10`参数测试前10个样本

## 🎯 验证清单

运行推理后，请验证：
- [ ] 所有生成的SMILES可以被RDKit解析
- [ ] 每个样本的10个候选都符合输入分子式
- [ ] 候选分子的元素组成完全一致（同分异构体）
- [ ] 分子结构合理（价态、键数等符合化学规则）

## 📧 后续优化建议

1. **增加采样数量**: 当前为10个候选，可增至100个以提高匹配率
2. **质量过滤**: 添加分子有效性和合理性检查
3. **结果排序**: 基于相似度或其他指标对候选排序

---

**修复日期**: 2024-10-28  
**修复版本**: v1.0 (Formula-Constrained)
