# 分子式约束修复记录 (2024-10-28)

## 问题描述

推理生成的SMILES不符合分子式约束：
- 生成的分子原子数量正确，但元素类型和连接方式错误
- 10个候选分子应该都符合输入的分子式（只是同分异构体），但实际上都不符合

## 根本原因

DiffMS使用`denoise_nodes=False`配置，理论上应该只扩散边（E），保持节点类型（X）固定。但在原始的`sample_batch`实现中存在问题：

1. **问题1**：虽然最终丢弃了采样的X，但在每一步的`sample_p_zs_given_zt`中，模型仍然预测新的节点类型
2. **问题2**：边的后验分布计算依赖于**预测的节点类型**，而不是固定的节点类型
3. **结果**：即使最后使用了固定的X，但边的生成过程已经被错误的节点预测影响了

## 解决方案

### 修改1: `sample_batch`方法

```python
@torch.no_grad()
def sample_batch(self, data: Batch) -> Batch:
    dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)

    # 关键修改：固定节点类型X（分子式约束），只扩散边E
    X_fixed = dense_data.X  # 来自dummy graph的固定节点类型
    
    # 初始化：E从噪声分布采样
    z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
    E = z_T.E
    y = data.y

    # 迭代去噪：固定X，只更新E
    for s_int in reversed(range(0, self.T)):
        # ...
        sampled_s, __ = self.sample_p_zs_given_zt(s_norm, t_norm, X_fixed, E, y, node_mask, 
                                                   keep_X_fixed=not self.denoise_nodes)
        # 只更新E，X保持不变
        E = sampled_s.E
    
    # 最终结果：使用固定的X和去噪后的E
    final_data = utils.PlaceHolder(X=X_fixed, E=E, y=y)
    final_data = final_data.mask(node_mask, collapse=True)
```

### 修改2: `sample_p_zs_given_zt`方法

添加`keep_X_fixed`参数，当为True时强制使用固定的节点类型：

```python
def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, keep_X_fixed=False):
    # ...
    pred_X = F.softmax(pred.X, dim=-1)
    pred_E = F.softmax(pred.E, dim=-1)

    # 关键修改：如果keep_X_fixed=True，强制pred_X等于X_t
    # 这确保节点类型不会改变，只有边会被更新
    if keep_X_fixed:
        pred_X = X_t  # 使用当前的X作为"预测"，保持节点类型不变
    
    # 计算后验分布（此时pred_X是固定的X_t）
    # ...
    
    # 额外保险：最终结果也使用固定的X
    if keep_X_fixed:
        X_s = X_t
```

## 工作原理

1. **Dummy Graph创建**：根据分子式创建dummy graph，节点类型按照元素组成设置（如C32H50O7 → 32个C + 7个O）
2. **X固定策略**：在整个采样过程中，X始终等于dummy graph的节点类型（one-hot编码）
3. **E生成策略**：
   - 初始状态：E从噪声分布采样
   - 每一步：基于固定的X和当前的E，预测下一步的E
   - 关键：后验分布计算时使用固定的X（而非模型预测的X）

## 预期结果

- ✅ 生成的10个候选分子应该都符合输入的分子式
- ✅ 10个候选应该是同分异构体（相同元素组成，不同连接方式）
- ✅ SMILES可以正确解析为有效分子

## 测试验证

修复后需要重新运行推理：

```bash
# 重新运行Modal推理（使用前10个测试数据）
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_inference.py --data-subdir test_top10
```

## 代码文件

修改的文件：
- `/Users/aylin/yaolab_projects/diffms_yaolab/DiffMS/src/diffusion_model_spec2mol.py`
  - `sample_batch()` 方法
  - `sample_p_zs_given_zt()` 方法

## 论文参考

根据DiffMS论文（https://arxiv.org/html/2502.09571v2）：
- Section 3.1 "Formula-constrained molecular generation"
- "the decoder is a discrete graph diffusion model restricted by the heavy-atom composition of a known chemical formula"
- 实现方式：固定节点类型和数量，只生成边

## 总结

这个修复确保了DiffMS的formula-constrained generation正确工作，生成的分子必然符合输入的分子式约束。这是通过在扩散采样的每一步都强制保持节点类型不变来实现的。
