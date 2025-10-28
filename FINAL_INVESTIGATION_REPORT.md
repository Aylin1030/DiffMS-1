# DiffMS推理问题完整调查报告

**日期**: 2025-10-28  
**状态**: 🔴 发现严重问题 - Validity = 0%

---

## 📊 问题概述

### 核心问题
生成的分子100%化学无效，主要是**价态错误**。

### 症状
```
❌ C(5), C(6), C(7), C(8) - 碳原子价态超过4
❌ O(3), O(4), O(5) - 氧原子价态超过2
```

### 与论文对比

| 数据集 | 论文Validity | 我们的Validity |
|--------|-------------|---------------|
| NPLIB1 | **100%** ✅ | N/A |
| MassSpecGym | **100%** ✅ | N/A |
| 自定义数据 | N/A | **0%** ❌ |
| MSG (5样本) | **100%** ✅ | **0%** ❌ |

---

## 🔍 完整调查过程

### 阶段1: 输出格式检查 ✅

**问题**: 论文用SMILES还是InChI？

**调查结果**:
- 论文使用**InChI**进行Top-K Accuracy评估
- 论文使用严格的`RDKit.SanitizeMol()`检查validity

**修复**:
```python
# 添加到convert_to_table.py
inchi = Chem.MolToInchi(mol)  # 论文标准
inchikey = Chem.MolToInchiKey(mol)  # 唯一标识

# 严格validity检查
Chem.SanitizeMol(mol)  # 与论文一致
```

**结果**: ✅ 输出格式已与论文一致

---

### 阶段2: Validity计算验证 ✅

**问题**: Lightning报告的validity=0是bug吗？

**调查结果**:
- ❌ **不是bug**，是真的无效
- 使用严格的RDKit sanitize检查后，确认0%有效
- 所有50个候选分子都有价态错误

**代码证据**:
```python
# DiffMS/src/utils.py
def is_valid(mol):
    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, 
                                             sanitizeFrags=True)  # 严格检查
    except:
        return False
    if len(mol_frags) > 1:  # 必须连通
        return False
    return True
```

**结果**: ✅ 确认validity=0%是真实问题

---

### 阶段3: 价态修正功能调查 ⚠️

**问题**: 为什么生成化学无效的分子？

**发现1**: 代码中存在价态修正函数
```python
# DiffMS/src/analysis/rdkit_functions.py
def correct_mol(m):
    """修正分子价态错误"""
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:  # 价态正确
            break
        else:  # 价态错误
            # 移除或降低键级来修正
            ...
    return mol, no_correct
```

**发现2**: 当前生成代码**没有使用**价态修正
```python
# DiffMS/src/diffusion_model_spec2mol.py - 原始代码
def sample_batch(self, data):
    ...
    for nodes, adj_mat in zip(X, E):
        mol = self.visualization_tools.mol_from_graphs(nodes, adj_mat)
        mols.append(mol)  # ❌ 直接使用，没有修正
    return mols
```

**修复尝试**:
```python
# 添加价态修正
if mol is not None:
    from rdkit import Chem
    from analysis.rdkit_functions import correct_mol
    try:
        editable_mol = Chem.RWMol(mol)
        corrected_mol, no_correct = correct_mol(editable_mol)
        if corrected_mol is not None:
            mol = corrected_mol
    except Exception as e:
        pass
```

**结果**: ❌ 仍然validity=0%

**原因**: 生成的边质量太差，`correct_mol()`无法修正严重的价态错误

---

### 阶段4: 模型配置检查 ✅

**问题**: 配置是否正确？

**关键配置**:
```yaml
# DiffMS/configs/dataset/msg.yaml
denoise_nodes: False  # ✅ 正确
```

**含义**:
- `denoise_nodes=False`: 只扩散**边**，不扩散**节点**
- 节点类型（原子）固定，来自分子式dummy graph
- 模型只预测键（边）

**验证**:
```python
# DiffMS/src/diffusion_model_spec2mol.py
X_t = X  # 保持节点不变
if self.denoise_nodes:
    X_t = F.one_hot(sampled_t.X, ...)  # 只有当denoise_nodes=True时才更新
E_t = F.one_hot(sampled_t.E, ...)  # 总是更新边
```

**结果**: ✅ 配置正确，逻辑正确

---

### 阶段5: 数据匹配度分析 ⚠️

**问题**: 测试数据与训练数据是否匹配？

**训练数据 (MSG)**:
- 分子大小: ~30-38原子
- 来源: MassSpecGym公共数据集

**我们的测试数据**:
- 分子大小: **44-88原子** ⚠️
- 来源: 自定义天然产物

**影响**:
```
训练范围: [20------30------40]
测试分子:                    [44------88]  ❌ 超出训练范围
```

模型可能没见过这么大的分子，导致：
- 边预测质量极差
- 价态错误严重
- 无法修正

---

## 🎯 根本原因分析

### 可能原因排序

#### 1. **数据域不匹配** ⭐⭐⭐⭐⭐
- **证据**: 测试分子(44-88原子) >> 训练分子(30-38原子)
- **影响**: 模型外推能力差，生成质量极低
- **验证方法**: 用官方测试数据测试

#### 2. **Checkpoint损坏/不完整** ⭐⭐⭐⭐
- **证据**: 论文报告100% validity，我们0%
- **影响**: 模型权重可能缺失或损坏
- **验证方法**: 
  - 检查checkpoint文件完整性
  - 用官方数据测试

#### 3. **Dummy Graph创建方式** ⭐⭐⭐
- **证据**: 节点按字母顺序排列 `[C,C,C...,O,O,O...]`
- **影响**: 与真实分子的原子排列不同
- **验证方法**: 尝试随机打乱节点

#### 4. **缺少关键后处理** ⭐⭐
- **证据**: 代码有`correct_mol()`但无法修正
- **影响**: 严重价态错误无法自动修正
- **验证方法**: 检查论文是否有其他后处理

---

## 📋 建议的下一步行动

### 🔴 紧急优先级

#### 1. 用官方数据验证模型 ⭐⭐⭐⭐⭐
```bash
# 获取MassSpecGym或NPLIB1官方测试集
# 运行推理
modal run diffms_inference.py --data-subdir official_test

# 期望结果：
# - 如果validity > 90%: 数据不匹配问题
# - 如果validity < 10%: 模型/checkpoint问题
```

#### 2. 验证Checkpoint完整性 ⭐⭐⭐⭐
```bash
# 检查文件大小
ls -lh /path/to/diffms_msg.ckpt

# 检查权重
python -c "
import torch
ckpt = torch.load('diffms_msg.ckpt')
print(f'Keys: {ckpt.keys()}')
print(f'State dict keys: {len(ckpt["state_dict"])}')
"
```

#### 3. 使用更小的测试分子 ⭐⭐⭐
```bash
# 筛选30-35原子的分子
# 测试validity是否提高
```

### 🟡 中等优先级

#### 4. 尝试dummy graph随机排列
#### 5. 检查是否有隐藏的配置参数
#### 6. 联系论文作者获取建议

---

## 📚 技术细节

### 分子生成流程

```
1. 输入: 质谱 + 分子式
   ↓
2. 创建dummy graph: 根据分子式 → [C,C,C...,O,O,O...]
   ↓
3. 扩散采样: X固定, E从噪声去噪
   ↓
4. mol_from_graphs: 组装分子
   ↓
5. (缺少) correct_mol: 修正价态 ← 我们添加了但无效
   ↓
6. 输出: RDKit Mol对象
```

### 价态错误示例

**生成的分子**:
```
C原子#6: 连接了6个键 ❌ (应该最多4个)
O原子#37: 连接了3个键 ❌ (应该最多2个)
```

**RDKit错误**:
```
Explicit valence for atom # 6 C, 6, is greater than permitted
Explicit valence for atom # 37 O, 3, is greater than permitted
```

---

## ✅ 已完成的工作

1. ✅ 输出格式改为InChI（与论文一致）
2. ✅ 使用严格的validity检查
3. ✅ 添加价态修正功能（虽然无效）
4. ✅ 验证配置正确性
5. ✅ 完整的代码调查
6. ✅ 详细的问题文档

---

## 🔴 结论

**主要问题**: 模型生成的边（键）质量极差，导致100%价态错误

**最可能原因**: 
1. 测试数据与训练数据域不匹配（分子太大）
2. Checkpoint可能有问题

**关键验证**: 
**必须用官方测试数据验证模型本身是否正常工作**

**当前状态**: 
- 代码已修复和优化
- 但模型生成质量仍然很差
- 需要进一步调查数据和checkpoint

---

## 📎 相关文档

- `docs/INFERENCE_SUCCESS_20251028.md` - 推理成功报告（已过时）
- `docs/VALENCE_CORRECTION_INVESTIGATION.md` - 价态修正调查
- `docs/FORMULA_CONSTRAINT_FIX_20251028.md` - 分子式约束修复
- `REAL_PROBLEM_REPORT.md` - 真实问题报告
- `INVESTIGATION_SUMMARY.md` - 调查总结

