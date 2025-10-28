# 价态修正功能调查报告

**日期**: 2025-10-28  
**问题**: 模型生成化学无效分子（价态错误）

---

## 🔍 调查发现

### 1. 问题根源

**当前生成方法** (`mol_from_graphs`):
- 直接将预测的原子和键添加到RDKit分子对象
- **没有任何化学约束检查**
- 可能产生价态错误（如C有5-8个键）

### 2. 代码中存在的修正功能

#### `correct_mol()` 函数
位置：`DiffMS/src/analysis/rdkit_functions.py`

```python
def correct_mol(m):
    mol = m
    
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:  # 价态正确
            break
        else:  # 价态错误
            # 找到价态错误的原子
            idx = atomid_valence[0]
            v = atomid_valence[1]
            
            # 找到该原子的所有键
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append((b.GetIdx(), type, ...))
            
            # 降低键级或移除键来修正价态
            # ...
            
    return mol, no_correct
```

**工作原理**：
1. 检查分子的价态
2. 如果有价态错误的原子，找到它的键
3. 移除或降低一个键的级别（3→2→1）
4. 重复直到所有原子价态正确

### 3. 为什么没有起作用？

#### 问题1: Mol对象不可编辑
```python
mol = mol_from_graphs(...)  # 返回Mol对象（只读）
corrected_mol = correct_mol(mol)  # 需要RWMol（可编辑）
```

**解决方案**：
```python
editable_mol = Chem.RWMol(mol)  # 转换为可编辑
corrected_mol = correct_mol(editable_mol)
```

#### 问题2: correct_mol可能返回None
如果价态错误太复杂，`correct_mol`可能无法修正，返回None。

#### 问题3: 严重的价态错误
生成的分子有多个原子价态严重错误（C(5), C(6), O(3)等），`correct_mol`可能无法处理。

---

## 🧪 测试结果

### 测试1: 添加correct_mol调用
- 修改：在`sample_batch`中添加价态修正
- 结果：Validity仍然 = 0%
- 原因：`correct_mol`可能无法修正严重的价态错误

### 测试2: 使用RWMol
- 修改：转换为RWMol对象
- 结果：待测试...

---

## 💡 可能的根本原因

### 假设1: 模型训练不足
- 模型本身预测的边就是错误的
- 训练数据不够或训练不充分

### 假设2: 配置不匹配
```yaml
# DiffMS/configs/dataset/msg.yaml
denoise_nodes: False  # ✅ 正确：只扩散边，不扩散节点
```

这个配置是正确的！意味着：
- 节点类型（原子）固定（来自分子式）
- 只预测边（键）

但如果边的预测质量很差，就会产生价态错误。

### 假设3: 数据不匹配

**论文测试的数据**：
- NPLIB1: 天然产物库
- MassSpecGym: 公共测试集
- 分子大小：~30-35原子

**我们的数据**：
- 自定义天然产物
- 分子大小：44-88原子（**超出训练范围**）

**影响**：模型可能对大分子的边预测质量很差。

---

## 🎯 下一步行动

### 优先级1: 测试官方数据
1. 获取NPLIB1或MassSpecGym官方测试集
2. 用官方数据测试validity
3. 如果官方数据validity>90%：说明是数据不匹配
4. 如果官方数据validity也低：说明是模型/checkpoint问题

### 优先级2: 检查checkpoint完整性
1. 验证checkpoint文件大小
2. 检查是否有缺失的权重
3. 尝试重新下载checkpoint

### 优先级3: 使用更小的测试分子
1. 筛选30-35原子的分子
2. 测试validity是否提高

---

## 📊 论文中的Validity

**Table 4 in DiffMS Paper**:
| Dataset | Validity | Top-1 | Top-10 |
|---------|----------|-------|--------|
| NPLIB1  | **100%** | 1.3%  | 4.9%   |
| MSG     | **100%** | 1.04% | 3.13%  |

**这说明**：
- 论文中模型生成的分子100%化学有效
- 要么模型预测边的质量很高
- 要么有有效的后处理修正

但我们的结果：
- Validity = 0%
- 即使添加了`correct_mol`仍然无效
- 说明生成的边质量极差

---

## 🔴 结论

**主要问题不是缺少价态修正，而是模型预测的边本身就很差**。

可能原因：
1. ❌ 数据不匹配（分子太大）
2. ❌ Checkpoint不完整/损坏
3. ❌ 配置不正确
4. ❌ Dummy graph创建方式不对

需要用**官方测试数据**验证模型本身是否正常。

