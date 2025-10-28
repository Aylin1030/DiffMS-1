# DiffMS推理快速修正参考

**目标**: 输出正确的canonical SMILES，而非"乱码"

---

## 🔧 关键修正（按建议清单）

### ✓ 检查点1: Checkpoint内容
```bash
python modal/debug_checkpoint.py
```
**验证**: ✓ 包含366个参数（304个decoder + 59个encoder）

### ✓ 检查点2 & 3: 配置字段
```python
# modal/diffms_inference.py

# ✗ 错误方式
cfg.general.test_only = str(checkpoint_path)  # 不要这样做
cfg.general.decoder = checkpoint_path         # 会重复加载

# ✓ 正确方式
cfg.general.test_only = True                  # 布尔值
cfg.general.decoder = None                     # checkpoint已包含
cfg.general.encoder = None                     # checkpoint已包含
```

### ✓ 检查点4: Formula字段
```tsv
# labels.tsv必须包含
spec                    formula         smiles          ...
MassSpecGymID0000201   C45H57N3O9      <可为空>        ...
```
**验证**: ✓ 所有谱图都有formula，格式正确

### ✓ 检查点5: Mol→SMILES转换 (最关键！)

**问题**: pkl文件包含Mol对象，需要转换为SMILES字符串

**解决方案**:
```python
# modal/convert_predictions_to_smiles.py

def mol_to_canonical_smiles(mol):
    """论文要求：canonical SMILES (无立体化学)"""
    if mol is None:
        return None
    
    # 1. 移除立体化学
    Chem.RemoveStereochemistry(mol)
    
    # 2. 转为canonical SMILES
    smiles = Chem.MolToSmiles(mol, canonical=True)
    
    # 3. 验证
    if Chem.MolFromSmiles(smiles) is None:
        return None
    
    return smiles
```

**使用**:
```bash
# 1. 推理生成pkl文件
modal run diffms_inference.py

# 2. 转换为SMILES (关键步骤！)
python modal/convert_predictions_to_smiles.py

# 3. 验证输出
# 输出: results_smiles/predictions_top1.tsv
#       results_smiles/predictions_all_candidates.tsv
```

### ✓ 检查点6: 路径与工作目录
```python
# 已在diffms_inference.py中设置
os.chdir("/root/src")           # 工作目录
sys.path.insert(0, "/root/src") # Python路径
```

### ✓ 检查点7: 版本兼容
```python
# Modal image配置
torch==2.0.1
rdkit==2023.3.2
pytorch-lightning==2.0.0
```

---

## 📋 完整推理流程（3步）

```bash
# 步骤1: 运行推理（生成pkl）
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
modal run diffms_inference.py --data-subdir msg_official_test5

# 步骤2: 转换为SMILES（关键！）
python convert_predictions_to_smiles.py

# 步骤3: 验证输出
python -c "
import pandas as pd
from rdkit import Chem

df = pd.read_csv('results_smiles/predictions_top1.tsv', sep='\t')
invalid = 0
for idx, row in df.iterrows():
    if not pd.isna(row['smiles']) and row['smiles'] != '':
        if Chem.MolFromSmiles(row['smiles']) is None:
            invalid += 1
            print(f'Invalid: {row[\"smiles\"]}')

print(f'Total: {len(df)}, Invalid: {invalid}')
"
```

---

## ⚠️ 常见错误

### 错误1: "乱码"输出
```python
# ✗ 错误：直接使用pkl
predictions = pickle.load(open('pred.pkl', 'rb'))
# 输出：<rdkit.Chem.rdchem.Mol object at 0x...>  # 这不是SMILES！

# ✓ 正确：转换为SMILES
mol = predictions[0][0]
smiles = Chem.MolToSmiles(mol, canonical=True)
# 输出："CCO"  # 这是正确的SMILES
```

### 错误2: 维度不匹配
```python
# ✓ 使用checkpoint的固定维度
dataset_infos.input_dims = {'X': 16, 'E': 5, 'y': 2061}
dataset_infos.output_dims = {'X': 8, 'E': 5, 'y': 2048}
```

### 错误3: 立体化学未移除
```python
# ✗ 错误：保留立体化学
smiles = Chem.MolToSmiles(mol, canonical=True)
# 可能输出："CC[C@@H](O)C"  # 包含@符号

# ✓ 正确：移除立体化学
Chem.RemoveStereochemistry(mol)
smiles = Chem.MolToSmiles(mol, canonical=True)
# 输出："CCC(O)C"  # 无立体化学
```

---

## 🎯 核心要点

| 要点 | 说明 |
|------|------|
| **输出不是SMILES** | pkl文件包含Mol对象，需要转换！ |
| **必须转换** | 使用 `convert_predictions_to_smiles.py` |
| **Canonical** | `Chem.MolToSmiles(mol, canonical=True)` |
| **无立体化学** | `Chem.RemoveStereochemistry(mol)` |
| **验证有效性** | `Chem.MolFromSmiles(smiles) is not None` |

---

## 📊 期望输出格式

```tsv
spec_id                 smiles
spec_0000              CCO
spec_0001              CC(C)O
spec_0002              CCCC
...
```

**不是**:
```
<rdkit.Chem.rdchem.Mol object at 0x...>
b'\x80\x03crdkit.Chem.rdchem\nMol\n...'
```

---

## ✅ 验证检查表

- [ ] Checkpoint包含encoder和decoder权重
- [ ] `cfg.general.test_only = True` (布尔值)
- [ ] `cfg.general.decoder = None` (避免重复加载)
- [ ] labels.tsv包含formula字段
- [ ] 运行了 `convert_predictions_to_smiles.py`
- [ ] 输出是TSV文件，包含spec_id和smiles列
- [ ] 所有SMILES都是字符串，不是对象
- [ ] 所有SMILES都可以被RDKit解析
- [ ] SMILES是canonical格式
- [ ] SMILES无立体化学(@符号)

---

**核心提醒**: 
1. **pkl → SMILES转换是必须的！**
2. **使用 `convert_predictions_to_smiles.py`**
3. **验证输出是字符串，不是对象**

---

生成日期: 2024-10-28

