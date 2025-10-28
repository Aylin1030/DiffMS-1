# DiffMS推理运行指南

## 🚀 快速开始

### 1. 上传测试数据到Modal

```bash
# 上传前10条测试数据
modal volume put diffms-data /Users/aylin/yaolab_projects/madgen_yaolab/msdata/test_top10 /test_top10

# 验证上传
modal volume ls diffms-data /test_top10
```

### 2. 运行推理

```bash
# 测试前10条（使用test_top10数据）
modal run modal/diffms_inference.py::main --max-count 10 --data-subdir test_top10

# 运行全部数据（使用processed_data）
modal run modal/diffms_inference.py::main
```

## 📂 数据目录

### 测试数据 (10条)
**本地路径**: `/Users/aylin/yaolab_projects/madgen_yaolab/msdata/test_top10/`
**Modal路径**: `/test_top10/`

包含:
- `labels.tsv` - 分子式信息
- `split.tsv` - 数据划分（10条test）
- `spec_files/` - 质谱文件（需要从原数据复制）

### 完整数据 (475条)
**本地路径**: `/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data/`
**Modal路径**: `/processed_data/`

## 📊 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-count` | None | 限制处理样本数，None=全部 |
| `--data-subdir` | "processed_data" | 数据子目录名称 |

## 📈 核心改进

### ✅ 已修复：分子式约束

**修复前** (错误):
```
输入: C32H50O7 (39个重原子)
生成: 固定20个C原子 ❌
```

**修复后** (正确):
```
输入: C32H50O7
生成: 32C + 7O = 39个重原子 ✅

输入: C36H58O8
生成: 36C + 8O = 44个重原子 ✅
```

### 代码修改

1. **分子式解析** (`DiffMS/src/mist/data/featurizers.py`):
```python
@staticmethod
def parse_formula(formula_str: str) -> dict:
    """解析分子式: C32H50O7 → {'C': 32, 'H': 50, 'O': 7}"""
```

2. **动态Dummy Graph** (`DiffMS/src/mist/data/featurizers.py`):
```python
@staticmethod
def create_dummy_graph_from_formula(formula_str: str) -> Data:
    """根据分子式创建正确大小和元素组成的dummy graph"""
```

3. **推理时使用** (`DiffMS/src/mist/data/datasets.py`):
```python
formula = spec.get_spectra_formula()
dummy_graph = GraphFeaturizer.create_dummy_graph_from_formula(formula)
```

## 📥 查看结果

推理完成后：

```bash
# 下载结果
modal volume get diffms-output /preds ./modal_results

# 转换为表格
python modal/convert_to_table.py modal_results/*.pkl --output_dir results
```

结果文件:
- `results/predictions_top1.tsv` - Top-1预测
- `results/predictions_all_candidates.tsv` - 所有候选

## 🔍 验证分子式匹配

```bash
# 查看预测的原子数是否与输入分子式匹配
cat results/predictions_top1.tsv | awk -F'\t' '{print $1, $3, $4}' | column -t
```

预期输出示例:
```
spectrum_id  smiles                    num_atoms
0           CC(C)...C=O                32       # 应≈C32的重原子数
1           CC1CCC...O                 38       # 应≈C33H52O5的重原子数
```

## ⚙️ 配置调整

如需修改配置，编辑 `modal/diffms_inference.py`:

```python
# 采样数量（生成多少个候选）
cfg.general.test_samples_to_generate = 10

# GPU类型
gpu="A100"  # 或 "H100", "T4", "A10G"

# 超时时间
timeout=4 * HOURS
```

## 🆘 问题排查

### 问题1: 数据文件未找到
```bash
# 检查Modal volume内容
modal volume ls diffms-data /test_top10
modal volume ls diffms-data /processed_data
```

### 问题2: 分子式解析错误
查看日志中的分子式解析结果，确保格式正确（如 `C32H50O7`）

### 问题3: 生成分子与分子式不匹配
- 检查 `labels.tsv` 中的 `formula` 列是否正确
- 确认使用的是修复后的代码

## 📚 相关文档

- **修复说明**: `docs/FORMULA_CONSTRAINT_FIX_20251028.md`
- **总结**: `SUMMARY.md`
- **Modal指南**: `modal/README.md`
