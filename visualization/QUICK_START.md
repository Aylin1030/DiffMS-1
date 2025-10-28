# 快速开始指南

## 🚀 一键启动

最简单的方式是运行启动脚本：

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/visualization
./run.sh
```

或者手动启动：

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab
uv run streamlit run visualization/app.py
```

**重要**: 必须使用 `uv run` 来确保在正确的虚拟环境中运行！

浏览器会自动打开 http://localhost:8501

## 📊 关于结果

**重要发现**：生成的SMILES中99%包含化学错误（如原子价错误），无法被RDKit解析。这表明：

1. **扩散模型生成的分子结构不合理** - 大多数预测的分子违反了化学规则
2. **只有约1%的SMILES是化学有效的** - 可以被RDKit正确解析并可视化
3. **所有"valid=True"的标记可能指的是其他验证** - 不是化学有效性验证

### 典型错误类型：
- `Explicit valence for atom X is greater than permitted` - 原子价超出允许范围
- `Can't kekulize mol` - 无法进行芳香性处理
- 其他化学结构不合理的错误

## 🔍 UI功能

### 侧边栏筛选
- 选择特定的Spectrum ID
- 按Rank范围筛选
- 仅显示有效分子（但大部分仍无法解析）

### 显示模式

#### 1. Grid View（网格视图）
- 一次查看多个分子
- 调整每行列数（2-5列）
- 快速浏览所有预测

#### 2. Detailed View（详细视图）
- 选择单个分子查看
- 显示分子结构图（如果可解析）
- 查看SMILES字符串
- 显示预测详情
- 计算分子属性（对于有效分子）

### 统计信息
- Total Predictions: 总预测数
- Unique Spectra: 独特谱图数量
- Valid Molecules: 标记为valid的分子数
- **Parseable SMILES**: 可以被RDKit解析的分子数（关键指标）
- Avg Atoms: 平均原子数

## ⚠️ 问题诊断

如果大部分分子显示"❌ Cannot parse SMILES"：
- 这是**正常的** - 说明模型生成的分子不符合化学规则
- 建议检查：
  - 模型训练是否正确
  - 是否需要后处理步骤来修正SMILES
  - 评估指标是否包含化学有效性检查

## 💡 建议

1. **重点关注可解析的1%分子** - 这些才是化学有效的
2. **分析失败原因** - 统计哪些类型的错误最常见
3. **考虑添加化学约束** - 在模型训练或生成过程中
4. **使用化学有效性作为评估指标** - 不仅仅是结构相似度

## 📝 导出数据

在数据表底部可以下载筛选后的结果为CSV文件。

