# Molecule Visualization Tool

用于DiffMS预测结果的分子结构可视化和分析工具。

## 📋 快速开始

### 一键启动
```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/visualization
./run.sh
```

或者手动启动：
```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab
uv run streamlit run visualization/app.py
```

浏览器会自动打开到 http://localhost:8501

## ⚠️ 重要发现

**只有1%的SMILES可以被成功解析！**

验证结果显示，100条预测中仅1条SMILES在化学上是有效的。其余99条包含化学错误（如原子价错误）。

详细分析请查看：`../docs/SMILES_VALIDATION_REPORT_20251028.md`

## 🎯 功能特性

### 1. 交互式Web界面
- 🔬 实时生成分子结构图
- 📊 统计信息展示
- 🔍 多维度筛选
- 📈 分子性质计算

### 2. 两种视图模式

#### Grid View（网格视图）
- 同时查看多个分子
- 可调节每行列数
- 适合快速浏览

#### Detailed View（详细视图）
- 单个分子详细信息
- SMILES字符串展示
- 分子属性计算
- 错误信息提示

### 3. 智能处理
- 自动尝试解析SMILES
- 即时生成分子图
- 清晰的错误提示
- 化学有效性验证

## 📁 文件结构

```
visualization/
├── app.py                  # Streamlit主应用
├── generate_images.py      # 批量生成分子图
├── models.py              # Pydantic数据模型
├── run.sh                 # 一键启动脚本
├── QUICK_START.md         # 快速开始指南
├── README.md              # 本文件
└── molecule_images/       # 生成的图片目录
    └── metadata.csv       # 元数据
```

## 🔍 UI功能说明

### 侧边栏筛选器
- **Spectrum ID**: 选择特定的谱图ID或查看全部
- **Rank Range**: 按排名范围筛选
- **Valid Only**: 仅显示标记为valid的分子

### 统计指标
- **Total Predictions**: 符合筛选条件的总预测数
- **Unique Spectra**: 独特的谱图数量
- **Valid Molecules**: 标记为valid的分子数
- **Parseable SMILES**: 可被RDKit解析的分子数（关键指标）
- **Avg Atoms**: 平均原子数

### 分子属性
对于化学有效的分子，显示：
- Molecular Weight (分子量)
- LogP (脂溶性)
- H-Bond Donors (氢键供体)
- H-Bond Acceptors (氢键受体)
- Rotatable Bonds (可旋转键)
- Aromatic Rings (芳香环)
- TPSA (拓扑极性表面积)

## 💻 技术栈

- **Python 3.12+**
- **RDKit** 2025.9.1 - 分子处理和可视化
- **Streamlit** 1.50.0 - Web界面
- **Pandas** 2.3.3 - 数据处理
- **Pillow** 11.3.0 - 图像处理
- **Pydantic** 2.12.3 - 数据验证

## 🐛 常见问题

### Q: 为什么大部分分子显示"Cannot parse SMILES"？
A: 这是正常现象。模型生成的99%的SMILES包含化学错误（如原子价违规），无法被RDKit解析。这表明模型需要改进。

### Q: "valid=True"是什么意思？
A: 这个标记可能只是指图结构完整性或其他内部验证，**不代表化学有效性**。

### Q: 如何找到化学有效的分子？
A: 查看"Parseable SMILES"指标，或在详细视图中逐个检查。能成功显示分子结构图的就是有效的。

### Q: 可以导出结果吗？
A: 可以。在页面底部有"Download Filtered Data"按钮，可导出CSV格式。

## 📊 批量生成图片

如果需要提前生成所有图片（虽然成功率只有1%）：

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab
uv run python visualization/generate_images.py
```

这会：
1. 读取 `results/predictions_all_candidates.tsv`
2. 尝试为每个SMILES生成PNG图片
3. 保存到 `visualization/molecule_images/` 目录
4. 创建 `metadata.csv` 元数据文件

## 🔬 建议

基于验证结果，建议：

1. **重新评估模型性能**
   - 添加化学有效性作为关键指标
   - 不能仅依赖结构相似度

2. **改进模型训练**
   - 添加化学约束
   - 验证训练数据质量
   - 引入后处理步骤

3. **更新验证pipeline**
   - 使用RDKit验证所有生成的SMILES
   - 统计并分析错误类型
   - 设置化学有效性阈值

## 📚 相关文档

- **详细分析报告**: `../docs/SMILES_VALIDATION_REPORT_20251028.md`
- **快速开始**: `QUICK_START.md`
- **模型信息**: `../DiffMS/`

## 📞 获取帮助

遇到问题？
1. 查看 `QUICK_START.md`
2. 阅读详细分析报告
3. 检查终端输出的错误信息
4. 确保所有依赖已正确安装

## 📝 许可证

遵循项目主许可证。
