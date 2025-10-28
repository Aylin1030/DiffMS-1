# DiffMS 质谱分子结构预测

基于DiffMS模型在Modal云端运行分子结构推理，支持分子式约束生成。

## 🚀 快速开始

### 1. 运行推理（Modal云端）

```bash
# 测试运行（前10个样本）
cd modal
modal run diffms_inference.py --data-subdir test_top10

# 完整运行
modal run diffms_inference.py
```

### 2. 查看结果

结果保存在Modal的`diffms-outputs` volume中，也可以查看本地的`modal/results/`目录。

详细指南见：**`RUN_INFERENCE.md`**

## 📁 项目结构

```
.
├── DiffMS/              # DiffMS模型源代码
├── modal/               # Modal云端推理脚本
│   ├── diffms_inference.py
│   ├── upload_to_modal.sh
│   └── results/
├── docs/                # 技术文档
├── RUN_INFERENCE.md    # 运行指南
└── SUMMARY.md          # 项目总结
```

## ✨ 关键特性

- **分子式约束**: 生成的分子严格符合输入的分子式
- **云端推理**: 使用Modal A100 GPU进行高效推理
- **批量处理**: 支持批量质谱数据处理
- **多候选生成**: 每个样本生成10个候选分子（同分异构体）

## 🔧 最近更新

**2024-10-28**: 修复分子式约束问题
- 修复了采样算法，确保生成的分子符合输入分子式
- 详见 `docs/FORMULA_CONSTRAINT_FIX_20251028.md`

## 📚 文档

- **运行指南**: `RUN_INFERENCE.md`
- **项目总结**: `SUMMARY.md`
- **技术文档**: `docs/`目录

## 🎯 使用场景

- 从质谱数据预测未知分子结构
- 代谢组学中的化合物鉴定
- 天然产物结构解析

## 📖 参考

- 论文: [DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra](https://arxiv.org/html/2502.09571v2)
- 原始代码: [DiffMS GitHub](https://github.com/coleygroup/DiffMS)

---

**维护**: Yao Lab  
**最后更新**: 2024-10-28

