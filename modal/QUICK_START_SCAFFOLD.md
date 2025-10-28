# 🚀 骨架约束推理 - 快速开始

**5分钟内完成骨架约束推理！**

---

## 📦 你的骨架信息

```
SMILES: CC(=CCCC(C1CCC2(C1(CCC3=C2CCC4C3(CCC(C4(C)C)O)C)C)C)C(=O)O)C
类型: 三萜类化合物骨架  
分子式: C30H48O3
重原子: 33个
```

---

## ⚡ 3步运行

### Step 1: 上传测试数据

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
bash upload_test_data.sh
```

### Step 2: 运行推理

```bash
modal run diffms_scaffold_inference.py
```

### Step 3: 下载结果

```bash
modal volume get diffms-outputs /outputs/smiles_scaffold ./results
```

**完成！** 🎉

---

## 📊 查看结果

```bash
# 查看 Top-1 预测
cat results/predictions_top1.tsv

# 查看包含骨架的候选分子
cat results/predictions_all_candidates.tsv | grep "True"
```

---

## 🔧 自定义骨架

### 使用苯环骨架（测试用）

```bash
modal run diffms_scaffold_inference.py \
    --scaffold-smiles "c1ccccc1" \
    --max-count 5
```

### 使用你自己的骨架

```bash
modal run diffms_scaffold_inference.py \
    --scaffold-smiles "YOUR_SMILES_HERE" \
    --max-count 10
```

---

## 📝 测试数据概览

你的测试数据（前10个样本）：

| Spec ID | 分子式 | 与骨架兼容? |
|---------|--------|------------|
| SPEC_4922 | C30H48O3 | ✅ 相同 |
| SPEC_6652 | C33H52O5 | ✅ 更大 |
| SPEC_4838 | C36H58O8 | ✅ 更大 |
| SPEC_5680 | C31H48O3 | ✅ 更大 |
| SPEC_6152 | C31H48O3 | ✅ 更大 |
| SPEC_9714 | C33H50O4 | ✅ 更大 |
| SPEC_5963 | C32H48O5 | ✅ 更大 |
| SPEC_7905 | C32H48O4 | ✅ 更大 |
| SPEC_10020 | C37H56O7 | ✅ 更大 |
| SPEC_6220 | C31H50O4 | ✅ 更大 |

**所有10个样本都与你的骨架兼容！** ✅

---

## ⏱️ 预期时间

- **上传数据**: ~30秒
- **推理（10个样本）**: ~2-3分钟
- **下载结果**: ~10秒

**总计**: < 5分钟

---

## 💰 成本

- **GPU**: A100 @ $1.10/hour
- **10个样本**: ~$0.05
- **完全可控**: 只在运行时收费

---

## 🆘 出问题了？

### 问题1: modal命令不存在

```bash
pip install modal
modal setup
```

### 问题2: Volume不存在

```bash
modal volume create diffms-data
modal volume create diffms-models
modal volume create diffms-outputs
```

### 问题3: 模型文件不存在

```bash
# 上传模型（如果还没有）
modal volume put diffms-models \
    /path/to/diffms_msg.ckpt \
    /models/diffms_msg.ckpt
```

---

## 📖 更多信息

- **完整指南**: `SCAFFOLD_INFERENCE_GUIDE.md`
- **技术文档**: `../docs/SCAFFOLD_CONSTRAINED_INFERENCE_20251028.md`

---

**准备好了吗？运行第一步！** 🏃‍♂️

```bash
bash upload_test_data.sh
```

