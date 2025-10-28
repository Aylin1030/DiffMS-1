# 🔧 故障排除指南

## ✅ 已修复：RDKit 导入错误

### 问题
```
ImportError: cannot import name 'rdMolOps' from 'rdkit.Chem'
```

### 原因
RDKit 的 `rdMolOps` 不能直接从 `rdkit.Chem` 导入。

### 解决方案
已修复 `DiffMS/src/inference/scaffold_hooks.py`：

**修改前**:
```python
from rdkit.Chem import rdFMCS, rdMolOps  # ❌
```

**修改后**:
```python
from rdkit import Chem  # ✅
# 直接使用 Chem.Mol.GetSubstructMatch 等方法
```

### 现在可以运行

```bash
modal run diffms_scaffold_inference.py
```

---

## 常见问题

### 问题 1: "Checkpoint文件不存在"

```
FileNotFoundError: Checkpoint文件不存在: /models/diffms_msg.ckpt
```

**解决**:
```bash
# 检查模型是否存在
modal volume ls diffms-models /models/

# 如果不存在，上传模型
modal volume put diffms-models \
    /path/to/your/diffms_msg.ckpt \
    /models/diffms_msg.ckpt
```

### 问题 2: "数据目录不存在"

```
FileNotFoundError: 数据目录不存在: /data/test_top10
```

**解决**:
```bash
# 上传测试数据
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
bash upload_test_data.sh
```

### 问题 3: "骨架与所有样本的分子式都不兼容"

```
ValueError: 骨架与所有目标分子式不兼容
```

**原因**: 骨架的分子式大于所有样本的分子式

**解决**: 使用更小的骨架
```bash
# 例如，使用苯环代替三萜
modal run diffms_scaffold_inference.py --scaffold-smiles "c1ccccc1"
```

### 问题 4: Hydra 配置错误

```
omegaconf.errors.ConfigAttributeError: Key 'scaffold_smiles' is not in struct
```

**解决**: 脚本已包含 `OmegaConf.set_struct(cfg, False)`，允许添加新字段。

如果仍有问题，检查 Hydra 版本：
```bash
pip show hydra-core
```

应该是 1.3.2 或更高版本。

---

## 验证步骤

### 1. 检查环境

```bash
# Modal CLI
modal --version

# 登录状态
modal token show
```

### 2. 检查 Volumes

```bash
# 数据 volume
modal volume ls diffms-data

# 模型 volume
modal volume ls diffms-models

# 输出 volume
modal volume ls diffms-outputs
```

### 3. 测试小批量

```bash
# 先测试 1 个样本
modal run diffms_scaffold_inference.py --max-count 1
```

---

## 调试技巧

### 查看详细日志

Modal 会自动显示日志。如果需要保存：

```bash
modal run diffms_scaffold_inference.py 2>&1 | tee inference.log
```

### 进入容器调试

```bash
# 启动交互式容器（需要修改脚本添加 shell）
modal shell diffms-scaffold-inference
```

### 检查文件是否正确挂载

在脚本开始时添加：

```python
import os
print("Files in /root/src/inference:")
print(os.listdir("/root/src/inference"))
```

---

## 性能优化

### 减少采样数量（测试用）

```python
cfg.general.test_samples_to_generate = 5  # 默认是 10
```

### 使用更小的 GPU

如果 A100 太贵，可以用 T4：

```python
@app.function(
    gpu="T4",  # 更便宜
    timeout=2 * HOURS,
)
```

---

## 成功标志

运行成功后应该看到：

```
✓ 骨架验证成功
✓ 10/10 个样本与骨架兼容
✓ 推理完成！
✓ 后处理完成！

结果统计:
  有效SMILES: 95/100 (95.0%)
  包含骨架: 87/100 (87.0%)

✅ 骨架约束推理全部完成！
```

---

## 联系支持

如果以上都无法解决问题：

1. 检查 Modal 平台状态: https://modal.com/status
2. 查看 Modal 文档: https://modal.com/docs
3. 检查 DiffMS 原始仓库: https://github.com/coleygroup/DiffMS

---

**最后更新**: 2024-10-28  
**状态**: RDKit 导入错误已修复 ✅

