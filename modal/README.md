# DiffMS Modal 云端推理

使用Modal云平台运行DiffMS分子结构预测，支持A100/H100等高性能GPU。

## 🚀 快速开始

### 1. 上传数据（首次运行）
```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal
./upload_to_modal.sh
```

### 2. 验证上传
```bash
modal run check_modal_volumes.py
```

### 3. 运行推理
```bash
# 测试运行（10个样本）
modal run diffms_inference.py --max-count 10

# 完整推理（478个样本）
modal run diffms_inference.py
```

### 4. 下载结果
```bash
modal volume get diffms-outputs /outputs ./modal_outputs
```

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `diffms_inference.py` | 主推理脚本 |
| `check_modal_volumes.py` | Volume验证脚本 |
| `upload_to_modal.sh` | 数据上传脚本 |
| `MODAL_GUIDE.md` | 详细使用指南和问题排查 |
| `README.md` | 本文件 |

## 🔧 主要修复

相比初始版本，修复了以下问题：

1. ✅ **模型加载**: 改用`load_from_checkpoint()`替代手动load_state_dict
2. ✅ **特征顺序**: 修正domain_features和extra_features创建顺序
3. ✅ **文件检查**: 启动时验证所有必需文件
4. ✅ **错误处理**: 添加详细的try-catch和日志
5. ✅ **GPU检测**: 改进GPU可用性检查逻辑

## 💡 使用建议

- **首次使用**: 先运行`--max-count 10`测试
- **GPU选择**: A100（40GB显存）适合大多数场景，H100更快
- **批量处理**: 可以分批运行避免超时
- **查看日志**: `modal app logs diffms-inference`

## 📚 详细文档

遇到问题？查看 [MODAL_GUIDE.md](MODAL_GUIDE.md) 获取：
- 详细的问题排查步骤
- Volume结构说明
- 性能优化建议
- 常见错误解决方案

## 🔗 相关项目

- **本地推理**: `/Users/aylin/yaolab_projects/diffms_yaolab/inference/`
- **数据预处理**: `/Users/aylin/yaolab_projects/madgen_yaolab/msdata/`

## 📝 版本历史

- **2025-10-28**: 修复模型加载和特征提取器问题
- **初始版本**: 基础Modal推理实现
