# 🚀 立即开始

## 方法1：使用启动脚本（推荐）

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/visualization
./run.sh
```

## 方法2：手动启动

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab
uv run streamlit run visualization/app.py
```

## 访问界面

浏览器会自动打开，或手动访问：
👉 **http://localhost:8501**

## 停止服务

在终端按 `Ctrl + C`

---

## ⚠️ 重要提示

- **必须使用 `uv run`** - 这确保在正确的虚拟环境中运行
- 首次运行会自动生成分子图片（需要几秒钟）
- 99%的SMILES无法解析是正常的（详见分析报告）

## 📚 更多信息

- 功能说明: `README.md`
- 快速指南: `QUICK_START.md`
- 详细分析: `../docs/SMILES_VALIDATION_REPORT_20251028.md`

