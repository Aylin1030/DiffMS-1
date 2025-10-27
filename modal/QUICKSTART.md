# 快速开始 - Modal云端推理

## 3分钟快速启动

### 1️⃣ 安装并认证Modal

```bash
# 安装Modal CLI
pip install modal

# 认证（会打开浏览器）
modal setup
```

### 2️⃣ 上传数据和模型

```bash
cd /Users/aylin/yaolab_projects/diffms_yaolab/modal

# 运行自动化上传脚本
bash setup_volumes.sh
```

等待上传完成（约5-10分钟，取决于网速）

### 3️⃣ 运行测试推理

```bash
# 测试10个数据点
modal run diffms_inference.py --max-count 10
```

### 4️⃣ 下载结果

```bash
# 下载所有输出到本地
modal volume get diffms-outputs /outputs ./local_outputs
```

---

## 完整推理

处理所有478个数据点：

```bash
modal run diffms_inference.py
```

---

## 文件说明

```
modal/
├── diffms_inference.py    # Modal推理主程序
├── setup_volumes.sh       # 数据上传脚本
├── README.md             # 详细说明文档
├── SETUP.md              # 详细配置指南
└── QUICKSTART.md         # 本文档
```

---

## 常用命令

```bash
# 查看已上传的数据
modal volume ls diffms-data /data

# 查看已上传的模型
modal volume ls diffms-models /models

# 查看推理结果
modal volume ls diffms-outputs /outputs

# 删除并重新上传（如需更新）
modal volume delete diffms-data
bash setup_volumes.sh
```

---

## 注意事项

✅ **首次运行会构建Docker镜像**，需要约5-10分钟  
✅ **后续运行会复用镜像**，启动只需30秒  
✅ **数据只需上传一次**，永久保存在Modal Volumes  
✅ **按使用付费**，不运行不计费  

---

## 帮助

- 详细配置：查看 `SETUP.md`
- 完整文档：查看 `README.md`
- Modal官方文档：https://modal.com/docs

