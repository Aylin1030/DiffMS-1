#!/usr/bin/env python
"""
环境测试脚本 - 验证所有依赖是否正确安装
"""

import sys
from pathlib import Path

# 添加DiffMS到路径
diffms_path = Path(__file__).parent.parent / "DiffMS"
sys.path.insert(0, str(diffms_path))

print("=" * 80)
print("DiffMS 环境测试")
print("=" * 80)

# 测试1: 基础导入
print("\n1️⃣  测试基础模块导入...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"  ✗ PyTorch导入失败: {e}")
    sys.exit(1)

# 测试2: PyTorch Lightning
print("\n2️⃣  测试PyTorch Lightning...")
try:
    from pytorch_lightning import Trainer
    import pytorch_lightning as pl
    print(f"  ✓ PyTorch Lightning {pl.__version__}")
except Exception as e:
    print(f"  ✗ PyTorch Lightning导入失败: {e}")
    sys.exit(1)

# 测试3: RDKit
print("\n3️⃣  测试RDKit...")
try:
    from rdkit import Chem
    print(f"  ✓ RDKit导入成功")
except Exception as e:
    print(f"  ✗ RDKit导入失败: {e}")
    sys.exit(1)

# 测试4: DiffMS模块
print("\n4️⃣  测试DiffMS模块...")
try:
    # 切换到src目录（很多配置文件在这里）
    import os
    os.chdir(str(diffms_path / "src"))
    
    from src import utils
    print(f"  ✓ src.utils")
    
    from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
    print(f"  ✓ Spec2MolDenoisingDiffusion")
    
    from src.datasets import spec2mol_dataset
    print(f"  ✓ spec2mol_dataset")
    
except Exception as e:
    print(f"  ✗ DiffMS模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 检查文件
print("\n5️⃣  检查必需文件...")
checkpoint = Path("/Users/aylin/Downloads/checkpoints/diffms_msg.ckpt")
if checkpoint.exists():
    size_gb = checkpoint.stat().st_size / (1024**3)
    print(f"  ✓ Checkpoint: {checkpoint} ({size_gb:.2f} GB)")
else:
    print(f"  ✗ Checkpoint不存在: {checkpoint}")

data_dir = Path("/Users/aylin/yaolab_projects/madgen_yaolab/msdata/processed_data")
if data_dir.exists():
    print(f"  ✓ 数据目录: {data_dir}")
else:
    print(f"  ✗ 数据目录不存在: {data_dir}")

# 测试6: 快速加载配置
print("\n6️⃣  测试加载checkpoint配置...")
try:
    checkpoint_data = torch.load(str(checkpoint), map_location='cpu')
    if 'hyper_parameters' in checkpoint_data:
        print(f"  ✓ Checkpoint包含hyper_parameters")
        if 'cfg' in checkpoint_data['hyper_parameters']:
            print(f"  ✓ Checkpoint包含cfg配置")
        else:
            print(f"  ⚠ Checkpoint的hyper_parameters中没有cfg")
    else:
        print(f"  ✗ Checkpoint没有hyper_parameters字段")
        print(f"  可用的键: {list(checkpoint_data.keys())}")
except Exception as e:
    print(f"  ✗ 加载checkpoint失败: {e}")

print("\n" + "=" * 80)
print("✅ 环境测试完成！所有基础模块都能正常导入。")
print("=" * 80)
print("\n下一步:")
print("  - 本地推理脚本现在应该能运行（即使在CPU上也能加载模型）")
print("  - Modal版本也应该能正常工作")

