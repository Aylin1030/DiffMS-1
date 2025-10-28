#!/usr/bin/env python
"""检查checkpoint文件结构"""
import torch
import sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/aylin/Downloads/checkpoints/diffms_msg.ckpt"

print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"\n✓ Checkpoint loaded successfully")
print(f"\n顶层键 (Top-level keys):")
for key in ckpt.keys():
    print(f"  - {key}")

if 'hyper_parameters' in ckpt:
    hp = ckpt['hyper_parameters']
    print(f"\nhyper_parameters 类型: {type(hp)}")
    if isinstance(hp, dict):
        print(f"hyper_parameters 键:")
        for key in hp.keys():
            print(f"  - {key}: {type(hp[key])}")
        
        if 'cfg' in hp:
            print(f"\n✓ 找到cfg配置")
        else:
            print(f"\n✗ 没有找到cfg配置")
            print(f"hyper_parameters的所有键: {list(hp.keys())}")
else:
    print(f"\n✗ 没有hyper_parameters键")

print(f"\n文件大小: {ckpt_path.stat().st_size / (1024**3):.2f} GB" if hasattr(ckpt_path, 'stat') else "")

