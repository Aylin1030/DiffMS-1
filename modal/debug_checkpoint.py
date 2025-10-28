"""
检查checkpoint结构和内容
"""
import torch
from pathlib import Path

def inspect_checkpoint(ckpt_path):
    """详细检查checkpoint内容"""
    print("=" * 80)
    print(f"检查Checkpoint: {ckpt_path}")
    print("=" * 80)
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 1. 顶层keys
    print("\n1. Checkpoint顶层keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  - {key}: dict with {len(checkpoint[key])} items")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  - {key}: Tensor {checkpoint[key].shape}")
        else:
            print(f"  - {key}: {type(checkpoint[key])}")
    
    # 2. state_dict结构
    if 'state_dict' in checkpoint:
        print("\n2. state_dict keys (前20个):")
        state_dict_keys = list(checkpoint['state_dict'].keys())
        for i, key in enumerate(state_dict_keys[:20]):
            shape = checkpoint['state_dict'][key].shape if isinstance(checkpoint['state_dict'][key], torch.Tensor) else "N/A"
            print(f"  {i+1:2d}. {key}: {shape}")
        
        if len(state_dict_keys) > 20:
            print(f"  ... (总共 {len(state_dict_keys)} 个keys)")
        
        # 3. 检查decoder/encoder权重
        print("\n3. Decoder权重 (以'decoder.'开头):")
        decoder_keys = [k for k in state_dict_keys if k.startswith('decoder.')]
        print(f"  找到 {len(decoder_keys)} 个decoder权重")
        if decoder_keys:
            for key in decoder_keys[:5]:
                print(f"    - {key}")
        
        print("\n4. Encoder权重 (以'encoder.'开头):")
        encoder_keys = [k for k in state_dict_keys if k.startswith('encoder.')]
        print(f"  找到 {len(encoder_keys)} 个encoder权重")
        if encoder_keys:
            for key in encoder_keys[:5]:
                print(f"    - {key}")
        
        # 4. 检查关键维度
        print("\n5. 关键维度信息:")
        for key in state_dict_keys:
            # 查找输入/输出层的维度
            if 'mlp_in_X' in key or 'mlp_in_E' in key or 'mlp_in_y' in key:
                print(f"  {key}: {checkpoint['state_dict'][key].shape}")
            if 'mlp_out_X' in key or 'mlp_out_E' in key or 'mlp_out_y' in key:
                print(f"  {key}: {checkpoint['state_dict'][key].shape}")
    
    # 5. hyper_parameters（如果存在）
    if 'hyper_parameters' in checkpoint:
        print("\n6. Hyper Parameters:")
        hparams = checkpoint['hyper_parameters']
        if isinstance(hparams, dict):
            for key in ['name', 'Xdim', 'Edim', 'ydim', 'Xdim_output', 'Edim_output', 'ydim_output']:
                if key in hparams:
                    print(f"  {key}: {hparams[key]}")
    
    print("\n" + "=" * 80)
    return checkpoint

if __name__ == "__main__":
    # 本地checkpoint路径
    ckpt_path = Path("/Users/aylin/Downloads/checkpoints/diffms_msg.ckpt")
    
    if ckpt_path.exists():
        checkpoint = inspect_checkpoint(ckpt_path)
        
        # 额外：检查是否包含完整模型
        print("\n额外检查:")
        print(f"  - 是否包含optimizer_states: {'optimizer_states' in checkpoint}")
        print(f"  - 是否包含lr_schedulers: {'lr_schedulers' in checkpoint}")
        print(f"  - 是否包含epoch: {'epoch' in checkpoint}")
        if 'epoch' in checkpoint:
            print(f"    Epoch: {checkpoint['epoch']}")
    else:
        print(f"Checkpoint文件不存在: {ckpt_path}")

