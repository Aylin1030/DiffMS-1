"""
检查Modal image是否包含所有必要的依赖
"""
import modal

app = modal.App("check-dependencies")

# 使用与diffms_inference.py相同的image配置
from pathlib import Path

DIFFMS_PATH = Path("/Users/aylin/yaolab_projects/diffms_yaolab/DiffMS")
DIFFMS_SRC_PATH = DIFFMS_PATH / "src"
DIFFMS_CONFIGS_PATH = DIFFMS_PATH / "configs"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget",
        "libxrender1", "libxext6", "libsm6", "libice6", "libx11-6", "libglib2.0-0"
    )
    .pip_install(
        "torch==2.0.1", "torchvision==0.15.2",
    )
    .pip_install(
        "torch-scatter==2.1.1", "torch-sparse==0.6.17",
    )
    .pip_install(
        "torch-geometric==2.3.1",
        "pytorch-lightning==2.0.0",
        "rdkit==2023.3.2",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "tqdm==4.65.0",
        "h5py==3.9.0",
        "networkx==3.1",
        "wandb",
        "matplotlib",
        "seaborn",
    )
    .add_local_dir(str(DIFFMS_SRC_PATH), "/root/src")
    .add_local_dir(str(DIFFMS_CONFIGS_PATH), "/root/configs")
)

@app.function(image=image)
def check_imports():
    """检查所有必要的包是否可以导入"""
    import sys
    results = []
    
    # 核心依赖
    packages = [
        ("torch", "PyTorch"),
        ("rdkit.Chem", "RDKit"),
        ("rdkit.Chem.Draw", "RDKit Draw"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("torch_geometric", "PyTorch Geometric"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            results.append(f"✓ {name}: OK")
        except ImportError as e:
            results.append(f"✗ {name}: FAILED - {e}")
    
    # 测试RDKit Draw功能
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        # 测试创建简单分子
        mol = Chem.MolFromSmiles("CCO")
        
        # 测试Draw功能
        img = Draw.MolToImage(mol, size=(200, 200))
        results.append(f"✓ RDKit Draw.MolToImage: OK")
        
        # 测试网格图功能
        img = Draw.MolsToGridImage([mol, mol], molsPerRow=2, subImgSize=(200, 200))
        results.append(f"✓ RDKit Draw.MolsToGridImage: OK")
        
    except Exception as e:
        results.append(f"✗ RDKit Draw: FAILED - {e}")
    
    # 测试图片保存
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import tempfile
        
        mol = Chem.MolFromSmiles("CCO")
        img = Draw.MolToImage(mol)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            results.append(f"✓ 图片保存: OK ({f.name})")
            
    except Exception as e:
        results.append(f"✗ 图片保存: FAILED - {e}")
    
    return "\n".join(results)

@app.local_entrypoint()
def main():
    print("=" * 60)
    print("检查Modal Image依赖")
    print("=" * 60)
    result = check_imports.remote()
    print(result)
    print("=" * 60)

