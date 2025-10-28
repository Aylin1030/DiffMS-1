"""
Scaffold-constrained inference hooks for DiffMS.
这个模块提供"骨架冻结/掩码/同构守护"的纯推理工具函数。
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable
import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


class Formula(Dict[str, int]):
    """分子式表示，例如 {'C': 6, 'H': 12, 'O': 6}"""
    pass


def smiles_to_mol(smiles: str) -> Mol:
    """将SMILES字符串转换为RDKit分子对象"""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return m


def formula_of(mol: Mol) -> Formula:
    """根据RDKit统计重原子元素计数，返回 {elem: count}"""
    f: Formula = Formula()
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym != "H":  # 只统计重原子
            f[sym] = f.get(sym, 0) + 1
    return f


def formula_subtract(F: Formula, G: Formula) -> Formula:
    """计算化学式差值 ΔF = F - G"""
    out: Formula = Formula(F)
    for k, v in G.items():
        out[k] = out.get(k, 0) - v
        if out[k] < 0:
            raise ValueError(f"ΔF negative for element {k}: {out[k]}")
    return out


def formula_to_string(f: Formula) -> str:
    """将化学式转换为字符串表示"""
    if not f:
        return "empty"
    return "".join(f"{elem}{count}" for elem, count in sorted(f.items()))


def parse_formula(formula_str: str) -> Formula:
    """
    解析化学式字符串，例如 'C6H12O6' -> {'C': 6, 'H': 12, 'O': 6}
    只提取重原子（非H）
    """
    import re
    f = Formula()
    # 匹配元素符号和数字
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula_str)
    for elem, count_str in matches:
        if elem and elem != 'H':  # 跳过氢
            count = int(count_str) if count_str else 1
            f[elem] = f.get(elem, 0) + count
    return f


def apply_formula_mask_to_logits(
    node_logits: torch.Tensor,
    remaining_formula: Formula,
    atom_type_vocab: List[str]
) -> torch.Tensor:
    """
    对节点logits应用化学式掩码：把不允许新增的元素（ΔF<=0）对应的原子类别置-inf
    
    Args:
        node_logits: shape (bs, n_nodes, n_atom_types)
        remaining_formula: 剩余可用的化学式
        atom_type_vocab: 原子类型列表，例如 ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    
    Returns:
        masked_logits: 应用掩码后的logits
    """
    mask = torch.ones_like(node_logits, dtype=torch.bool)
    
    for idx, elem in enumerate(atom_type_vocab):
        # 如果该元素在剩余化学式中数量<=0，则禁止采样
        if elem in remaining_formula and remaining_formula[elem] <= 0:
            mask[..., idx] = False
    
    # 将禁止的位置设为-inf
    masked_logits = torch.where(mask, node_logits, torch.tensor(-1e10).to(node_logits.device))
    return masked_logits


def contains_scaffold(candidate: Mol, scaffold: Mol, use_chirality: bool = False) -> bool:
    """
    VF2子图同构检查：验证候选分子是否包含指定骨架
    
    Args:
        candidate: 候选分子
        scaffold: 骨架分子
        use_chirality: 是否考虑手性
    
    Returns:
        True if candidate contains scaffold as substructure
    """
    if candidate is None or scaffold is None:
        return False
    try:
        return candidate.HasSubstructMatch(scaffold, useChirality=use_chirality)
    except Exception:
        return False


def quick_valence_check(mol: Mol) -> bool:
    """
    快速价态检查：确保所有原子价态合法
    
    Args:
        mol: RDKit分子对象
    
    Returns:
        True if all atoms have valid valence
    """
    if mol is None:
        return False
    
    try:
        # RDKit的Sanitize会检查价态
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def quick_local_checks(mol: Mol) -> bool:
    """
    轻量合法性检查：价态/连通性等
    
    Args:
        mol: RDKit分子对象
    
    Returns:
        True if molecule passes basic validity checks
    """
    if mol is None:
        return False
    
    try:
        # 检查价态
        Chem.SanitizeMol(mol)
        
        # 检查是否有原子
        if mol.GetNumAtoms() == 0:
            return False
        
        return True
    except Exception:
        return False


def is_generation_finished(remaining_formula: Formula) -> bool:
    """
    检查生成是否完成：ΔF全为0（所有元素都已用完）
    
    Args:
        remaining_formula: 剩余化学式
    
    Returns:
        True if all elements are used up
    """
    return all(v == 0 for v in remaining_formula.values())


def update_remaining_formula(
    remaining_formula: Formula,
    added_atoms: List[str]
) -> Formula:
    """
    更新剩余化学式：从ΔF中减去新增的原子
    
    Args:
        remaining_formula: 当前剩余化学式
        added_atoms: 新增的原子列表，例如 ['C', 'N', 'O']
    
    Returns:
        updated_formula: 更新后的剩余化学式
    """
    updated = Formula(remaining_formula)
    for atom in added_atoms:
        if atom in updated:
            updated[atom] -= 1
            if updated[atom] < 0:
                # 超出限制，返回None表示非法
                return None
    return updated


def create_scaffold_mask(
    scaffold: Mol,
    n_nodes: int,
    scaffold_indices: Optional[List[int]] = None
) -> torch.Tensor:
    """
    创建骨架掩码：标记哪些节点属于骨架（不可修改）
    
    Args:
        scaffold: 骨架分子
        n_nodes: 图中总节点数
        scaffold_indices: 骨架原子在图中的索引（如果已知）
    
    Returns:
        mask: shape (n_nodes,), True表示该节点属于骨架
    """
    if scaffold_indices is None:
        # 假设骨架原子在前面
        scaffold_size = scaffold.GetNumAtoms()
        scaffold_indices = list(range(scaffold_size))
    
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    for idx in scaffold_indices:
        if idx < n_nodes:
            mask[idx] = True
    
    return mask


def apply_attachment_mask_to_logits(
    edge_logits: torch.Tensor,
    scaffold_mask: torch.Tensor,
    attachment_indices: Optional[List[int]] = None
) -> torch.Tensor:
    """
    对边logits应用锚点掩码：只允许在白名单锚点处接枝
    
    Args:
        edge_logits: shape (bs, n, n, n_edge_types)
        scaffold_mask: shape (n,), 标记骨架节点
        attachment_indices: 允许接枝的节点索引白名单
    
    Returns:
        masked_logits: 应用掩码后的edge logits
    """
    if attachment_indices is None:
        # 如果没有指定，允许所有骨架节点接枝
        return edge_logits
    
    bs, n, _, n_edge_types = edge_logits.shape
    mask = torch.zeros(n, dtype=torch.bool, device=edge_logits.device)
    
    for idx in attachment_indices:
        if idx < n:
            mask[idx] = True
    
    # 创建边掩码：只有当至少一个端点是允许的锚点时才允许该边
    edge_mask = mask.unsqueeze(0) | mask.unsqueeze(1)  # (n, n)
    edge_mask = edge_mask.unsqueeze(0).unsqueeze(-1).expand(bs, n, n, n_edge_types)
    
    # 将不允许的边置-inf
    masked_logits = torch.where(edge_mask, edge_logits, torch.tensor(-1e10).to(edge_logits.device))
    return masked_logits


def get_atom_types_from_dense(X: torch.Tensor, atom_type_vocab: List[str]) -> List[List[str]]:
    """
    从稠密节点表示中提取原子类型
    
    Args:
        X: shape (bs, n, n_atom_types), one-hot或softmax后的概率
        atom_type_vocab: 原子类型列表
    
    Returns:
        batch中每个分子的原子类型列表
    """
    batch_atoms = []
    atom_indices = torch.argmax(X, dim=-1)  # (bs, n)
    
    for mol_indices in atom_indices:
        mol_atoms = [atom_type_vocab[idx.item()] for idx in mol_indices]
        batch_atoms.append(mol_atoms)
    
    return batch_atoms


# 高级功能：骨架对齐和覆写
def align_scaffold_to_graph(
    scaffold: Mol,
    current_mol: Mol
) -> Optional[Tuple[List[int], List[int]]]:
    """
    将骨架对齐到当前分子图，返回原子映射
    
    Args:
        scaffold: 骨架分子
        current_mol: 当前生成的分子
    
    Returns:
        (scaffold_indices, current_indices): 原子索引对应关系
        如果无法对齐则返回None
    """
    if scaffold is None or current_mol is None:
        return None
    
    try:
        # 使用子结构匹配找到骨架在当前分子中的位置
        # 直接使用 Chem.Mol.GetSubstructMatch
        match = current_mol.GetSubstructMatch(scaffold)
        if not match:
            return None
        
        scaffold_indices = list(range(scaffold.GetNumAtoms()))
        current_indices = list(match)
        
        return scaffold_indices, current_indices
    except Exception:
        return None


def freeze_scaffold_in_dense_graph(
    X: torch.Tensor,
    E: torch.Tensor,
    scaffold: Mol,
    atom_type_vocab: List[str],
    edge_type_vocab: List[str],
    scaffold_indices: Optional[List[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在稠密图表示中冻结骨架：将骨架的节点和边覆写为确定性的one-hot
    
    Args:
        X: shape (bs, n, n_atom_types)
        E: shape (bs, n, n, n_edge_types)
        scaffold: 骨架分子
        atom_type_vocab: 原子类型词表
        edge_type_vocab: 边类型词表
        scaffold_indices: 骨架原子在图中的索引
    
    Returns:
        (X_frozen, E_frozen): 冻结骨架后的图
    """
    if scaffold_indices is None:
        # 默认假设骨架在前面
        scaffold_indices = list(range(scaffold.GetNumAtoms()))
    
    X_frozen = X.clone()
    E_frozen = E.clone()
    
    # 冻结节点：设置为one-hot
    for local_idx, graph_idx in enumerate(scaffold_indices):
        if local_idx >= scaffold.GetNumAtoms() or graph_idx >= X.shape[1]:
            continue
        
        atom = scaffold.GetAtomWithIdx(local_idx)
        atom_symbol = atom.GetSymbol()
        
        if atom_symbol in atom_type_vocab:
            atom_type_idx = atom_type_vocab.index(atom_symbol)
            # 将该节点设为one-hot
            X_frozen[:, graph_idx, :] = 0
            X_frozen[:, graph_idx, atom_type_idx] = 1
    
    # 冻结边：设置骨架内部的边
    for bond in scaffold.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        if begin_idx >= len(scaffold_indices) or end_idx >= len(scaffold_indices):
            continue
        
        begin_graph_idx = scaffold_indices[begin_idx]
        end_graph_idx = scaffold_indices[end_idx]
        
        if begin_graph_idx >= E.shape[1] or end_graph_idx >= E.shape[2]:
            continue
        
        # 获取键类型
        bond_type = bond.GetBondType()
        edge_type_str = str(bond_type).split('.')[-1].lower()  # 例如 'SINGLE' -> 'single'
        
        # 映射到edge_type_vocab（需要根据实际vocab调整）
        # DiffMS使用的是 0=no_edge, 1=single, 2=double, 3=triple, 4=aromatic
        bond_type_map = {
            'single': 1,
            'double': 2,
            'triple': 3,
            'aromatic': 4
        }
        
        if edge_type_str in bond_type_map:
            edge_type_idx = bond_type_map[edge_type_str]
            if edge_type_idx < E.shape[-1]:
                # 设置为one-hot（对称）
                E_frozen[:, begin_graph_idx, end_graph_idx, :] = 0
                E_frozen[:, begin_graph_idx, end_graph_idx, edge_type_idx] = 1
                E_frozen[:, end_graph_idx, begin_graph_idx, :] = 0
                E_frozen[:, end_graph_idx, begin_graph_idx, edge_type_idx] = 1
    
    return X_frozen, E_frozen


# 导出的主要函数列表
__all__ = [
    'Formula',
    'smiles_to_mol',
    'formula_of',
    'formula_subtract',
    'formula_to_string',
    'parse_formula',
    'apply_formula_mask_to_logits',
    'contains_scaffold',
    'quick_valence_check',
    'quick_local_checks',
    'is_generation_finished',
    'update_remaining_formula',
    'create_scaffold_mask',
    'apply_attachment_mask_to_logits',
    'get_atom_types_from_dense',
    'align_scaffold_to_graph',
    'freeze_scaffold_in_dense_graph',
]

