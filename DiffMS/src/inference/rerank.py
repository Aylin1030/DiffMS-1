"""
Spectrum-based reranking utilities for DiffMS predictions.
快速谱打分器 + 可选高精度重排
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdchem import Mol


def fast_spec_score(
    mol: Mol,
    spectrum_peaks: np.ndarray,
    ppm_tolerance: float = 20.0
) -> float:
    """
    快速启发式谱匹配分数：基于中性损失/ppm容差的匹配数量
    
    Args:
        mol: 候选分子
        spectrum_peaks: 质谱峰，shape (n_peaks, 2)，每行是 [m/z, intensity]
        ppm_tolerance: ppm容差
    
    Returns:
        score: 匹配分数（越高越好）
    """
    if mol is None or spectrum_peaks is None or len(spectrum_peaks) == 0:
        return -1e10
    
    try:
        # 计算分子质量
        mol_mass = Descriptors.ExactMolWt(mol)
        
        # 获取所有峰的m/z和强度
        mz_values = spectrum_peaks[:, 0]
        intensities = spectrum_peaks[:, 1]
        
        # 计算可能的中性损失（分子质量 - 峰m/z）
        neutral_losses = mol_mass - mz_values
        
        # 简单启发式：统计有多少峰在合理范围内（中性损失为正且不太大）
        valid_peaks_mask = (neutral_losses > 0) & (neutral_losses < mol_mass)
        
        # 加权分数：用强度加权
        if valid_peaks_mask.any():
            matched_intensities = intensities[valid_peaks_mask]
            score = np.sum(matched_intensities)
        else:
            score = 0.0
        
        # 额外奖励：如果分子质量接近最大峰的m/z（可能是[M+H]+）
        max_peak_mz = mz_values[np.argmax(intensities)]
        mass_diff = abs(mol_mass - max_peak_mz)
        mass_diff_ppm = (mass_diff / mol_mass) * 1e6 if mol_mass > 0 else 1e10
        
        if mass_diff_ppm < ppm_tolerance:
            score += np.max(intensities) * 2.0  # 额外奖励
        
        return float(score)
    
    except Exception as e:
        # 计算失败，返回负分
        return -1e10


def cosine_similarity_score(
    mol: Mol,
    spectrum_embed: torch.Tensor,
    encoder_model: Optional[Any] = None
) -> float:
    """
    基于编码器特征的余弦相似度打分（需要encoder模型）
    
    Args:
        mol: 候选分子
        spectrum_embed: 谱编码向量
        encoder_model: 可选的编码器模型
    
    Returns:
        cosine_similarity: 余弦相似度分数
    """
    if mol is None or encoder_model is None:
        return -1.0
    
    try:
        # TODO: 这里需要根据实际的encoder实现来生成分子的嵌入
        # 目前返回一个占位分数
        return 0.0
    except Exception:
        return -1.0


def accurate_spec_score(
    mol: Mol,
    spectrum_peaks: np.ndarray,
    use_cfm: bool = False
) -> float:
    """
    高精度谱打分：可选用CFM-ID/MetFrag等外部碎裂模拟器
    
    Args:
        mol: 候选分子
        spectrum_peaks: 质谱峰
        use_cfm: 是否使用CFM-ID（需要额外安装）
    
    Returns:
        score: 高精度匹配分数
    """
    if use_cfm:
        # TODO: 接入CFM-ID或其他高精度碎裂预测器
        # 目前降级到快速打分
        return fast_spec_score(mol, spectrum_peaks)
    else:
        # 默认使用快速打分
        return fast_spec_score(mol, spectrum_peaks)


def rerank_by_spectrum(
    candidates: List[Mol],
    spectrum_peaks: np.ndarray,
    top_k_pre: int = 64,
    use_accurate_rerank: bool = False,
    ppm_tolerance: float = 20.0
) -> List[Mol]:
    """
    基于质谱重排候选分子
    
    Args:
        candidates: 候选分子列表
        spectrum_peaks: 质谱峰，shape (n_peaks, 2)
        top_k_pre: 第一轮快速打分后保留的top-k
        use_accurate_rerank: 是否对top-k使用高精度重排
        ppm_tolerance: ppm容差
    
    Returns:
        reranked_candidates: 重排后的分子列表（按分数从高到低）
    """
    if not candidates:
        return []
    
    # 第一轮：快速打分
    scored_candidates = []
    for mol in candidates:
        if mol is not None:
            score = fast_spec_score(mol, spectrum_peaks, ppm_tolerance)
            scored_candidates.append((score, mol))
    
    # 按快速分数排序
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 取top-k进入第二轮
    top_k_candidates = scored_candidates[:min(top_k_pre, len(scored_candidates))]
    
    if use_accurate_rerank and len(top_k_candidates) > 0:
        # 第二轮：高精度重排
        accurate_scored = []
        for fast_score, mol in top_k_candidates:
            accurate_score = accurate_spec_score(mol, spectrum_peaks)
            accurate_scored.append((accurate_score, mol))
        
        # 按高精度分数重新排序
        accurate_scored.sort(key=lambda x: x[0], reverse=True)
        return [mol for _, mol in accurate_scored]
    else:
        # 只用快速分数
        return [mol for _, mol in top_k_candidates]


def rerank_by_multiple_criteria(
    candidates: List[Mol],
    spectrum_peaks: np.ndarray,
    formula: Optional[Dict[str, int]] = None,
    scaffold: Optional[Mol] = None,
    weights: Optional[Dict[str, float]] = None
) -> List[Tuple[float, Mol]]:
    """
    多准则重排：综合谱匹配、化学式匹配、骨架匹配等
    
    Args:
        candidates: 候选分子列表
        spectrum_peaks: 质谱峰
        formula: 目标化学式
        scaffold: 目标骨架
        weights: 各准则的权重字典
    
    Returns:
        scored_candidates: [(总分, 分子), ...] 按分数从高到低排序
    """
    if weights is None:
        weights = {
            'spectrum': 1.0,
            'formula': 0.5,
            'scaffold': 0.5,
            'validity': 0.3
        }
    
    scored_candidates = []
    
    for mol in candidates:
        if mol is None:
            continue
        
        total_score = 0.0
        
        # 1. 谱匹配分数
        spec_score = fast_spec_score(mol, spectrum_peaks)
        total_score += weights.get('spectrum', 1.0) * spec_score
        
        # 2. 化学式匹配分数
        if formula is not None:
            formula_score = score_formula_match(mol, formula)
            total_score += weights.get('formula', 0.5) * formula_score
        
        # 3. 骨架匹配分数
        if scaffold is not None:
            scaffold_score = 1.0 if mol.HasSubstructMatch(scaffold) else 0.0
            total_score += weights.get('scaffold', 0.5) * scaffold_score
        
        # 4. 有效性分数
        validity_score = 1.0 if is_valid_molecule(mol) else 0.0
        total_score += weights.get('validity', 0.3) * validity_score
        
        scored_candidates.append((total_score, mol))
    
    # 排序
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return scored_candidates


def score_formula_match(mol: Mol, target_formula: Dict[str, int]) -> float:
    """
    计算分子与目标化学式的匹配分数
    
    Args:
        mol: 候选分子
        target_formula: 目标化学式，例如 {'C': 6, 'H': 12, 'O': 6}
    
    Returns:
        score: 匹配分数（完全匹配为1.0，否则根据差异打折）
    """
    if mol is None:
        return 0.0
    
    try:
        # 获取分子的实际化学式（只统计重原子）
        mol_formula = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol != 'H':  # 跳过氢
                mol_formula[symbol] = mol_formula.get(symbol, 0) + 1
        
        # 计算差异
        all_elements = set(target_formula.keys()) | set(mol_formula.keys())
        total_diff = 0
        total_atoms = sum(target_formula.values())
        
        for elem in all_elements:
            target_count = target_formula.get(elem, 0)
            mol_count = mol_formula.get(elem, 0)
            total_diff += abs(target_count - mol_count)
        
        # 归一化分数
        if total_atoms == 0:
            return 0.0
        
        score = max(0.0, 1.0 - total_diff / (2 * total_atoms))
        return score
    
    except Exception:
        return 0.0


def is_valid_molecule(mol: Mol) -> bool:
    """
    检查分子是否有效（价态、连通性等）
    
    Args:
        mol: 候选分子
    
    Returns:
        True if molecule is valid
    """
    if mol is None:
        return False
    
    try:
        Chem.SanitizeMol(mol)
        if mol.GetNumAtoms() == 0:
            return False
        return True
    except Exception:
        return False


def filter_by_scaffold(
    candidates: List[Mol],
    scaffold: Mol,
    strict: bool = True
) -> List[Mol]:
    """
    按骨架过滤候选分子
    
    Args:
        candidates: 候选分子列表
        scaffold: 骨架分子
        strict: 严格模式（必须包含骨架），否则优先包含骨架的排在前面
    
    Returns:
        filtered_candidates: 过滤/排序后的候选列表
    """
    if scaffold is None:
        return candidates
    
    if strict:
        # 严格模式：只保留包含骨架的
        return [mol for mol in candidates if mol is not None and mol.HasSubstructMatch(scaffold)]
    else:
        # 宽松模式：优先排序包含骨架的
        with_scaffold = [mol for mol in candidates if mol is not None and mol.HasSubstructMatch(scaffold)]
        without_scaffold = [mol for mol in candidates if mol is not None and not mol.HasSubstructMatch(scaffold)]
        return with_scaffold + without_scaffold


def deduplicate_candidates(
    candidates: List[Mol],
    use_inchi: bool = True
) -> List[Mol]:
    """
    去重候选分子（基于InChI或SMILES）
    
    Args:
        candidates: 候选分子列表
        use_inchi: 使用InChI去重（更严格），否则使用canonical SMILES
    
    Returns:
        unique_candidates: 去重后的分子列表
    """
    seen = set()
    unique = []
    
    for mol in candidates:
        if mol is None:
            continue
        
        try:
            if use_inchi:
                key = Chem.MolToInchi(mol)
            else:
                key = Chem.MolToSmiles(mol, canonical=True)
            
            if key not in seen:
                seen.add(key)
                unique.append(mol)
        except Exception:
            # 无法转换，跳过
            continue
    
    return unique


# 导出函数
__all__ = [
    'fast_spec_score',
    'cosine_similarity_score',
    'accurate_spec_score',
    'rerank_by_spectrum',
    'rerank_by_multiple_criteria',
    'score_formula_match',
    'is_valid_molecule',
    'filter_by_scaffold',
    'deduplicate_candidates',
]

