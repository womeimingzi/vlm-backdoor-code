from collections import defaultdict
from typing import Dict, List, Optional

import torch

from experiments.shared.exp1b_projection import extract_orthogonal_directions


def get_2d_keys(state_dict: dict, skip_embeddings: bool = True) -> List[str]:
    """Filter state_dict to 2D weight matrices suitable for SVD analysis."""
    keys = []
    for k, v in state_dict.items():
        if v.dim() != 2:
            continue
        if skip_embeddings and "embeddings" in k:
            continue
        keys.append(k)
    return sorted(keys)


def per_matrix_svd(
    bd_state: dict,
    clean_state: dict,
    keys: List[str],
    rank: Optional[int] = None,
) -> Dict[str, tuple]:
    """Compute SVD(delta W) for each weight matrix key."""
    result = {}
    for k in keys:
        dW = bd_state[k].float() - clean_state[k].float()
        if rank is not None and rank + 10 < min(dW.shape):
            q = rank + 10
            U, S, V = torch.svd_lowrank(dW, q=q, niter=4)
            Vh = V.T
        else:
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        result[k] = (U, S, Vh)
    return result


def extract_orthogonal_directions_multimatrix(
    svd_bd: dict,
    svd_bn: dict,
    keys: List[str],
    k: int,
    angle_threshold: float = 50.0,
) -> Dict[str, list]:
    """Extract orthogonal directions independently for multiple matrices."""
    result = {}
    for key in keys:
        _, _, Vh_bd = svd_bd[key]
        _, _, Vh_bn = svd_bn[key]
        effective_k = min(k, Vh_bd.shape[0], Vh_bn.shape[0])
        if effective_k < 2:
            continue
        dirs = extract_orthogonal_directions(Vh_bd, Vh_bn, effective_k, angle_threshold)
        if dirs:
            result[key] = dirs
    return result


def projection_purify_multimatrix(
    bd_state: dict,
    clean_state: dict,
    directions_dict: Dict[str, list],
) -> dict:
    """Apply projection removal to multiple weight matrices."""
    purified = {k: v.clone() for k, v in bd_state.items()}
    for key, directions in directions_dict.items():
        if not directions or key not in bd_state:
            continue
        W_bd = bd_state[key].float()
        W_clean = clean_state[key].float()
        dW = W_bd - W_clean
        d_vectors = [d for d, _ in directions]
        D = torch.stack(d_vectors, dim=1)
        projected = dW @ D @ D.T
        purified[key] = W_bd - projected
    return purified


def compare_directions_multimatrix(dirs_true: dict, dirs_pseudo: dict) -> dict:
    """Compare ground-truth and pseudo-benign directions across matrices."""
    cos_sims = {}
    for key in dirs_true:
        if key in dirs_pseudo and dirs_true[key] and dirs_pseudo[key]:
            d_true = dirs_true[key][0][0]
            d_pseudo = dirs_pseudo[key][0][0]
            cos = float(torch.abs(d_true.double() @ d_pseudo.double()))
            cos_sims[key] = round(cos, 6)

    if not cos_sims:
        return {
            "n_matrices_with_true_dirs": len(dirs_true),
            "n_matrices_with_pseudo_dirs": len(dirs_pseudo),
            "n_matrices_with_both": 0,
            "mean_cos_sim": None,
            "median_cos_sim": None,
        }

    values = list(cos_sims.values())
    group_stats = defaultdict(list)
    for key, cos in cos_sims.items():
        parts = key.split(".")
        if parts[0] == "encoder" and len(parts) >= 5:
            func_group = ".".join(parts[3:-1])
            group_stats[func_group].append(cos)

    group_means = {g: round(sum(v) / len(v), 4) for g, v in group_stats.items()}
    return {
        "n_matrices_with_true_dirs": len(dirs_true),
        "n_matrices_with_pseudo_dirs": len(dirs_pseudo),
        "n_matrices_with_both": len(cos_sims),
        "mean_cos_sim": round(sum(values) / len(values), 6),
        "median_cos_sim": round(sorted(values)[len(values) // 2], 6),
        "min_cos_sim": round(min(values), 6),
        "max_cos_sim": round(max(values), 6),
        "group_means": group_means,
        "per_matrix": cos_sims,
    }
