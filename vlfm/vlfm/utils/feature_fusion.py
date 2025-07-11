# vlfm/utils/feature_fusion.py
import math
import torch
import torch.nn.functional as F

def per_query_gated_fusion(Qb: torch.Tensor, Qn: torch.Tensor) -> torch.Tensor:
    """
    Fuse two VLM image‐feature tensors in language space using per‐query gated fusion.
    Args:
        feat1: Tensor of shape (K, D)
        feat2: Tensor of shape (K, D)
    Returns:
        fused: Tensor of shape (K, D)
    """
    # Qb, Qn: (K, D)
    K, D = Qb.shape

    # compute per-slot dot / sqrt(D)
    scores = (Qb * Qn).sum(dim=1) / math.sqrt(D)     # (K,)

    # gate via sigmoid
    gates = torch.sigmoid(scores).unsqueeze(1)      # (K,1)

    # novelty weight
    novelty = (1 - gates) * Qn                     # (K, D)

    # residual fuse + optional layer‐norm
    fused = Qb + novelty
    return fused

def compute_cosine_similarity(feat1: torch.Tensor, feat2: torch.Tensor) -> float:
    # flatten (32,256) -> (1, 8192) -> unsqueeze (1,1,8192) and (1,8192,1)
    feat1 = F.normalize(feat1.view(1, -1), dim=-1).unsqueeze(1)  # (1, 1, 8192)
    feat2 = F.normalize(feat2.view(1, -1), dim=-1).unsqueeze(2)  # (1, 8192, 1)   
    sim = torch.bmm(feat1, feat2).mean().item()
    return sim