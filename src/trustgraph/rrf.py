from __future__ import annotations
from typing import List, Tuple
from collections import defaultdict

def rrf_fuse(*rank_lists: List[Tuple[str, float]], k: int = 60, k_rrf: float = 60.0) -> List[str]:
    """
    Reciprocal Rank Fusion. Each list is [(doc_id, score)], assumed already ranked desc.
    Returns list of fused doc_ids.
    """
    scores = defaultdict(float)
    for rl in rank_lists:
        for rank, (doc_id, _) in enumerate(rl[:k], start=1):
            scores[doc_id] += 1.0 / (k_rrf + rank)
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
