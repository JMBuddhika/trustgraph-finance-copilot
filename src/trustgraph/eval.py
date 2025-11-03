from __future__ import annotations
from typing import List, Dict, Any
import math

# --- Retrieval metrics ---
def precision_at_k(gt_ids: List[str], pred_ids: List[str], k: int) -> float:
    pred = pred_ids[:k]
    if not pred: return 0.0
    hit = sum(1 for x in pred if x in gt_ids)
    return hit / len(pred)

def recall_at_k(gt_ids: List[str], pred_ids: List[str], k: int) -> float:
    pred = pred_ids[:k]
    if not gt_ids: return 0.0
    hit = sum(1 for x in pred if x in gt_ids)
    return hit / len(gt_ids)

def mean_reciprocal_rank(gt_ids: List[str], pred_ids: List[str]) -> float:
    for i, x in enumerate(pred_ids, start=1):
        if x in gt_ids:
            return 1.0 / i
    return 0.0

def average_precision(gt_ids: List[str], pred_ids: List[str]) -> float:
    ap, hits = 0.0, 0
    for i, x in enumerate(pred_ids, start=1):
        if x in gt_ids:
            hits += 1
            ap += hits / i
    return ap / max(1, len(gt_ids))

def ndcg_at_k(gt_ranked: List[str], pred_ids: List[str], k: int) -> float:
    # Binary relevance
    dcg = 0.0
    for i, x in enumerate(pred_ids[:k], start=1):
        rel = 1.0 if x in gt_ranked else 0.0
        dcg += (2**rel - 1) / math.log2(i+1)
    idcg = 0.0
    for i in range(1, min(k, len(gt_ranked)) + 1):
        idcg += (2**1 - 1) / math.log2(i+1)
    return dcg / idcg if idcg > 0 else 0.0

# --- Generation metrics (simple) ---
def exact_match(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()

def token_f1(pred: str, gold: str) -> float:
    pa = pred.lower().split()
    ga = gold.lower().split()
    common = set(pa) & set(ga)
    if not pa or not ga: return 0.0
    prec = len(common)/len(pa)
    rec = len(common)/len(ga)
    if prec+rec==0: return 0.0
    return 2*prec*rec/(prec+rec)
