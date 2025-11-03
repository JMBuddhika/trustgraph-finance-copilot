from __future__ import annotations
import pickle, json
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

from .config import PATHS, MODELS, SETTINGS
from .rrf import rrf_fuse


# ---------------------------
# Helpers
# ---------------------------
def _load_corpus() -> List[Dict[str, Any]]:
    if not PATHS.corpus_jsonl.exists():
        raise FileNotFoundError("No corpus found. Run ingestion first.")
    data = []
    with PATHS.corpus_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _tokenize(s: str) -> List[str]:
    return s.lower().split()


def _meta_passage(rec: Dict[str, Any]) -> str:
    """
    Encode a passage with lightweight metadata so dense retrieval can respect ticker/form.
    """
    tkr = rec.get("ticker", "")
    frm = rec.get("form", "")
    txt = rec.get("text", "")
    return f"passage: [TICKER: {tkr}] [FORM: {frm}] {txt}"


# ---------------------------
# Index building
# ---------------------------
def build_indexes():
    PATHS.index_dir.mkdir(parents=True, exist_ok=True)
    corpus = _load_corpus()

    # ----- Dense index (FAISS) with metadata-aware passages
    embed = SentenceTransformer(MODELS.embed_model)
    _ = embed.encode(["query: warmup"], normalize_embeddings=True)  # warm-up to load model weights
    X = embed.encode(
        [_meta_passage(r) for r in corpus],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X.astype(np.float32))
    faiss.write_index(index, str(PATHS.faiss_index))
    with PATHS.faiss_meta.open("wb") as f:
        pickle.dump({"ids": [r["id"] for r in corpus], "meta": corpus}, f)

    # ----- BM25 (include minimal metadata tokens to help lexical match)
    bm25_docs = [f"{r.get('ticker','')} {r.get('form','')} {r.get('text','')}" for r in corpus]
    tokenized = [_tokenize(t) for t in bm25_docs]
    bm25 = BM25Okapi(tokenized)
    with PATHS.bm25_corpus.open("wb") as f:
        pickle.dump(
            {"bm25": bm25, "ids": [r["id"] for r in corpus], "texts": bm25_docs, "meta": corpus},
            f,
        )

    print("Indexes built.")


def _load_dense() -> Tuple[faiss.IndexFlatIP, Dict[str, Any]]:
    index = faiss.read_index(str(PATHS.faiss_index))
    with PATHS.faiss_meta.open("rb") as f:
        meta = pickle.load(f)
    return index, meta


def _load_bm25() -> Dict[str, Any]:
    with PATHS.bm25_corpus.open("rb") as f:
        return pickle.load(f)


# ---------------------------
# Retrieval
# ---------------------------
def retrieve(
    query: str,
    k: int = SETTINGS.final_k,
    rerank: bool = True,
    ticker: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval (dense + BM25 -> RRF) with optional ticker-awareness.
    If `ticker` is provided, we bias both dense and lexical lists and prefer matches.
    """
    # Dense
    index, meta = _load_dense()
    embed = SentenceTransformer(MODELS.embed_model)
    q_str = f"query: {query}"
    if ticker:
        q_str = f"{q_str} [TICKER: {ticker.upper()}]"
    qv = embed.encode([q_str], normalize_embeddings=True)
    D, I = index.search(qv.astype(np.float32), SETTINGS.topk_dense)
    dense_hits = [(meta["ids"][idx], float(D[0][i])) for i, idx in enumerate(I[0])]

    # Optional ticker-aware reorder for dense (preserve relative order otherwise)
    if ticker:
        tkr = ticker.strip().upper()
        id2score = dict(dense_hits)
        dense_ids = [doc_id for doc_id, _ in dense_hits]
        dense_ids_t = [i for i in dense_ids if str(_find_meta(meta, i).get("ticker", "")).upper() == tkr]
        dense_ids_nt = [i for i in dense_ids if i not in dense_ids_t]
        dense_hits = [(i, id2score[i]) for i in (dense_ids_t + dense_ids_nt)]

    # BM25
    bm = _load_bm25()
    tokenized_query = _tokenize(query + (f" {ticker}" if ticker else ""))
    scores = bm["bm25"].get_scores(tokenized_query)
    top_bm_idx = np.argsort(scores)[-SETTINGS.topk_bm25:][::-1]
    bm25_hits = []
    if ticker:
        tkr = ticker.strip().upper()
        for i in top_bm_idx:
            doc_id = bm["ids"][i]
            rec = _find_meta({"meta": bm["meta"]}, doc_id)
            base = float(scores[i])
            # small boost if ticker matches (keeps order but improves RRF ranks)
            boost = 1.15 if str(rec.get("ticker", "")).upper() == tkr else 1.0
            bm25_hits.append((doc_id, base * boost))
    else:
        bm25_hits = [(bm["ids"][i], float(scores[i])) for i in top_bm_idx]

    # RRF fusion
    fused_ids = rrf_fuse(dense_hits, bm25_hits, k=SETTINGS.fusion_k)
    id2rec = {r["id"]: r for r in bm["meta"]}
    cands = [id2rec[i] for i in fused_ids if i in id2rec]

    # Prefer ticker matches at the top (soft bias)
    if ticker:
        tkr = ticker.strip().upper()
        c_t = [r for r in cands if str(r.get("ticker", "")).upper() == tkr]
        c_nt = [r for r in cands if str(r.get("ticker", "")).upper() != tkr]
        cands = c_t + c_nt

    if not rerank:
        return cands[:k]

    # Cross-encoder rerank (optional, local HF model). Append ticker hint to query for extra bias.
    try:
        ce = CrossEncoder(MODELS.reranker_model)
        ce_query = f"{query} [TICKER: {ticker.upper()}]" if ticker else query
        pairs = [(ce_query, r["text"]) for r in cands[:SETTINGS.rerank_k]]
        ce_scores = ce.predict(pairs)
        order = np.argsort(ce_scores)[::-1]
        reranked = [cands[i] for i in order]
        return reranked[:k]
    except Exception:
        # If reranker isn't available, return fused list
        return cands[:k]


def _find_meta(meta_pack: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
    """
    Utility: find meta record by doc_id from a meta pack (with key 'meta': list[dict]).
    """
    for r in meta_pack["meta"]:
        if r.get("id") == doc_id:
            return r
    return {}
