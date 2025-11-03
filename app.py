from __future__ import annotations
import os, json, re
from typing import Dict, Any, List

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from src.trustgraph.config import MODELS, PATHS, SETTINGS, GROQ_API_KEY
from src.trustgraph.sec_ingest import ingest
from src.trustgraph.indexes import build_indexes, retrieve
from src.trustgraph.generation import answer_with_evidence
from src.trustgraph.sqlstore import SQLStore
from src.trustgraph.eval import (
    precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k,
    exact_match, token_f1
)

# ---------------------------------------
# App setup
# ---------------------------------------
load_dotenv()
st.set_page_config("TrustGraph RAG — Verifiable Finance Copilot", layout="wide")
st.title("TrustGraph RAG — Verifiable Finance Copilot")
st.caption("Free stack: Groq + DuckDB + FAISS + BM25 | Executable citations | SEC filings")

if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY is not set. Answering will fail until you add it to your .env.", icon="⚠️")

# ---------------------------------------
# Sidebar: data ops
# ---------------------------------------
with st.sidebar:
    st.subheader("Data & Index")
    tickers_str = st.text_input("Tickers (space-separated)", "AAPL MSFT NVDA").strip()
    tickers = tickers_str.split() if tickers_str else []
    forms = st.multiselect("Forms", ["10-K","10-Q"], ["10-K","10-Q"])
    limit = st.number_input("Filings per form", 1, 6, 1)

    if st.button("Ingest SEC"):
        try:
            with st.status("Downloading + parsing SEC filings...", expanded=True):
                ingest(tickers, forms, int(limit))
            st.success("Ingestion done.")
        except Exception as e:
            st.error(
                "Ingestion failed. Tip: recent sec-edgar-downloader needs SEC_COMPANY and SEC_EMAIL "
                "in your .env. Example:\nSEC_COMPANY=TrustGraphRAG\nSEC_EMAIL=you@domain.com\n\n"
                f"Details: {e}"
            )

    if st.button("Build Indexes (BM25 + FAISS)"):
        try:
            with st.status("Building indexes...", expanded=True):
                build_indexes()
            st.success("Indexes ready.")
        except Exception as e:
            st.error(f"Failed to build indexes: {e}")

    st.divider()
    st.subheader("DuckDB")
    if st.button("List Tables"):
        try:
            store = SQLStore()
            st.session_state["tables"] = store.tables()
            store.close()
        except Exception as e:
            st.error(f"DuckDB error: {e}")

    if "tables" in st.session_state:
        st.code("\n".join(st.session_state["tables"]), language="text")

# ---------------------------------------
# Question UI
# ---------------------------------------
st.markdown("### Ask a question")
q = st.text_input(
    "Example: What drove NVIDIA's YoY gross margin change? Show numbers and quotes.",
    ""
)
ticker_hint = st.text_input("Optional ticker hint (improves SQL planning)", "")

colL, colR = st.columns([1.25, 1])

if st.button("Answer"):
    if not q.strip():
        st.warning("Enter a question.")
    else:
        try:
            with st.spinner("Retrieving, planning SQL, and generating answer..."):
                qa = answer_with_evidence(q, ticker_hint or None)

            # Keep the last QA around for metrics
            st.session_state["qa_last"] = qa
            st.session_state["last_q"] = q
            st.session_state["last_ticker"] = ticker_hint or ""

            # -------- Left: Answer & Claims
            with colL:
                ribbon = f"Faithfulness: **{qa.faithfulness:.2f}**"
                if qa.abstained:
                    ribbon += "  |  ❗ *Auto-abstained due to weak evidence*"
                st.markdown(ribbon)
                st.markdown(qa.answer_markdown or "_No answer produced._")

                if qa.claims:
                    st.markdown("#### Claims")
                    for i, c in enumerate(qa.claims, 1):
                        st.write(
                            f"{i}. {c.text}  "
                            f"*(docs: {', '.join(c.doc_refs) or '-'}; "
                            f"sql: {', '.join(c.sql_refs) or '-'})*"
                        )

            # -------- Right: Evidence
            with colR:
                st.markdown("#### Evidence")
                doc_map: Dict[str, Dict[str, Any]] = qa.citations.get("docs", {}) or {}
                sql_views: Dict[str, str] = qa.citations.get("sql", {}) or {}
                plans = qa.citations.get("plans", []) or []

                if doc_map:
                    st.markdown("**Text snippets**")
                    # Keep insertion order: Python 3.8+ preserves dict order
                    for key, rec in list(doc_map.items())[:10]:
                        try:
                            meta = f"{rec['ticker']} {rec['form']} {rec['accession']}"
                            snippet = rec["text"][:380] + ("..." if len(rec["text"]) > 380 else "")
                            st.code(f"[{key}] {meta}\n{snippet}", language="markdown")
                        except Exception as e:
                            st.error(f"Doc render error: {e}")

                if sql_views:
                    st.markdown("**SQL results (reproducible)**")
                    try:
                        store = SQLStore()
                        for sid, csv_preview in sql_views.items():
                            st.write(f"[{sid}]")
                            if isinstance(csv_preview, str) and csv_preview.startswith("ERROR:"):
                                st.error(csv_preview)
                            else:
                                st.code(str(csv_preview)[:1000], language="csv")

                            with st.expander(f"Re-run {sid}"):
                                sql_text = next((p["sql"] for p in plans if p.get("id")==sid), None)
                                if sql_text:
                                    st.code(sql_text, language="sql")
                                    if st.button(f"Execute {sid}", key=f"exec_{sid}"):
                                        try:
                                            df = store.query(sql_text)
                                            st.dataframe(df, use_container_width=True)
                                        except Exception as e:
                                            st.error(str(e))
                        store.close()
                    except Exception as e:
                        st.error(f"DuckDB query error: {e}")

        except Exception as e:
            st.error(f"Answer pipeline failed: {e}")

# ---------------------------------------
# Evaluation (beta)
# ---------------------------------------
st.markdown("### Evaluation")

qa = st.session_state.get("qa_last")
last_ticker = st.session_state.get("last_ticker", "")
if not qa:
    st.info("Ask a question first to see metrics.")
else:
    # ---- Retrieval metrics
    with st.expander("Retrieval metrics", True):
        st.caption(
            "If you know the true relevant doc IDs, paste them. "
            "Otherwise we use a simple heuristic based on the Ticker hint."
        )

        # predicted doc ids in the order we showed evidence
        pred_ids: List[str] = []
        for _, rec in (qa.citations.get("docs", {}) or {}).items():
            if isinstance(rec, dict) and "id" in rec:
                pred_ids.append(rec["id"])

        # heuristic GT: any retrieved doc whose ticker matches the hint
        gt_ids_default: List[str] = []
        if last_ticker:
            tkr = last_ticker.strip().upper()
            for _, rec in (qa.citations.get("docs", {}) or {}).items():
                if str(rec.get("ticker","")).upper() == tkr:
                    gt_ids_default.append(rec["id"])
            # unique, preserve order
            gt_ids_default = list(dict.fromkeys(gt_ids_default))

        manual_ids = st.text_input(
            "Gold doc IDs (comma-separated, optional)",
            ", ".join(gt_ids_default)
        )
        gt_ids = [x.strip() for x in manual_ids.split(",") if x.strip()]

        if not pred_ids:
            st.info("No retrieved docs available.")
        else:
            k = min(5, len(pred_ids))
            if gt_ids:
                p_at_k = precision_at_k(gt_ids, pred_ids, k)
                r_at_k = recall_at_k(gt_ids, pred_ids, k)
                mrr = mean_reciprocal_rank(gt_ids, pred_ids)
                ndcg = ndcg_at_k(gt_ids, pred_ids, k)
                st.write(f"**k = {k}**")
                st.write(f"Precision@{k}: **{p_at_k:.2f}**")
                st.write(f"Recall@{k}: **{r_at_k:.2f}**")
                st.write(f"MRR: **{mrr:.2f}**")
                st.write(f"nDCG@{k}: **{ndcg:.2f}**")
                st.code(f"pred_ids = {pred_ids}\ngt_ids   = {gt_ids}", language="python")
            else:
                st.warning("No gold doc IDs provided (and no ticker hint). Add IDs above to compute metrics.")

    # ---- Generation metrics
    with st.expander("Generation metrics", True):
        st.caption("Provide a short *reference answer* to compute EM/F1/Similarity (optional).")

        ref = st.text_area("Reference answer (optional)", "")
        if ref.strip():
            try:
                em = exact_match(qa.answer_markdown or "", ref)
                f1 = token_f1(qa.answer_markdown or "", ref)

                # semantic similarity via same embed model (normalized → cosine = dot)
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(MODELS.embed_model)
                    vecs = model.encode([qa.answer_markdown or "", ref], normalize_embeddings=True)
                    cos = float(np.dot(vecs[0], vecs[1]))
                except Exception:
                    cos = float("nan")

                st.write(f"Exact Match: **{1.0 if em else 0.0:.2f}**")
                st.write(f"Token F1: **{f1:.2f}**")
                st.write(f"Semantic similarity (cosine): **{cos:.2f}**")
            except Exception as e:
                st.error(f"Metric error: {e}")
        else:
            st.info("Add a reference answer above to compute EM/F1/Similarity.")

    with st.expander("Judge details", False):
        st.markdown(
            "- **Faithfulness** is computed by an LLM judge on your answer vs. text/SQL evidence.\n"
            "- If you consistently see 0.00, ensure the judge prompt/parser patches in `verification.py` are applied.\n"
            "- Keep temperature ~0.0–0.1 for stability."
        )
