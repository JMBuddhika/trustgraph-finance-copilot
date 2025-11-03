from __future__ import annotations
import os, json, re
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv

from src.trustgraph.config import MODELS, PATHS
from src.trustgraph.sec_ingest import ingest
from src.trustgraph.indexes import build_indexes, retrieve
from src.trustgraph.generation import answer_with_evidence
from src.trustgraph.sqlstore import SQLStore

load_dotenv()

st.set_page_config("TrustGraph RAG — Verifiable Finance Copilot", layout="wide")

st.title("TrustGraph RAG — Verifiable Finance Copilot")
st.caption("Free stack: Groq + DuckDB + FAISS + BM25 | Executable citations | SEC filings")

with st.sidebar:
    st.subheader("Data & Index")
    tickers = st.text_input("Tickers (space-separated)", "AAPL MSFT NVDA").strip().split()
    forms = st.multiselect("Forms", ["10-K","10-Q"], ["10-K","10-Q"])
    limit = st.number_input("Filings per form", 1, 6, 1)
    if st.button("Ingest SEC"):
        with st.status("Downloading + parsing SEC filings...", expanded=True):
            ingest(tickers, forms, limit)
        st.success("Ingestion done.")

    if st.button("Build Indexes (BM25 + FAISS)"):
        with st.status("Building indexes...", expanded=True):
            build_indexes()
        st.success("Indexes ready.")

    st.divider()
    st.subheader("DuckDB")
    if st.button("List Tables"):
        store = SQLStore()
        st.session_state["tables"] = store.tables()
        store.close()

    if "tables" in st.session_state:
        st.code("\n".join(st.session_state["tables"]), language="text")

st.markdown("### Ask a question")
q = st.text_input("Example: What drove NVIDIA's YoY gross margin change? Show numbers and quotes.", "")
ticker_hint = st.text_input("Optional ticker hint (improves SQL planning)", "")

colL, colR = st.columns([1.2, 1])

if st.button("Answer"):
    if not q.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Retrieving, planning SQL, and generating answer..."):
            qa = answer_with_evidence(q, ticker_hint or None)
        with colL:
            ribbon = f"Faithfulness: **{qa.faithfulness:.2f}**"
            if qa.abstained:
                ribbon += "  |  ❗ *Auto-abstained due to weak evidence*"
            st.markdown(ribbon)
            st.markdown(qa.answer_markdown or "_No answer produced._")

            if qa.claims:
                st.markdown("#### Claims")
                for i, c in enumerate(qa.claims, 1):
                    st.write(f"{i}. {c.text}  *(docs: {', '.join(c.doc_refs) or '-'}; sql: {', '.join(c.sql_refs) or '-'})*")

        with colR:
            st.markdown("#### Evidence")
            docs = qa.citations.get("docs", {})
            sqlv = qa.citations.get("sql", {})
            plans = qa.citations.get("plans", [])

            if docs:
                st.markdown("**Text snippets**")
                for key, rec in list(docs.items())[:10]:
                    meta = f"{rec['ticker']} {rec['form']} {rec['accession']}"
                    snippet = rec['text'][:380] + ("..." if len(rec['text'])>380 else "")
                    st.code(f"[{key}] {meta}\n{snippet}", language="markdown")

            if sqlv:
                st.markdown("**SQL results (reproducible)**")
                store = SQLStore()
                for sid, csv_preview in sqlv.items():
                    st.write(f"[{sid}]")
                    if csv_preview.startswith("ERROR:"):
                        st.error(csv_preview)
                    else:
                        st.code(csv_preview[:1000], language="csv")
                    # Reproduce button
                    with st.expander(f"Re-run {sid}"):
                        # Find the original SQL from plans
                        sql = next((p["sql"] for p in plans if p.get("id")==sid), None)
                        if sql:
                            st.code(sql, language="sql")
                            if st.button(f"Execute {sid}", key=f"exec_{sid}"):
                                try:
                                    df = store.query(sql)
                                    st.dataframe(df, use_container_width=True)
                                except Exception as e:
                                    st.error(str(e))
                store.close()
