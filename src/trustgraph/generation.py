from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os, json, re, textwrap

import pandas as pd
from groq import Groq
from pydantic import BaseModel

from .config import MODELS, SETTINGS, GROQ_API_KEY
from .indexes import retrieve
from .sqlstore import SQLStore

# -------------------------
# Data models
# -------------------------

class Claim(BaseModel):
    text: str
    doc_refs: List[str] = []
    sql_refs: List[str] = []

class QAResult(BaseModel):
    answer_markdown: str
    claims: List[Claim]
    citations: Dict[str, Any]
    faithfulness: float
    abstained: bool

# -------------------------
# LLM clients & prompts
# -------------------------

def _groq_client():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment/.env")
    return Groq(api_key=GROQ_API_KEY)

SYS_SOLVER = """You are a precise financial analyst bot.
You must ground every claim in: (a) quoted spans from filings text, and/or (b) SQL results provided.
Never invent numbers or text. If evidence is weak, say 'Not enough evidence'. Keep answers concise.
Use [#] footnotes like [1a], [S1] to refer to evidence ids that exist.
"""

SYS_SQL_PLANNER = """You are a SQL planning assistant.

Rules:
- You may ONLY reference the tables explicitly listed under "Valid tables".
- Use the column names exactly as shown.
- Prefer concise SQL; compute aggregates as needed.
- If no matching table exists, return an empty array [].

Output strictly a JSON array (no code fences, no prose), e.g.:
[
  {"id":"S1","sql":"SELECT ...","rationale":"..."}
]
"""

SYS_CLAIM_BINDER = """Given the user question, the retrieved text snippets, and SQL results (CSV previews),
write a short answer with [#] footnotes per claim (e.g., [1a] for text, [S1] for SQL).
Then list the atomic claims you made. Be concise and factual. If evidence is insufficient, abstain.

Return ONLY a JSON object (no prose, no code fences) with keys:
{
  "answer_markdown": "...",
  "claims":[{"text":"...","doc_refs":["1a","2a"],"sql_refs":["S1"]}]
}
"""

# -------------------------
# SQL planning helpers
# -------------------------

def _list_tables_for_prompt(store: SQLStore, like: str | None = None) -> Dict[str, List[str]]:
    # Filter to tables related to ticker if hint is provided
    return store.table_summaries(like=like)

def _valid_sql_references(sql: str, valid_tables: Dict[str, List[str]]) -> bool:
    # Very simple whitelist: require at least one known table name to appear in the SQL
    lower_sql = sql.lower()
    for t in valid_tables.keys():
        if t.lower() in lower_sql:
            return True
    return False

def plan_sql(question: str, store: SQLStore, ticker_hint: str | None = None) -> List[Dict[str, str]]:
    client = _groq_client()
    catalog = _list_tables_for_prompt(store, like=ticker_hint)
    if not catalog:
        # no tables at all for the hint; expose everything to the planner
        catalog = _list_tables_for_prompt(store, like=None)

    schema_text = "\n".join(f"- {t}: {', '.join(cols[:16])}" for t, cols in list(catalog.items())[:80])
    prompt = f"""Question: {question}

Valid tables (DuckDB):
{schema_text}
"""

    r = client.chat.completions.create(
        model=MODELS.groq_chat_model,
        messages=[
            {"role":"system","content":SYS_SQL_PLANNER},
            {"role":"user","content":prompt},
        ],
        temperature=0.1,
    )
    raw = (r.choices[0].message.content or "").strip()

    try:
        arr = json.loads(raw)
        if not isinstance(arr, list):
            return []
    except Exception:
        return []

    # Keep only plans that reference allowed tables
    filtered: List[Dict[str,str]] = []
    for p in arr:
        sql = (p or {}).get("sql","")
        if _valid_sql_references(sql, catalog):
            pid = (p or {}).get("id") or f"S{len(filtered)+1}"
            filtered.append({"id": pid, "sql": sql, "rationale": (p or {}).get("rationale","")})

    return filtered[:3]

def _synthesize_yoy_segment_sql(store: SQLStore, ticker_hint: str | None) -> List[Dict[str,str]]:
    """
    Deterministic fallback: find a table for the ticker that has Year, Segment, and a revenue column.
    Build a YoY-by-segment query for 2023 vs 2024.
    """
    tables = store.tables()
    if ticker_hint:
        like = ticker_hint.lower()
        tables = [t for t in tables if like in t.lower()] or tables

    for t in tables:
        cols = [c.lower() for c in store.schema_of(t)["column_name"].tolist()]
        if ("year" in cols) and ("segment" in cols) and (("revenue_usd_m" in cols) or ("revenue" in cols)):
            rev_col = "Revenue_USD_M" if "revenue_usd_m" in cols else "Revenue"
            sql = f"""
WITH base AS (
  SELECT CAST(Year AS VARCHAR) AS Year, Segment, {rev_col} AS Revenue
  FROM {t}
  WHERE Year IN ('2023','2024')
),
agg AS (
  SELECT Year, Segment, SUM(Revenue) AS Revenue
  FROM base
  GROUP BY 1,2
),
wide AS (
  SELECT
    Segment,
    SUM(CASE WHEN Year='2023' THEN Revenue ELSE 0 END) AS rev_2023,
    SUM(CASE WHEN Year='2024' THEN Revenue ELSE 0 END) AS rev_2024
  FROM agg
  GROUP BY 1
)
SELECT
  Segment,
  rev_2023,
  rev_2024,
  (rev_2024 - rev_2023) AS yoy_delta_usd,
  CASE WHEN rev_2023=0 THEN NULL
       ELSE (rev_4 = rev_2024) -- dummy to keep syntax highlighters calm
  END;
"""
            # Fix the accidental dummy line and compute pct properly
            sql = sql.replace("(rev_4 = rev_2024) -- dummy to keep syntax highlighters calm",
                              "(rev_2024 - rev_2023)*100.0/rev_2023")
            return [{"id":"S_auto1","sql": sql, "rationale": f"Auto YoY by segment on {t}"}]
    return []

def run_sql_plans(plans: List[Dict[str,str]], store: SQLStore) -> Dict[str, str]:
    views = {}
    for p in plans:
        sid = p.get("id") or f"S{len(views)+1}"
        sql = p.get("sql","")
        try:
            df = store.query(sql)
            views[sid] = df.head(50).to_csv(index=False)
        except Exception as e:
            views[sid] = f"ERROR: {e}"
    return views

# -------------------------
# Main answer function
# -------------------------

def answer_with_evidence(question: str, ticker_hint: str | None = None) -> QAResult:
    # 1) Retrieve text evidence (ticker-aware)
    docs = retrieve(question, k=SETTINGS.final_k, rerank=True, ticker=ticker_hint)
    doc_map = {f"{i+1}a": d for i, d in enumerate(docs)}  # 1a, 2a...
    text_snips = []
    for key, d in doc_map.items():
        snip = d["text"]
        if len(snip) > 450:
            snip = snip[:450] + "..."
        text_snips.append(f"[{key}] {snip}")

    # 2) Plan and execute SQL
    store = SQLStore()
    plans = plan_sql(question, store, ticker_hint=ticker_hint)

    sql_views = run_sql_plans(plans, store)

    # Fallback: synthesize a correct YoY-by-segment SQL if plans invalid or all errored
    if not sql_views or all(v.startswith("ERROR:") for v in sql_views.values()):
        auto_plans = _synthesize_yoy_segment_sql(store, ticker_hint)
        if auto_plans:
            auto_views = run_sql_plans(auto_plans, store)
            if any(not v.startswith("ERROR:") for v in auto_views.values()):
                plans = auto_plans
                sql_views = auto_views

    store.close()

    # 3) Bind claims & draft answer (JSON only)
    client = _groq_client()
    sql_ids = ", ".join([p["id"] for p in plans]) if plans else "(none)"
    binder_user = f"""Question: {question}

Text evidence:
{chr(10).join(text_snips)}

SQL results (CSV preview) — IDs available: {sql_ids}
{chr(10).join([f"[{k}]\\n{v[:800]}" for k,v in sql_views.items()])}
"""

    r = client.chat.completions.create(
        model=MODELS.groq_chat_model,
        messages=[
            {"role":"system","content":SYS_CLAIM_BINDER},
            {"role":"user","content":binder_user},
        ],
        temperature=0.1,
    )
    raw = (r.choices[0].message.content or "").strip()

    # Parse binder JSON robustly
    try:
        js = json.loads(raw)
    except Exception:
        # Fallback: craft a minimal answer if the LLM didn't return JSON
        js = {"answer_markdown": raw, "claims": []}

    # Normalize claims
    claims = []
    for c in js.get("claims", []):
        claims.append(Claim(text=c.get("text",""), doc_refs=c.get("doc_refs",[]), sql_refs=c.get("sql_refs",[])))

    # Compose citation map
    citations = {"docs": doc_map, "sql": sql_views, "plans": plans}

    # 4) Judge faithfulness (robust parser)
    from .verification import score_faithfulness
    faith = score_faithfulness(question, js.get("answer_markdown",""), text_snips, sql_views)

    # 5) Abstain policy
    abstain = (faith < SETTINGS.min_faithfulness) or ("not enough evidence" in js.get("answer_markdown","").lower())

    return QAResult(
        answer_markdown=js.get("answer_markdown",""),
        claims=claims,
        citations=citations,
        faithfulness=faith,
        abstained=abstain,
    )
