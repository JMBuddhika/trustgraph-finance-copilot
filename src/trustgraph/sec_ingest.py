from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from sec_edgar_downloader import Downloader

from .config import PATHS, SETTINGS
from .sqlstore import SQLStore

import os

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def chunk_text(text: str, size=SETTINGS.chunk_chars, overlap=SETTINGS.chunk_overlap):
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i+size])
        i += max(1, size - overlap)
    return out

def parse_filing_html(fp: Path) -> Dict[str, Any]:
    html = fp.read_text(errors="ignore", encoding="utf-8", newline="")
    soup = BeautifulSoup(html, "lxml")

    # Get plain text paragraphs
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    text = soup.get_text(separator=" ")
    text = clean_text(text)

    # Extract tables to DataFrames with pandas.read_html (robust to messy HTML)
    tables = []
    try:
        for i, df in enumerate(pd.read_html(str(fp), flavor="lxml")):
            df.columns = [str(c) for c in df.columns]
            tables.append(df)
    except Exception:
        pass

    return {"text": text, "tables": tables}

def save_tables_to_duckdb(store: SQLStore, tables: List[pd.DataFrame], meta_prefix: str) -> List[str]:
    names = []
    for i, df in enumerate(tables):
        name = f'{meta_prefix}_tbl{i}'
        # sanitize
        name = re.sub(r"[^A-Za-z0-9_]", "_", name).lower()
        store.register_df(name, df)
        names.append(name)
    return names

def detect_meta_from_path(path: Path):
    # path structure from sec-edgar-downloader:
    # .../data/raw/sec/<ticker>/<form>/<accession>/*.html
    parts = path.parts
    # Find ticker & form heuristically
    ticker = "UNKNOWN"
    form = "UNK"
    if len(parts) >= 4:
        ticker = parts[-4]
        form = parts[-3]
    accession = path.parent.name
    return ticker, form, accession

def ingest(tickers: List[str], forms: List[str], limit: int = 2):
    company = os.getenv("SEC_COMPANY", "TrustGraphRAG")
    email = os.getenv("SEC_EMAIL")
    if not email:
        raise RuntimeError("SEC_EMAIL missing. Add it to .env (required by sec-edgar-downloader).")

    dl = Downloader(company, email, str(PATHS.raw_sec))
    store = SQLStore()  # opens DuckDB

    corpus_out = PATHS.corpus_jsonl
    if corpus_out.exists():
        # append mode
        pass

    with corpus_out.open("a", encoding="utf-8") as fout:
        for tkr in tickers:
            for fm in forms:
                tqdm.write(f"Downloading {fm} for {tkr} (limit={limit})")
                try:
                    dl.get(fm, tkr, amount=limit)
                except Exception as e:
                    tqdm.write(f"Download error for {tkr} {fm}: {e}")
                    continue

                html_files = list(Path(PATHS.raw_sec, tkr, fm).rglob("*.htm*"))
                for fp in tqdm(html_files, desc=f"{tkr}-{fm}", unit="file"):
                    try:
                        parsed = parse_filing_html(fp)
                    except Exception as e:
                        tqdm.write(f"Parse failed {fp}: {e}")
                        continue

                    ticker, form, accession = detect_meta_from_path(fp)
                    meta_prefix = f"{ticker}_{form}_{accession}"

                    # Save tables
                    tbl_names = save_tables_to_duckdb(store, parsed["tables"], meta_prefix)

                    # Chunk text and dump JSONL with metadata
                    for j, ch in enumerate(chunk_text(parsed["text"])):
                        rec = {
                            "id": f"{meta_prefix}_chunk{j}",
                            "ticker": ticker,
                            "form": form,
                            "accession": accession,
                            "file": str(fp),
                            "text": ch,
                            "tables": tbl_names,  # all tables from this filing
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    store.close()
    print("Ingestion complete.")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT NVDA")
    ap.add_argument("--forms", nargs="+", default=["10-K","10-Q"])
    ap.add_argument("--limit", type=int, default=2, help="How many filings per form")
    args = ap.parse_args()
    ingest(args.tickers, args.forms, args.limit)

if __name__ == "__main__":
    main()
