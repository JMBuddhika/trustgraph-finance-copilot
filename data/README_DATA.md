# Data Seed (Sample)

This folder contains a **small synthetic seed** so you can demo the app without downloading SEC filings.

## What’s inside
- `processed/corpus.jsonl` — 5 toy chunks across AAPL, MSFT, NVDA with metadata and table references.
- `sql/finance.duckdb` — DuckDB file with 3 tiny tables (if your environment supported building it here).
  - If you **don't** see `finance.duckdb`, use `sql/seed.sql` to create it locally.

## How to use
1. Copy the entire `data/` directory into your project root (so paths match `./data/...`).  
2. If `sql/finance.duckdb` is missing:
   ```bash
   duckdb data/sql/finance.duckdb -c ".read data/sql/seed.sql"
   ```
   or in Python:
   ```python
   import duckdb
   con = duckdb.connect("data/sql/finance.duckdb")
   con.execute(open("data/sql/seed.sql").read())
   con.close()
   ```
3. Build retrieval indexes from the seed corpus (optional but useful):
   ```bash
   uv run -c "from src.trustgraph.indexes import build_indexes; build_indexes()"
   ```

## Notes
- The text is synthetic and safe to share.  
- The schema uses intuitive names: `Revenue_USD_M`, `GrossMargin_Pct`, `Year`/`Quarter`, `Segment`.  
- The toy data is **not real financials**—it’s just for UI and pipeline demos.
