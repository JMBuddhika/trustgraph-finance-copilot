from __future__ import annotations
import duckdb, pandas as pd, os
from dataclasses import dataclass
from typing import List, Dict, Any
from .config import PATHS

@dataclass
class SQLStore:
    path: str = str(PATHS.duckdb_path)
    conn: duckdb.DuckDBPyConnection | None = None

    def __post_init__(self):
        self.conn = duckdb.connect(self.path)
        self.conn.execute("PRAGMA threads=4;")

    def register_df(self, name: str, df: pd.DataFrame):
        self.conn.register(name, df)
        # persist as a duckdb table
        self.conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {name};")

    def tables(self) -> List[str]:
        return [r[0] for r in self.conn.execute("SHOW TABLES;").fetchall()]

    def schema_of(self, table: str) -> pd.DataFrame:
        return self.conn.execute(f"DESCRIBE {table};").fetchdf()

    def query(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).fetchdf()

    def table_summaries(self, like: str | None = None) -> Dict[str, List[str]]:
        out = {}
        for t in self.tables():
            if like and like.lower() not in t.lower():
                continue
            cols = self.schema_of(t)["column_name"].tolist()
            out[t] = cols
        return out

    def close(self):
        if self.conn: self.conn.close()
