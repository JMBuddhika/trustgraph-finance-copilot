from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    data: Path = root / "data"
    raw_sec: Path = Path(os.getenv("SEC_DOWNLOAD_DIR", str((root / "data" / "raw" / "sec"))))
    processed: Path = root / "data" / "processed"
    corpus_jsonl: Path = processed / "corpus.jsonl"
    index_dir: Path = root / "data" / "index"
    faiss_index: Path = index_dir / "faiss.index"
    faiss_meta: Path = index_dir / "faiss_meta.pkl"
    bm25_corpus: Path = index_dir / "bm25_corpus.pkl"
    duckdb_path: Path = Path(os.getenv("DUCKDB_PATH", str(root / "data" / "sql" / "finance.duckdb")))
    cache: Path = root / "data" / "cache"

@dataclass
class Models:
    embed_model: str = os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    groq_chat_model: str = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

@dataclass
class Settings:
    topk_dense: int = 30
    topk_bm25: int = 30
    fusion_k: int = 60
    final_k: int = 10
    rerank_k: int = 12
    min_faithfulness: float = 0.58  # below -> abstain
    chunk_chars: int = 900
    chunk_overlap: int = 150

PATHS = Paths()
MODELS = Models()
SETTINGS = Settings()

for p in [PATHS.data, PATHS.raw_sec, PATHS.processed, PATHS.index_dir, PATHS.duckdb_path.parent, PATHS.cache]:
    p.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    # We won't crash, but app will warn.
    pass
