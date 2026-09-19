from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Central configuration for the RAG project."""

    project_root: Path = Path(__file__).resolve().parent
    docs_dir: Path = Path(os.getenv("DOCS_DIR", "docs"))
    index_dir: Path = Path(os.getenv("INDEX_DIR", "index/faiss"))

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))

    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    reranker_device: str = os.getenv("RERANKER_DEVICE", "cpu")
    hybrid_candidate_k: int = int(os.getenv("HYBRID_CANDIDATE_K", "20"))

    def resolve_docs_dir(self) -> Path:
        path = self.docs_dir
        return path if path.is_absolute() else self.project_root / path

    def resolve_index_dir(self) -> Path:
        path = self.index_dir
        return path if path.is_absolute() else self.project_root / path
