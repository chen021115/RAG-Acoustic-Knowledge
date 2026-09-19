from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name: str, device: str = "cpu") -> HuggingFaceEmbeddings:
    """Create a local Hugging Face embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_faiss_index(
    documents: Sequence[Document],
    embeddings: HuggingFaceEmbeddings,
    index_dir: str | Path,
    manifest_extra: dict | None = None,
) -> FAISS:
    if not documents:
        raise ValueError("Cannot build an index from zero documents.")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    store = FAISS.from_documents(list(documents), embeddings)
    store.save_local(str(index_dir))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_chunks": len(documents),
        "sources": sorted({str(d.metadata.get("source", "unknown")) for d in documents}),
    }
    if manifest_extra:
        manifest.update(manifest_extra)

    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return store


def load_faiss_index(
    index_dir: str | Path,
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    index_dir = Path(index_dir)
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_dir}. Run `python main.py build` first."
        )

    # LangChain stores the docstore metadata in a local pickle file. Only load
    # indexes you created yourself or otherwise trust.
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve(store: FAISS, query: str, top_k: int = 4) -> list[Document]:
    return store.similarity_search(query, k=top_k)


def retrieve_with_distance(
    store: FAISS, query: str, top_k: int = 4
) -> list[tuple[Document, float]]:
    """Return documents with FAISS distance (smaller generally means closer)."""
    return store.similarity_search_with_score(query, k=top_k)
