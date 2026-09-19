from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def _doc_key(doc: Document) -> str:
    m = doc.metadata
    return f"{m.get('source','')}|{m.get('page','')}|{m.get('chunk_id','')}|{hash(doc.page_content)}"


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for English-heavy papers plus Chinese queries."""
    text = text.lower()
    parts = re.findall(r"[a-z0-9_\-]+|[\u4e00-\u9fff]+", text)
    tokens: list[str] = []
    for p in parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", p):
            tokens.extend(list(p))
            if len(p) >= 2:
                tokens.extend(p[i:i+2] for i in range(len(p)-1))
        else:
            tokens.append(p)
    return tokens


def load_documents_from_faiss_pickle(index_dir: str | Path) -> list[Document]:
    """Read documents already stored in LangChain's local FAISS index.pkl.

    This does NOT rebuild embeddings or parse PDFs. Only use an index.pkl you created/trust.
    """
    p = Path(index_dir) / "index.pkl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Keep your existing FAISS index in place.")
    with p.open("rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)
    docs: list[Document] = []
    for i in sorted(index_to_docstore_id):
        doc_id = index_to_docstore_id[i]
        doc = docstore.search(doc_id)
        if isinstance(doc, Document):
            docs.append(doc)
    if not docs:
        raise ValueError("No documents found in existing FAISS index.pkl")
    return docs


@dataclass
class BM25Index:
    documents: list[Document]
    tokenized_corpus: list[list[str]]
    bm25: BM25Okapi

    @classmethod
    def build(cls, documents: Iterable[Document]) -> "BM25Index":
        docs = list(documents)
        corpus = [tokenize(d.page_content) for d in docs]
        return cls(docs, corpus, BM25Okapi(corpus))

    def search(self, query: str, k: int = 20) -> list[tuple[Document, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
        return [(self.documents[i], float(scores[i])) for i in order]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"documents": self.documents, "tokenized_corpus": self.tokenized_corpus}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}. Run `python main.py build-bm25` once.")
        with path.open("rb") as f:
            obj = pickle.load(f)
        docs = obj["documents"]
        corpus = obj["tokenized_corpus"]
        return cls(docs, corpus, BM25Okapi(corpus))


def reciprocal_rank_fusion(
    vector_docs: list[Document],
    bm25_docs: list[Document],
    top_k: int,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[Document]:
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}
    for rank, doc in enumerate(vector_docs, 1):
        key = _doc_key(doc); docs_by_key[key] = doc
        scores[key] = scores.get(key, 0.0) + vector_weight / (rrf_k + rank)
    for rank, doc in enumerate(bm25_docs, 1):
        key = _doc_key(doc); docs_by_key[key] = doc
        scores[key] = scores.get(key, 0.0) + bm25_weight / (rrf_k + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [docs_by_key[k] for k in ordered]


def hybrid_retrieve(
    store,
    bm25_index: BM25Index,
    query: str,
    final_k: int = 8,
    candidate_k: int = 20,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[Document]:
    vector_docs = store.similarity_search(query, k=candidate_k)
    bm25_docs = [d for d, _ in bm25_index.search(query, k=candidate_k)]
    return reciprocal_rank_fusion(
        vector_docs, bm25_docs, top_k=final_k,
        vector_weight=vector_weight, bm25_weight=bm25_weight,
    )
