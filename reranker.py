from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class BGEReranker:
    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        self.model = CrossEncoder(model_name_or_path, device=device)

    def rerank(self, query: str, documents: list[Document], top_k: int = 4) -> list[Document]:
        if not documents:
            return []
        pairs = [(query, d.page_content) for d in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
        return [d for d, _ in ranked[:top_k]]
