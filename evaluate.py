from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import Settings
from vector_store import create_embeddings, load_faiss_index, retrieve


def keyword_hit(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight retrieval evaluation")
    parser.add_argument("--cases", default="eval_questions.example.json")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    settings = Settings()
    top_k = args.top_k or settings.top_k
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))

    embeddings = create_embeddings(
        settings.embedding_model, device=settings.embedding_device
    )
    store = load_faiss_index(settings.resolve_index_dir(), embeddings)

    hits = 0
    for case in cases:
        question = case["question"]
        keywords = case.get("expected_keywords", [])
        docs = retrieve(store, question, top_k=top_k)
        merged = "\n".join(doc.page_content for doc in docs)
        ok = keyword_hit(merged, keywords)
        hits += int(ok)
        print(f"[{'HIT' if ok else 'MISS'}] {question}")
        print(f"  expected_keywords={keywords}")

    total = len(cases)
    rate = hits / total if total else 0.0
    print(f"\nKeyword Hit@{top_k}: {hits}/{total} = {rate:.2%}")
    print("Note: this is a retrieval sanity metric, not a full answer-quality metric.")


if __name__ == "__main__":
    main()
