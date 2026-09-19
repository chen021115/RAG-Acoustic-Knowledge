from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import Settings
from data_process import build_chunks_from_directory
from rag_chain import answer_question, create_llm, format_sources, format_context
from hybrid_retriever import BM25Index, hybrid_retrieve, load_documents_from_faiss_pickle
from reranker import BGEReranker
from vector_store import (
    build_faiss_index,
    create_embeddings,
    load_faiss_index,
    retrieve_with_distance,
)


def get_settings() -> Settings:
    return Settings()


def cmd_build(args: argparse.Namespace) -> None:
    settings = get_settings()
    docs_dir = settings.resolve_docs_dir()
    index_dir = settings.resolve_index_dir()

    print(f"[1/3] Loading PDFs from: {docs_dir}")
    chunks = build_chunks_from_directory(
        docs_dir,
        chunk_size=args.chunk_size or settings.chunk_size,
        chunk_overlap=args.chunk_overlap or settings.chunk_overlap,
    )
    print(f"      Generated {len(chunks)} chunks")

    print(f"[2/3] Loading embedding model: {settings.embedding_model}")
    embeddings = create_embeddings(
        settings.embedding_model, device=settings.embedding_device
    )

    print(f"[3/3] Building FAISS index at: {index_dir}")
    build_faiss_index(
        chunks,
        embeddings,
        index_dir,
        manifest_extra={
            "embedding_model": settings.embedding_model,
            "chunk_size": args.chunk_size or settings.chunk_size,
            "chunk_overlap": args.chunk_overlap or settings.chunk_overlap,
        },
    )
    print("Done.")


def load_store():
    settings = get_settings()
    embeddings = create_embeddings(
        settings.embedding_model, device=settings.embedding_device
    )
    return settings, load_faiss_index(settings.resolve_index_dir(), embeddings)


def cmd_search(args: argparse.Namespace) -> None:
    settings, store = load_store()
    results = retrieve_with_distance(store, args.question, top_k=args.top_k or settings.top_k)

    for idx, (doc, distance) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"\n[{idx}] distance={distance:.4f} | {source} | page={page}")
        print(doc.page_content[:800].strip())


def cmd_ask(args: argparse.Namespace) -> None:
    settings, store = load_store()
    llm = create_llm(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.temperature,
    )
    result = answer_question(
        store,
        args.question,
        top_k=args.top_k or settings.top_k,
        llm=llm,
    )
    print("\n=== Answer ===\n")
    print(result.answer)
    print("\n=== Retrieved sources ===\n")
    print(format_sources(result.documents))


def cmd_chat(args: argparse.Namespace) -> None:
    settings, store = load_store()
    llm = create_llm(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.temperature,
    )
    top_k = args.top_k or settings.top_k

    print("RAG interactive mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        result = answer_question(store, question, top_k=top_k, llm=llm)
        print("\nRAG> " + result.answer)
        print("\nSources:\n" + format_sources(result.documents))



def bm25_path(settings: Settings) -> Path:
    return settings.resolve_index_dir().parent / "bm25.pkl"


def cmd_build_bm25(_: argparse.Namespace) -> None:
    settings = get_settings()
    docs = load_documents_from_faiss_pickle(settings.resolve_index_dir())
    print(f"Loaded {len(docs)} existing chunks from FAISS index.pkl (no PDF parsing, no re-embedding).")
    index = BM25Index.build(docs)
    out = bm25_path(settings)
    index.save(out)
    print(f"BM25 index saved: {out}")


def _hybrid_docs(question: str, final_k: int, candidate_k: int, use_reranker: bool):
    settings, store = load_store()
    bm25 = BM25Index.load(bm25_path(settings))
    candidates = hybrid_retrieve(store, bm25, question, final_k=candidate_k, candidate_k=candidate_k)
    if use_reranker:
        rr = BGEReranker(settings.reranker_model, device=settings.reranker_device)
        return settings, rr.rerank(question, candidates, top_k=final_k)
    return settings, candidates[:final_k]


def cmd_hybrid_search(args: argparse.Namespace) -> None:
    final_k = args.top_k or get_settings().top_k
    _, docs = _hybrid_docs(args.question, final_k, args.candidate_k, False)
    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] {d.metadata.get('source','unknown')} | page={d.metadata.get('page','?')}")
        print(d.page_content[:800].strip())


def cmd_hybrid_ask(args: argparse.Namespace) -> None:
    final_k = args.top_k or get_settings().top_k
    settings, docs = _hybrid_docs(args.question, final_k, args.candidate_k, args.rerank)
    llm = create_llm(settings.llm_model, settings.llm_api_key, settings.llm_base_url, settings.temperature)
    prompt = ("你是水声与信号处理方向的专业文献问答助手。只能依据给定检索上下文回答；关键结论用[1][2]标注；证据不足明确说明。\n\n"
              f"用户问题：{args.question}\n\n检索上下文：\n{format_context(docs)}")
    answer = llm.invoke(prompt).content
    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Retrieved sources ===\n")
    print(format_sources(docs))


def cmd_stats(_: argparse.Namespace) -> None:
    settings = get_settings()
    manifest_path = settings.resolve_index_dir() / "manifest.json"
    if not manifest_path.exists():
        print("No manifest found. Run `python main.py build` first.")
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Water-acoustic literature RAG knowledge base"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Parse PDFs and build FAISS index")
    p_build.add_argument("--chunk-size", type=int, default=None)
    p_build.add_argument("--chunk-overlap", type=int, default=None)
    p_build.set_defaults(func=cmd_build)

    p_search = sub.add_parser("search", help="Run semantic retrieval only")
    p_search.add_argument("question")
    p_search.add_argument("--top-k", type=int, default=None)
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="Run retrieval + LLM generation")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive RAG QA")
    p_chat.add_argument("--top-k", type=int, default=None)
    p_chat.set_defaults(func=cmd_chat)


    p_bm25 = sub.add_parser("build-bm25", help="Build lightweight BM25 index from existing FAISS chunks; does not rebuild FAISS")
    p_bm25.set_defaults(func=cmd_build_bm25)

    p_hs = sub.add_parser("hybrid-search", help="FAISS + BM25 reciprocal-rank-fusion retrieval")
    p_hs.add_argument("question")
    p_hs.add_argument("--top-k", type=int, default=None)
    p_hs.add_argument("--candidate-k", type=int, default=20)
    p_hs.set_defaults(func=cmd_hybrid_search)

    p_ha = sub.add_parser("hybrid-ask", help="Hybrid RAG; add --rerank for BGE reranking")
    p_ha.add_argument("question")
    p_ha.add_argument("--top-k", type=int, default=None)
    p_ha.add_argument("--candidate-k", type=int, default=20)
    p_ha.add_argument("--rerank", action="store_true")
    p_ha.set_defaults(func=cmd_hybrid_ask)

    p_stats = sub.add_parser("stats", help="Show current index metadata")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
