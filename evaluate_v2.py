from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from config import Settings
from hybrid_retriever import BM25Index, hybrid_retrieve
from reranker import BGEReranker
from vector_store import create_embeddings, load_faiss_index


def is_hit(docs, expected):
    if not expected:
        return False, None
    for rank, d in enumerate(docs, 1):
        s = str(d.metadata.get("source", "")).lower()
        if any(x.lower() in s for x in expected):
            return True, rank
    return False, None


def metrics(rows):
    n = len(rows) or 1
    return {
        "questions": len(rows),
        "hit_rate": sum(r["hit"] for r in rows) / n,
        "mrr": sum((1.0/r["rank"]) if r["rank"] else 0.0 for r in rows) / n,
        "avg_latency_s": sum(r["latency_s"] for r in rows) / n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="eval_questions_30.json")
    p.add_argument("--mode", choices=["faiss","hybrid","rerank","all"], default="all")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--out", default="evaluation_results.csv")
    args = p.parse_args()

    s = Settings()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    emb = create_embeddings(s.embedding_model, device=s.embedding_device)
    store = load_faiss_index(s.resolve_index_dir(), emb)
    bm25 = BM25Index.load(s.resolve_index_dir().parent / "bm25.pkl") if args.mode in {"hybrid","rerank","all"} else None
    reranker = None
    if args.mode in {"rerank","all"}:
        reranker = BGEReranker(s.reranker_model, device=s.reranker_device)

    modes = [args.mode] if args.mode != "all" else ["faiss","hybrid","rerank"]
    all_rows = []
    for mode in modes:
        rows=[]
        print(f"\n=== {mode.upper()} ===")
        for i, case in enumerate(cases,1):
            q=case["question"]; expected=case.get("expected_source_contains",[])
            t0=time.perf_counter()
            if mode=="faiss":
                docs=store.similarity_search(q,k=args.top_k)
            else:
                candidates=hybrid_retrieve(store,bm25,q,final_k=args.candidate_k,candidate_k=args.candidate_k)
                docs=candidates[:args.top_k] if mode=="hybrid" else reranker.rerank(q,candidates,top_k=args.top_k)
            latency=time.perf_counter()-t0
            hit,rank=is_hit(docs,expected)
            row={"mode":mode,"id":i,"question":q,"hit":int(hit),"rank":rank or "","latency_s":round(latency,4),"top_sources":" | ".join(str(d.metadata.get('source','')) for d in docs)}
            rows.append(row); all_rows.append(row)
            print(f"[{i:02d}] {'HIT' if hit else 'MISS'} rank={rank or '-'} {q}")
        print(json.dumps(metrics(rows),ensure_ascii=False,indent=2))

    with Path(args.out).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=all_rows[0].keys()); w.writeheader(); w.writerows(all_rows)
    print(f"\nSaved: {args.out}")

if __name__=="__main__":
    main()
