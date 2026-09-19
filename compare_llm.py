from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from config import Settings
from hybrid_retriever import BM25Index, hybrid_retrieve
from rag_chain import create_llm, format_context, format_sources
from reranker import BGEReranker
from vector_store import create_embeddings, load_faiss_index


def main():
    p=argparse.ArgumentParser(description="Generate Pure-LLM vs RAG answers for manual comparison")
    p.add_argument("--cases",default="eval_questions_30.json")
    p.add_argument("--limit",type=int,default=10)
    p.add_argument("--top-k",type=int,default=4)
    p.add_argument("--candidate-k",type=int,default=20)
    p.add_argument("--out",default="pure_llm_vs_rag.csv")
    a=p.parse_args(); s=Settings()
    cases=json.loads(Path(a.cases).read_text(encoding="utf-8"))[:a.limit]
    emb=create_embeddings(s.embedding_model,device=s.embedding_device)
    store=load_faiss_index(s.resolve_index_dir(),emb)
    bm25=BM25Index.load(s.resolve_index_dir().parent/"bm25.pkl")
    rr=BGEReranker(s.reranker_model,device=s.reranker_device)
    llm=create_llm(s.llm_model,s.llm_api_key,s.llm_base_url,s.temperature)
    rows=[]
    for i,c in enumerate(cases,1):
        q=c["question"]; print(f"[{i}/{len(cases)}] {q}")
        pure=llm.invoke("请直接根据你自身知识回答下面问题，不要假装引用外部文献。\n\n"+q).content
        candidates=hybrid_retrieve(store,bm25,q,final_k=a.candidate_k,candidate_k=a.candidate_k)
        docs=rr.rerank(q,candidates,top_k=a.top_k)
        prompt=("你是水声专业文献问答助手。只能依据给定文献上下文回答；关键结论用[1][2]标注；证据不足就明确说明。\n\n"
                f"问题：{q}\n\n文献上下文：\n{format_context(docs)}")
        rag=llm.invoke(prompt).content
        rows.append({"question":q,"pure_llm_answer":pure,"rag_answer":rag,"rag_sources":format_sources(docs),
                     "manual_factuality_1_5":"","manual_citation_1_5":"","manual_completeness_1_5":"","notes":""})
    with Path(a.out).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(f"Saved: {a.out}\nFill the manual score columns after reviewing answers against the cited PDFs.")
if __name__=="__main__": main()
