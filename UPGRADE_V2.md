# V2 incremental upgrade (reuse existing FAISS)

This patch does **not** require downloading your PDFs again and does **not** rebuild your FAISS embeddings.

## 1. Install only one new lightweight dependency

```powershell
pip install rank-bm25
```

`sentence-transformers` is already in the original project.

## 2. Build BM25 from the chunks already stored in `index/faiss/index.pkl`

```powershell
python main.py build-bm25
```

This only tokenizes the existing ~2331 chunks and writes `index/bm25.pkl`. It does not parse PDFs and does not run BGE embedding over the corpus.

## 3. Test Hybrid Search

```powershell
python main.py hybrid-search "水声多径为什么影响定位精度？" --top-k 5
```

## 4. Test Hybrid RAG (without reranker first)

```powershell
python main.py hybrid-ask "水声多径为什么影响定位精度？"
```

## 5. Add BGE Reranker

Set `.env`:

```env
RERANKER_MODEL=D:/your/local/path/bge-reranker-base
RERANKER_DEVICE=cpu
```

If Hugging Face is reachable you may keep `BAAI/bge-reranker-base`. Given your earlier network timeout, a local path is safer.

Then:

```powershell
python main.py hybrid-ask "水声多径为什么影响定位精度？" --rerank
```

The architecture is now:

`Query -> FAISS Top20 + BM25 Top20 -> RRF Fusion -> BGE Reranker -> Top4 -> Qwen`

## 6. Retrieval evaluation: 30 professional questions

```powershell
python evaluate_v2.py --mode all --top-k 5
```

It writes `evaluation_results.csv` and compares:

- FAISS
- FAISS + BM25 Hybrid
- Hybrid + BGE Reranker

Metrics:

- Hit@K (source-title based sanity metric)
- MRR
- average retrieval latency

## 7. Pure LLM vs RAG comparison

To avoid spending too many API tokens, start with 10 questions:

```powershell
python compare_llm.py --limit 10
```

It writes `pure_llm_vs_rag.csv`. Review each pair against the cited PDFs and fill the manual 1-5 columns for factuality, citation support, and completeness.

## Important

Do **not** run `python main.py build` unless you intentionally change PDFs, chunk size, chunk overlap, or the embedding model. Your current FAISS index is reused as-is.
