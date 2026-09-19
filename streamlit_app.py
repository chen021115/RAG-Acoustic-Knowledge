from __future__ import annotations

import streamlit as st

from config import Settings
from rag_chain import answer_question, create_llm, format_sources
from vector_store import create_embeddings, load_faiss_index

st.set_page_config(page_title="Water Acoustic RAG", page_icon="🌊", layout="wide")
st.title("🌊 水声领域文献知识库 RAG")
st.caption("PDF → Chunking → BGE Embedding → FAISS → LLM with citations")

settings = Settings()


@st.cache_resource
def load_resources():
    embeddings = create_embeddings(
        settings.embedding_model, device=settings.embedding_device
    )
    store = load_faiss_index(settings.resolve_index_dir(), embeddings)
    llm = create_llm(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.temperature,
    )
    return store, llm


with st.sidebar:
    st.subheader("检索参数")
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=settings.top_k)
    st.write(f"Embedding: `{settings.embedding_model}`")
    st.write(f"LLM: `{settings.llm_model}`")

question = st.text_area(
    "请输入水声/信号处理相关问题",
    placeholder="例如：声速剖面变化为什么会影响水下定位精度？",
    height=100,
)

if st.button("检索并回答", type="primary", disabled=not question.strip()):
    try:
        store, llm = load_resources()
        with st.spinner("正在检索文献并生成回答..."):
            result = answer_question(store, question, top_k=top_k, llm=llm)
        st.subheader("回答")
        st.markdown(result.answer)
        st.subheader("检索来源")
        st.code(format_sources(result.documents), language="text")
        with st.expander("查看检索片段"):
            for idx, doc in enumerate(result.documents, start=1):
                st.markdown(
                    f"**[{idx}] {doc.metadata.get('source')} - 第 {doc.metadata.get('page')} 页**"
                )
                st.write(doc.page_content)
    except Exception as exc:
        st.error(str(exc))
