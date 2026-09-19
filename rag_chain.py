from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from vector_store import retrieve


SYSTEM_PROMPT = """你是水声与信号处理方向的专业文献问答助手。
请严格依据给定的检索上下文回答，不要把模型自身记忆当作文献事实。

要求：
1. 只有当上下文能够支持结论时才回答；证据不足时明确说“当前知识库中未检索到足够证据”。
2. 关键技术结论后使用 [1]、[2] 这样的编号引用对应检索片段。
3. 不要编造论文题目、作者、公式、实验数值或页码。
4. 若不同文献片段存在差异，请指出差异，不要强行合并。
5. 回答尽量结构化、简洁，但保留必要的专业术语。
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "用户问题：\n{question}\n\n检索上下文：\n{context}\n\n请给出答案。",
        ),
    ]
)


@dataclass
class RAGResult:
    answer: str
    documents: list[Document]


def format_context(documents: Iterable[Document]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        blocks.append(
            f"[{idx}] 来源：{source}；页码：{page}；chunk：{chunk_id}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)


def format_sources(documents: Iterable[Document]) -> str:
    lines: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        lines.append(f"[{idx}] {source}，第 {page} 页，chunk {chunk_id}")
    return "\n".join(lines)


def create_llm(
    model: str,
    api_key: str,
    base_url: str = "",
    temperature: float = 0.1,
) -> ChatOpenAI:
    if not api_key:
        raise ValueError("LLM_API_KEY is empty. Configure .env before generation.")

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def answer_question(
    store,
    question: str,
    top_k: int,
    llm: ChatOpenAI,
) -> RAGResult:
    documents = retrieve(store, question, top_k=top_k)
    context = format_context(documents)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return RAGResult(answer=answer, documents=documents)
