from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Light cleanup that keeps paragraph/equation boundaries as much as possible."""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    """Load a PDF page-by-page and preserve source/page metadata."""
    reader = PdfReader(str(pdf_path))
    pages: list[Document] = []

    for page_index, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        if not text:
            continue

        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_index,
                    "file_path": str(pdf_path.resolve()),
                },
            )
        )
    return pages


def load_pdf_directory(docs_dir: str | Path) -> list[Document]:
    """Recursively load every PDF in a directory."""
    docs_dir = Path(docs_dir)
    pdf_files = sorted(docs_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {docs_dir}. Put papers under docs/ first."
        )

    documents: list[Document] = []
    for pdf_path in pdf_files:
        documents.extend(load_pdf_pages(pdf_path))
    return documents


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Split page documents into retrieval chunks while keeping metadata."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", ";", "，", ",", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))

    for idx, chunk in enumerate(chunks):
        chunk.metadata = {
            **chunk.metadata,
            "chunk_id": idx,
            "char_count": len(chunk.page_content),
        }
    return chunks


def build_chunks_from_directory(
    docs_dir: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    pages = load_pdf_directory(docs_dir)
    return split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
