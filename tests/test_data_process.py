from data_process import clean_text, split_documents
from langchain_core.documents import Document


def test_clean_text_removes_null_and_duplicate_spaces():
    text = "A\x00  B\n\n\nC"
    assert clean_text(text) == "A B\n\nC"


def test_split_preserves_metadata():
    docs = [Document(page_content="水声定位。" * 100, metadata={"source": "demo.pdf", "page": 1})]
    chunks = split_documents(docs, chunk_size=80, chunk_overlap=10)
    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "demo.pdf"
    assert chunks[0].metadata["page"] == 1
    assert "chunk_id" in chunks[0].metadata
