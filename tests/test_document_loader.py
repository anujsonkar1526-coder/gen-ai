import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.document_loader import (
    clean_page_text,
    chunk_documents,
    extract_pdf_pages,
    get_pdf_metadata,
)


def test_clean_page_text():
    raw = "  Hello   world! \n\n\n\n\nNew   line  "
    cleaned = clean_page_text(raw)
    assert cleaned == "Hello world!\n\nNew line"
    assert clean_page_text("") == ""
    assert clean_page_text(None) == ""


def test_chunk_documents_preserves_page_metadata():
    docs = [
        Document(
            page_content="Orient BlackSwan was incorporated in 1948. " * 30,
            metadata={"source": "History_2024-25.pdf", "page": 2, "total_pages": 91},
        ),
        Document(
            page_content="Women's Health in Sri Lanka by Margaret Jones. " * 30,
            metadata={"source": "History_2024-25.pdf", "page": 4, "total_pages": 91},
        ),
    ]

    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 2
    for chunk in chunks:
        # Crucial requirement: page metadata MUST be preserved on every chunk
        assert "page" in chunk.metadata
        assert chunk.metadata["page"] in [2, 4]
        assert chunk.metadata["source"] == "History_2024-25.pdf"
        assert chunk.metadata["total_pages"] == 91
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith("p")


def test_extract_pdf_pages_missing_file():
    with pytest.raises(FileNotFoundError) as exc_info:
        extract_pdf_pages("non_existent_path/no_file.pdf")
    assert "not found" in str(exc_info.value).lower()


def test_get_pdf_metadata_existing(tmp_path):
    # Test with real PDF if available in data/
    real_pdf = Path("data/History_2024-25.pdf")
    if real_pdf.exists():
        meta = get_pdf_metadata(str(real_pdf))
        assert meta["exists"] is True
        assert meta["pages"] == 91
        assert meta["size_mb"] > 0
    else:
        # Non-existing fallback test
        meta = get_pdf_metadata(str(tmp_path / "missing.pdf"))
        assert meta["exists"] is False
        assert meta["pages"] == 0
