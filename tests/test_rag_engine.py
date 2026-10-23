import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.rag_engine import (
    HistoryRAGService,
    normalize_response,
    format_context_passages,
    extract_supporting_sources,
    MissingAPIKeyError,
    APIQuotaError,
    InvalidAPIKeyError,
    NOT_FOUND_RESPONSE,
)


def test_normalize_response_exact_and_variants():
    assert normalize_response("Not found") == "Not found"
    assert normalize_response("Not found.") == "Not found"
    assert normalize_response("not found!") == "Not found"
    assert normalize_response("  not found  ") == "Not found"
    assert normalize_response("Not found: no evidence in context") == "Not found"
    assert normalize_response("Orient BlackSwan was incorporated in 1948 [Page 2]") == "Orient BlackSwan was incorporated in 1948 [Page 2]"


def test_format_context_passages():
    docs = [
        Document(page_content="Text on page 2", metadata={"page": 2, "source": "test.pdf"}),
        Document(page_content="Text on page 4", metadata={"page": 4, "source": "test.pdf"}),
    ]
    formatted = format_context_passages(docs)
    assert "Passage 1 (Source: test.pdf, Page: 2)" in formatted
    assert "Text on page 2" in formatted
    assert "Passage 2 (Source: test.pdf, Page: 4)" in formatted


def test_extract_supporting_sources():
    docs = [
        Document(page_content="Short excerpt", metadata={"page": 5, "source": "test.pdf", "chunk_id": "p5_c0"}),
        Document(page_content="Another excerpt", metadata={"page": 9, "source": "test.pdf", "chunk_id": "p9_c0"}),
    ]
    sources = extract_supporting_sources(docs)
    assert len(sources) == 2
    assert sources[0]["page"] == 5
    assert sources[0]["excerpt"] == "Short excerpt"
    assert sources[1]["page"] == 9


def test_grounded_answer_and_citations(tmp_path, sample_documents, mock_embeddings, mock_llm):
    # Setup temporary chroma vectorstore
    persist_dir = str(tmp_path / "rag_chroma")
    vector_store = Chroma.from_documents(
        documents=sample_documents,
        embedding=mock_embeddings,
        persist_directory=persist_dir,
    )

    rag = HistoryRAGService(
        vector_store=vector_store,
        llm=mock_llm,
        api_key="test-key",
    )

    res = rag.query("When was Orient BlackSwan incorporated?")

    assert res["is_found"] is True
    assert "1948" in res["answer"]
    assert "[Page 2]" in res["answer"]
    assert len(res["sources"]) > 0
    assert any(s["page"] == 2 for s in res["sources"])


def test_not_found_behavior_for_out_of_domain_question(tmp_path, sample_documents, mock_embeddings, mock_llm):
    persist_dir = str(tmp_path / "rag_chroma_nf")
    vector_store = Chroma.from_documents(
        documents=sample_documents,
        embedding=mock_embeddings,
        persist_directory=persist_dir,
    )

    rag = HistoryRAGService(
        vector_store=vector_store,
        llm=mock_llm,
        api_key="test-key",
    )

    res = rag.query("Who designed the iPhone in 2007? (out_of_domain)")

    # Crucial requirement: Must return EXACTLY 'Not found'
    assert res["answer"] == NOT_FOUND_RESPONSE
    assert res["is_found"] is False
    assert res["sources"] == []


def test_empty_query_returns_not_found(tmp_path, sample_documents, mock_embeddings, mock_llm):
    persist_dir = str(tmp_path / "rag_chroma_empty")
    vector_store = Chroma.from_documents(
        documents=sample_documents,
        embedding=mock_embeddings,
        persist_directory=persist_dir,
    )

    rag = HistoryRAGService(
        vector_store=vector_store,
        llm=mock_llm,
        api_key="test-key",
    )

    res = rag.query("   ")
    assert res["answer"] == NOT_FOUND_RESPONSE
    assert res["is_found"] is False
    assert res["sources"] == []


def test_missing_api_key_raises_error():
    mock_store = MagicMock()
    with pytest.raises(MissingAPIKeyError):
        HistoryRAGService(vector_store=mock_store, api_key="")


def test_api_quota_error_handling():
    mock_store = MagicMock()
    # Mock retriever raising quota error
    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = Exception("429 ResourceExhausted: Quota exceeded for quota metric")
    mock_store.as_retriever.return_value = mock_retriever

    mock_llm = MagicMock()
    rag = HistoryRAGService(vector_store=mock_store, llm=mock_llm, api_key="dummy-key")

    with pytest.raises(APIQuotaError):
        rag.query("Test question")


def test_invalid_api_key_error_handling():
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = Exception("API_KEY_INVALID: API key not valid")
    mock_store.as_retriever.return_value = mock_retriever

    mock_llm = MagicMock()
    rag = HistoryRAGService(vector_store=mock_store, llm=mock_llm, api_key="dummy-key")

    with pytest.raises(InvalidAPIKeyError):
        rag.query("Test question")
