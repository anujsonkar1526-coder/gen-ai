import shutil
import pytest
from src.vector_store import (
    index_documents,
    get_vectorstore,
    is_vectorstore_indexed,
    get_index_stats,
)


def test_index_and_retrieve(tmp_path, sample_documents, mock_embeddings):
    persist_dir = str(tmp_path / "test_chroma")

    # Before indexing
    assert is_vectorstore_indexed(persist_dir) is False

    # Index
    vector_store = index_documents(
        documents=sample_documents,
        persist_directory=persist_dir,
        embedding_function=mock_embeddings,
        batch_size=2,
    )

    # After indexing
    assert is_vectorstore_indexed(persist_dir) is True
    stats = get_index_stats(persist_dir)
    assert stats["indexed"] is True
    assert stats["count"] == len(sample_documents)

    # Reload persistent store
    reloaded_store = get_vectorstore(
        persist_directory=persist_dir,
        embedding_function=mock_embeddings,
    )

    retriever = reloaded_store.as_retriever(search_kwargs={"k": 2})
    results = retriever.invoke("Orient BlackSwan publishing")

    assert len(results) > 0
    # Check that retrieved document contains page metadata
    assert "page" in results[0].metadata
    assert results[0].metadata["source"] == "History_2024-25.pdf"


def test_index_empty_documents_raises_value_error(tmp_path, mock_embeddings):
    persist_dir = str(tmp_path / "empty_chroma")
    with pytest.raises(ValueError) as exc:
        index_documents(
            documents=[],
            persist_directory=persist_dir,
            embedding_function=mock_embeddings,
        )
    assert "cannot index an empty list" in str(exc.value).lower()
