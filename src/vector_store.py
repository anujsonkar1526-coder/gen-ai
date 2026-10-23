import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    CHROMA_PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
    get_api_key,
)

DEFAULT_COLLECTION_NAME = "orient_blackswan_history"


def get_embedding_model(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns a Google Generative AI embeddings client.
    Defaults to models/gemini-embedding-001 (active stable Gemini embedding model).
    """
    key = api_key if api_key is not None else get_api_key()
    if not key:
        raise ValueError(
            "Gemini API key is required to initialize GoogleGenerativeAIEmbeddings. "
            "Please configure GEMINI_API_KEY or GOOGLE_API_KEY in your .env file."
        )

    target_model = model_name or EMBEDDING_MODEL or "models/gemini-embedding-001"
    return GoogleGenerativeAIEmbeddings(
        model=target_model,
        google_api_key=key,
    )


def is_vectorstore_indexed(
    persist_directory: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> bool:
    """
    Checks if a persistent ChromaDB directory exists and contains indexed documents.
    Does not require initializing embedding models or making API calls.
    """
    target_dir = persist_directory or CHROMA_PERSIST_DIRECTORY
    dir_path = Path(target_dir)

    if not dir_path.is_dir():
        return False

    try:
        client = chromadb.PersistentClient(path=str(dir_path))
        existing_collections = [c.name for c in client.list_collections()]
        if collection_name not in existing_collections:
            return False

        coll = client.get_collection(name=collection_name)
        return coll.count() > 0
    except Exception:
        return False


def get_index_stats(
    persist_directory: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Returns statistics about the ChromaDB index without calling external APIs.
    """
    target_dir = persist_directory or CHROMA_PERSIST_DIRECTORY
    dir_path = Path(target_dir)

    if not dir_path.is_dir():
        return {
            "indexed": False,
            "count": 0,
            "directory": str(dir_path),
            "collection_name": collection_name,
        }

    try:
        client = chromadb.PersistentClient(path=str(dir_path))
        collections = [c.name for c in client.list_collections()]
        if collection_name not in collections:
            return {
                "indexed": False,
                "count": 0,
                "directory": str(dir_path),
                "collection_name": collection_name,
            }

        coll = client.get_collection(name=collection_name)
        count = coll.count()
        return {
            "indexed": count > 0,
            "count": count,
            "directory": str(dir_path),
            "collection_name": collection_name,
        }
    except Exception as e:
        return {
            "indexed": False,
            "count": 0,
            "directory": str(dir_path),
            "collection_name": collection_name,
            "error": str(e),
        }


def get_vectorstore(
    persist_directory: Optional[str] = None,
    embedding_function: Optional[Any] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """
    Loads an existing persistent Chroma vector store.
    """
    target_dir = persist_directory or CHROMA_PERSIST_DIRECTORY
    embeddings = embedding_function or get_embedding_model()

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(target_dir),
        embedding_function=embeddings,
    )


def index_documents(
    documents: List[Document],
    persist_directory: Optional[str] = None,
    embedding_function: Optional[Any] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 80,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Chroma:
    """
    Indexes a list of documents into persistent local ChromaDB storage in batches.
    Prevents API rate limiting and large payload failures.
    """
    if not documents:
        raise ValueError("Cannot index an empty list of documents.")

    target_dir = persist_directory or CHROMA_PERSIST_DIRECTORY
    embeddings = embedding_function or get_embedding_model()

    os.makedirs(target_dir, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(target_dir),
        embedding_function=embeddings,
    )

    total_docs = len(documents)
    for start_idx in range(0, total_docs, batch_size):
        end_idx = min(start_idx + batch_size, total_docs)
        batch = documents[start_idx:end_idx]
        vector_store.add_documents(batch)
        if progress_callback:
            progress_callback(end_idx, total_docs)

    return vector_store
