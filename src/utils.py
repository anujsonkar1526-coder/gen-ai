import os
from pathlib import Path
from typing import Dict, Any, Optional

from config import (
    PDF_PATH,
    CHROMA_PERSIST_DIRECTORY,
    is_api_key_configured,
    get_api_key,
)
from src.document_loader import get_pdf_metadata
from src.vector_store import get_index_stats


def get_system_status(pdf_path: Optional[str] = None, chroma_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Collects complete diagnostic status of the environment, document, and vector index.
    """
    target_pdf = pdf_path or PDF_PATH
    target_chroma = chroma_dir or CHROMA_PERSIST_DIRECTORY

    pdf_meta = get_pdf_metadata(target_pdf)
    chroma_stats = get_index_stats(target_chroma)
    api_configured = is_api_key_configured()

    return {
        "api_key_configured": api_configured,
        "api_key_masked": f"{get_api_key()[:6]}...{get_api_key()[-4:]}" if api_configured and len(get_api_key()) > 10 else ("Configured" if api_configured else "Missing"),
        "pdf": pdf_meta,
        "vector_store": chroma_stats,
        "is_ready": (
            api_configured
            and pdf_meta["exists"]
            and chroma_stats["indexed"]
        ),
    }


def format_citation_markdown(source: Dict[str, Any]) -> str:
    """
    Formats a single source reference into clean markdown.
    """
    page = source.get("page", "Unknown")
    excerpt = source.get("excerpt", "").strip()
    return f"**Page {page}**: \"{excerpt}\""
