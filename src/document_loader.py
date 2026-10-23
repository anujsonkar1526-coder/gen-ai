import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_page_text(text: str) -> str:
    """
    Cleans extracted raw text while preserving linguistic structures and word boundaries.
    """
    if not text:
        return ""
    # Normalize horizontal spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Remove trailing/leading spaces on lines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    # Normalize excessive consecutive newlines to maximum of 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: str) -> List[Document]:
    """
    Extracts text from each page of a PDF file, strictly preserving page metadata.
    Page numbering is 1-indexed (Page 1 = first page of PDF).

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        ValueError: If the file is not a valid PDF or yields zero extractable text.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as e:
        raise ValueError(f"Failed to read PDF file '{path.name}': {str(e)}") from e

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"PDF document '{path.name}' contains 0 pages.")

    documents: List[Document] = []
    source_name = path.name

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1  # 1-indexed page number
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        cleaned = clean_page_text(raw_text)
        if cleaned:
            doc = Document(
                page_content=cleaned,
                metadata={
                    "source": source_name,
                    "page": page_num,
                    "total_pages": total_pages,
                },
            )
            documents.append(doc)

    if not documents:
        raise ValueError(
            f"PDF document '{source_name}' contains no extractable text. "
            "It may be composed entirely of scanned images without OCR."
        )

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Splits page-level documents into overlapping chunks while preserving original
    page and source metadata across every split chunk.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata with unique chunk identifier
    for idx, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 1)
        chunk.metadata["chunk_id"] = f"p{page}_c{idx}"

    return chunks


def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    End-to-end extraction and chunking pipeline for a PDF document.
    """
    pages = extract_pdf_pages(pdf_path)
    return chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def get_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Returns summary metadata about the target PDF file.
    """
    path = Path(pdf_path)
    if not path.is_file():
        return {
            "exists": False,
            "filename": path.name,
            "path": str(path),
            "size_bytes": 0,
            "size_mb": 0.0,
            "pages": 0,
        }

    size_bytes = path.stat().st_size
    size_mb = round(size_bytes / (1024 * 1024), 2)

    try:
        reader = pypdf.PdfReader(str(path))
        page_count = len(reader.pages)
    except Exception:
        page_count = 0

    return {
        "exists": True,
        "filename": path.name,
        "path": str(path),
        "size_bytes": size_bytes,
        "size_mb": size_mb,
        "pages": page_count,
    }
