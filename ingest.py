#!/usr/bin/env python3
"""
Orient BlackSwan History Bot - Document Ingestion & Vector Indexing Script
Extracts text from PDF, preserves page metadata, chunks with overlap,
and builds a persistent local ChromaDB vector store.
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PDF_PATH,
    CHROMA_PERSIST_DIRECTORY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    get_api_key,
    is_api_key_configured,
)
from src.document_loader import load_and_chunk_pdf, get_pdf_metadata
from src.vector_store import (
    index_documents,
    is_vectorstore_indexed,
    get_index_stats,
    get_embedding_model,
)


def run_ingestion(
    pdf_path: str = PDF_PATH,
    persist_dir: str = CHROMA_PERSIST_DIRECTORY,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    force_reindex: bool = False,
) -> bool:
    print("=" * 60)
    print("  ORIENT BLACKSWAN HISTORY BOT - INGESTION PIPELINE")
    print("=" * 60)

    # 1. Check API Key
    if not is_api_key_configured():
        print("❌ Error: Gemini API Key is not configured!")
        print("   Please set GEMINI_API_KEY in your .env file or environment.")
        print("   See .env.example for details.")
        return False

    print(f"🔑 Gemini API Key: Configured")
    print(f"🧠 Embedding Model: {EMBEDDING_MODEL}")

    # 2. Check Document
    pdf_meta = get_pdf_metadata(pdf_path)
    if not pdf_meta["exists"]:
        print(f"❌ Error: PDF file not found at: {pdf_path}")
        print("   Please ensure the PDF is placed at the specified path.")
        return False

    print(f"📄 Target PDF: {pdf_meta['filename']} ({pdf_meta['size_mb']} MB, {pdf_meta['pages']} pages)")
    print(f"📂 Persist Directory: {persist_dir}")

    # 3. Check existing index
    if is_vectorstore_indexed(persist_dir) and not force_reindex:
        stats = get_index_stats(persist_dir)
        print(f"ℹ️  Index already exists with {stats['count']} chunks.")
        print("   Use --force to overwrite and re-index the document.")
        print("=" * 60)
        return True

    # Safely clean up previous or partial index directory
    if Path(persist_dir).exists():
        print("🔄 Safely cleaning up previous or partial index directory...")
        import shutil
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            print(f"⚠️  Warning: could not cleanly remove {persist_dir}: {e}")

    # 4. Extract and Chunk PDF
    print(f"\n[1/3] Extracting text and chunking PDF (size={chunk_size}, overlap={chunk_overlap})...")
    start_time = time.time()
    try:
        chunks = load_and_chunk_pdf(
            pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    except Exception as e:
        print(f"❌ Error during PDF extraction: {e}")
        return False

    extraction_time = time.time() - start_time
    print(f"   ✓ Extracted and generated {len(chunks)} chunks in {extraction_time:.2f}s")
    if chunks:
        pages_covered = len({c.metadata.get('page') for c in chunks})
        print(f"   ✓ Preserved metadata across {pages_covered} distinct pages.")

    # 5. Initialize Embeddings Model (with gemini-embedding-001 candidate support)
    print(f"\n[2/3] Initializing Google Generative AI Embeddings ({EMBEDDING_MODEL})...")
    try:
        embeddings = get_embedding_model()
        # Verify connectivity
        embeddings.embed_query("ping")
    except Exception as e:
        err_msg = str(e).lower()
        if "404" in err_msg or "not found" in err_msg or "not supported" in err_msg:
            candidates = ["models/gemini-embedding-001", "gemini-embedding-001", "models/embedding-001", "embedding-001"]
            matched = None
            for c in candidates:
                try:
                    test_emb = get_embedding_model(model_name=c)
                    test_emb.embed_query("ping")
                    matched = test_emb
                    print(f"   ℹ️  Switched to active candidate model: {c}")
                    break
                except Exception:
                    continue
            if matched:
                embeddings = matched
            else:
                print(f"❌ Error initializing embeddings: {e}")
                return False
        else:
            print(f"❌ Error initializing embeddings: {e}")
            return False
    print("   ✓ Embeddings client initialized and verified successfully.")

    # 6. Index into ChromaDB
    print(f"\n[3/3] Generating embeddings and persisting to ChromaDB...")
    index_start = time.time()

    def progress_callback(processed: int, total: int):
        pct = (processed / total) * 100
        sys.stdout.write(f"\r   Indexing progress: {processed}/{total} chunks ({pct:.1f}%)")
        sys.stdout.flush()

    try:
        index_documents(
            documents=chunks,
            persist_directory=persist_dir,
            embedding_function=embeddings,
            batch_size=80,
            progress_callback=progress_callback,
        )
        print()
    except Exception as e:
        print(f"\n❌ Error during vector indexing: {e}")
        return False

    indexing_time = time.time() - index_start
    total_time = time.time() - start_time

    print(f"   ✓ Successfully indexed and persisted to {persist_dir} in {indexing_time:.2f}s")
    print("\n" + "=" * 60)
    print(f"🎉 Ingestion Complete! Total time: {total_time:.2f}s")
    print(f"   Total chunks stored: {len(chunks)}")
    print("   You can now run: streamlit run app.py")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Orient BlackSwan History Bot - Ingestion & Indexing"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=PDF_PATH,
        help=f"Path to PDF document (default: {PDF_PATH})",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=CHROMA_PERSIST_DIRECTORY,
        help=f"ChromaDB persist directory (default: {CHROMA_PERSIST_DIRECTORY})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk character size (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Chunk character overlap (default: {CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if persistent index exists",
    )

    args = parser.parse_args()

    success = run_ingestion(
        pdf_path=args.pdf,
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_reindex=args.force,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
