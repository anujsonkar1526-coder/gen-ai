#!/usr/bin/env python3
"""
Orient BlackSwan History Bot - Embedding Smoke Test
Verifies the Gemini embedding integration and dimensions without leaking API keys.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

from config import get_api_key, is_api_key_configured, EMBEDDING_MODEL
from src.vector_store import get_embedding_model


def run_smoke_test():
    print("=" * 60)
    print("  ORIENT BLACKSWAN HISTORY BOT - EMBEDDING SMOKE TEST")
    print("=" * 60)

    if not is_api_key_configured():
        print("❌ Error: Valid Gemini API Key not found in .env!")
        print("   Please ensure your actual GEMINI_API_KEY is placed in .env.")
        return False

    print("🔑 API Key Status: Configured")
    print(f"🧠 Target Model: {EMBEDDING_MODEL}")

    try:
        try:
            embeddings = get_embedding_model()
            test_text = "Orient BlackSwan History 2024-25"
            vector = embeddings.embed_query(test_text)
            active_model = EMBEDDING_MODEL
        except Exception as initial_e:
            err_msg = str(initial_e).lower()
            if "404" in err_msg or "not found" in err_msg or "not supported" in err_msg:
                candidates = ["models/gemini-embedding-001", "gemini-embedding-001", "models/embedding-001", "embedding-001"]
                matched = None
                for c in candidates:
                    try:
                        test_emb = get_embedding_model(model_name=c)
                        vector = test_emb.embed_query(test_text)
                        matched = (test_emb, c)
                        break
                    except Exception:
                        continue
                if matched:
                    embeddings, active_model = matched
                else:
                    raise initial_e
            else:
                raise initial_e

        dim = len(vector)
        print(f"✅ Smoke Test Passed!")
        print(f"   Active Model: {active_model}")
        print(f"   Generated embedding vector with dimension: {dim}")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"❌ Smoke Test Failed: {type(e).__name__}: {str(e)}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
