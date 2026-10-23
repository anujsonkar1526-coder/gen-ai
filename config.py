import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# API Key configuration (accepts GEMINI_API_KEY or GOOGLE_API_KEY)
def get_api_key() -> str:
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()
    if key in [
        "your_gemini_api_key_here",
        "your_api_key_here",
        "your_google_api_key_here",
        "None",
    ]:
        return ""
    return key

def is_api_key_configured() -> bool:
    return bool(get_api_key())

# Document & Persistence Paths
PDF_PATH = os.getenv("PDF_PATH", str(BASE_DIR / "data" / "History_2024-25.pdf"))
CHROMA_PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "chroma_db")
)

# Gemini Model Names (gemini-1.5-flash is universally supported across SDKs)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Embedding Model (models/gemini-embedding-001 is the active stable Gemini embedding model)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Chunking & Retrieval Parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Fallback phrase required by specification when information is absent or insufficient
NOT_FOUND_RESPONSE = "Not found"
