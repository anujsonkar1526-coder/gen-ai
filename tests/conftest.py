import hashlib
import pytest
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class DeterministicMockEmbeddings(Embeddings):
    """
    Mock embeddings generator producing deterministic float vectors.
    Enables testing ChromaDB without network access or live API credentials.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        # Hash text to produce deterministic vector
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        vec = [((seed + i * 31) % 1000) / 1000.0 for i in range(self.dim)]
        # Normalize
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class MockGroundedLLM(BaseChatModel):
    """
    Mock LLM adhering to Orient BlackSwan strict grounding and 'Not found' rules.
    """

    default_response: str = "Orient BlackSwan was incorporated in 1948 [Page 2]."

    def _generate(
        self,
        messages: List[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        full_text = " ".join([m.content for m in messages if isinstance(m.content, str)])

        # Check if the question is absent from the context passages
        if "out_of_domain" in full_text.lower() or "iphone" in full_text.lower() or "not in document" in full_text.lower():
            content = "Not found"
        elif "1948" in full_text or "incorporated" in full_text.lower():
            content = "Orient BlackSwan was incorporated in 1948 with an emphasis on publishing quality [Page 2]."
        elif "sri lanka" in full_text.lower() or "margaret jones" in full_text.lower():
            content = "The book 'Critical History of Women’s Health in Modern Sri Lanka' is authored by Margaret Jones [Page 4]."
        else:
            content = self.default_response

        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "mock-grounded-gemini"


@pytest.fixture
def mock_embeddings():
    return DeterministicMockEmbeddings()


@pytest.fixture
def mock_llm():
    return MockGroundedLLM()


@pytest.fixture
def sample_documents() -> List[Document]:
    return [
        Document(
            page_content="Orient BlackSwan is one of India’s best known and most respected publishing houses. Incorporated in 1948, the consistent emphasis of our publishing programme has been on quality.",
            metadata={"source": "History_2024-25.pdf", "page": 2, "total_pages": 91, "chunk_id": "p2_c0"},
        ),
        Document(
            page_content="Critical History of Women’s Health in Modern Sri Lanka by Margaret Jones examines colonial healthcare structures and midwifery reforms.",
            metadata={"source": "History_2024-25.pdf", "page": 4, "total_pages": 91, "chunk_id": "p4_c0"},
        ),
        Document(
            page_content="The Mughal Empire and its Decline provides an authoritative study of agrarian crisis and administrative breakdowns during the later Mughal period.",
            metadata={"source": "History_2024-25.pdf", "page": 12, "total_pages": 91, "chunk_id": "p12_c0"},
        ),
    ]
