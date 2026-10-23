import re
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    GEMINI_MODEL,
    TOP_K,
    NOT_FOUND_RESPONSE,
    get_api_key,
)


class RAGException(Exception):
    """Base exception for RAG pipeline errors."""
    pass


class MissingAPIKeyError(RAGException):
    """Raised when the Gemini API key is missing or not configured."""
    pass


class InvalidAPIKeyError(RAGException):
    """Raised when the Gemini API key is rejected as invalid or unauthorized."""
    pass


class APIQuotaError(RAGException):
    """Raised when the Gemini API quota limit is reached."""
    pass


class IndexNotReadyError(RAGException):
    """Raised when the vector store is not found or has not been indexed."""
    pass


SYSTEM_PROMPT = """You are the Orient BlackSwan History Bot, an academic RAG assistant for the Orient BlackSwan History 2024-25 catalogue and publication documents.

CRITICAL INSTRUCTIONS - STRICT ADHERENCE REQUIRED:
1. STRICT DOCUMENT GROUNDING: Answer the question ONLY and ENTIRELY from the provided Context Passages below. You must NEVER use outside knowledge, prior general knowledge, or assumptions not explicitly stated in the context.
2. STRICT "Not found" RULE: If the answer cannot be determined directly and factually from the provided Context Passages, or if the evidence is insufficient, ambiguous, or absent, respond with EXACTLY:
Not found
Do NOT add any apologies, commentary, pleasantries, or outside facts when the evidence is missing. Output only: Not found
3. CITATIONS & EXCERPTS: When answering based on evidence, you must include the specific PDF page number(s) (e.g., [Page X]) in your explanation and provide a brief supporting excerpt from the text.
4. NO SHORTCUTS: Formulate a clear, direct, and scholarly response backed only by the text.
"""

USER_PROMPT_TEMPLATE = """Context Passages:
{context}

Question:
{question}

Answer:"""


def format_context_passages(documents: List[Document]) -> str:
    """
    Formats retrieved LangChain documents into indexed passages with explicit page metadata.
    """
    if not documents:
        return "No passages retrieved."

    formatted = []
    for idx, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Document")
        formatted.append(
            f"--- Passage {idx} (Source: {source}, Page: {page}) ---\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted)


def extract_supporting_sources(documents: List[Document], max_excerpt_len: int = 250) -> List[Dict[str, Any]]:
    """
    Extracts structured page and excerpt metadata from retrieved documents.
    """
    sources = []
    seen = set()

    for doc in documents:
        page = doc.metadata.get("page", 1)
        source_name = doc.metadata.get("source", "History_2024-25.pdf")
        content = doc.page_content.strip()
        
        # Deduplicate identical page and prefix
        key = (page, content[:60])
        if key in seen:
            continue
        seen.add(key)

        excerpt = content if len(content) <= max_excerpt_len else content[:max_excerpt_len].rstrip() + "..."
        sources.append({
            "page": page,
            "source": source_name,
            "excerpt": excerpt,
            "full_content": content,
            "chunk_id": doc.metadata.get("chunk_id", ""),
        })

    return sources


def normalize_response(raw_answer: str) -> str:
    """
    Normalizes the response to strictly 'Not found' if the model output indicates
    absence of evidence or matches variants of 'not found'.
    """
    stripped = raw_answer.strip()

    # Exact or near-exact match
    if stripped.lower() in ["not found", "not found.", "not found!", "not-found", "'not found'"]:
        return NOT_FOUND_RESPONSE

    # If the response starts with "Not found" or indicates insufficient evidence
    if re.match(r"^not\s+found\b", stripped, re.IGNORECASE):
        # If it's just "Not found: reasons...", enforce strict "Not found"
        return NOT_FOUND_RESPONSE

    return stripped


class HistoryRAGService:
    """
    Orchestrates retrieval and generation for the Orient BlackSwan History Bot.
    """

    def __init__(
        self,
        vector_store: Any,
        llm: Optional[Any] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        top_k: int = TOP_K,
    ):
        self.vector_store = vector_store
        # If api_key is explicitly provided (even as empty string), do not fall back to get_api_key()
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = get_api_key()

        self.model_name = model_name or GEMINI_MODEL
        self.top_k = top_k

        if llm is not None:
            self.llm = llm
        else:
            if not self.api_key:
                raise MissingAPIKeyError(
                    "Gemini API key is not configured. Please supply a valid GEMINI_API_KEY."
                )
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.0,  # Zero temperature for strict deterministic grounding
            )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT_TEMPLATE),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def query(self, question: str) -> Dict[str, Any]:
        """
        Executes RAG pipeline:
        1. Semantic retrieval with top_k passages.
        2. Strict evidence-grounded generation.
        3. Returns answer, is_found flag, citations, and excerpts.
        """
        clean_question = question.strip()
        if not clean_question:
            return {
                "answer": NOT_FOUND_RESPONSE,
                "is_found": False,
                "sources": [],
                "retrieved_docs": [],
            }

        # 1. Retrieve relevant passages
        try:
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            docs: List[Document] = retriever.invoke(clean_question)
        except Exception as e:
            err_msg = str(e).lower()
            if "quota" in err_msg or "resource_exhausted" in err_msg or "429" in err_msg:
                raise APIQuotaError("Gemini API quota exceeded during embedding retrieval.") from e
            if "api key" in err_msg or "invalid" in err_msg or "unauthorized" in err_msg:
                raise InvalidAPIKeyError("Invalid or unauthorized Gemini API key.") from e
            raise RAGException(f"Retrieval error: {str(e)}") from e

        if not docs:
            return {
                "answer": NOT_FOUND_RESPONSE,
                "is_found": False,
                "sources": [],
                "retrieved_docs": [],
            }

        # Format context
        context_str = format_context_passages(docs)

        # 2. Invoke Gemini LLM
        try:
            raw_response = self.chain.invoke({
                "context": context_str,
                "question": clean_question,
            })
        except Exception as e:
            err_msg = str(e).lower()
            if "quota" in err_msg or "resource_exhausted" in err_msg or "429" in err_msg:
                raise APIQuotaError(
                    "Gemini API quota limit reached. Please wait a moment or check your Google AI Studio plan."
                ) from e
            if "api key" in err_msg or "invalid" in err_msg or "unauthorized" in err_msg:
                raise InvalidAPIKeyError(
                    "Invalid or unauthorized Gemini API key. Please verify your GEMINI_API_KEY."
                ) from e
            raise RAGException(f"LLM generation failed: {str(e)}") from e

        # 3. Post-process answer and citations
        normalized = normalize_response(raw_response)
        is_found = (normalized != NOT_FOUND_RESPONSE)

        sources = extract_supporting_sources(docs) if is_found else []

        return {
            "answer": normalized,
            "is_found": is_found,
            "sources": sources,
            "retrieved_docs": docs,
        }
