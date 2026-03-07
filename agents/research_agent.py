"""
Research Agent for DocIntel

Responsible for:
- retrieving relevant document chunks
- constructing context
- generating an answer using the LLM
"""

from providers.llm_provider import get_llm_provider
from config.settings import (
    MAX_CONTEXT_CHUNKS,
    VECTOR_TOP_K,
)

class ResearchAgent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = get_llm_provider()

    def _build_context(self, documents):
        """
        Combine retrieved documents into a structured context string
        including metadata for traceability.
        """

        context_blocks = []

        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")

            block = f"""
                    [Source: {source} | Page: {page}]
                    {doc.page_content}
                    """
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def generate(self, query: str, k: int = VECTOR_TOP_K):

        retrieved_docs = self.retriever.retrieve(query, k=k)

        # limit chunks used for LLM context
        context_docs = retrieved_docs[:MAX_CONTEXT_CHUNKS]

        context = self._build_context(context_docs)

        prompt = f"""
        You are a financial research assistant.

        Answer the question using ONLY the information in the provided context.

        Rules:
        - Do NOT use outside knowledge.
        - Do NOT guess.
        - If the answer is not present in the context, say:
        "The information is not available in the provided documents."
        - When possible, rely on the exact wording from the context.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        answer = self.llm.invoke(prompt)

        seen = set()
        sources = []

        for doc in context_docs:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            entry = f"{src} — page {page}"

            if entry not in seen:
                sources.append(entry)
                seen.add(entry)

        return {
            "draft_answer": answer,
            "documents": context_docs,
            "context": context,
            "sources": sources,
        }