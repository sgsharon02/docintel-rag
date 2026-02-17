"""
Research Agent for DocIntel

Responsible for:
- retrieving relevant document chunks
- constructing context
- generating an answer using the LLM
"""

from providers.llm_provider import get_llm_provider


class ResearchAgent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = get_llm_provider()

    def _build_context(self, documents):
        """
        Combine retrieved documents into a context string.
        """
        return "\n\n".join([doc.page_content for doc in documents])

    def generate(self, query: str, k: int = 5):
        """
        Returns:
        - draft_answer
        - retrieved_docs
        - context
        """

        retrieved_docs = self.retriever.retrieve(query, k=k)

        context = self._build_context(retrieved_docs)

        prompt = f"""
        You are a financial research assistant.

        Use ONLY the context below to answer the question.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        answer = self.llm.invoke(prompt)

        return {
            "draft_answer": answer,
            "documents": retrieved_docs,
            "context": context,
        }
