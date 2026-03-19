"""
Cross-Encoder Reranker for DocIntel

Re-ranks retrieved documents using a cross-encoder model
to improve relevance before passing context to the LLM.
"""

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents, top_k: int = 5):
        """
        Re-rank retrieved documents using cross-encoder scoring.
        """

        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        for score, doc in zip(scores, documents):

            if doc.metadata is None:
                doc.metadata = {}

            doc.metadata["rerank_score"] = round(float(score), 4)

        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        top_k = min(top_k, len(scored_docs))

        reranked_docs = [doc for _, doc in scored_docs[:top_k]]

        return reranked_docs