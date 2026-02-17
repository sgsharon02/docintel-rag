"""
BM25 Retriever for DocIntel

Implements keyword-based retrieval using rank-bm25.
"""

from rank_bm25 import BM25Okapi
import re


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.documents = None
        self.tokenized_corpus = None

    def _tokenize(self, text: str):
        return re.findall(r"\w+", text.lower())

    def build(self, documents):
        """
        Build BM25 index from chunked documents.
        """
        self.documents = documents

        corpus = [doc.page_content for doc in documents]
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, k: int = 5):
        if self.bm25 is None:
            raise ValueError("BM25 index not built yet.")

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [self.documents[i] for i in ranked_indices]
