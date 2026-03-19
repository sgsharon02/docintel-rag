"""
Hybrid Retriever for DocIntel

Combines vector retrieval and BM25 retrieval using
score normalization and weighted ranking.
"""
from retriever.reranker import CrossEncoderReranker

from config.settings import (
    HYBRID_TOP_K,
    MAX_CONTEXT_CHUNKS,
    HYBRID_VECTOR_WEIGHT,
    HYBRID_BM25_WEIGHT
)

class HybridRetriever:
    def __init__(self, vector_index, bm25_retriever):
        self.vector_index = vector_index
        self.bm25_retriever = bm25_retriever
        self.reranker = CrossEncoderReranker()

    def retrieve(self, query: str, k: int = HYBRID_TOP_K):
        """
        Retrieve documents using hybrid retrieval.
        """

        vector_results = self.vector_index.vectorstore.similarity_search_with_score(
            query, k=k
        )

        bm25_results = self.bm25_retriever.retrieve(query, k=k)

        vector_scores = {}
        bm25_scores = {}
        all_docs = {}

        ### Normalize vector scores
        if vector_results:
            max_vector_score = max([score for _, score in vector_results])
        else:
            max_vector_score = 1

        for doc, score in vector_results:
            key = f"{doc.metadata.get('source')}:{doc.metadata.get('page')}:{doc.metadata.get('chunk_index')}"
            vector_scores[key] = score / max_vector_score
            all_docs[key] = doc

        ### BM25 scores are implicit — assign rank score
        for rank, doc in enumerate(bm25_results):
            key = f"{doc.metadata.get('source')}:{doc.metadata.get('page')}:{doc.metadata.get('chunk_index')}"
            bm25_scores[key] = (k - rank) / k
            all_docs[key] = doc

        ### Combine scores
        hybrid_results = []

        for key, doc in all_docs.items():
            v_score = vector_scores.get(key, 0)
            b_score = bm25_scores.get(key, 0)

            hybrid_score = (HYBRID_VECTOR_WEIGHT * v_score    + HYBRID_BM25_WEIGHT * b_score)
            # attach retrieval scores to metadata
            doc.metadata["vector_score"] = round(float(v_score), 4)
            doc.metadata["bm25_score"] = round(float(b_score), 4)
            doc.metadata["hybrid_score"] = round(float(hybrid_score), 4)
            
            hybrid_results.append((doc, hybrid_score))

        # sort by hybrid score
        hybrid_results.sort(key=lambda x: x[1], reverse=True)

        # candidate set before reranking
        candidate_docs = [doc for doc, _ in hybrid_results[:k]]

        # cross-encoder reranking
        reranked_docs = self.reranker.rerank(
            query,
            candidate_docs,
            top_k=MAX_CONTEXT_CHUNKS
        )

        return reranked_docs
