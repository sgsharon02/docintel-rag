"""
Hybrid Retriever for DocIntel

Combines vector retrieval and BM25 retrieval using
score normalization and weighted ranking.
"""

class HybridRetriever:
    def __init__(self, vector_index, bm25_retriever):
        self.vector_index = vector_index
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str, k: int = 5):
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
            key = doc.page_content.strip()
            vector_scores[key] = score / max_vector_score
            all_docs[key] = doc

        ### BM25 scores are implicit — assign rank score
        for rank, doc in enumerate(bm25_results):
            key = doc.page_content.strip()
            bm25_scores[key] = (k - rank) / k
            all_docs[key] = doc

        ### Combine scores
        hybrid_results = []

        for key, doc in all_docs.items():
            v_score = vector_scores.get(key, 0)
            b_score = bm25_scores.get(key, 0)

            hybrid_score = 0.7 * v_score + 0.3 * b_score

            hybrid_results.append((doc, hybrid_score))

        hybrid_results.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in hybrid_results[:k]]
