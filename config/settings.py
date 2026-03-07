"""
DocIntel Configuration Settings
Central place for retrieval and ingestion parameters.
"""

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
VECTOR_TOP_K = 5
BM25_TOP_K = 5
HYBRID_TOP_K = 20

# Hybrid scoring weight
HYBRID_VECTOR_WEIGHT = 0.7
HYBRID_BM25_WEIGHT = 0.3


# Generation
MAX_CONTEXT_CHUNKS = 5
