"""
Ingestion Pipeline for DocIntel

Handles:
- document loading
- chunking
- FAISS index persistence
- BM25 index build
- hybrid retriever creation
"""

import os
import json
from ingestion.hash_utils import compute_docs_hash

from ingestion.document_loader import DocumentLoader
from chunking.document_chunker import DocumentChunker
from retriever.vector_index import VectorIndex
from retriever.bm25_index import BM25Retriever
from retriever.hybrid_retriever import HybridRetriever


class IngestionPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path

        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()

        self.vector_index = VectorIndex()
        self.bm25 = BM25Retriever()

    def run(self):
        print("\nRunning ingestion pipeline...\n")

        ### Load Documents
        if os.path.isdir(self.data_path):
            docs = self.loader.load_directory(self.data_path)
        else:
            docs = self.loader.load_pdf(self.data_path)
        print("Documents loaded:", len(docs))

        ### Chunk Documents
        chunks = self.chunker.chunk_documents(docs)
        print("Chunks created:", len(chunks))

        ### Compute Document Hash
        doc_hash = compute_docs_hash(chunks)

        hash_file = os.path.join(self.vector_index.index_dir, "doc_hash.json")
        stored_hash = None

        if os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                stored_hash = json.load(f).get("hash")

        ### Vector Index (FAISS)
        if self.vector_index.exists() and stored_hash == doc_hash:
            print("Loading FAISS index (hash match)...")
            self.vector_index.load()
        else:
            print("Rebuilding FAISS index (hash changed or missing)...")
            self.vector_index.build(chunks)
            self.vector_index.save()

            os.makedirs(self.vector_index.index_dir, exist_ok=True)
            with open(hash_file, "w") as f:
                json.dump({"hash": doc_hash}, f)

        ### BM25 Index
        self.bm25.build(chunks)

        ### Hybrid Retriever
        hybrid = HybridRetriever(self.vector_index, self.bm25)

        print("Ingestion pipeline ready.\n")

        return hybrid
