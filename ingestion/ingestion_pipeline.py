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
import logging
from ingestion.hash_utils import compute_docs_hash

from ingestion.document_loader import DocumentLoader
from chunking.document_chunker import DocumentChunker
from retriever.vector_index import VectorIndex
from retriever.bm25_index import BM25Retriever
from retriever.hybrid_retriever import HybridRetriever
from datetime import datetime

logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self, data_path: str):
        logger.info(f"Initializing IngestionPipeline with data_path: {data_path}")
        self.data_path = data_path

        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()

        self.vector_index = VectorIndex()
        self.bm25 = BM25Retriever()
        logger.info("IngestionPipeline components initialized")

    def run(self):
        logger.info("\n" + "="*50)
        logger.info("STARTING INGESTION PIPELINE")
        logger.info("="*50)

        try:
            ### Load Documents
            logger.info(f"Loading documents from: {self.data_path}")
            if os.path.isdir(self.data_path):
                docs = self.loader.load_directory(self.data_path)
            else:
                docs = self.loader.load_pdf(self.data_path)
            logger.info(f"Documents loaded: {len(docs)} documents")

            ### Chunk Documents
            logger.info("Chunking documents...")
            chunks = self.chunker.chunk_documents(docs)
            logger.info(f"Chunks created: {len(chunks)} chunks")

            ### Compute Document Hash
            logger.info("Computing document hash...")
            doc_hash = compute_docs_hash(chunks)
            logger.info(f"Document hash computed: {doc_hash}")

            hash_file = os.path.join(self.vector_index.index_dir, "doc_hash.json")
            stored_hash = None

            if os.path.exists(hash_file):
                with open(hash_file, "r") as f:
                    stored_hash = json.load(f).get("hash")
                logger.info(f"Stored hash found: {stored_hash}")

            ### Vector Index (FAISS)
            if self.vector_index.exists() and stored_hash == doc_hash:
                logger.info("Hash match detected. Loading existing FAISS index...")
                self.vector_index.load()
                logger.info("FAISS index loaded successfully")
            else:
                logger.info("Hash mismatch or index missing. Rebuilding FAISS index...")
                self.vector_index.build(chunks)
                logger.info("Building FAISS index...")
                self.vector_index.save()
                logger.info("FAISS index built and saved")

                os.makedirs(self.vector_index.index_dir, exist_ok=True)
                with open(hash_file, "w") as f:
                    json.dump({"hash": doc_hash}, f)
                logger.info(f"Hash file saved: {hash_file}")

            logger.info("Writing manifest...")
            self._write_manifest(docs, chunks)
            logger.info("Manifest written")

            ### BM25 Index
            logger.info("Building BM25 index...")
            self.bm25.build(chunks)
            logger.info("BM25 index built successfully")

            ### Hybrid Retriever
            logger.info("Creating hybrid retriever...")
            hybrid = HybridRetriever(self.vector_index, self.bm25)
            logger.info("Hybrid retriever created successfully")

            logger.info("="*50)
            logger.info("INGESTION PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*50 + "\n")

            return hybrid

        except Exception as e:
            logger.error(f" !! INGESTION PIPELINE FAILED: {str(e)}", exc_info=True)
            raise

    def _write_manifest(self, docs, chunks):
        logger.debug(f"Creating manifest with {len(docs)} documents and {len(chunks)} chunks")
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "documents": list(
                set(d.metadata.get("source", "unknown") for d in docs)
            ),
            "num_pages": len(docs),
            "num_chunks": len(chunks),
            "index_dir": self.vector_index.index_dir,
        }

        manifest_path = os.path.join(
            self.vector_index.index_dir,
            "manifest.json",
        )

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.debug(f"Manifest saved to: {manifest_path}")