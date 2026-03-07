"""
Vector Index Builder for DocIntel

Creates and persists a FAISS vector store from chunked documents.
"""

import os
from langchain_community.vectorstores import FAISS
from providers.embedding_provider import get_embedding_provider
from config.settings import VECTOR_TOP_K

class VectorIndex:
    def __init__(self, index_dir: str = "index_store"):
        self.embedding_provider = get_embedding_provider()
        self.vectorstore = None
        self.index_dir = index_dir

    ### Build Index
    def build(self, documents):
        self.vectorstore = FAISS.from_documents(
            documents,
            self.embedding_provider
        )
        return self.vectorstore

   
    ### Save Index
    def save(self):
        if self.vectorstore is None:
            raise ValueError("No vectorstore to save.")

        os.makedirs(self.index_dir, exist_ok=True)
        self.vectorstore.save_local(self.index_dir)

   
    ### Load Index
    def load(self):
        if not os.path.exists(self.index_dir):
            raise ValueError("Index directory does not exist.")

        self.vectorstore = FAISS.load_local(
            self.index_dir,
            self.embedding_provider,
            allow_dangerous_deserialization=True
        )

        return self.vectorstore

   
    ### Exists Check
    def exists(self):
        return os.path.exists(self.index_dir)

   
    ### Retriever Interface
    def as_retriever(self, k: int = VECTOR_TOP_K):
        if self.vectorstore is None:
            raise ValueError("Vector index not built or loaded.")

        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
