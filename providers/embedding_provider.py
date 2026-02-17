"""
Embedding Provider Abstraction Layer for DocIntel
Supports IBM watsonx, GCP Vertex AI, local embeddings, and a mock provider.
"""

import os
import numpy as np
from langchain_core.embeddings import Embeddings


### Mock Embedding Provider
class MockEmbeddingProvider(Embeddings):
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_documents(self, texts):
        return [np.random.rand(self.dim).tolist() for _ in texts]

    def embed_query(self, text):
        return np.random.rand(self.dim).tolist()


### IBM Embedding Provider
class IBMEmbeddingProvider(Embeddings):
    def __init__(self):
        from langchain_ibm import WatsonxEmbeddings

        model_id = os.getenv(
            "IBM_EMBEDDING_MODEL",
            "ibm/slate-30m-english-rtrvr"
        )

        self.embeddings = WatsonxEmbeddings(
            model_id=model_id,
            url=os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
            project_id=os.getenv("IBM_PROJECT_ID"),
        )

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text):
        return self.embeddings.embed_query(text)


### GCP Embedding Provider
class GCPEmbeddingProvider(Embeddings):
    def __init__(self):
        from langchain_google_vertexai import VertexAIEmbeddings

        model_name = os.getenv(
            "GCP_EMBEDDING_MODEL",
            "textembedding-gecko@003"
        )

        self.embeddings = VertexAIEmbeddings(model_name=model_name)

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text):
        return self.embeddings.embed_query(text)


### Local Embedding Provider (SentenceTransformers)
class LocalEmbeddingProvider(Embeddings):
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "all-MiniLM-L6-v2"
        )

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


def get_embedding_provider():
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").lower()

    if provider == "ibm":
        return IBMEmbeddingProvider()
    elif provider == "gcp":
        return GCPEmbeddingProvider()
    elif provider == "local":
        return LocalEmbeddingProvider()
    else:
        return MockEmbeddingProvider()
