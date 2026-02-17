"""
LLM Provider Abstraction Layer for DocIntel

This module allows the RAG pipeline to switch between
IBM watsonx and GCP Vertex AI without changing application logic.
"""

from abc import ABC, abstractmethod
import os

### Base Interface
class BaseLLMProvider(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass


### IBM watsonx Provider
class IBMLLMProvider(BaseLLMProvider):
    def __init__(self):
        import os
        from langchain_ibm import WatsonxLLM
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        model_id = os.getenv("IBM_MODEL_ID", "ibm/granite-3-3-8b-instruct")
        project_id = os.getenv("IBM_PROJECT_ID", "skills-network")

        params = {
            GenParams.MAX_NEW_TOKENS: 200,
            GenParams.TEMPERATURE: 0.2,
        }

        self.llm = WatsonxLLM(
            model_id=model_id,
            url=os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
            project_id=project_id,
            params=params,
        )

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt)

### GCP Vertex AI Provider
class GCPLLMProvider(BaseLLMProvider):
    def __init__(self):
        from langchain_google_vertexai import VertexAI

        model_name = os.getenv("GCP_MODEL_ID", "gemini-1.5-pro")

        self.llm = VertexAI(
            model_name=model_name,
            temperature=0.2,
        )

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt)
    
### llama LLM Provider
class LlamaLocalProvider(BaseLLMProvider):
    def __init__(self):
        from langchain_ollama import OllamaLLM

        model = os.getenv("LOCAL_LLM_MODEL", "llama3")
        self.llm = OllamaLLM(model=model)

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt)
    
### Mock LLM Provider
class MockLLMProvider(BaseLLMProvider):
    def invoke(self, prompt: str) -> str:
        return "Mock response: LLM provider not configured yet."



def get_llm_provider():
    provider = os.getenv("LLM_PROVIDER", "mock").lower()

    if provider == "ibm":
        return IBMLLMProvider()
    elif provider == "gcp":
        return GCPLLMProvider()
    elif provider == "llama":
        return LlamaLocalProvider()
    elif provider == "mock":
        return MockLLMProvider()
    else:
        raise ValueError(f"Unsupported provider: {provider}")
