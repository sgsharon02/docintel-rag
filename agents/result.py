from typing import List
from langchain_core.documents import Document
from pydantic import BaseModel


class RAGResult(BaseModel):

    answer: str
    verification: str
    sources: List[str]
    context: str
    documents: List[Document]