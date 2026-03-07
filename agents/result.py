from dataclasses import dataclass
from typing import List, Any


@dataclass
class RAGResult:
    answer: str
    context: str
    sources: List[str]
    verification: Any
    documents: List[Any]
