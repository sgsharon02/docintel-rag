from typing import List, TypedDict


class WorkflowState(TypedDict, total=False):

    query: str
    draft_answer: str
    documents: List
    context: str
    sources: List[str]
    verification: str
    retry_count: int