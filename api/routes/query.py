from fastapi import APIRouter, Depends
from pydantic import BaseModel
import time

from api.dependencies import get_workflow

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


def serialize_documents(docs):
    return [
        {
            "page_content": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ]


@router.post("/query")
def run_query(req: QueryRequest, workflow=Depends(get_workflow)):

    start = time.time()

    result = workflow.run(req.query)

    latency = time.time() - start

    return {
        "answer": result.answer,
        "verification": result.verification,
        "sources": result.sources,
        "context": result.context,
        "documents": serialize_documents(result.documents),
        "latency": latency,
        }