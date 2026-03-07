from fastapi import APIRouter
from api.state import ingestion_state

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok",
            "index_status": ingestion_state["status"]
            }
