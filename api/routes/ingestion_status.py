from fastapi import APIRouter
from api.state import ingestion_state
import os

router = APIRouter()

INDEX_DIR = "index_store"


@router.get("/ingestion-status")
def ingestion_status():

    # If ingestion currently running
    if ingestion_state["status"] == "building":
        return {"status": "building"}

    # If index exists
    if os.path.exists(os.path.join(INDEX_DIR, "manifest.json")):
        return {"status": "ready"}

    # Otherwise
    return {"status": "not_ready"}