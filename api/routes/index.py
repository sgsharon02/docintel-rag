from fastapi import APIRouter
import shutil
import os

from api.dependencies import clear_workflow
from api.state import ingestion_state

router = APIRouter()

INDEX_DIR = "index_store"


@router.post("/reset-index")
def reset_index():

    # Prevent reset during ingestion
    if ingestion_state["status"] == "building":
        return {
            "status": "error",
            "message": "Cannot reset while ingestion is running"
        }

    # Delete index directory
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    # Recreate empty index directory
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Clear workflow cache
    clear_workflow()

    # Reset ingestion state
    ingestion_state["status"] = "not_ready"

    return {
        "status": "index_reset",
        "message": "Index cleared successfully"
    }