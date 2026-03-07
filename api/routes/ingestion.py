from pathlib import Path
from fastapi import APIRouter, BackgroundTasks
from ingestion.ingestion_pipeline import IngestionPipeline
from agents.workflow import AgentWorkflow
from api.dependencies import set_workflow
from api.state import ingestion_state

router = APIRouter()


def run_ingestion():
    ingestion_state["status"] = "building"

    file_path = Path("data") / "annualreport-2023.pdf"
    pipeline = IngestionPipeline(str(file_path))
    hybrid = pipeline.run()

    workflow = AgentWorkflow(hybrid)
    set_workflow(workflow)

    ingestion_state["status"] = "ready"


@router.post("/ingest")
def ingest(background_tasks: BackgroundTasks):

    if ingestion_state["status"] == "building":
        return {
            "status": "already_running"
        }

    background_tasks.add_task(run_ingestion)

    return {
        "status": "started",
        "message": "Index building in background"
    }