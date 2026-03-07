from ingestion.ingestion_pipeline import IngestionPipeline
from agents.workflow import AgentWorkflow

# Global workflow container
_workflow = None


def get_workflow():
    if _workflow is None:
        raise RuntimeError(
            "Workflow not initialized. Run ingestion first."
        )

    return _workflow


def initialize_workflow(data_path="data/"):
    global _workflow

    pipeline = IngestionPipeline(data_path)
    hybrid = pipeline.run()
    _workflow = AgentWorkflow(hybrid)

    return _workflow


def set_workflow(workflow):
    global _workflow
    _workflow = workflow


def clear_workflow():
    global _workflow
    _workflow = None