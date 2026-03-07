"""
DocIntel Demo Script

Runs predefined queries against the document corpus
and prints formatted results with timing + logging.
"""

import os
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
from dotenv import load_dotenv
load_dotenv()

from ingestion.ingestion_pipeline import IngestionPipeline
from agents.workflow import AgentWorkflow
from retriever.retrieval_logger import log_retrieval


DEMO_QUERIES = [
    "What were the specific quarterly dividend increases for JPMorgan Chase between the third quarter of 2023 and the first quarter of 2024?",
    "What risks does the company highlight in the report?",
    "What sustainability initiatives are mentioned?",
]


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_demo():
    print_header("DOCINTEL DEMO")

    # Ingestion
    start = time.time()
    pipeline = IngestionPipeline("data/")
    hybrid = pipeline.run()
    ingestion_time = time.time() - start

    print(f"Ingestion ready ({ingestion_time:.2f}s)\n")

    workflow = AgentWorkflow(hybrid)

    # Query Loop
    for i, query in enumerate(DEMO_QUERIES, start=1):
        print_header(f"QUERY {i}")
        print("Question:")
        print(query)

        start = time.time()
        result = workflow.run(query)
        latency = time.time() - start

        # Logging retrieval
        log_retrieval(
            query,
            result.documents,
            result.verification.get("status", "UNKNOWN")
            if isinstance(result.verification, dict)
            else None,
        )

        print("\nAnswer:\n")
        print(result.answer)

        print("\nSources:")
        for s in result.sources:
            print(f"  • {s}")

        print("\nVerification:")
        print(result.verification)

        print(f"\nLatency: {latency:.2f}s")

    print_header("DEMO COMPLETE")


if __name__ == "__main__":
    run_demo()
