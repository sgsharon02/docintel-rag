from dotenv import load_dotenv
load_dotenv()

from providers.llm_provider import get_llm_provider
llm = get_llm_provider()

from ingestion.ingestion_pipeline import IngestionPipeline

print("\nDocIntel starting...\n")

pipeline = IngestionPipeline("data/annualreport-2023.pdf")

print("Pipeline ready. Running query...\n")
hybrid = pipeline.run()


# Agent Workflow
print("\nStarting Agent Workflow...\n")

from agents.workflow import AgentWorkflow

workflow = AgentWorkflow(hybrid)

query = "What were the specific quarterly dividend increases for JPMorgan Chase between the third quarter of 2023 and the first quarter of 2024?"

result = workflow.run(query)

print("\nFINAL ANSWER:\n")
print(result.answer)

print("\nVERIFICATION REPORT:\n")
print(result.verification)


# Evaluation
print("\nFINAL EVALUATION REPORT:\n")

from evaluation.rag_eval import (
    llm_answer_score,
    retrieval_recall_at_k,
    answer_grounded,
    print_eval_report,
)

recall = retrieval_recall_at_k(
    hybrid.retrieve(query),
    expected_keywords=["1.05", "1.15"],
)

grounding_score = answer_grounded(
    result.answer,
    result.context,
)

print_eval_report(query, result.answer, result.verification)
print("\nRetrieval Recall:", recall)
print("Grounding Score:", grounding_score)

score = llm_answer_score(
    llm,
    query,
    result.answer,
    result.context,
)

print("LLM Evaluation Score:", score)

print("\nSOURCES:\n")
for s in result.sources:
    print("-", s)
