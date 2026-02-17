from dotenv import load_dotenv
load_dotenv()

from providers.llm_provider import get_llm_provider
llm = get_llm_provider()

from ingestion.ingestion_pipeline import IngestionPipeline

print("\nDocIntel starting...\n")

pipeline = IngestionPipeline("data/annualreport-2023.pdf")

print("Pipeline ready. Running query...\n")
hybrid = pipeline.run()


# -------------------------
# Agent Workflow
# -------------------------
print("\nStarting Agent Workflow...\n")

from agents.workflow import AgentWorkflow

workflow = AgentWorkflow(hybrid)

query = "What were the specific quarterly dividend increases for JPMorgan Chase between the third quarter of 2023 and the first quarter of 2024?"

result = workflow.run(query)

print("\nFINAL ANSWER:\n")
print(result["answer"])

print("\nVERIFICATION REPORT:\n")
print(result["verification"])

# -------------------------
# Evaluation
# -------------------------
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
    result["answer"],
    result["context"],
)

print_eval_report(query, result["answer"], result["verification"])
print("\nRetrieval Recall:", recall)
print("Grounding Score:", grounding_score)

score = llm_answer_score(
    llm,
    query,
    result["answer"],
    result["context"],
)

print("LLM Evaluation Score:", score)



# from dotenv import load_dotenv
# from providers.llm_provider import get_llm_provider

# load_dotenv()

# print("\nCreating model:\n")

# llm = get_llm_provider()

# response = llm.invoke("Explain what financial risk disclosure means in one sentence.")
# print("\nMODEL RESPONSE:\n")
# print(response)


# from providers.embedding_provider import get_embedding_provider

# print("\nCreating embedding:\n")

# emb = get_embedding_provider()

# vec = emb.embed_query("financial risk disclosure")
# print(len(vec))

# from ingestion.document_loader import DocumentLoader

# print("\nLoading Document:\n")

# loader = DocumentLoader()
# query = "What were the specific quarterly dividend increases for JPMorgan Chase between the third quarter of 2023 and the first quarter of 2024?"

# docs = loader.load_pdf("data/annualreport-2023.pdf")

# print("Loaded pages:", len(docs))
# print(docs[0].metadata)
# print(docs[0].page_content[:200])

# print("\nChunking Document:\n")
# from chunking.document_chunker import DocumentChunker
# chunker = DocumentChunker()
# chunks = chunker.chunk_documents(docs)
# print("Chunks created:", len(chunks))
# print(chunks[0].metadata)
# print(chunks[1].page_content[:200])

# print("\nCreating Retriever:\n")
# from retriever.vector_index import VectorIndex
# vector_index = VectorIndex()
# if vector_index.exists():
#     print("Loading FAISS index...")
#     vector_index.load()
# else:
#     print("Building FAISS index...")
#     vector_index.build(chunks)
#     vector_index.save()

# retriever = vector_index.as_retriever()
# results = retriever.invoke("What were the specific quarterly dividend increases for JPMorgan Chase between the third quarter of 2023 and the first quarter of 2024?")
# print("Retrieved:", len(results))
# print(results[0].metadata)
# print(results[0].page_content[:200])

# print("\nCreating BM25Retriever:\n")
# from retriever.bm25_index import BM25Retriever
# bm25 = BM25Retriever()
# bm25.build(chunks)
# results = bm25.retrieve(query, k=3)

# print("BM25 results:", len(results))
# print(results[0].metadata)
# print(results[0].page_content[:200])

# print("\nCreating Hybrid Retriever:\n")
# from retriever.hybrid_retriever import HybridRetriever
# hybrid = HybridRetriever(vector_index, bm25)
# results = hybrid.retrieve(query, k=3)
# print("Hybrid results:", len(results))
# print(results[0].metadata)
# print(results[0].page_content[:200])

# print("\nCalling research agent:\n")
# from agents.research_agent import ResearchAgent
# research_agent = ResearchAgent(hybrid)
# query = "What were the specific quarterly dividend increases for JPMorgan Chase between the third quarter of 2023 and the first quarter of 2024?"

# research_result = research_agent.generate(query)
# print("Answer:")
# print(research_result["draft_answer"])

# print("\nCalling verification agent:\n")
# from agents.verification_agent import VerificationAgent
# verification_agent = VerificationAgent()


# research_result = research_agent.generate(query)

# verification = verification_agent.verify(
#     question=query,
#     answer=research_result["draft_answer"],
#     context=research_result["context"],
# )

# print("\nVerification:")
# print(verification)

# from agents.workflow import AgentWorkflow

# workflow = AgentWorkflow(hybrid)

# result = workflow.run(query)

# print("\nFINAL ANSWER:\n")
# print(result["answer"])

# print("\nVERIFICATION REPORT:\n")
# print(result["verification"])


# print("\nFINAL EVALUATION REPORT:\n")

# from evaluation.rag_eval import (
#     llm_answer_score,
#     retrieval_recall_at_k,
#     answer_grounded,
#     print_eval_report,
# )

# result = workflow.run(query)

# recall = retrieval_recall_at_k(
#     hybrid.retrieve(query),
#     expected_keywords=["1.05", "1.15"],
# )

# grounding_score = answer_grounded(
#     result["answer"],
#     result["context"],
# )

# print_eval_report(query, result["answer"], result["verification"])
# print("\nRetrieval Recall:", recall)
# print("Grounding Score:", grounding_score)

# score = llm_answer_score(
#     llm,
#     query,
#     result["answer"],
#     result["context"],
# )

# print("LLM Evaluation Score:", score)
