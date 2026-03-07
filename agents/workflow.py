"""
Agent Workflow for DocIntel

Orchestrates:
Research Agent → Verification Agent → Optional retry
"""

import logging
from agents.research_agent import ResearchAgent
from agents.verification_agent import VerificationAgent
from retriever.retrieval_logger import log_retrieval
from agents.result import RAGResult

logger = logging.getLogger(__name__)

class AgentWorkflow:
    def __init__(self, retriever):
        logger.info("Initializing AgentWorkflow")
        self.research_agent = ResearchAgent(retriever)
        self.verification_agent = VerificationAgent()
        logger.info("AgentWorkflow initialized successfully")

    def run(self, question: str):
        """
        Full agent pipeline:
        research → verify → optional retry
        """
        logger.info(f"Starting workflow for question: {question}")

        # Step 1 — Research
        logger.info("Step 1: Starting research agent...")
        research_result = self.research_agent.generate(question)
        logger.info(f"Research complete. Found {len(research_result.get('documents', []))} documents")

        draft_answer = research_result["draft_answer"]
        context = research_result["context"]
        logger.debug(f"Draft answer length: {len(draft_answer)} characters")

        # Step 2 — Verification
        logger.info("Step 2: Starting verification agent...")
        verification_report = self.verification_agent.verify(
            question=question,
            answer=draft_answer,
            context=context,
        )
        logger.info(f"Verification complete. Report: {verification_report}")

        log_retrieval(question, research_result["documents"], verification_report)

        # Optional retry logic
        if "NOT_SUPPORTED" in verification_report:
            logger.warning("Verification returned NOT_SUPPORTED. Retrying research...")
            research_result = self.research_agent.generate(question)
            logger.info(f"Retry research complete. Found {len(research_result.get('documents', []))} documents")

            draft_answer = research_result["draft_answer"]
            context = research_result["context"]

            logger.info("Step 2 (retry): Starting verification agent...")
            verification_report = self.verification_agent.verify(
                question=question,
                answer=draft_answer,
                context=context,
            )
            logger.info(f"Retry verification complete. Report: {verification_report}")
        else:
            logger.info("Verification passed. No retry needed.")

        logger.info("Workflow completed successfully")
        return RAGResult(
            answer=draft_answer,
            verification=verification_report,
            context=context,
            sources=research_result["sources"],
            documents=research_result["documents"],
        )