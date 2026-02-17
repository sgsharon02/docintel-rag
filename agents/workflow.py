"""
Agent Workflow for DocIntel

Orchestrates:
Research Agent → Verification Agent → Optional retry
"""

from agents.research_agent import ResearchAgent
from agents.verification_agent import VerificationAgent


class AgentWorkflow:
    def __init__(self, retriever):
        self.research_agent = ResearchAgent(retriever)
        self.verification_agent = VerificationAgent()

    def run(self, question: str):
        """
        Full agent pipeline:
        research → verify → optional retry
        """

        # Step 1 — Research
        research_result = self.research_agent.generate(question)

        draft_answer = research_result["draft_answer"]
        context = research_result["context"]

        # Step 2 — Verification
        verification_report = self.verification_agent.verify(
            question=question,
            answer=draft_answer,
            context=context,
        )

        # Optional retry logic
        if "NOT_SUPPORTED" in verification_report:
            research_result = self.research_agent.generate(question)

            draft_answer = research_result["draft_answer"]
            context = research_result["context"]

            verification_report = self.verification_agent.verify(
                question=question,
                answer=draft_answer,
                context=context,
            )

        return {
            "answer": draft_answer,
            "verification": verification_report,
            "context": context,
        }
    