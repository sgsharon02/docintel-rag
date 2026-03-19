"""
LangGraph Workflow for DocIntel
"""

import logging
from langgraph.graph import StateGraph, END

from agents.research_agent import ResearchAgent
from agents.verification_agent import VerificationAgent
from agents.workflow_state import WorkflowState
from agents.result import RAGResult
from config.settings import MAX_RETRIES

logger = logging.getLogger(__name__)


class AgentWorkflow:

    def __init__(self, retriever):

        logger.info("Initializing AgentWorkflow")

        self.research_agent = ResearchAgent(retriever)
        self.verification_agent = VerificationAgent()

        self.graph = self._build_graph()

        logger.info("AgentWorkflow initialized successfully")

        try:
            graph_png = self.graph.get_graph().draw_mermaid_png()

            with open("workflow_graph.png", "wb") as f:
                f.write(graph_png)

            logger.info("Workflow graph visualization saved")

        except Exception as e:
            logger.warning(f"Graph visualization failed: {e}")

    def _research_node(self, state: WorkflowState):

        query = state["query"]
        retry_count = state.get("retry_count", 0)

        logger.info("Step 1: Running research agent")
        logger.info(f"Query: {query}")
        logger.info(f"Retry count: {retry_count}")

        result = self.research_agent.generate(query)

        logger.info(f"Research complete. Found {len(result['documents'])} documents")

        return {
            "draft_answer": result["draft_answer"],
            "documents": result["documents"],
            "context": result["context"],
            "sources": result["sources"],
            "retry_count": retry_count,
        }

    def _verification_node(self, state: WorkflowState):

        logger.info("Step 2: Starting verification agent")

        report = self.verification_agent.verify(
            question=state["query"],
            answer=state["draft_answer"],
            context=state["context"],
        )

        logger.info(f"Verification complete. Report: {report}")

        return {"verification": report}

    def _verification_router(self, state: WorkflowState):

        report = state["verification"].lower()
        retry_count = state.get("retry_count", 0)

        if "not_supported" in report:

            if retry_count >= MAX_RETRIES:
                logger.warning("Max retries reached. Ending workflow.")
                return "finish"

            logger.warning("Verification returned NOT_SUPPORTED. Retrying research...")

            state["retry_count"] = retry_count + 1
            return "retry"

        logger.info("Verification passed. No retry needed.")
        return "finish"

    def _build_graph(self):

        workflow = StateGraph(WorkflowState)

        workflow.add_node("research", self._research_node)
        workflow.add_node("verify", self._verification_node)

        workflow.set_entry_point("research")

        workflow.add_edge("research", "verify")

        workflow.add_conditional_edges(
            "verify",
            self._verification_router,
            {
                "retry": "research",
                "finish": END,
            },
        )

        return workflow.compile()

    def run(self, query: str):

        logger.info("==================================================")
        logger.info("Running LangGraph workflow")

        state = self.graph.invoke(
            {
                "query": query,
                "retry_count": 0
            }
        )

        result = RAGResult(
            answer=state.get("draft_answer"),
            verification=state.get("verification"),
            sources=state.get("sources", []),
            context=state.get("context", ""),
            documents=state.get("documents", []),
        )

        logger.info("Workflow completed successfully")
        logger.info("==================================================")

        return result