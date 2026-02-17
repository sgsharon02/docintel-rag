"""
Verification Agent for DocIntel

Checks whether the generated answer is supported by retrieved context.
Produces a verification report.
"""

from providers.llm_provider import get_llm_provider


class VerificationAgent:
    def __init__(self):
        self.llm = get_llm_provider()

    def verify(self, question: str, answer: str, context: str):
        """
        Verify that the answer is grounded in the provided context.
        """

        prompt = f"""
        You are a verification assistant.

        Your job is to determine whether the answer is fully supported
        by the provided context.

        If the answer contains unsupported claims, hallucinations,
        or missing evidence, clearly explain why.

        Context:
        {context}

        Question:
        {question}

        Answer:
        {answer}

        Provide:
        1. Verification status (SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED)
        2. Explanation
        """

        verification_report = self.llm.invoke(prompt)

        return verification_report
