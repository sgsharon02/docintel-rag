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
        You are a verification assistant for a Retrieval-Augmented Generation system.

        Your job is to determine whether the answer is supported ONLY by the provided context.

        IMPORTANT RULES:
        - Only use the information present in the context.
        - Do NOT invent sections, page numbers, or sources.
        - If the evidence is not explicitly present in the context, mark NOT_SUPPORTED.
        - If the answer is fully supported, mark SUPPORTED.

        Context:
        {context}

        Question:
        {question}

        Answer:
        {answer}

        Return your result STRICTLY in this JSON format:

        {{
            "status": "SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED",
            "explanation": "short reasoning referencing the context text only"
        }}
        """

        verification_report = self.llm.invoke(prompt)

        return verification_report



