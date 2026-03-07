"""
RAG Evaluation Utilities for DocIntel
"""

from typing import List

def retrieval_recall_at_k(
    retrieved_docs: List,
    expected_keywords: List[str],
):
    """
    Simple recall metric:
    checks whether expected keywords appear in retrieved documents.
    """

    texts = []

    for doc in retrieved_docs:
        if isinstance(doc, dict):
            texts.append(doc.get("page_content", ""))
        else:
            texts.append(getattr(doc, "page_content", ""))

    combined_text = " ".join(texts).lower()

    hits = sum(
        1 for keyword in expected_keywords
        if keyword.lower() in combined_text
    )

    if not expected_keywords:
        return 0.0

    return hits / len(expected_keywords)

### Grounding Check
def answer_grounded(answer: str, context: str):
    """
    Simple grounding heuristic:
    checks if answer phrases appear in context.
    """

    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    overlap = answer_words.intersection(context_words)

    if len(answer_words) == 0:
        return 0

    return len(overlap) / len(answer_words)


### Manual Evaluation Helper
def print_eval_report(question, answer, verification):
    print("\n--- RAG EVALUATION REPORT ---\n")
    print("Question:")
    print(question)
    print("\nAnswer:")
    print(answer)
    print("\nVerification:")
    print(verification)


# LLM Evaluation Helper
def llm_answer_score(llm, question, answer, context):
    prompt = f"""
You are evaluating a RAG system.

Question:
{question}

Answer:
{answer}

Context:
{context}

Score the answer from 1 to 5 based on:
- factual correctness
- grounding in context
- completeness

Return only the number.
"""

    return llm.invoke(prompt)
