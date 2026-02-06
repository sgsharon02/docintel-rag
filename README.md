# DocIntel — Trust-Aware Multi-Agent RAG for Financial Documents

DocIntel is a **production-oriented Retrieval-Augmented Generation (RAG) system** designed to answer questions from **financial and regulatory documents** with **verification, hallucination detection, and cloud-agnostic AI backends**.

Unlike naïve document chatbots, DocIntel focuses on **trust, traceability, and reliability** when working with **long, structured PDFs containing tables, disclosures, and risk statements**.

---

## Problem

Large language models struggle with long financial documents because they:

* Hallucinate unsupported claims
* Misinterpret tables and footnotes
* Fail on scanned PDFs
* Provide answers without verification
* Cannot reliably retrieve numeric disclosures

Financial and regulatory documents require **grounded, auditable answers**, not probabilistic summaries.

DocIntel addresses this by combining **hybrid retrieval, agent workflows, and verification loops**.

---

## Architecture Overview

DocIntel uses a **multi-agent RAG workflow**:

```
User Question
      ↓
Scope Agent
      ↓
Hybrid Retrieval Agent (Vector + BM25)
      ↓
Research Agent (Draft Answer)
      ↓
Verification Agent (Fact-Check)
      ↓
Final Answer + Verification Report
```

---

## Core Features

### Document Ingestion

* PDF ingestion pipeline
* OCR support for scanned documents (Docling)
* Metadata extraction (page, section, document source)
* Custom document-aware chunking

---

### Hybrid Retrieval

DocIntel combines:

* Vector similarity search
* BM25 keyword retrieval

This improves:

* Numeric retrieval
* Disclosure lookups
* Section-specific queries

---

### Multi-Agent Workflow

DocIntel uses four agents:

| Agent              | Responsibility                                      |
| ------------------ | --------------------------------------------------- |
| Scope Agent        | Determines if question is answerable from documents |
| Retrieval Agent    | Fetches relevant chunks                             |
| Research Agent     | Generates draft answer                              |
| Verification Agent | Validates claims against sources                    |

---

### Verification & Self-Correction

The verification agent:

* Checks answer claims against retrieved text
* Detects unsupported statements
* Verifies numeric values
* Triggers a single retry if needed

If verification fails:

```
"Not supported by documents."
```

---

### Cloud-Agnostic AI Backends

DocIntel supports interchangeable providers:

* IBM watsonx (Granite / Llama)
* GCP Vertex AI (Gemini)

The RAG pipeline remains unchanged while swapping model providers via configuration.

This demonstrates **AI infrastructure portability**.

---

## Output Format

Each response includes a verification report:

```json
{
  "answer": "...",
  "verification_report": {
    "supported_claims": [],
    "unsupported_claims": [],
    "confidence": 0.0
  }
}
```

This makes the system **auditable and explainable**.

---

## Evaluation

DocIntel includes lightweight evaluation:

* Retrieval Recall@K
* Manual Q&A validation
* Answer-to-source overlap checks

These provide early signals of **RAG reliability**.

---

## Example Use Cases

DocIntel is designed for questions like:

* “What were the key liquidity risk factors in 2023?”
* “Where is CET1 ratio mentioned and what value is reported?”
* “What assumptions were used for fair value measurement?”
* “How did credit risk exposure change year-over-year?”

These queries are difficult for naïve RAG systems.

---

## Repository Structure (Planned)

```
docintel-rag/
│
├── data/
├── ingestion/
├── chunking/
├── retriever/
├── agents/
├── verification/
├── evaluation/
├── providers/
└── app/
```

---

## Non-Goals

DocIntel intentionally does **not** include:

* Financial advice or prediction
* Real-time trading data
* Model fine-tuning
* Autonomous learning systems
* Multi-user authentication
* Production deployment infrastructure

The focus is **reliable document intelligence**, not chatbot features.

---

## Tech Stack

* LangGraph
* LangChain
* Docling (OCR)
* ChromaDB / FAISS
* BM25 retrieval
* IBM watsonx
* Google Vertex AI
* Python

---

## Project Status

Active development.

Initial milestone:

* Document ingestion
* Hybrid retrieval
* Multi-agent workflow
* Verification loop
* Cloud-agnostic LLM switching

---

## Author

Built as part of an applied AI engineering project exploring **trust-aware RAG systems for high-risk document domains**.

---
