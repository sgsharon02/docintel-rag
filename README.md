# 📄 DocIntel — Trust-Aware RAG for Financial Documents

DocIntel is a **production-oriented Retrieval-Augmented Generation (RAG) system** designed to answer questions from **financial and regulatory documents** with **verification, source attribution, and explainability**.

Unlike naïve document chatbots, DocIntel is designed for:

→ **grounded, evidence-backed answers**
→ **page-level source traceability**  
→ **automated verification of claims**    
→ **full retrieval transparency**   

---

## 🚨 Problem

Large language models struggle with financial documents because they:

- hallucinate unsupported claims  
- misinterpret tables and disclosures  
- fail on long, structured PDFs  
- provide answers without evidence  

In high-stakes domains like finance, answers must be:

→ **accurate**  
→ **auditable**  
→ **verifiable**

---

## 🧠 Architecture

DocIntel uses a structured RAG pipeline:

```

User Query
↓
Hybrid Retrieval (Vector + BM25)
↓
Cross-Encoder Reranking
↓
Research Agent (LLM)
↓
Verification Agent
↓
Final Answer + Sources + Verification Report

````

---

## ⚙️ Core Features

### 📥 Document Ingestion
- PDF ingestion pipeline  
- Metadata extraction (source, page, section)  
- Structure-aware chunking with overlap

---

### 🔎 Hybrid Retrieval
Combines:
- semantic search (FAISS / embeddings)  
- keyword search (BM25)  

Improves:
- numeric lookup  
- financial disclosures  
- section-level retrieval  

---

### 🎯 Cross-Encoder Reranking
- Re-ranks retrieved chunks using deep semantic scoring  
- Improves precision of top-K results  

---

### 🔁 LangGraph Workflow
- Structured agent pipeline:
  - Research → Verification → Retry (if needed)  
- Enables controlled, reliable reasoning  

---

### 🛡️ Verification Layer
- Validates whether answer is supported by retrieved context  
- Detects unsupported claims / hallucinations  
- Returns structured verification report  

---

### 🔍 Explainable Debug UI
- Chunk-level inspection  
- Section grouping  
- Retrieval scores:
  - Vector score  
  - BM25 score  
  - Hybrid score  
  - Rerank score  

→ Makes retrieval **transparent and debuggable**

---
## 🔌 Multi-Provider LLM Architecture

DocIntel is designed with a **provider-agnostic LLM abstraction layer**, allowing seamless switching between different model providers without changing application logic.

Supported providers:

- IBM watsonx (Granite / hosted models)
- Google Vertex AI (Gemini)
- Local models via Ollama (llama3)
- Mock provider (for testing)

The provider is selected via configuration:

```env
LLM_PROVIDER=llama
```
---

## 📊 Evaluation

DocIntel includes lightweight evaluation:

- **Grounding Score** → how much of the answer is supported by context  
- **LLM Score (1–5)** → correctness, completeness, and relevance  

---

## 🧪 Example Queries

- “What were the specific quarterly dividend increases for JPMorgan Chase between the third quarter of 2023 and the first quarter of 2024?”  
- “Where is CET1 ratio mentioned and what value is reported?”  
- “What are the key liquidity risk factors?”  

---

## 🛠️ Tech Stack

- Python  
- FastAPI  
- Streamlit  
- LangChain  
- LangGraph  
- FAISS  
- BM25 (rank-bm25)  
- SentenceTransformers (Cross-Encoder)  

LLM Providers:
- IBM watsonx  
- Google Vertex AI  
- Local Ollama (llama3)  

---

## 🚀 Running the Project

### 1. Clone Repository

```bash
git clone https://github.com/sgsharon02/docintel-rag.git
cd docintel-rag
````

---

### 2. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

### 3. Configure Environment

Create `.env`:

```env
LLM_PROVIDER=llama
EMBEDDING_PROVIDER=local
LOCAL_LLM_MODEL=llama3
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

### 4. Start Backend

```bash
uvicorn api.main:app --reload
```

---

### 5. Start UI

```bash
streamlit run app/streamlit_app.py
```

---

### 6. Ingest Documents

* Place PDFs in `/data`
* Use UI → “Run Ingestion”

---

### 7. Query

Ask questions in the UI and inspect:

* retrieved chunks
* scores
* verification report

---

## 📁 Project Structure

```
docintel-rag/
├── agents/
├── ingestion/
├── chunking/
├── retriever/
├── providers/
├── evaluation/
├── api/
├── app/
├── data/
└── index_store/
```

---

## 🎯 What This Project Demonstrates

* End-to-end RAG system design
* Hybrid retrieval + reranking
* Multi-agent workflows (LangGraph)
* Verification and grounding
* Explainable AI systems

---

## 📌 Key Insight

> Generating answers is easy.
> **Verifying and explaining them is the real challenge.**

---

## 👨‍💻 Author

Built as an applied AI engineering project focused on
**trustworthy RAG systems for high-risk document domains.**