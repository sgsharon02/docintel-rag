import os
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import time
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from providers.llm_provider import get_llm_provider
from evaluation.rag_eval import (
    llm_answer_score,
    answer_grounded,
)

load_dotenv()

API_URL = "http://localhost:8000"


# Helpers
def truncate_text(text, max_chars=500):
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")

    if last_space == -1:
        return truncated + "..."

    return truncated[:last_space] + "..."


def render_chunk(doc, preview_chars=400):

    source = doc["metadata"].get("source", "unknown")
    page = doc["metadata"].get("page", "?")
    chunk_idx = doc["metadata"].get("chunk_index", "?")
    rank = doc["metadata"].get("rank", "?")

    st.markdown(
        f"""
        **🔢 Rank #{rank} — {source} (Page {page}, Chunk {chunk_idx})**
        """
    )

    preview = truncate_text(doc["page_content"], preview_chars)
    st.markdown(preview)

    with st.expander("🔎 View full chunk"):
        st.text_area("Full chunk text", doc["page_content"], height=250)

    st.caption(f"Source: {source} — page {page}")

    score_badges = []

    def badge(label, value):
        return f"`{label}: {value}`"

    if "hybrid_score" in doc["metadata"]:
        score_badges.append(badge("Hybrid", doc["metadata"]["hybrid_score"]))

    if "rerank_score" in doc["metadata"]:
        score_badges.append(badge("Rerank", doc["metadata"]["rerank_score"]))

    if "vector_score" in doc["metadata"]:
        score_badges.append(badge("Vector", doc["metadata"]["vector_score"]))

    if "bm25_score" in doc["metadata"]:
        score_badges.append(badge("BM25", doc["metadata"]["bm25_score"]))

    if score_badges:
        st.markdown(" ".join(score_badges))

    st.divider()


# Page Config
st.set_page_config(page_title="DocIntel", layout="wide")

st.title("📄 DocIntel — Document Intelligence Assistant")

st.caption(
    "Pipeline: Hybrid Retrieval → Reranker → Research Agent → Verification"
)

st.caption("Hybrid RAG • Verification • Source Attribution")


# Sidebar — Ingestion
st.sidebar.header("📥 Ingestion")

data_path = st.sidebar.text_input("Document folder", "data/")

pdf_files = []
if os.path.exists(data_path):
    pdf_files = [f for f in os.listdir(data_path) if f.endswith(".pdf")]

selected_docs = st.sidebar.multiselect(
    "Select documents to index",
    pdf_files,
    default=pdf_files,
)

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    save_path = os.path.join("data", uploaded_file.name)

    if not os.path.exists(save_path):
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success(f"Saved {uploaded_file.name}")


# Ingestion Trigger
if st.sidebar.button("Run Ingestion"):

    with st.spinner("Starting ingestion..."):
        resp = requests.post(f"{API_URL}/ingest").json()

        st.sidebar.info(resp.get("message", "Ingestion started"))

    time.sleep(1)

    status = requests.get(f"{API_URL}/ingestion-status").json()

    if status["status"] == "building":
        st.sidebar.warning("Index building...")
    elif status["status"] == "ready":
        st.sidebar.success("Index ready")


if st.sidebar.button("Reset Index"):
    requests.post(f"{API_URL}/reset-index")
    st.sidebar.success("Index reset")

# API Health
try:
    health = requests.get(f"{API_URL}/health").json()
    api_ok = health.get("status") == "ok"
except:
    api_ok = False

if api_ok:
    st.sidebar.success("API running")
else:
    st.sidebar.error("API not reachable")

# Index Status
try:
    status = requests.get(f"{API_URL}/ingestion-status").json()["status"]

    if status == "ready":
        st.sidebar.success("Index ready")
    elif status == "building":
        st.sidebar.warning("Index building...")
    else:
        st.sidebar.warning("Index not built")

except:
    st.sidebar.error("Cannot check index status")

# Manifest
manifest_path = os.path.join("index_store", "manifest.json")

if os.path.exists(manifest_path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    st.sidebar.markdown("### 📊 Index Info")
    st.sidebar.write("Documents:", manifest["documents"])
    st.sidebar.write("Pages:", manifest["num_pages"])
    st.sidebar.write("Chunks:", manifest["num_chunks"])
    st.sidebar.write("Built:", manifest["timestamp"])


# LangGraph Workflow Image
if os.path.exists("workflow_graph.png"):
    st.sidebar.markdown("### LangGraph Workflow")
    st.sidebar.image("workflow_graph.png", caption="LangGraph Workflow")


# Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat Input
query = st.chat_input("Ask a question about the documents...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.status("Running DocIntel pipeline...", expanded=False) as status:

            start = time.time()

            resp = requests.post(
                f"{API_URL}/query",
                json={"query": query}
            ).json()

            latency = time.time() - start

            status.update(label="Completed", state="complete")

        answer = resp.get("answer", "")
        documents = resp.get("documents", [])
        context = resp.get("context", "")
        verification = resp.get("verification", {})
        sources = resp.get("sources", [])

        st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Sources
        st.markdown("### 📌 Sources")
        for s in sources:
            st.write(f"- {s}")

        # Context
        with st.expander("📚 Retrieved Context (Model Input)"):
            st.text_area("Context sent to model", context, height=300)

        # Verification
        st.markdown("### 🛡 Verification")
        st.write(verification)
        st.caption(f"⏱ Response time: {latency:.2f}s")

        # Debug Panel
        with st.expander(f"🔎 Retrieval Debug Panel ({len(documents)} chunks)"):

            for i, doc in enumerate(documents, start=1):
                doc["metadata"]["rank"] = i

            section_groups = {}

            for doc in documents:
                section = doc["metadata"].get("section", "Unknown Section")
                section_groups.setdefault(section, []).append(doc)

            for section, docs in section_groups.items():

                st.markdown(f"### 📂 Section: {section}")

                for doc in docs:

                    render_chunk(doc)

                    if doc["metadata"].get("block_type") == "table":
                        st.warning("📊 Table Chunk")

        # Evaluation
        with st.expander("📈 Evaluation Metrics"):

            llm = get_llm_provider()
            grounding = answer_grounded(answer, context)
            llm_score = llm_answer_score(llm, query, answer, context)

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Grounding Score",
                    value=f"{grounding:.2f}"
                )
                st.caption("Grounding: how much of the answer is supported by retrieved context.")

            with col2:
                st.metric(
                    label="LLM Answer Score",
                    value=llm_score
                )
                st.caption( "LLM Score (1–5): evaluates correctness, completeness, and relevance of the answer.")