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
    retrieval_recall_at_k,
    answer_grounded,
)

load_dotenv()

API_URL = "http://localhost:8000"


# ----------------------------
# Helpers
# ----------------------------

def truncate_text(text, max_chars=500):
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")

    if last_space == -1:
        return truncated + "..."

    return truncated[:last_space] + "..."


def render_chunk(doc, idx=None, preview_chars=400):
    """Clean chunk display component"""

    source = doc["metadata"].get("source", "unknown")
    page = doc["metadata"].get("page", "?")
    chunk_idx = doc["metadata"].get("chunk_index", "?")

    title = f"📄 {source} — Page {page} — Chunk {chunk_idx}"

    if idx is not None:
        title = f"Chunk {idx} • {title}"

    st.markdown(f"**{title}**")

    preview = truncate_text(doc["page_content"], preview_chars)
    st.markdown(preview)

    with st.expander("🔎 View full chunk"):
        st.text_area(
            "Full chunk text",
            doc["page_content"],
            height=250
        )

    st.caption(f"Source: {source} — page {page}")
    st.divider()


# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(page_title="DocIntel", layout="wide")

st.title("📄 DocIntel — Document Intelligence Assistant")
st.caption("Hybrid RAG • Verification • Source Attribution")


# ----------------------------
# Sidebar — Ingestion
# ----------------------------

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


# ----------------------------
# Ingestion Trigger
# ----------------------------

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


# ----------------------------
# API Health
# ----------------------------

try:
    health = requests.get(f"{API_URL}/health").json()
    api_ok = health.get("status") == "ok"
except:
    api_ok = False

if api_ok:
    st.sidebar.success("API running")
else:
    st.sidebar.error("API not reachable")


# ----------------------------
# Index Status
# ----------------------------

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


# ----------------------------
# Clear Session
# ----------------------------

if st.sidebar.button("Clear Session"):
    st.session_state.clear()
    st.sidebar.info("UI session cleared")


# ----------------------------
# Manifest
# ----------------------------

manifest_path = os.path.join("index_store", "manifest.json")

if os.path.exists(manifest_path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    st.sidebar.markdown("### 📊 Index Info")
    st.sidebar.write("Documents:", manifest["documents"])
    st.sidebar.write("Pages:", manifest["num_pages"])
    st.sidebar.write("Chunks:", manifest["num_chunks"])
    st.sidebar.write("Built:", manifest["timestamp"])


# ----------------------------
# Query History
# ----------------------------

st.sidebar.markdown("### 🕑 Query History")

if "history" not in st.session_state:
    st.session_state["history"] = []

for q in st.session_state["history"][-5:]:
    st.sidebar.write("•", q)


# ----------------------------
# Query Interface
# ----------------------------

st.subheader("🔍 Ask a Question")

query = st.text_input("Enter a question")

if query:

    if not st.session_state["history"] or st.session_state["history"][-1] != query:
        st.session_state["history"].append(query)

    with st.spinner("Querying API..."):

        start = time.time()

        resp = requests.post(
            f"{API_URL}/query",
            json={"query": query}
        ).json()

        latency = time.time() - start

    answer = resp.get("answer", "")
    documents = resp.get("documents", [])
    context = resp.get("context", "")
    verification = resp.get("verification", {})
    sources = resp.get("sources", [])


    # ----------------------------
    # Answer
    # ----------------------------

    st.markdown("### ✅ Answer")
    st.write(answer)


    # ----------------------------
    # Sources
    # ----------------------------

    st.markdown("### 📌 Sources")

    for s in sources:
        st.write(f"- {s}")


    # ----------------------------
    # Retrieved Context
    # ----------------------------

    with st.expander("📚 Retrieved Context (Model Input)"):

        st.text_area(
            "Context sent to model",
            context,
            height=300
        )


    # ----------------------------
    # Verification
    # ----------------------------

    st.markdown("### 🛡 Verification")

    st.write(verification)

    st.caption(f"⏱ Response time: {latency:.2f}s")


    # ----------------------------
    # Retrieval Debug Panel
    # ----------------------------

    with st.expander(f"🔎 Retrieval Debug Panel ({len(documents)} chunks)"):

        section_groups = {}

        for doc in documents:
            section = doc["metadata"].get("section", "Unknown Section")
            section_groups.setdefault(section, []).append(doc)

        for section, docs in section_groups.items():

            st.markdown(f"### 📂 Section: {section}")

            for i, doc in enumerate(docs, start=1):

                render_chunk(doc, idx=i)

                if doc["metadata"].get("block_type") == "table":
                    st.warning("📊 Table Chunk")


    # ----------------------------
    # Evaluation Metrics
    # ----------------------------

    with st.expander("📈 Evaluation Metrics"):

        llm = get_llm_provider()

        recall = retrieval_recall_at_k(
            documents,
            expected_keywords=["1.05", "1.15"],
        )

        grounding = answer_grounded(
            answer,
            context,
        )

        llm_score = llm_answer_score(
            llm,
            query,
            answer,
            context,
        )

        st.metric("Retrieval Recall", recall)
        st.metric("Grounding Score", grounding)
        st.metric("LLM Evaluation Score", llm_score)