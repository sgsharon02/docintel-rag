import os
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import time
import json
import requests

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from dotenv import load_dotenv
from providers.llm_provider import get_llm_provider
from evaluation.rag_eval import (
    llm_answer_score,
    retrieval_recall_at_k,
    answer_grounded,
)

load_dotenv()

API_URL = "http://localhost:8000"


# Page config
st.set_page_config(page_title="DocIntel", layout="wide")

st.title("📄 DocIntel — Document Intelligence Assistant")
st.caption("Hybrid RAG • Verification • Source Attribution")


# Sidebar — Ingestion
st.sidebar.header("📥 Ingestion")

data_path = st.sidebar.text_input("Document folder", "data/")

# Multi-doc selection
pdf_files = []
if os.path.exists(data_path):
    pdf_files = [f for f in os.listdir(data_path) if f.endswith(".pdf")]

selected_docs = st.sidebar.multiselect(
    "Select documents to index",
    pdf_files,
    default=pdf_files,
)

# Upload
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    save_path = os.path.join("data", uploaded_file.name)
    if not os.path.exists(save_path):
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Saved {uploaded_file.name}")

# Run ingestion via API
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


# Reset index
if st.sidebar.button("Reset Index"):
    requests.post(f"{API_URL}/reset-index")
    st.sidebar.success("Index reset")



# Session status (API + index)
try:
    health = requests.get(f"{API_URL}/health").json()
    api_ok = health.get("status") == "ok"
except:
    api_ok = False

if api_ok:
    st.sidebar.success("API running")
else:
    st.sidebar.error("API not reachable")

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


# Clear UI session
if st.sidebar.button("Clear Session"):
    st.session_state.clear()
    st.sidebar.info("UI session cleared")


# Manifest display
manifest_path = os.path.join("index_store", "manifest.json")

if os.path.exists(manifest_path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    st.sidebar.markdown("### 📊 Index Info")
    st.sidebar.write("Documents:", manifest["documents"])
    st.sidebar.write("Pages:", manifest["num_pages"])
    st.sidebar.write("Chunks:", manifest["num_chunks"])
    st.sidebar.write("Built:", manifest["timestamp"])

# Query history
st.sidebar.markdown("### 🕑 Query History")
if "history" not in st.session_state:
    st.session_state["history"] = []

for q in st.session_state["history"][-5:]:
    st.sidebar.write("•", q)


# Query interface
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

    
    # Answer
    st.markdown("### ✅ Answer")
    st.write(answer)

    
    # Sources
    st.markdown("### 📌 Sources")
    for s in sources:
        st.write(f"- {s}")

    
    # Context panel
    with st.expander("View Retrieved Context"):
        for i, doc in enumerate(documents, start=1):
            st.markdown(f"**Chunk {i}**")
            st.write(doc["page_content"][:800])
            st.caption(
                f"Source: {doc['metadata'].get('source')} — page {doc['metadata'].get('page')}"
            )
            st.divider()

    
    # Verification
    st.markdown("### 🛡 Verification")
    st.write(verification)
    st.caption(f"⏱ Response time: {latency:.2f}s")

    
    # Debug panel (grouped by section)
    with st.expander(f"🔎 Retrieval Debug Panel ({len(documents)} chunks)"):
        section_groups = {}

        for doc in documents:
            section = doc["metadata"].get("section", "Unknown Section")
            section_groups.setdefault(section, []).append(doc)

        for section, docs in section_groups.items():
            st.markdown(f"## 📂 Section: {section} ({len(docs)} chunks)")

            for d in docs:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Metadata**")
                    st.json(d["metadata"])

                with col2:
                    st.write("**Preview**")
                    st.write(d["page_content"][:300])

                if d["metadata"].get("block_type") == "table":
                    st.warning("📊 Table Chunk")

                st.divider()

    
    # Evaluation panel
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

        st.write("Retrieval Recall:", recall)
        st.write("Grounding Score:", grounding)
        st.write("LLM Evaluation Score:", llm_score)