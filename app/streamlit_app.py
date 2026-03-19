import sys
import tempfile
from pathlib import Path

# Ensure app/ is on sys.path so sibling imports work when launched via
# `streamlit run app/streamlit_app.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from rag_pipeline import load_and_split_pdf, get_embeddings, index_documents
from agent import build_graph

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Financial Advisor", layout="centered")
st.title("AI Financial Advisor")
st.markdown(
    "Upload a financial document (PDF), ask a question, and get an AI-powered "
    "portfolio summary, risk analysis, and draft client email."
)

# ---------------------------------------------------------------------------
# File upload & indexing
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a financial PDF", type=["pdf"])

if uploaded_file is not None:
    if st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            docs = load_and_split_pdf(tmp_path)
            embeddings = get_embeddings()
            vectorstore = index_documents(docs, embeddings)

            st.session_state["vectorstore"] = vectorstore
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["num_chunks"] = len(docs)
            st.session_state["result"] = None  # clear stale result from previous file

    st.success(
        f"Document ready: **{st.session_state['file_name']}** "
        f"({st.session_state['num_chunks']} chunks indexed)"
    )

    # -----------------------------------------------------------------------
    # Question input & run button
    # -----------------------------------------------------------------------
    st.divider()
    question = st.text_input(
        "Ask a question about the document",
        placeholder="e.g. What are the key financial highlights and risks?",
    )

    if st.button("Run Analysis"):
        if not question.strip():
            st.warning("Please enter a question before running the analysis.")
        else:
            with st.spinner("Running agent — this may take 15–30 seconds..."):
                agent = build_graph(st.session_state["vectorstore"])
                result = agent.invoke({"question": question})
                st.session_state["result"] = result

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    if st.session_state.get("result"):
        result = st.session_state["result"]
        st.divider()

        st.subheader("Portfolio Summary")
        st.write(result["summary"])

        st.subheader("Risk Notes")
        st.write(result["risk_notes"])

        st.subheader("Draft Client Email")
        st.write(result["draft_email"])

        st.info("Audit log entry saved to `logs/audit_log.jsonl`")
