# AI Financial Advisor RAG Agent

A portfolio project that demonstrates a Retrieval-Augmented Generation (RAG) agent for financial advising. Users can upload financial PDF documents (reports, prospectuses, filings) and ask natural language questions. The agent retrieves relevant context from the documents using ChromaDB vector search and generates grounded answers via Azure OpenAI.

## Tech Stack
- **LangChain / LangGraph** — orchestration and agent logic
- **ChromaDB** — local vector store for document embeddings
- **Azure OpenAI** — LLM and embedding models
- **Streamlit** — web UI
- **PyPDF** — PDF ingestion
