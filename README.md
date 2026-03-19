AI Financial Advisor RAG Agent
A LangGraph-powered agent that reads financial documents and answers questions about them — generating a portfolio summary, risk breakdown, and draft client email from a single query.
Built with LangChain, LangGraph, Azure OpenAI, and Streamlit.
What it does
Upload any financial PDF (10-K, earnings report, portfolio statement). Ask a question. The agent runs four steps in sequence:

Retrieve — pulls the most relevant chunks from the document using vector similarity search
Summarize — writes a concise portfolio summary from the retrieved content
Flag risks — identifies and categorizes risk factors across market, regulatory, liquidity, and concentration categories
Draft email — writes a professional client-facing email based on the findings

Every run is logged to logs/audit_log.jsonl with a timestamp, the question, all outputs, and chunk count.
Tech stack

LangChain — document loading, chunking, embeddings, retrieval
LangGraph — multi-node agent orchestration
Azure OpenAI — GPT-4o for generation, text-embedding-ada-002 for embeddings
Streamlit — demo UI
Python — built and tested on Python 3.12

Project structure
ai-financial-advisor/
├── app/
│   ├── rag_pipeline.py      # PDF ingestion, embedding, retrieval chain
│   ├── agent.py             # LangGraph workflow (4 nodes)
│   ├── audit_logger.py      # JSONL audit logging
│   └── streamlit_app.py     # Streamlit UI
├── data/                    # Drop PDFs here
├── logs/                    # Audit logs written here
├── requirements.txt
└── .env                     # Not committed — see setup below
Setup

Clone the repo and install dependencies:

bashpip install -r requirements.txt
```

2. Create a `.env` file in the project root:
```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

Drop a financial PDF into the data/ folder
Run the Streamlit app:

bashstreamlit run app/streamlit_app.py
Or run the agent directly from the terminal:
bashpython app/agent.py
Notes
This is a portfolio project. The in-memory vector store resets when the app restarts — this is intentional for simplicity. A production version would swap in a persistent store like Azure AI Search or Chroma.
