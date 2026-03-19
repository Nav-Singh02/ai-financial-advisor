import logging
import sys
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict

from rag_pipeline import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_API_VERSION,
    DATA_DIR,
    load_and_split_pdf,
    get_embeddings,
    index_documents,
)
from audit_logger import log_interaction

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared LLM instance (load_dotenv already called by rag_pipeline import)
# ---------------------------------------------------------------------------
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    context: Optional[list[Document]]
    summary: Optional[str]
    risk_notes: Optional[str]
    draft_email: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_context(docs: list[Document]) -> str:
    """Convert retrieved Document chunks into a numbered text block for LLM prompts."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i} | source: {source}, page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------
def build_graph(vectorstore):
    """Build and compile the LangGraph agentic pipeline.

    Args:
        vectorstore: An InMemoryVectorStore with .similarity_search(query, k) -> list[Document]

    Returns:
        CompiledStateGraph — invoke with {"question": str}
    """

    def retrieve(state: AgentState) -> dict:
        logger.info("Node: retrieve | question=%s", state["question"])
        docs = vectorstore.similarity_search(state["question"], k=5)
        logger.info("Retrieved %d chunks", len(docs))
        return {"context": docs}

    def summarize(state: AgentState) -> dict:
        logger.info("Node: summarize")
        context_text = _format_context(state["context"])
        messages = [
            SystemMessage(content=(
                "You are a senior financial analyst. "
                "Using only the provided document excerpts, write a concise portfolio summary "
                "(3-5 sentences). Cover key financial metrics, performance highlights, and "
                "overall portfolio positioning. Do not speculate beyond the provided text."
            )),
            HumanMessage(content=(
                f"Document excerpts:\n\n{context_text}\n\n"
                "Please write the portfolio summary now."
            )),
        ]
        response = llm.invoke(messages)
        return {"summary": response.content}

    def flag_risk(state: AgentState) -> dict:
        logger.info("Node: flag_risk")
        context_text = _format_context(state["context"])
        messages = [
            SystemMessage(content=(
                "You are a risk analyst. "
                "Review the following document excerpts and identify the key risk factors "
                "mentioned or implied. List each risk with a brief explanation. "
                "Focus on: market risks, credit risks, liquidity risks, regulatory risks, "
                "and concentration risks. Only report risks evidenced in the text."
            )),
            HumanMessage(content=(
                f"Document excerpts:\n\n{context_text}\n\n"
                "Please list the key risk factors now."
            )),
        ]
        response = llm.invoke(messages)
        return {"risk_notes": response.content}

    def draft_email(state: AgentState) -> dict:
        logger.info("Node: draft_email")
        messages = [
            SystemMessage(content=(
                "You are a financial advisor writing to a client. "
                "Compose a professional, clear, and empathetic email that: "
                "(1) summarizes their portfolio performance, "
                "(2) highlights key risks they should be aware of, and "
                "(3) closes with a reassuring next-steps statement. "
                "Tone: professional but approachable. Length: 3-4 paragraphs."
            )),
            HumanMessage(content=(
                f"Portfolio Summary:\n{state['summary']}\n\n"
                f"Risk Notes:\n{state['risk_notes']}\n\n"
                "Please draft the client email now."
            )),
        ]
        response = llm.invoke(messages)
        log_interaction(
            question=state["question"],
            summary=state["summary"],
            risk_notes=state["risk_notes"],
            draft_email=response.content,
            source_chunks=state["context"],
        )
        return {"draft_email": response.content}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("summarize", summarize)
    graph.add_node("flag_risk", flag_risk)
    graph.add_node("compose_email", draft_email)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", "flag_risk")
    graph.add_edge("flag_risk", "compose_email")
    graph.add_edge("compose_email", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Test / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}.")
        print("Add a financial PDF to the data/ folder and re-run.")
        sys.exit(1)

    test_pdf = pdf_files[0]
    print(f"Loading: {test_pdf.name}\n")

    docs = load_and_split_pdf(test_pdf)
    embeddings = get_embeddings()
    vectorstore = index_documents(docs, embeddings)

    agent = build_graph(vectorstore)

    question = "What are the key financial highlights and risks for this portfolio?"
    print(f"Running agent with question:\n  {question}\n")

    result = agent.invoke({"question": question})

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(result["summary"])

    print("\n" + "=" * 60)
    print("RISK NOTES")
    print("=" * 60)
    print(result["risk_notes"])

    print("\n" + "=" * 60)
    print("DRAFT EMAIL")
    print("=" * 60)
    print(result["draft_email"])
