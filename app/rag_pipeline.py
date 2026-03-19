import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")


# ---------------------------------------------------------------------------
# Document loading & splitting
# ---------------------------------------------------------------------------
def load_and_split_pdf(
    pdf_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Load a PDF and split it into overlapping text chunks."""
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    logger.info(f"Loaded {len(pages)} pages from {pdf_path.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    logger.info(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def get_embeddings() -> AzureOpenAIEmbeddings:
    """Create an Azure OpenAI embeddings client."""
    return AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        openai_api_version="2023-05-15",
    )


# ---------------------------------------------------------------------------
# Vectorstore
# ---------------------------------------------------------------------------
def index_documents(
    docs: list[Document],
    embeddings: AzureOpenAIEmbeddings,
) -> InMemoryVectorStore:
    logger.info(f"Embedding and indexing {len(docs)} chunks...")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vectors = embeddings.embed_documents(texts)
    vectorstore = InMemoryVectorStore(embedding=embeddings)
    vectorstore.add_texts(texts=texts, metadatas=metadatas, embeddings=vectors)
    logger.info(f"Indexed {len(docs)} chunks into InMemoryVectorStore")
    return vectorstore


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------
def build_rag_chain(vectorstore: InMemoryVectorStore, k: int = 5):
    """Build a retrieval-augmented generation chain using Azure OpenAI GPT-4o.

    Returns a LangChain Runnable that accepts {"input": question} and returns
    {"input": ..., "context": list[Document], "answer": str}.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
    )

    system_prompt = (
        "You are a financial analyst assistant. Use only the provided context "
        "to answer the question. If the answer is not contained in the context, "
        "say 'I cannot find this information in the provided documents.' "
        "Be precise with numbers, dates, and financial figures.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------
def query(chain, question: str) -> dict:
    """Run a question through the RAG chain.

    Returns:
        {"answer": str, "context": list[Document]}
    """
    logger.info(f"Query: {question}")
    result = chain.invoke({"input": question})
    logger.info(f"Answer length: {len(result['answer'])} chars")
    return {"answer": result["answer"], "context": result["context"]}


# ---------------------------------------------------------------------------
# Test / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}.")
        print("Add a financial PDF (e.g. an annual report) to the data/ folder and re-run.")
        raise SystemExit(1)

    test_pdf = pdf_files[0]
    print(f"Using: {test_pdf.name}\n")

    # Ingest
    docs = load_and_split_pdf(test_pdf)
    print(f"Loaded {len(docs)} chunks\n")

    embeddings = get_embeddings()
    vectorstore = index_documents(docs, embeddings)

    # Build chain
    chain = build_rag_chain(vectorstore, k=5)

    # Run a sample question
    test_question = "What is the total revenue reported for the most recent fiscal year?"
    print(f"Question: {test_question}\n")

    result = query(chain, test_question)
    print(f"Answer:\n{result['answer']}\n")

    print(f"Sources ({len(result['context'])} chunks retrieved):")
    for i, doc in enumerate(result["context"], 1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"  [{i}] {Path(src).name}, page {page}")
