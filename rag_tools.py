from functools import lru_cache
from pathlib import Path

from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_PATH = Path(
    "documents/retail_dataset_guide.txt"
)

CHROMA_PATH = "data/chroma_db"

COLLECTION_NAME = "retail_knowledge"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD DOCUMENT
# ============================================================

def load_document():

    if not DOCUMENT_PATH.exists():

        raise FileNotFoundError(
            f"Knowledge document not found: "
            f"{DOCUMENT_PATH}"
        )

    return DOCUMENT_PATH.read_text(
        encoding="utf-8"
    )


# ============================================================
# SPLIT DOCUMENT
# ============================================================

def split_document(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    return splitter.create_documents(
        [text]
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# CREATE / LOAD VECTOR DATABASE
# ============================================================

@lru_cache(maxsize=1)
def get_vector_store():

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    # Check whether the collection already contains documents
    try:
        existing = vector_store.get()

        existing_ids = existing.get(
            "ids",
            []
        )

    except Exception:
        existing_ids = []

    # Create the knowledge base automatically
    # when running on a fresh machine/cloud server.
    if not existing_ids:

        text = load_document()

        chunks = split_document(
            text
        )

        vector_store.add_documents(
            chunks
        )

    return vector_store


# ============================================================
# RAG SEARCH TOOL
# ============================================================

@tool
def search_knowledge_base(
    question: str
) -> str:
    """Search the retail knowledge base.

    Use this for questions about:
    - dataset definitions
    - column meanings
    - business definitions
    - currency
    - forecasting concepts
    - information contained in the project documentation
    """

    if not question or not question.strip():

        return (
            "Please provide a question to search "
            "the knowledge base."
        )

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        question.strip(),
        k=2,
    )

    if not documents:

        return (
            "No relevant information was found "
            "in the knowledge base."
        )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return context


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )
    print(
        "MASHA RAG TOOL TEST"
    )
    print(
        "========================================"
    )

    question = (
        "What does Customer ID mean?"
    )

    result = search_knowledge_base.invoke(
        {
            "question": question
        }
    )

    print("\nQuestion:")
    print(question)

    print("\nRetrieved information:")
    print(result)