from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD CHROMA VECTOR DATABASE
# ============================================================

def load_vector_store():

    embeddings = create_embedding_model()

    vector_store = Chroma(
        collection_name="retail_knowledge",
        persist_directory="data/chroma_db",
        embedding_function=embeddings
    )

    return vector_store


# ============================================================
# RAG SEARCH TOOL
# ============================================================

@tool
def search_knowledge_base(question: str) -> str:
    """Search the retail knowledge base for information about
    the dataset, columns, business definitions, currency,
    and forecasting concepts.

    Use this tool when the user asks about information that
    should come from the project documentation rather than
    numerical calculations from the CSV.
    """

    vector_store = load_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=2
    )

    if not documents:
        return "No relevant information was found in the knowledge base."

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return context


# ============================================================
# TEST TOOL
# ============================================================

if __name__ == "__main__":

    result = search_knowledge_base.invoke(
        {
            "question": "What does Customer ID mean?"
        }
    )

    print("========================================")
    print("RAG TOOL TEST")
    print("========================================")

    print("\nRetrieved information:")
    print(result)