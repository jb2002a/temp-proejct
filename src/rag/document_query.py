
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from src.rag.document_db import get_index_from_chroma


def index_query(query: str) -> str:
    """
    Query the index
    """
    
    index = get_index_from_chroma()
    
    retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
    )

    query_engine = RetrieverQueryEngine(retriever=retriever)
    response = query_engine.query(query)

    return response

if __name__ == "__main__":
    response = index_query("What is the purpose of the guideline?")
    print(response)