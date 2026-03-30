
from src.rag.config import Settings
from src.rag.document_db import save_index_to_chroma, get_index_from_chroma
from src.rag.document_query import index_query

def save_index_to_db() -> None:
    """
    Save the index to the database
    """
    save_index_to_chroma()
    print("Index saved to ChromaDB")

def query_index(query: str) -> str:
    """
    Query the index
    """
    print("Index retrieved from ChromaDB")
    result = index_query(query)
    print("Query executed")
    return result

if __name__ == "__main__":
    save_index_to_db()
    query = "What is the purpose of the guideline?"
    response = query_index(query)
    print(response)