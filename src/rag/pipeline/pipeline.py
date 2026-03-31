
from src.rag.config import Settings
from src.rag.document_db import save_index_to_chroma
from src.rag.document_query import query_index
from src.ragas.ragas_generate_testset import generate_test_dataset

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
    result = query_index(query)
    print("Query executed")
    return result

def generate_dataset() -> None:
    """
    Generate a test dataset
    """
    generate_test_dataset()
    print("Test dataset generated")

if __name__ == "__main__":
    generate_dataset()