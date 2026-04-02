
from src.rag.config import Settings
from src.rag.document_db import get_index_from_chroma
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response

def get_query_engine() -> BaseQueryEngine:
    """
    Query the index
    """
    index = get_index_from_chroma()
    query_engine = index.as_query_engine()

    return query_engine

def query_index(query: str) -> Response:
    """
    Query the index
    """
    query_engine = get_query_engine()
    response = query_engine.query(query)
    return response

def get_contexts(response: Response) -> list[str]:
    """
    Get the contexts from the response
    """
    return [node.node.get_content() for node in response.source_nodes]

if __name__ == "__main__":
    # python -m src.rag.document_query
    user_input = "향후 10년의 CHD 위험도 예측"
    response = query_index(user_input)
    
    print(get_contexts(response))
    print("*" * 100)
    print(response.response)

    