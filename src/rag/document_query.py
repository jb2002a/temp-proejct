
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
    response = query_index("경구혈당강하제의 단독요법 실패 시, 심혈관질환 예방을 위해 어떤 약제를 추가하는 것이 권장되며 그 이유는 무엇인가요?")
    
    print(get_contexts(response))
    