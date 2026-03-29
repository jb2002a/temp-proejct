from src.rag.document_retriever import query_vector_store_index, get_vector_store_index
from src.graph.state import State

def retriever_node(state: State) -> dict:
    """
    Retriever node
    """
    query = state.get("query") or ""
    
    vector_store_index = get_vector_store_index()
    
    nodes = query_vector_store_index(query, vector_store_index)
    return {"nodes": nodes}
    