from src.rag.document_retriever import query_vector_store_index, get_vector_store_index
from src.rag.remove_wrong_characters import clean_pdf_text
from src.graph.state import State

def retriever_node(state: State) -> dict:
    """
    Retriever node and return the context with post-processing
    """
    query = state.get("query") or ""
    
    vector_store_index = get_vector_store_index()
    nodes = query_vector_store_index(query, vector_store_index)

    texts = [clean_pdf_text(nws.text) for nws in (nodes or [])]

    # Ragas 점수 계산
    return {"context": texts}
    