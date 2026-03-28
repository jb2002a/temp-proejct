import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.rag.document_chuncking import chunk_document
from src.rag.document_loader import load_documents
from typing import List
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from src.rag.config_db import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME    
from src.rag.config_db import EMBEDDING_MODEL_NAME

def get_vector_db() -> ChromaVectorStore:
    """
    Delete the collection if it exists to avoid duplicate data and create a new one.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH) 
    
    # Delete the collection if it exists
    collections = client.list_collections()
    if CHROMA_COLLECTION_NAME in [c.name for c in collections]:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"Deleted collection: {CHROMA_COLLECTION_NAME}")

    # Create a new collection
    chroma_collection = client.create_collection(CHROMA_COLLECTION_NAME)
    print(f"Created collection: {CHROMA_COLLECTION_NAME}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

def get_embed_model() -> HuggingFaceEmbedding:
    """
    Get the embedding model
    """
    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    
def save_chunks_to_chroma(chunks: List[BaseNode]) -> None:
    """
    Embedding chunks with VectorStoreIndex and save the index to the ChromaDB
    """
    
    vector_store = get_vector_db()
    embed_model = get_embed_model()

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # index는 저장소 관리 객체라고 이해하는게 편함(저장,청킹,검색 등 모두 수행)
    index = VectorStoreIndex(
        storage_context=storage_context,
        nodes = chunks,
        embed_model=embed_model,
        show_progress=True
    )


if __name__ == "__main__":
    # python -m src.rag.document_save_to_vectordb
    documents = load_documents()
    chunks = chunk_document(documents)
    save_chunks_to_chroma(chunks)
    print(f"Saving Done, Total chunks: {len(chunks)}")
