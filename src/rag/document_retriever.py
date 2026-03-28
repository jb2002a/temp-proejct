from src.rag.document_save_to_vectordb import get_embed_model
from src.rag.config_db import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from typing import List
from llama_index.core.schema import NodeWithScore


def get_vector_store_index() -> VectorStoreIndex:
    """
    Connect to the ChromaDB and return the vector store
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH) 
    chroma_collection = client.get_collection(CHROMA_COLLECTION_NAME)

    embed_model = get_embed_model()

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return index

def query_vector_store_index(query: str) -> List[NodeWithScore]:
    """
    Query the vector store index and return the results
    """
    index = get_vector_store_index()
    retriever = index.as_retriever()
    nodes = retriever.retrieve(query)
    return nodes

if __name__ == "__main__":
    # python -m src.rag.document_retriever
    nodes = query_vector_store_index("진료지침 개정 범위와 목적은 뭐야?")
    print(nodes)