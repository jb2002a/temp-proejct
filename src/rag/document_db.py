from src.rag.config import Settings
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from src.rag.document_loader import load_documents

def save_index_to_chroma() -> None:
    """
    create the index and save to the ChromaDB
    """
    documents = load_documents()
    storage_context = get_storage_context()

    VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, show_progress=True)

def get_index_from_chroma() -> VectorStoreIndex:
    """
    Get the index from the ChromaDB
    """
    storage_context = get_storage_context()
    index = VectorStoreIndex.from_vector_store(vector_store=storage_context.vector_store)
    return index

def get_storage_context() -> StorageContext:
    """
    Get the storage context
    """
    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_or_create_collection("guideline_collection")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    print("노드(청크) 수:", collection.count())
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

if __name__ == "__main__":
    save_index_to_chroma()
    print("Index saved to ChromaDB")