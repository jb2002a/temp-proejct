import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.rag.document_chuncking import chunk_document
from src.rag.document_loader import load_documents, pdf_file_path
from typing import List
from llama_index.core.schema import BaseNode

def save_chunks_to_chroma(chunks: List[BaseNode]) -> None:
    """
    Save chunks to ChromaDB, Get the existing collection if it exists, create a new one if it doesn't exist.
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = client.get_or_create_collection("diabetes_clinical_guideline_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    vector_store.add(chunks, show_progress=True)

if __name__ == "__main__":
    documents = load_documents(pdf_file_path)
    chunks = chunk_document(documents)
    save_chunks_to_chroma(chunks)

