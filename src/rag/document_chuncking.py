from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.core.schema import BaseNode
from typing import List
from src.rag.document_loader import load_documents, pdf_file_path

def chunk_document(document: List[Document]) -> List[BaseNode]:
    """ 
    Chunk the document into smaller chunks
    """
    splitter = SentenceSplitter()
    all_chunks = []
    for doc in document:
        chunks = splitter.get_nodes_from_documents([doc], show_progress=True)
        all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":  
    documents = load_documents(pdf_file_path)
    chunks = chunk_document(documents)
    for chunk in chunks:
        print(chunk.text)
        print("-" * 100)

