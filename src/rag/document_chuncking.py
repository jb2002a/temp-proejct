from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.core.schema import BaseNode
from typing import List
from src.rag.document_loader import load_documents
from src.rag.config_db import CHUKING_LOGIC


#노드 :원본 문서(Document)를 특정 크기로 자른 조각(Chunk)을 객체화한 것, 청크단위 객체라고 보면됨
def chunk_document(document: List[Document]) -> List[BaseNode]:
    """ 
    Chunk the document into smaller chunks
    """
    all_chunks = []
    for doc in document:
        chunks = CHUKING_LOGIC.get_nodes_from_documents([doc])
        all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":  
    documents = load_documents()
    chunks = chunk_document(documents)
    for chunk in chunks:
        print(chunk.text)
        print("-" * 100)
    print(f"Chunking Done, Total chunks: {len(chunks)}")

