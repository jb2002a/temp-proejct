
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from src.rag.common.clients import get_vector_store_from_chroma
from src.rag.common.config import PDF_FILE_PATH

# 전처리 과정
# 문서 로드 -> 청킹 -> 임베딩&벡터스토어 저장

def load_documents() -> List[Document]:
    """
    Load a single pdf file and return a document object
    To save the context, use mode="single" parameter
    """
    loader = PyMuPDFLoader(file_path=PDF_FILE_PATH,mode="single")
    documents = loader.load()
    print(f"Load Done, Total pages: {len(documents)}")
    return documents

def split_document(documents: List[Document]) -> List[Document]:
    """
    Split the document into smaller chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
    all_splits = text_splitter.split_documents(documents)
    print(f"Split Done, Total chunks: {len(all_splits)}")
    return all_splits

def save_chunks_to_chroma(all_splits: List[Document]) -> None:
    """
    Save the chunks to the ChromaDB
    """
    vector_store = get_vector_store_from_chroma()
    document = load_documents()
    all_splits = split_document(document)

    document_ids = vector_store.add_documents(documents=all_splits)
    print("Save Done")

if __name__ == "__main__":
    # python -m src.rag.pre_processing.document_pre_proccessing
    documents = load_documents()   
    all_splits = split_document(documents)
    save_chunks_to_chroma(all_splits)


