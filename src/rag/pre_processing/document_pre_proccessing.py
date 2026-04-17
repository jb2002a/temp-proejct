from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter

from typing import List
from src.rag.common.clients import get_vector_store_from_chroma
from src.rag.common.config import HTML_FILE_PATH
from dotenv import load_dotenv

# 전처리 과정
# 문서 로드(API 제한떄문에 따로 저장시키고 읽는 로직으로 변경) -> 청킹 -> 임베딩&벡터스토어 저장
load_dotenv(override=True)

def read_split_result() -> str:
    with open(HTML_FILE_PATH, "r", encoding="utf-8") as f:
        html_string = f.read()
    return html_string

def split_html(html_string: str) -> List[Document]:
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    html_splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on,
        max_chunk_size=1000,
        chunk_overlap=100,
        elements_to_preserve=["table"]
        )
    html_header_splits = html_splitter.split_text(html_string)

    #저장
    #헤더 무시하는 문제 발생함
    #ocr 모드로 켜져서 오류난거일수도있고
    #load 문서확인해봐야할듯.
    #그리고 옵저빙 랭스미스로 다옮겨
    with open("all_splits.txt", "w", encoding="utf-8") as f:
        for i, split in enumerate(html_header_splits):
            f.write(f"[{i+1}] {split.metadata['Header 1']}\n")
            f.write(split.page_content)
            f.write("\n\n")

    return html_header_splits

def save_chunks_to_chroma(all_splits: List[Document]) -> None:
    """
    Reset the ChromaDB and Save the chunks to the ChromaDB 
    """
    vector_store = get_vector_store_from_chroma()
    vector_store.reset_collection() 
    print("Reset Done")
    print("Save started")
    vector_store.add_documents(documents=all_splits)
    print(f"Saved Done, Total saved nodes: {vector_store._collection.count()}")

if __name__ == "__main__":
    # python -m src.rag.pre_processing.document_pre_proccessing
    documents = read_split_result()   
    all_splits = split_html(documents)
    save_chunks_to_chroma(all_splits)
    


