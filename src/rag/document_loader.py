from llama_index.readers.file import PDFReader
from llama_index.core import Document
from typing import List
from pathlib import Path
from llama_index.core import SimpleDirectoryReader

from src.rag.config_db import pdf_folder_path

def load_documents(pdf_folder_path: Path) -> List[Document]:
    """
    Load all pdf files in the PDF folder and return a list of document objects
    """
    reader = SimpleDirectoryReader(
      input_dir=str(pdf_folder_path),
      required_exts=[".pdf"],              # PDF만 읽기
      recursive=False,                     # 하위 폴더까지면 True
      file_extractor={".pdf": PDFReader()} # 명시적으로 PDF reader 지정
  )
  
    documents = reader.load_data()
    return documents

if __name__ == "__main__":
    documents = load_documents(pdf_folder_path)   
    print(f"Loading Done, Total documents: {len(documents)}")

