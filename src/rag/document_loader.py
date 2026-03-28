from llama_index.core import Document
from typing import List
from llama_index.core import SimpleDirectoryReader
from src.rag.config_db import PDF_FOLDER_PATH, PDF_READER

def load_documents() -> List[Document]:
    """
    Load all pdf files in the PDF folder and return a list of document objects
    """
    reader = SimpleDirectoryReader(
      input_dir=str(PDF_FOLDER_PATH),
      required_exts=[".pdf"],              # PDF만 읽기
      recursive=False,                     # 하위 폴더까지면 True
      file_extractor={".pdf": PDF_READER} # 명시적으로 PDF reader 지정
  )
  
    documents = reader.load_data()
    return documents

if __name__ == "__main__":
    documents = load_documents()   
    print(f"Loading Done, Total documents: {len(documents)}")

