from llama_index.core import Document
from typing import List
from llama_index.core import SimpleDirectoryReader

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PDF_FOLDER_PATH = PROJECT_ROOT / "pdfs"

def load_documents() -> List[Document]:
    """
    Load all pdf files in the PDF folder and return a list of document objects
    """
    documents = SimpleDirectoryReader(input_dir=PDF_FOLDER_PATH).load_data(show_progress=True)
  
    return documents

if __name__ == "__main__":
    # python -m src.rag.document_loader
    documents = load_documents()   
    print(f"Loading Done, Total documents: {len(documents)}")

