from llama_index.readers.file import PDFReader
from llama_index.core import Document
from typing import List
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
pdf_file_path = PROJECT_ROOT / "pdfs" / "당뇨병 임상진료지침.pdf"

def load_documents(pdf_file_path: Path) -> List[Document]:
    """
    Load the document from the PDF file
    """
    reader = PDFReader()
    documents = reader.load_data(pdf_file_path)
    return documents

if __name__ == "__main__":
    documents = load_documents(pdf_file_path)
    print(documents)

