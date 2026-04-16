from langchain_upstage import UpstageDocumentParseLoader
from dotenv import load_dotenv
from src.rag.common.config import PDF_FILE_PATH
import json

load_dotenv(override=True)

def load_documents():
    loader = UpstageDocumentParseLoader(PDF_FILE_PATH, ocr="standard", output_format="html")
    documents = loader.load()
    return documents

if __name__ == "__main__": 
    # python -m src.rag.pre_processing.loader_test
    documents = load_documents()
    print(f"Load Done, Total pages: {len(documents)}")

    #저장(html)
    with open("documents.html", "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.page_content)
            f.write("\n\n---\n\n")

    print(f"Documents saved to documents.html")


        