from langchain_upstage import UpstageDocumentParseLoader
from dotenv import load_dotenv
from src.rag.common.config import PDF_FILE_PATH

load_dotenv(override=True)

def load_documents():
    loader = UpstageDocumentParseLoader(PDF_FILE_PATH)
    documents = loader.load()
    return documents

if __name__ == "__main__": 
    # python -m src.rag.pre_processing.loader_test
    documents = load_documents()
    print(f"Load Done, Total pages: {len(documents)}")
    for d in documents[:3]:
        print(f"- {d.page_content=}")
        