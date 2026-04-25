from langchain_core.documents import Document
from typing import List
from src.rag.common.clients import get_vector_store_from_chroma
from src.rag.common.config import JSONL_FILE_PATH
from dotenv import load_dotenv
from langsmith import traceable
import re
import json

load_dotenv(override=True)


@traceable 
def load_jsonl_file() -> List[Document]:

    docs = []
    with open(JSONL_FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = (row.get("page_content") or "").strip()
            if not text:
                continue
            metadata = {
                "source_file": row.get("source_file", "na"),
                "page": row.get("page", "na"),
                "current_subject": row.get("current_subject", "na"),
                "category": row.get("category", "na"),
                "id": row.get("id", "na"),
            }

            docs.append(Document(page_content=text, metadata=metadata))

    return docs

@traceable 
def save_chunks_to_chroma(documents: List[Document]) -> None:
    """
    Reset the ChromaDB and Save the chunks to the ChromaDB 
    """
    
    vector_store = get_vector_store_from_chroma()
    vector_store.reset_collection() 
    print("Reset Done")
    print("Save started")
    vector_store.add_documents(documents=documents)
    print(f"Saved Done, Total saved nodes: {vector_store._collection.count()}")


if __name__ == "__main__":
    # python -m src.rag.pre_processing.json_to_vector_store
    documents = load_jsonl_file()
    save_chunks_to_chroma(documents)


