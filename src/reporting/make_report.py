from src.rag.config_db import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    PDF_FOLDER_PATH,
    EMBEDDING_MODEL_NAME,
    REPORT_FOLDER_PATH,
    TEST_DATASET_PATH,
)
from src.rag.document_retriever import get_vector_store_index, query_vector_store_index
from datetime import datetime
import json

# 허깅페이스 로깅 제거
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING) 

def make_deafult_report() -> str:
    """
    Make a default report String
    """
    report = ""

    report += "*" * 100 + "\n"
    report += f"CHROMA_DB_PATH: {CHROMA_DB_PATH}\n"
    report += f"CHROMA_COLLECTION_NAME: {CHROMA_COLLECTION_NAME}\n"
    report += f"PDF_FOLDER_PATH: {PDF_FOLDER_PATH}\n"
    report += f"EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}\n"
    report += f"REPORT_FOLDER_PATH: {REPORT_FOLDER_PATH}\n"
    report += f"TEST_DATASET_PATH: {TEST_DATASET_PATH}\n"
    report += "*" * 100 + "\n"
    report += "\n"
    report += "\n"
    report += "\n"
    return report

def make_report() -> None:
    """
    Make a report of the document and save it to a text file
    """
    report = make_deafult_report()

    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)

    print(f"Loading model started, this may take a while...")
    VectorStoreIndex = get_vector_store_index()

    for index_test, test in enumerate(test_dataset):
        print(f"Processing test {index_test + 1} of {len(test_dataset)} started")

        query = test["query"]
        ground_truth_context = test["ground_truth_context"]

        report += f"Test {index_test + 1}\n"
        report += f"Query: {query}\n"
        report += f"Ground Truth Context: {ground_truth_context}\n"
        report += "=" * 100 + "\n"

        nodes = query_vector_store_index(query, VectorStoreIndex)

        for index, node in enumerate(nodes):
            report += f"Index: {index}\n"
            report += f"Text: {node.text}\n"
            report += f"Score: {node.score}\n"
            report += "=" * 100 + "\n"
        report += "\n"
        report += "\n"
        print(f"Processed test {index_test + 1} is done")
        print("=" * 100 + "\n\n")

    print(f"All tests are done")
    save_report(report)

def save_report(report: str) -> None:
    """
    Save the report to a text file
    """
    if not REPORT_FOLDER_PATH.exists():
        REPORT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {REPORT_FOLDER_PATH}")
    report_name = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    with open(REPORT_FOLDER_PATH / report_name, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {REPORT_FOLDER_PATH / report_name}")

if __name__ == "__main__":
    # python -m src.reporting.make_report
    make_report()
