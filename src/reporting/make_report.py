from src.rag.config_db import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, PDF_FOLDER_PATH, EMBEDDING_MODEL_NAME, REPORT_FOLDER_PATH
from src.rag.document_retriever import query_vector_store_index
from datetime import datetime

def make_report(query : str) -> None:
    """
    Make a report of the document and save it to a text file
    """
    report = ""

    report += f"Query: {query}\n"
    report += "=" * 100 + "\n"
    report += f"CHROMA_DB_PATH: {CHROMA_DB_PATH}\n"
    report += f"CHROMA_COLLECTION_NAME: {CHROMA_COLLECTION_NAME}\n"
    report += f"PDF_FOLDER_PATH: {PDF_FOLDER_PATH}\n"
    report += f"EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}\n"
    report += f"REPORT_FOLDER_PATH: {REPORT_FOLDER_PATH}\n"
    report += "=" * 100 + "\n"

    #NodeWithScore 객체를 반환, (Node,Score로 구성)
    nodes = query_vector_store_index(query)

    for index, node in enumerate(nodes):
        report += f"Index: {index}\n"
        report += f"Text: {node.text}\n"
        report += f"Score: {node.score}\n"
        report += "=" * 100 + "\n"

    if not REPORT_FOLDER_PATH.exists():
        REPORT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {REPORT_FOLDER_PATH}")

    report_name = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    with open(REPORT_FOLDER_PATH / report_name, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {REPORT_FOLDER_PATH / report_name}")


if __name__ == "__main__":
    # python -m src.reporting.make_report
    make_report("진료지침 개정 범위와 목적은 뭐야?")