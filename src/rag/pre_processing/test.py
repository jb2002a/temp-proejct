# document 그대로 어떤형식으로 저장되는지 확인하는 테스트

from src.rag.pre_processing.document_pre_proccessing import load_documents

documents = load_documents()

#문서 저장
document = documents[0].page_content
with open("document.txt", "w", encoding="utf-8") as f:
    f.write(document)

# python -m src.rag.pre_processing.test