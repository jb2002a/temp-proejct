from src.rag.common.clients import get_model, get_vector_store_from_chroma
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

def generate_response_and_context(query: str) -> dict[str, list[Document]]:
    """Generate a response and context."""
    model = get_model()
    vector_store = get_vector_store_from_chroma()
    retriever = vector_store.as_retriever(
        search_kwargs={
        "filter": {
            "Header 1": {
                "$nin": ["↘ 참고문헌", "↘ 참고 문헌","↘ 권고도출 자료원", "↘ 권고도출자료원"]
            }
        },
        }
    )

    retrieved_docs = retriever.invoke(query)
    messages = build_messages(query, retrieved_docs)

    response = model.invoke(messages)
    return {"response": response.content, "retrieved_docs": retrieved_docs}

def build_messages(query: str, retrieved_docs: list[Document]):

    docs_info = "\n\n".join(
        f"[{i+1}] {doc.metadata['Header 1']}\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)
    )

    system = SystemMessage(
        content=(
            "너는 한국어 의료 QA 어시스턴트다. "
            "질문에 대하여, 주어진 컨텍스트만 근거로 답하고, 근거가 없으면 모른다고 답해라. "
        )
    )

    human = HumanMessage(
        content=(
            "### 질문\n"
            f"{query}\n\n"

            "### 헤더 + 컨텍스트\n"
            f"{docs_info}\n\n"
        )
    )

    return [system, human]

if __name__ == "__main__":
    # python -m src.rag.post_processing.response

    query = "1. America Diabetes Association. Report of the expert committee on the diagnosis and classification of diabetes mellitus. Diabetes Care 2002;25(1):S5-S20 이라는 참고문헌 부분이 존재하는지 검색"

    result = generate_response_and_context(query=query)
    response, retrieved_docs = result["response"], result["retrieved_docs"]
   # 저장

    with open("test_response.txt", "w", encoding="utf-8") as f:
        f.write(response)
        f.write("\n\n")
        for idx, doc in enumerate(retrieved_docs):
            f.write(f"[{idx+1}] {doc.metadata['Header 1']}\n\n")
            f.write(doc.page_content)
            f.write("\n\n")
    
    print("Work done!")