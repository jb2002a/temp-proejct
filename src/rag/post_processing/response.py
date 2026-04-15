from src.rag.common.clients import get_model, get_vector_store_from_chroma
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

def generate_response_and_context(query: str) -> dict[str, list[str]]:
    """Generate a response and context."""
    model = get_model()
    vector_store = get_vector_store_from_chroma()
    retriever = vector_store.as_retriever()

    retrieved_docs = retriever.invoke(query)
    messages = build_messages(query, retrieved_docs)

    response = model.invoke(messages)
    return {"response": response.content, "retrieved_docs": [doc.page_content for doc in retrieved_docs]}

def build_messages(query: str, retrieved_docs: list[Document]):
    docs_content = "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(retrieved_docs)
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
            "### 컨텍스트\n"
            f"{docs_content}\n\n"
        )
    )

    return [system, human]

if __name__ == "__main__":
    # python -m src.rag.post_processing.response
    result = generate_response_and_context(query="노인 당뇨병에서 혈당조절이 안되는 환자인 경우 치료법")
    response, retrieved_docs = result["response"], result["retrieved_docs"]
    print(response)
    print("\n\n")
    print(retrieved_docs)





