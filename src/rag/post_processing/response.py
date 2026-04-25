from src.rag.common.clients import get_model, get_vector_store_from_chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

EXCLUDED_SUBJECTS = [
    "# ↘ 참고문헌",
    "# 참고문헌",
    "# ↘ 권고도출 자료원",
    "# 권고도출 자료원",
]

@traceable
def generate_response_and_context(query: str) -> dict[str, list[Document]]:
    """Generate a response and context."""
    model = get_model()
    vector_store = get_vector_store_from_chroma()
    retriever = vector_store.as_retriever()

    retrieved_docs = retriever.invoke(query)
    messages = build_messages(query, retrieved_docs)

    response = model.invoke(messages)
    return {"response": response.content, "retrieved_docs": retrieved_docs}

@traceable
def build_messages(query: str, retrieved_docs: list[Document]):

    docs_info = "\n\n".join(
        f"[{i+1}] {doc.metadata['current_subject']}\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)
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

@traceable
def pipeline_response(query: str) -> dict[str, list[Document]]:
    result = generate_response_and_context(query=query)
    response, retrieved_docs = result["response"], result["retrieved_docs"]
    return {"response": response, "retrieved_docs": retrieved_docs}

if __name__ == "__main__":
    # python -m src.rag.post_processing.response
    query = "혈당조절시 조심해야할점은 뭐야?"
    result = pipeline_response(query=query)
    print(f"response: {result['response']}")
    print("-"*100)
    print(f"retrieved_docs: {result['retrieved_docs']}")
    
