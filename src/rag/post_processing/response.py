from src.rag.post_processing.prompt_genearte import create_prompt_with_retriever_context
from src.rag.common.clients import get_model, get_vector_store_from_chroma



def generate_response(query: str) -> str:
    model = get_model()
    vector_store = get_vector_store_from_chroma()
    prompt = create_prompt_with_retriever_context(query, vector_store)

    response = model.invoke(prompt)
    return response.content

if __name__ == "__main__":
    # python -m src.rag.post_processing.response
    response = generate_response("노인 당뇨병에서 혈당조절이 안되는 환자인 경우 치료법")
    print(response)




