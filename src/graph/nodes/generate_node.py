from src.graph.state import State
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


def generate_node(state: State) -> dict:
    """
    Generate node
    """
    query = state.get("query") or ""

    texts = [nws.text for nws in (state.get("nodes") or [])]
    context = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(texts))

    prompt = create_prompt_template()   
    model = get_google_genai_client()
    
    sequence = prompt | model
    
    result = sequence.invoke({"context": context, "query": query})

    return {"answer": result.content}

def get_google_genai_client() -> ChatGoogleGenerativeAI:
    """
    Get GenAI client ; for this project, we use Gemini 2.5 Flash Lite
    """
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key is None:
        raise ValueError("GOOGLE_API_KEY is not set")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    return model


def create_prompt_template() -> ChatPromptTemplate:
    """
    Create prompt
    """ 
    system_prompt = """당신은 제공된 "참고 문서"만을 근거로 답하는 어시스턴트입니다.
    규칙:
    - 참고 문서에 근거가 없는 내용은 추측하지 말고, "제공된 문서에서 확인할 수 없습니다"라고 말하세요.
    - 답할 때 사용한 근거는 반드시 대괄호 번호로 인용하세요. 예: [1], [2].
    - 참고 문서와 모순되면 문서를 우선하세요.
    - 불필요하게 길게 쓰지 말고, 질문에 맞게 구조화해 답하세요."""

    human_prompt= """아래는 검색으로 가져온 참고 문서 조각입니다. 각 블록 앞의 [번호]는 인용에 사용합니다.
    {context}

    ------------------------------
    query: {query}
    
    위 참고 문서만을 근거로 질문에 답하세요. 근거가 없으면 그렇다고 명시하세요.
    """

    template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_prompt)]
    )

    return template

