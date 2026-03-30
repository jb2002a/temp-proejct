
from llama_index.core.retrievers import VectorIndexRetriever
from src.rag.document_db import get_index_from_chroma
from llama_index.core.prompts import RichPromptTemplate
from src.rag.config import Settings
from typing import Required

def get_prompt(context_str: str, query_str: str) -> str:
    template_str = """You answer using only the numbered document excerpts below (retrieval-augmented generation).

    Rules:
    - Ground every factual claim in those excerpts. Do not invent facts or use outside knowledge.
    - If the excerpts are empty or do not support an answer, say the context is insufficient and do not guess.
    - When you use information from an excerpt, cite its bracket label, e.g. [1], [2].
    - Write the answer in the same language as the user's question.

    --- Retrieved excerpts ---
    {{ context_str }}
    ---

    User question: {{ query_str }}

    Answer:
    """
    qa_template = RichPromptTemplate(template_str)
    return qa_template.format(context_str=context_str, query_str=query_str)

def index_query(query: Required[str]) -> str:
    """
    Query the index
    """
    
    index = get_index_from_chroma()

    retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
    )

    retrieved_nodes = retriever.retrieve(query)
    context_str = "\n\n".join([f"[{index+1}] {node.text}" for index,node in enumerate(retrieved_nodes)])
    print(context_str)

    prompt = get_prompt(context_str=context_str, query_str=query)

    llm = Settings.llm
    response = llm.complete(prompt)

    return response

if __name__ == "__main__":
    response = index_query("What is the purpose of the guideline?")
    print("-"*50)
    print(response)