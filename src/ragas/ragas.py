import src.rag.document_loader as document_loader
from dotenv import load_dotenv
import os
from ragas.llms import llm_factory, InstructorBaseRagasLLM
from openai import OpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.testset import Testset, TestsetGenerator
from llama_index.core import Document
from typing import List



def generate_test_dataset() -> List[Testset]:
    document_list: List[Document] = document_loader.load_documents()
    
    llm = get_llm()
    embedding_model = get_huggingface_embeddings()
    
    generator = TestsetGenerator(llm=llm, embedding_model=embedding_model)
    dataset = generator.generate_with_llamaindex_docs(document_list, testset_size=10)
    return dataset

def get_llm() -> InstructorBaseRagasLLM:
    
    load_dotenv(override=True)   
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory("gpt-4o-mini", client=client)
    return llm

def get_huggingface_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model="BAAI/bge-m3")

if __name__ == "__main__":
    dataset = generate_test_dataset()
    print(dataset)