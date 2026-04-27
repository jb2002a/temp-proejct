
from ragas.testset import TestsetGenerator
from pandas import DataFrame
from src.rag.common.clients import get_model, get_embed_model
from src.rag.pre_processing.json_to_vector_store import load_jsonl_file
from pathlib import Path
import asyncio
from src.rag.post_processing.response import EXCLUDED_SUBJECTS
from langchain_core.documents import Document
import os

async def generate_test_dataset_and_store() -> DataFrame:
    llm = get_model()
    embedding_model = get_embed_model()

    os.environ["LANGSMITH_TRACING"] = "false"

    documents = load_jsonl_file()
    documents = sort_out_excluded_subjects(documents)

    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embedding_model)
    try:
        testset = generator.generate_with_langchain_docs(documents=documents, testset_size=30)
    except Exception as e:
        print(f"Testset generation failed: {type(e).__name__}: {e}")
        raise
    return testset.to_pandas()

def sort_out_excluded_subjects(retrieved_docs: list[Document]) -> list[Document]:
    return [doc for doc in retrieved_docs if doc.metadata["current_subject"] not in EXCLUDED_SUBJECTS]

if __name__ == "__main__":
    # python -m src.rag.dataset.dataset_generate
    df = asyncio.run(generate_test_dataset_and_store())

    output_path = Path("src/test_set/testset.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Test dataset generated and stored as {output_path.name}")

