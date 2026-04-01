from llama_index.core import Settings

from src.rag.document_loader import load_documents
from ragas.testset import TestsetGenerator

import pathlib

from pandas import DataFrame
from ragas.dataset_schema import EvaluationDataset
import json
from pathlib import Path

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TESTSET_FOLDER_PATH = PROJECT_ROOT / "test_set"
TESTSET_FILE_PATH = PROJECT_ROOT / "test_set" / "testset.jsonl"


def generate_test_dataset_and_store() -> DataFrame:
    documents = load_documents()

    llm = Settings.llm
    embedding_model = Settings.embed_model

    llm_context = "생성하는 질문(user query)과 정답(reference answer)은 모두 한국어로만 작성하세요."

    generator = TestsetGenerator.from_llama_index(llm=llm, embedding_model=embedding_model, llm_context=llm_context)
    testset = generator.generate_with_llamaindex_docs(documents=documents, testset_size=20)

    df = testset.to_pandas()

    if not TESTSET_FOLDER_PATH.exists():
        TESTSET_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    df.to_json(TESTSET_FILE_PATH, orient="records", lines=True, force_ascii=False)

    return df

def get_test_dataset() -> EvaluationDataset:
     p = Path(TESTSET_FILE_PATH)
     with p.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
     return EvaluationDataset.from_list(data)

if __name__ == "__main__":
    # python -m src.ragas.ragas_testset
    generate_test_dataset_and_store()
    print("Test dataset generated and stored")
    dataset = get_test_dataset()
    print(dataset)