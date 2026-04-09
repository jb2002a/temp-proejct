
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
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

    llm_context = "반드시 모든 질문과 답변은 한국어로 작성하세요."

    generator = TestsetGenerator.from_llama_index(llm=llm, embedding_model=embedding_model, llm_context=llm_context)
    testset = generator.generate_with_llamaindex_docs(documents=documents, testset_size=20)

    df = testset.to_pandas()
    return df

def upload_pandas_with_langsmith(df: DataFrame) -> None:

if __name__ == "__main__":
    # python -m src.ragas.ragas_testset
    generate_test_dataset_and_store()
    print("Test dataset generated and stored")