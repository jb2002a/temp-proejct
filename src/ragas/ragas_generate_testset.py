from llama_index.core import Settings

from src.rag.document_loader import load_documents
from ragas.testset import TestsetGenerator

import pathlib

from pandas import DataFrame

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TESTSET_FOLDER_PATH = PROJECT_ROOT / "test_set"
TESTSET_FILE_PATH = PROJECT_ROOT / "test_set" / "testset.jsonl"


def generate_test_dataset() -> DataFrame:
    documents = load_documents()
    documents = documents[10:150]

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