
from ragas.testset import TestsetGenerator
import pathlib
from langsmith import Client
import os
import pandas as pd

from pandas import DataFrame
from src.rag.common.clients import get_advanced_model, get_embed_model
from src.rag.pre_processing.document_pre_proccessing import load_documents


def generate_test_dataset_and_store() -> DataFrame:
    llm = get_advanced_model()
    embedding_model = get_embed_model()
    documents = load_documents()

    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embedding_model)
    testset = generator.generate_with_langchain_docs(documents=documents, testset_size=1)
    df = testset.to_pandas()

    return df

def upload_pandas_with_langsmith(df: DataFrame) -> None:
    client = Client()

    input_keys = ["user_input", "reference_contexts"]
    output_keys = ["reference", "response"]

    client.upload_dataframe(df, name="ragas_testset", input_keys=input_keys, output_keys=output_keys)

if __name__ == "__main__":
    # python -m src.rag.dataset.dataset_generate
    generate_test_dataset_and_store()
    print("Test dataset generated and stored")