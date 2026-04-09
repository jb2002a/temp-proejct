
from ragas.testset import TestsetGenerator
from langsmith import Client

from pandas import DataFrame
from src.rag.common.clients import get_advanced_model, get_embed_model
from src.rag.pre_processing.document_pre_proccessing import load_documents

import asyncio
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop import (
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
)

async def generate_test_dataset_and_store() -> DataFrame:
    llm = get_advanced_model()

    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
        (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
        (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
    ]

    for query, _ in distribution:
        prompts = await query.adapt_prompts("korean", llm=llm)
        query.set_prompts(**prompts)

    embedding_model = get_embed_model()
    documents = load_documents()

    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embedding_model)
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=1,
        query_distribution=distribution,  # adapt된 distribution 전달
    )
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