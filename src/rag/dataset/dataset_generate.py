
from ragas.testset import TestsetGenerator
from pandas import DataFrame
from src.rag.common.clients import get_gemini_model, get_embed_model
from src.rag.pre_processing.document_pre_proccessing import load_documents, split_document
from pathlib import Path
import traceback
from datetime import datetime
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers import MultiHopSpecificQuerySynthesizer
import asyncio

async def generate_test_dataset_and_store() -> DataFrame:
    llm = get_gemini_model()
    embedding_model = get_embed_model()
    documents = load_documents()

    query_distribution = [
      (SingleHopSpecificQuerySynthesizer(llm=llm), 0.7),
      (MultiHopSpecificQuerySynthesizer(llm=llm), 0.3),
    ]
    
    chunks = split_document(documents)

    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embedding_model)
    try:
        testset = generator.generate_with_chunks(chunks=chunks, testset_size=30, query_distribution=query_distribution)
    except Exception as e:
        print(f"Testset generation failed: {type(e).__name__}: {e}")
        raise
    return testset.to_pandas()

if __name__ == "__main__":
    # python -m src.rag.dataset.dataset_generate
    df = asyncio.run(generate_test_dataset_and_store())

    testset_name = f"testset_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    output_path = Path("src/test_set/") / testset_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Test dataset generated and stored as {testset_name}")

