
from ragas.testset import TestsetGenerator
from pandas import DataFrame
from src.rag.common.clients import get_model, get_embed_model
from src.rag.pre_processing.document_pre_proccessing import load_documents, split_document
from pathlib import Path

def generate_test_dataset_and_store() -> DataFrame:
    llm = get_model()

    embedding_model = get_embed_model()
    documents = load_documents()
    chunks = split_document(documents)

    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embedding_model)
    testset = generator.generate_with_chunks(chunks=chunks, testset_size=20)
    return testset.to_pandas()

if __name__ == "__main__":
    # python -m src.rag.dataset.dataset_generate
    df = generate_test_dataset_and_store()
    output_path = Path("src/test_set/testset.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("Test dataset generated and stored")

