import asyncio
import json
import os
import pathlib

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas import experiment
from ragas.backends.local_csv import LocalCSVBackend
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from src.rag.document_query import get_query_engine, get_contexts

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TESTSET_FILE_PATH = PROJECT_ROOT / "test_set" / "testset.jsonl"

class ExperimentResult(BaseModel):
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str | None = None
    reference_contexts: list[str] | None = None
    faithfulness: float
    answer_relevancy: float

@experiment(ExperimentResult)
async def run_evaluation(row, llm, embedding_model):

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embedding_model)

    faith_result = await faithfulness.ascore(
        user_input=row.user_input,
        response=row.response,
        retrieved_contexts=row.retrieved_contexts
    )

    relevancy_result = await answer_relevancy.ascore(
        user_input=row.user_input,
        response=row.response
    )

    return ExperimentResult(
        user_input=row.user_input,
        response=row.response,
        retrieved_contexts=row.retrieved_contexts,
        reference=getattr(row, "reference", None),
        reference_contexts=getattr(row, "reference_contexts", None),
        faithfulness=faith_result.value,
        answer_relevancy=relevancy_result.value
    )    


def build_eval_dataset_from_jsonl_and_query_engine() -> EvaluationDataset:
    query_engine = get_query_engine()
    rows: list[dict] = []

    with TESTSET_FILE_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            user_input = record["user_input"]

            response_obj = query_engine.query(user_input)
            response_text = response_obj.response
            retrieved_contexts = get_contexts(response_obj)

            rows.append(
                {
                    "user_input": user_input,
                    "response": response_text,
                    "retrieved_contexts": retrieved_contexts,
                    "reference": record.get("reference"),
                    "reference_contexts": record.get("reference_contexts"),
                }
            )

    return EvaluationDataset.from_list(rows)

async def main():
    load_dotenv()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory("gpt-4o-mini", client=client, max_tokens=8192, temperature=0.0)
    embedding_model = HuggingFaceEmbeddings(model="BAAI/bge-m3")

    backend = LocalCSVBackend(root_dir=str(PROJECT_ROOT / "ragas_store"))
    dataset = build_eval_dataset_from_jsonl_and_query_engine()
    exp_results = await run_evaluation.arun(
        dataset=dataset,
        backend=backend,
        llm=llm,
        embedding_model=embedding_model,
    )
    

if __name__ == "__main__":
    # python -m src.ragas.evaluate
    asyncio.run(main())