import asyncio
import ast
import csv
import os
import pathlib
from collections.abc import Callable

from langsmith import traceable
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas import experiment
from ragas.backends.local_csv import LocalCSVBackend
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from src.rag.common.config import BASIC_MODEL, EMBEDING_MODEL, TESTSET_FILE_PATH
from src.rag.post_processing.response import generate_response_and_context

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent



class ExperimentResult(BaseModel):
    # 질문 (user query)
    user_input: str
    # 모델의 답변
    response: str
    # 모델이 참조한 문서 컨텍스트
    reference: str | None = None
    # 테스트셋의 정답 컨텍스트
    retrieved_contexts: list[str]
    # 테스트셋의 정답
    reference_contexts: list[str] | None = None
    # 근거 문서에 기반해 답변했는가?
    faithfulness: float
    # 질문의 의도에 맞는 답변인가?
    answer_relevancy: float
    # 관련 있는 문서가 상위에 노출되었는가?
    context_precision: float
    # 필요한 정보를 빠짐없이 가져왔는가?
    context_recall: float


@experiment(ExperimentResult)
async def run_evaluation(row, llm, embedding_model):
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embedding_model)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)


    faith_result = await faithfulness.ascore(
        user_input=row.user_input,
        response=row.response,
        retrieved_contexts=row.retrieved_contexts
    )

    relevancy_result = await answer_relevancy.ascore(
        user_input=row.user_input,
        response=row.response
    )

    context_precision_result = await context_precision.ascore(
        user_input=row.user_input,
        reference=row.reference ,
        retrieved_contexts=row.retrieved_contexts
    )

    context_recall_result = await context_recall.ascore(
        user_input=row.user_input,
        reference=row.reference,
        retrieved_contexts=row.retrieved_contexts
    )

    return ExperimentResult(
        user_input=row.user_input,
        response=row.response,
        retrieved_contexts=row.retrieved_contexts,
        reference=getattr(row, "reference", None),
        reference_contexts=getattr(row, "reference_contexts", None),
        faithfulness=faith_result.value,
        answer_relevancy=relevancy_result.value,
        context_precision=context_precision_result.value,
        context_recall=context_recall_result.value,

    )

@traceable
def create_query_engine() -> Callable[[str], tuple[str, list[str]]]:
    def query(user_input: str) -> tuple[str, list[str]]:
        result = generate_response_and_context(query=user_input)
        response_text = result["response"]
        retrieved_contexts = [doc.page_content for doc in result["retrieved_docs"]]
        return response_text, retrieved_contexts

    return query

@traceable
def _parse_reference_contexts(value: str | None) -> list[str] | None:
    if value is None or value == "":
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]

@traceable
def build_eval_dataset_from_csv_and_query_engine() -> EvaluationDataset:
    query_engine = create_query_engine()
    rows: list[dict] = []

    with TESTSET_FILE_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for record in reader:
            user_input = (record.get("user_input") or "").strip()
            if not user_input:
                continue

            response_text, retrieved_contexts = query_engine(user_input)
            rows.append(
                {
                    "user_input": user_input,
                    "response": response_text,
                    "retrieved_contexts": retrieved_contexts,
                    "reference": (record.get("reference") or None),
                    "reference_contexts": _parse_reference_contexts(record.get("reference_contexts")),
                }
            )

    return EvaluationDataset.from_list(rows)

@traceable
async def pipeline_evaluate():
    load_dotenv(override=True)
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory(BASIC_MODEL, client=client, max_tokens=8192, temperature=0.0)
    embedding_model = HuggingFaceEmbeddings(model=EMBEDING_MODEL)

    backend = LocalCSVBackend(root_dir=str(PROJECT_ROOT / "ragas_store"))
    dataset = build_eval_dataset_from_csv_and_query_engine()
    await run_evaluation.arun(
        dataset=dataset,
        backend=backend,
        llm=llm,
        embedding_model=embedding_model,
    )
    
if __name__ == "__main__":
    # python -m src.evaluate.evaluate
    asyncio.run(pipeline_evaluate())