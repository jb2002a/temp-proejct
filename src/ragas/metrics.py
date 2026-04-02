import json
import numpy as np

from ragas.metrics.result import MetricResult
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
)
from ragas.metrics.collections.answer_relevancy.util import (
    AnswerRelevanceInput,
    AnswerRelevanceOutput,
    AnswerRelevancePrompt,
)
from ragas.metrics.collections.faithfulness.util import (
    StatementGeneratorPrompt,
    NLIStatementPrompt,
)
from ragas.metrics.collections.context_precision.util import ContextPrecisionPrompt
from ragas.metrics.collections.context_recall.util import ContextRecallPrompt


class KoStatementGeneratorPrompt(StatementGeneratorPrompt):
    instruction = (
        StatementGeneratorPrompt.instruction
        + "\n생성하는 statements는 반드시 한국어로 작성하세요."
    )


class KoNLIStatementPrompt(NLIStatementPrompt):
    instruction = (
        NLIStatementPrompt.instruction
        + "\n각 statement의 reason은 반드시 한국어로 작성하세요."
    )


class KoContextPrecisionPrompt(ContextPrecisionPrompt):
    instruction = (
        ContextPrecisionPrompt.instruction
        + "\nreason은 반드시 한국어로 작성하세요."
    )


class KoContextRecallPrompt(ContextRecallPrompt):
    instruction = (
        ContextRecallPrompt.instruction
        + "\n각 classification의 reason은 반드시 한국어로 작성하세요."
    )


class KoAnswerRelevancePrompt(AnswerRelevancePrompt):
    instruction = (
        AnswerRelevancePrompt.instruction
        + "\n생성하는 question은 반드시 한국어로 작성하세요."
    )

# 1) Faithfulness: statement별 verdict/reason을 reason에 담기
class FaithfulnessWithReason(Faithfulness):
    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.statement_generator_prompt = KoStatementGeneratorPrompt()
        self.nli_statement_prompt = KoNLIStatementPrompt()

    async def ascore(self, user_input: str, response: str, retrieved_contexts: list[str]) -> MetricResult:
        statements = await self._create_statements(user_input, response)
        if not statements:
            return MetricResult(value=float("nan"), reason="No statements generated")

        context_str = "\n".join(retrieved_contexts)
        verdicts = await self._create_verdicts(statements, context_str)
        score = self._compute_score(verdicts)

        detail = [
            {"statement": s.statement, "verdict": s.verdict, "reason": s.reason}
            for s in verdicts.statements
        ]
        return MetricResult(value=float(score), reason=json.dumps(detail, ensure_ascii=False))


# 2) ContextPrecision: context별 verdict/reason을 reason에 담기
class ContextPrecisionWithReason(ContextPrecision):
    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.prompt = KoContextPrecisionPrompt()

    async def ascore(self, user_input: str, reference: str, retrieved_contexts: list[str]) -> MetricResult:
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not reference:
            raise ValueError("reference cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        verdicts: list[int] = []
        reasons: list[dict] = []

        for context in retrieved_contexts:
            input_data = self.prompt.input_model(question=user_input, context=context, answer=reference)
            prompt_string = self.prompt.to_string(input_data)
            result = await self.llm.agenerate(prompt_string, self.prompt.output_model)

            verdicts.append(result.verdict)
            reasons.append(
                {
                    "context": context,
                    "verdict": result.verdict,
                    "reason": result.reason,
                }
            )

        score = self._calculate_average_precision(verdicts)
        return MetricResult(value=float(score), reason=json.dumps(reasons, ensure_ascii=False))


# 3) ContextRecall: classification별 attributed/reason을 reason에 담기
class ContextRecallWithReason(ContextRecall):
    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.prompt = KoContextRecallPrompt()

    async def ascore(self, user_input: str, retrieved_contexts: list[str], reference: str) -> MetricResult:
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not reference:
            raise ValueError("reference cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        context = "\n".join(retrieved_contexts)
        input_data = self.prompt.input_model(question=user_input, context=context, answer=reference)
        prompt_string = self.prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_string, self.prompt.output_model)

        if not result.classifications:
            return MetricResult(value=float("nan"), reason="No classifications generated")

        attributions = [c.attributed for c in result.classifications]
        score = sum(attributions) / len(attributions)

        detail = [
            {"statement": c.statement, "attributed": c.attributed, "reason": c.reason}
            for c in result.classifications
        ]
        return MetricResult(value=float(score), reason=json.dumps(detail, ensure_ascii=False))


# 4) AnswerRelevancy: 기본 출력에 reason 필드가 없으므로, 근거를 직접 구성
class AnswerRelevancyWithReason(AnswerRelevancy):
    def __init__(self, llm, embeddings, **kwargs):
        super().__init__(llm=llm, embeddings=embeddings, **kwargs)
        self.prompt = KoAnswerRelevancePrompt()

    async def ascore(self, user_input: str, response: str) -> MetricResult:
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not response:
            raise ValueError("response cannot be empty")

        generated_questions = []
        noncommittal_flags = []

        for _ in range(self.strictness):
            input_data = AnswerRelevanceInput(response=response)
            prompt_string = self.prompt.to_string(input_data)
            result = await self.llm.agenerate(prompt_string, AnswerRelevanceOutput)

            if result.question:
                generated_questions.append(result.question)
                noncommittal_flags.append(result.noncommittal)

        if not generated_questions:
            return MetricResult(value=0.0, reason="No generated questions")

        all_noncommittal = bool(np.all(noncommittal_flags))

        question_vec = np.asarray(await self.embeddings.aembed_text(user_input)).reshape(1, -1)
        gen_question_vec = np.asarray(await self.embeddings.aembed_texts(generated_questions)).reshape(
            len(generated_questions), -1
        )

        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(question_vec, axis=1)
        cosine_sim = np.dot(gen_question_vec, question_vec.T).reshape(-1) / norm

        score = float(cosine_sim.mean() * int(not all_noncommittal))

        reason_payload = {
            "generated_questions": generated_questions,
            "noncommittal_flags": [int(x) for x in noncommittal_flags],
            "all_noncommittal": all_noncommittal,
            "cosine_similarity": [float(x) for x in cosine_sim.tolist()],
        }
        return MetricResult(value=score, reason=json.dumps(reason_payload, ensure_ascii=False))