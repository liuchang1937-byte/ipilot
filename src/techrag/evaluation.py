from __future__ import annotations

import json
import re
from collections.abc import Sequence

from .models import Answer, EvaluationCase, EvaluationReport
from .rag import TechRAG
from .text import tokenize


def generate_evals(rag: TechRAG, *, num_evals: int = 10) -> list[EvaluationCase]:
    """Generate and persist small synthetic evaluation cases from indexed chunks."""

    rows = rag.database.chunk_rows()[:num_evals]
    cases: list[EvaluationCase] = []
    for row in rows:
        prompt = (
            "Create one answerable question from the context. Keep the context language. "
            'Return JSON only: {"question":"...","answer":"..."}.\n\n'
            f"Context:\n{row[2]}"
        )
        response = rag.chat_model.complete([{"role": "user", "content": prompt}])
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            continue
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        question, answer = payload.get("question"), payload.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        case = EvaluationCase(question, (str(row[5]),), answer)
        rag.database.insert_evaluation(question, [str(row[5])], answer)
        cases.append(case)
    return cases


insert_evals = generate_evals


def load_evals(rag: TechRAG) -> list[EvaluationCase]:
    return [
        EvaluationCase(question, tuple(sources), answer)
        for question, sources, answer in rag.database.evaluation_rows()
    ]


def answer_evals(rag: TechRAG, cases: Sequence[EvaluationCase] | None = None) -> list[Answer]:
    return [rag.ask(case.question) for case in (cases or load_evals(rag))]


def evaluate(
    rag: TechRAG,
    cases: Sequence[EvaluationCase] | None = None,
    *,
    top_k: int | None = None,
    evaluate_answers: bool = False,
) -> EvaluationReport:
    """Evaluate retrieval Recall@K/MRR and optional reference-answer token coverage."""

    selected = list(cases or load_evals(rag))
    if not selected:
        return EvaluationReport(0, 0.0, 0.0, None)
    hits = 0
    reciprocal_ranks = 0.0
    coverages: list[float] = []
    for case in selected:
        results = rag.search(case.question, top_k=top_k)
        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if result.source in case.expected_sources
            ),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks += 1 / rank
        if evaluate_answers and case.reference_answer:
            generated = rag.ask(case.question, top_k=top_k).text
            expected_tokens = set(tokenize(case.reference_answer))
            generated_tokens = set(tokenize(generated))
            coverages.append(
                len(expected_tokens & generated_tokens) / max(len(expected_tokens), 1)
            )
    return EvaluationReport(
        cases=len(selected),
        recall_at_k=hits / len(selected),
        mean_reciprocal_rank=reciprocal_ranks / len(selected),
        answer_coverage=sum(coverages) / len(coverages) if coverages else None,
    )
