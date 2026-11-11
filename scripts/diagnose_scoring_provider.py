"""Retry one authorised failed scoring case and record only safe error categories."""

from __future__ import annotations

import json
from pathlib import Path

from techrag import TechRAG
from techrag.interview import QUESTIONING, InterviewEngine, InterviewSession
from techrag.providers import DashScopeDeepSeekProvider
from techrag.scoring_benchmark import load_scoring_dataset

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticProvider:
    def __init__(self, provider):
        self.provider = provider
        self.diagnostics = []

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def complete(self, messages):
        try:
            return self.provider.complete(messages)
        except Exception as error:
            message = str(error).lower()
            if any(
                term in message
                for term in (
                    "insufficient balance",
                    "insufficient_balance",
                    "insufficient funds",
                    "余额不足",
                )
            ):
                category = "insufficient_balance"
            elif any(term in message for term in ("rate limit", "rate_limit", "too many requests")):
                category = "rate_limit"
            elif any(
                term in message for term in ("unauthorized", "invalid api key", "authentication")
            ):
                category = "authentication"
            elif any(term in message for term in ("timeout", "timed out")):
                category = "timeout"
            else:
                category = "other_provider_failure"
            status = getattr(error, "status_code", None)
            self.diagnostics.append(
                {
                    "error_type": type(error).__name__,
                    "status_code": status if isinstance(status, int) else None,
                    "category": category,
                }
            )
            raise


def main():
    case = next(c for c in load_scoring_dataset()["cases"] if c["id"] == "S-011")
    provider = DiagnosticProvider(DashScopeDeepSeekProvider())
    with TechRAG(_chat_model=provider, _embedder=provider) as rag:
        session = InterviewSession(
            id="diagnostic-S-011",
            mode="technical",
            phase=QUESTIONING,
            target_rounds=5,
            difficulty_policy="locked",
            difficulty=2,
            domains=(),
            current_question=case["question"],
            current_reference_answer=case["reference_answer"],
            current_scoring_points=case["scoring_points"],
            question_answers=list(case["answers"]),
        )
        evaluation = InterviewEngine(rag)._evaluate(session, case["answers"][-1])
    result = {
        "case_id": case["id"],
        "scoring_method": evaluation["scoring_method"],
        "diagnostics": provider.diagnostics,
    }
    output = ROOT / "reports/evidence/provider-diagnostic.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
