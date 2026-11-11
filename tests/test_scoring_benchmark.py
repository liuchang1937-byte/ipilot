import json

import pytest

from techrag.experiments import require_reviewed
from techrag.scoring_benchmark import (
    load_scoring_dataset,
    run_scoring_benchmark,
    scoring_summary,
)

pytest_plugins = ["test_knowledge"]


def test_reviewed_scoring_set_covers_six_answer_types_and_three_points():
    data = load_scoring_dataset()
    assert len(data["cases"]) == 24
    assert {c["category"] for c in data["cases"]} == {
        "correct",
        "partial",
        "incorrect",
        "omission",
        "equivalent",
        "correction",
    }
    assert all(set(c["expected_statuses"]) == {"P1", "P2", "P3"} for c in data["cases"])
    assert data["label_status"] == "human_reviewed"
    require_reviewed(data, data["cases"])


def test_unverified_points_and_fallbacks_are_not_removed_from_agreement():
    records = [
        {
            "case_id": "a",
            "expected_statuses": {"P1": "mastered", "P2": "incorrect"},
            "elapsed_seconds": 1,
            "evaluation": {
                "scoring_method": "fallback",
                "point_reviews": [
                    {"point_id": "P1", "status": "mastered"},
                    {"point_id": "P2", "status": "unverified"},
                ],
            },
        },
        {
            "case_id": "a",
            "expected_statuses": {"P1": "mastered", "P2": "incorrect"},
            "elapsed_seconds": 1,
            "evaluation": {
                "scoring_method": "model",
                "point_reviews": [
                    {"point_id": "P1", "status": "mastered"},
                    {"point_id": "P2", "status": "incorrect"},
                ],
            },
        },
    ]
    summary = scoring_summary(records)
    assert summary["agreement_all_points"] == 0.75
    assert summary["agreement_verified_points"] == 1
    assert summary["unverified_rate"] == 0.25 and summary["fallback_cases"] == 1
    assert summary["repeat_modal_agreement"] == 0.5
    assert summary["independent_cases"] == 1 and summary["point_observations"] == 4


def test_scoring_runner_uses_production_prompt_and_quote_validator_without_saving(
    knowledge, tmp_path
):
    service, model, _ = knowledge
    calls = []

    def complete(messages):
        calls.append(messages[-1]["content"])
        return json.dumps(
            {
                "overall": 85,
                "point_reviews": [
                    {
                        "point_id": "P1",
                        "status": "mastered",
                        "reason": "职责正确。",
                        "evidence": [{"answer_index": 2, "quote": "宿主执行工具"}],
                    },
                    {
                        "point_id": "P2",
                        "status": "mastered",
                        "reason": "虚构引用。",
                        "evidence": [{"answer_index": 1, "quote": "并不存在的原文"}],
                    },
                ],
            }
        )

    model.complete = complete
    path = tmp_path / "scores.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "x",
                        "category": "correction",
                        "question": "谁执行？",
                        "reference_answer": "宿主执行并回填结果。",
                        "scoring_points": "宿主执行；回填结果",
                        "answers": ["模型执行", "宿主执行工具"],
                        "expected_statuses": {"P1": "mastered", "P2": "not_addressed"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    before = service.rag.stats()
    report = run_scoring_benchmark(
        service.rag, dataset_path=path, repeats=2, output_dir=tmp_path / "reports"
    )
    assert len(calls) == 2 and all("严格但公平" in c and "编号回答" in c for c in calls)
    assert report["summary"]["agreement_all_points"] == 0.5
    assert report["summary"]["unverified_rate"] == 0.5
    assert service.rag.stats() == before
    assert report["records"][0]["evaluation"]["point_reviews"][1]["status"] == "unverified"
    assert (tmp_path / "reports" / f"{report['id']}.json").is_file()
    assert (tmp_path / "reports" / "checkpoints" / f"{report['id']}.json").is_file()
    with pytest.raises(ValueError, match="human_reviewed"):
        run_scoring_benchmark(service.rag, dataset_path=path, reviewed_only=True)
