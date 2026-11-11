import json
from pathlib import Path

import pytest

from techrag.benchmark import VARIANTS, load_dataset
from techrag.experiments import (
    EVIDENCE_DATASET,
    paired_comparisons,
    require_reviewed,
    run_experiment,
    select_cases,
    validate_holdout,
)
from techrag.interview_questions import load_question_directory

pytest_plugins = ["test_knowledge"]


def test_component_ablations_change_one_option_from_hybrid():
    baseline = VARIANTS["hybrid"][1]
    for name, flag in (("mqe", "mqe"), ("hyde", "hyde"), ("hybrid_reranked", "rerank")):
        options = VARIANTS[name][1]
        assert getattr(options, flag) is True
        for other in ("mqe", "hyde", "rerank", "mode", "top_k", "decompose"):
            if other != flag:
                assert getattr(options, other) == getattr(baseline, other)


def test_v2_labels_are_reviewed_grouped_and_resolve_to_real_corpus():
    data = load_dataset(EVIDENCE_DATASET)
    assert len(data["cases"]) == 120
    assert data["label_status"] == "human_reviewed"
    require_reviewed(data, data["cases"])
    assert sum(c["split"] == "dev" for c in data["cases"]) == 80
    assert sum(c["split"] == "test" for c in data["cases"]) == 40
    validate_holdout(data["cases"])
    questions = load_question_directory(Path("data/interview_questions"))
    available = {f"interview-question://{q.id}" for q in questions}
    assert {s for c in data["cases"] for s in c["expected_sources"]} <= available
    selected = select_cases(data, "dev", 6, 42)
    assert selected == select_cases(data, "dev", 6, 42)
    assert {c["category"] for c in selected} == {
        "compound",
        "cross_domain",
        "paraphrase",
        "terminology",
        "unanswerable",
    }
    with pytest.raises(ValueError, match="human_reviewed"):
        require_reviewed(data | {"label_status": "seed_requires_human_review"}, selected)


@pytest.mark.parametrize("key", ["group", "expected_sources"])
def test_holdout_rejects_shared_groups_or_sources(key):
    cases = [
        {"id": "a", "split": "dev", "group": "a", "expected_sources": ["a"]},
        {"id": "b", "split": "test", "group": "b", "expected_sources": ["b"]},
    ]
    cases[1][key] = cases[0][key]
    with pytest.raises(ValueError, match="分组或标注来源"):
        validate_holdout(cases)


def test_reviewed_flag_needs_per_case_provenance():
    data = {"label_status": "human_reviewed"}
    with pytest.raises(ValueError, match="复核记录"):
        require_reviewed(data, [{"id": "x"}])
    require_reviewed(
        data,
        [
            {
                "id": "x",
                "review": {
                    "approved": True,
                    "reviewer": "reviewer",
                    "reviewed_at": "2026-10-12",
                },
            }
        ],
    )


def test_paired_deltas_average_repeats_and_bootstrap_independent_groups():
    records = []
    for case in range(20):
        for repeat in (1, 2, 3):
            for variant, value in (("hybrid", 0.4), ("mqe", 0.6)):
                records.append(
                    {
                        "case_id": str(case),
                        "group": str(case // 2),
                        "repeat": repeat,
                        "variant": variant,
                        "error": None,
                        "recall_at_k": value,
                    }
                )
    row = paired_comparisons(records)[0]
    assert row["paired_cases"] == 20 and row["paired_observations"] == 60
    assert row["independent_groups"] == 10
    assert row["delta"] == 0.2 and row["bootstrap_95_ci"] == [0.2, 0.2]
    # A failed candidate must not be paired against the successful baseline.
    records[1]["error"] = "failed"
    assert paired_comparisons(records)[0]["paired_observations"] == 59
    assert paired_comparisons(records[:4])[0]["bootstrap_95_ci"] is None


def test_repeated_experiment_saves_runs_selection_and_failure_denominators(knowledge, tmp_path):
    service, model, reranker = knowledge
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "name": "test",
                "label_status": "seed_requires_human_review",
                "cases": [
                    {
                        "id": "a",
                        "question": "Agent架构？",
                        "category": "paraphrase",
                        "split": "dev",
                        "group": "a",
                        "expected_sources": ["agent.md"],
                        "answerable": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = run_experiment(
        service.rag,
        dataset_path=path,
        service=service,
        variants=["hybrid", "mqe"],
        repeats=2,
        answer_judge=False,
        output_dir=tmp_path / "reports",
    )
    assert len(report["runs"]) == 2 and len(report["records"]) == 4
    assert report["summaries"][0]["completed"] == 2
    assert report["claim_ready"] is False
    assert service.rag.chat_model is model and service.reranker is reranker
    assert (tmp_path / "reports" / f"{report['id']}.json").exists()
    assert (tmp_path / "reports" / report["id"] / "selected-dataset.json").exists()

    def fail(*args, **kwargs):
        raise RuntimeError("provider failed")

    service.answer = fail
    report = run_experiment(
        service.rag,
        dataset_path=path,
        service=service,
        variants=["hybrid"],
        repeats=1,
        answer_judge=False,
        output_dir=tmp_path / "failed",
    )
    summary = report["summaries"][0]
    assert summary["recall_at_k"] is None
    assert summary["end_to_end_recall_at_k"] == 0
    assert summary["completion_rate"] == 0 and report["claim_ready"] is False
