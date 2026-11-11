import json
import math

import pytest

from techrag.benchmark import BenchmarkBlockedError, _metrics, load_dataset, run_benchmark

pytest_plugins = ["test_knowledge"]


def test_compound_recall_counts_every_required_source_and_refusal_separately():
    case = {"expected_sources": ["a", "b"]}
    result = {
        "sections": [{"trace": {"selected_sources": ["x", "a", "a"]}, "insufficient": False}],
        "sources": [{"source": "a"}],
    }
    metrics = _metrics(case, result)
    assert metrics["recall_at_k"] == 0.5 and metrics["reciprocal_rank"] == 0.5
    assert metrics["hit"] and not metrics["all_sources_found"]
    assert metrics["precision_at_k"] == pytest.approx(1 / 3)
    assert metrics["ndcg_at_k"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert metrics["correct_refusal"] is None
    absent = {"expected_sources": []}
    assert not _metrics(absent, result)["correct_refusal"]
    result["sections"][0]["insufficient"] = True
    result["sources"] = []
    assert _metrics(absent, result)["correct_refusal"]
    assert _metrics(absent, result)["recall_at_k"] is None


def test_seed_dataset_is_valid_and_covers_different_question_types():
    data = load_dataset()
    assert len(data["cases"]) == 20
    assert data["label_status"] == "seed_requires_human_review"
    assert {c["category"] for c in data["cases"]} >= {
        "compound",
        "comparison",
        "unanswerable",
        "cross_domain",
        "paraphrase",
    }


def test_pipeline_comparison_counts_calls_preserves_providers_and_saves_results(
    knowledge,
    tmp_path,
):
    service, model, reranker = knowledge
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        json.dumps(
            {
                "name": "test",
                "cases": [
                    {
                        "id": "multi",
                        "question": "Agent 架构？Function Calling 原理？",
                        "domains": [],
                        "expected_sources": ["agent.md", "tools.md"],
                        "answerable": True,
                    },
                    {
                        "id": "absent",
                        "question": "没有答案",
                        "domains": ["empty"],
                        "expected_sources": [],
                        "answerable": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report = run_benchmark(
        service.rag,
        dataset_path=dataset,
        service=service,
        variants=["vector", "reranked"],
        output_dir=tmp_path / "reports",
    )
    assert service.rag.chat_model is model and service.reranker is reranker
    assert len(report["records"]) == 4
    assert len(report["shared_plans"]) == 2
    assert sum("知识库问题拆解器" in p for p in model.prompts) == 2
    vector, ranked = report["summaries"]
    assert vector["calls"]["rerank_calls"] == 0
    assert ranked["calls"]["rerank_calls"] > 0 and ranked["reranked_cases"] == 1
    assert ranked["recall_at_k"] == 1 and ranked["correct_refusal_rate"] == 1
    saved = json.loads((tmp_path / "reports" / f"{report['id']}.json").read_text("utf-8"))
    assert saved == report
    assert (tmp_path / "reports" / f"{report['id']}.md").is_file()
    assert all(r["result"]["sources"] for r in report["records"] if r["answerable"])
    for row in report["records"]:
        t = row["timings_seconds"]
        assert t["total"] == pytest.approx(t["decomposition"] + t["pipeline"], abs=2e-6)
        assert t["retrieval"] >= t["embedding"] + t["vector_search"]
        assert t["pipeline"] >= t["retrieval"] + t["rerank"] + t["generation"]
    assert ranked["latency_seconds"]["total"]["p95"] >= ranked["mean_seconds"] - 0.001


def test_graded_ndcg_keeps_subquestion_rankings_separate_and_truncates_at_k():
    case = {"expected_sources": ["a", "b"], "source_relevance": {"a": 3, "b": 1}}
    result = {
        "options": {"top_k": 2},
        "sections": [
            {"trace": {"selected_sources": ["b", "a", "ignored"]}, "insufficient": False},
            {"trace": {"selected_sources": ["a"]}, "insufficient": False},
        ],
        "sources": [],
    }
    metrics = _metrics(case, result)
    ideal = 7 + 1 / math.log2(3)
    assert metrics["precision_at_k"] == 0.75
    assert metrics["ndcg_at_k"] == pytest.approx(((1 + 7 / math.log2(3)) / ideal + 7 / ideal) / 2)
    assert metrics["recall_at_k"] == 1 and metrics["reciprocal_rank"] == 1
    assert "ignored" not in metrics["retrieved_sources"]


def test_benchmark_answer_judge_has_separate_cost_and_can_be_disabled(knowledge, tmp_path):
    service, _, _ = knowledge
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "x",
                        "question": "Agent 架构？",
                        "expected_sources": ["agent.md"],
                        "answerable": True,
                        "reference_points": ["Agent包含规划"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class Judge:
        calls = 0

        def complete(self, messages):
            self.calls += 1
            data = json.loads(messages[-1]["content"])
            return json.dumps(
                {
                    "claims": [
                        {
                            "text": data["answer"],
                            "supported": True,
                            "correct": True,
                            "evidence": [
                                {"source_id": 1, "quote": "Agent 包括规划"},
                                {"source_id": 2, "quote": "Function Calling"},
                            ],
                            "citations": [
                                {"id": 1, "supported": True},
                                {"id": 2, "supported": True},
                            ],
                            "reason": "支持",
                        }
                    ],
                    "reference_points": [{"index": 1, "covered": True, "reason": "覆盖"}],
                    "answer_relevancy": {"score": 0.8, "reason": "相关"},
                }
            )

    judge = Judge()
    report = run_benchmark(
        service.rag,
        dataset_path=dataset,
        service=service,
        judge=judge,
        variants=["reranked"],
        output_dir=tmp_path / "judged",
    )
    summary = report["summaries"][0]
    assert judge.calls == summary["judge_calls"] == 1
    assert summary["answer_quality"]["answer_correctness"] == 1
    assert summary["metric_samples"]["faithfulness"] == 1
    assert summary["judged_cases"] == 1 and summary["judge_failures"] == 0
    assert report["answer_judge"]["uses_answer_model"] is False
    report = run_benchmark(
        service.rag,
        dataset_path=dataset,
        service=service,
        judge=judge,
        answer_judge=False,
        variants=["vector"],
        output_dir=tmp_path / "disabled",
    )
    summary = report["summaries"][0]
    assert judge.calls == 1 and summary["judge_calls"] == 0
    assert summary["metric_samples"]["faithfulness"] == 0
    assert summary["answer_quality"]["faithfulness"] is None
    assert summary["latency_seconds"]["judge"]["mean"] is None


@pytest.mark.parametrize(
    "extra",
    [
        {"reference_points": [""]},
        {"reference_points": "not-a-list"},
        {"reference_answer": 1},
        {"source_relevance": {"agent.md": 0}},
        {"source_relevance": {"missing": 3}},
        {"source_relevance": {"agent.md": True}},
    ],
)
def test_invalid_reference_labels_fail_before_model_calls(tmp_path, extra):
    case = {"id": "x", "question": "x", "expected_sources": ["agent.md"], "answerable": True}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"cases": [{**case, **extra}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_missing_sources_and_duplicate_variants_fail_before_external_calls(knowledge, tmp_path):
    service, model, _ = knowledge
    dataset = tmp_path / "missing.json"
    dataset.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "x",
                        "question": "x",
                        "expected_sources": ["missing"],
                        "answerable": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    for variants in [["vector"], ["vector", "vector"], []]:
        with pytest.raises(ValueError):
            run_benchmark(service.rag, dataset_path=dataset, variants=variants)
    assert not model.prompts


def test_failed_cases_are_visible_and_do_not_corrupt_providers(knowledge, tmp_path):
    service, model, reranker = knowledge
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "x",
                        "question": "x",
                        "expected_sources": ["agent.md"],
                        "answerable": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    def fail(*args, **kwargs):
        raise RuntimeError("private-provider-details")

    service.answer = fail
    report = run_benchmark(
        service.rag,
        dataset_path=dataset,
        service=service,
        variants=["hybrid"],
        output_dir=tmp_path / "results",
    )
    assert report["summaries"][0]["failures"] == 1
    assert report["summaries"][0]["recall_at_k"] is None
    assert "private-provider-details" not in json.dumps(report)
    assert service.rag.chat_model is model and service.reranker is reranker


def test_checkpoint_resumes_successful_rows_without_repeating_plan_or_generation(
    knowledge, tmp_path
):
    service, model, _ = knowledge
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "x",
                        "question": "Agent 架构？",
                        "expected_sources": ["agent.md"],
                        "answerable": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def interrupt(*_args):
        raise RuntimeError("interrupted")

    options = dict(
        dataset_path=path,
        service=service,
        variants=["hybrid", "hybrid_reranked"],
        answer_judge=False,
        output_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoint",
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        run_benchmark(service.rag, progress=interrupt, **options)
    plans_before = sum("知识库问题拆解器" in p for p in model.prompts)
    report = run_benchmark(service.rag, **options)
    assert report["resume"]["reused_records"] == 1
    assert len(report["records"]) == 2
    assert sum("知识库问题拆解器" in p for p in model.prompts) == plans_before == 1
    # Changed input must not be merged into existing results.
    data = json.loads(path.read_text("utf-8"))
    data["cases"][0]["question"] = "changed"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="断点"):
        run_benchmark(service.rag, **options)


def test_retrieval_only_runs_real_context_selection_without_answers_or_refusal_claims(
    knowledge, tmp_path
):
    service, model, _ = knowledge
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "x",
                        "question": "Agent 架构？",
                        "expected_sources": ["agent.md"],
                        "answerable": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = run_benchmark(
        service.rag,
        dataset_path=path,
        service=service,
        variants=["hybrid_reranked"],
        retrieval_only=True,
        output_dir=tmp_path,
    )
    row = report["records"][0]
    assert report["evaluation_mode"] == "retrieval_only"
    assert row["recall_at_k"] == 1 and row["result"]["retrieved_contexts"]
    assert row["correct_refusal"] is None and row["false_refusal"] is None
    assert not report["answer_judge"]["enabled"]
    assert not any("严谨的知识库问答助手" in p for p in model.prompts)
    assert row["timings_seconds"]["generation"] == 0


def test_balance_failure_stops_before_following_cases_and_restores_providers(knowledge, tmp_path):
    service, model, _ = knowledge
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "x",
                        "question": "Agent 架构？",
                        "expected_sources": ["agent.md"],
                        "answerable": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class BalanceFailure(RuntimeError):
        status_code = 402

    calls = []

    def fail(_messages):
        calls.append(1)
        raise BalanceFailure("private-provider-detail")

    model.complete = fail
    with pytest.raises(BenchmarkBlockedError, match="余额不足"):
        run_benchmark(
            service.rag,
            dataset_path=path,
            service=service,
            variants=["hybrid", "expanded"],
            checkpoint_dir=tmp_path / "checkpoint",
        )
    assert calls == [1] and service.rag.chat_model is model
    record = (tmp_path / "checkpoint" / "blocked.json").read_text("utf-8")
    assert "private-provider-detail" not in record


def test_resume_retries_failed_decomposition_instead_of_freezing_its_fallback(knowledge, tmp_path):
    service, model, _ = knowledge
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "x",
                        "question": "Agent 架构？",
                        "expected_sources": ["agent.md"],
                        "answerable": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    original = model.complete

    def bad_plan(messages):
        if "知识库问题拆解器" in messages[-1]["content"]:
            return "not json"
        return original(messages)

    model.complete = bad_plan
    options = dict(
        dataset_path=path,
        service=service,
        variants=["hybrid"],
        retrieval_only=True,
        checkpoint_dir=tmp_path / "checkpoint",
        output_dir=tmp_path,
    )
    first = run_benchmark(service.rag, **options)
    assert first["records"][0]["result"]["warnings"]
    model.complete = original
    second = run_benchmark(service.rag, **options)
    assert not second["records"][0]["result"]["warnings"]
    assert second["resume"]["reused_records"] == 0
    assert len(list((tmp_path / "checkpoint" / "attempts").glob("*.json"))) == 2
