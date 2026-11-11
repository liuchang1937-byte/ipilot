import importlib.util
import json
from pathlib import Path

import pytest

pytest_plugins = ["test_knowledge"]


def study_module():
    spec = importlib.util.spec_from_file_location(
        "study_runner", Path("scripts/run_full_evaluation.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_worker_snapshot_resume_and_provider_restoration(knowledge, tmp_path):
    service, model, _ = knowledge
    module = study_module()
    module._RAG = service.rag
    module._SERVICE = service
    case = {
        "id": "x",
        "question": "Agent 架构？",
        "expected_sources": ["agent.md"],
        "answerable": True,
        "split": "dev",
        "group": "x",
    }
    task = {
        "case": case,
        "dataset": {"name": "test", "cases": [case]},
        "variants": ["hybrid", "hybrid_reranked"],
        "repeat": 1,
        "directory": str(tmp_path),
        "retrieval_only": True,
        "answer_judge": False,
    }
    result = module._run_case(task)
    before = list(model.prompts)
    module._run_case(task)
    report = json.loads(Path(result["report_path"]).read_text("utf-8"))
    assert report["resume"]["reused_records"] == 2
    assert model.prompts == before and service.rag.chat_model is model
    task["case"] = case | {"question": "changed"}
    with pytest.raises(ValueError, match="案例数据改变"):
        module._run_case(task)


def test_selection_is_frozen_and_failed_or_degraded_runs_cannot_select(tmp_path):
    module = study_module()
    protocol = {"selection_rule": "test"}

    def row(variant, recall, latency):
        return {
            "variant": variant,
            "failures": 0,
            "degraded_cases": 0,
            "completion_rate": 1,
            "recall_at_k": recall,
            "mrr": 1,
            "latency_seconds": {"retrieval": {"mean": latency}},
        }

    dev = {
        "summaries": [row("hybrid", 1, 1), row("hybrid_reranked", 0.9, 0.5), row("expanded", 1, 10)]
    }
    selected = module._freeze_selection(tmp_path, dev, protocol)
    assert selected["selected_variant"] == "hybrid"
    assert selected["holdout_variants"] == ["hybrid", "hybrid_reranked"]
    dev["summaries"][1]["recall_at_k"] = 1
    with pytest.raises(ValueError, match="冻结选型"):
        module._freeze_selection(tmp_path, dev, protocol)
    dev["summaries"][1]["degraded_cases"] = 1
    with pytest.raises(ValueError, match="失败或检索降级"):
        module._freeze_selection(tmp_path, dev, protocol)


def test_runner_lock_prevents_duplicate_launch_and_releases_after_exit(tmp_path):
    module = study_module()
    with module.runner_lock(tmp_path):
        with pytest.raises(ValueError, match="正在运行"):
            with module.runner_lock(tmp_path):
                pytest.fail("duplicate launch acquired lock")
    with module.runner_lock(tmp_path):
        assert (tmp_path / "runner.lock").exists()
