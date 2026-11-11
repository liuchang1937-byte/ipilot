import copy
import json

import pytest

from techrag.answer_evaluation import _validate_and_score, evaluate_answer


@pytest.fixture
def judgment():
    payload = {"question": "模型和宿主分别做什么？", "answerable": True,
               "answer": "模型生成参数。[1] 模型直接执行工具。",
               "reference_points": ["模型生成参数", "宿主执行工具"],
               "contexts": [{"id": 1, "content": "模型生成参数，宿主执行工具。"}]}
    result = {"sections": [{"insufficient": False}]}
    value = {
        "claims": [
            {"text": "模型生成参数。[1]", "supported": True, "correct": True,
             "evidence": [{"source_id": 1, "quote": "模型生成参数"}],
             "citations": [{"id": 1, "supported": True}], "reason": "有依据"},
            {"text": "模型直接执行工具。", "supported": False, "correct": False,
             "evidence": [], "citations": [], "reason": "职责错误"},
        ],
        "reference_points": [{"index": 1, "covered": True, "reason": "覆盖"},
                             {"index": 2, "covered": False, "reason": "错误"}],
        "answer_relevancy": {"score": .6, "reason": "部分回应"},
    }
    return payload, result, value


def test_quality_scores_use_validated_claim_counts_and_reference_coverage(judgment):
    payload, result, value = judgment
    review = _validate_and_score(value, payload, result)
    assert review["metrics"] == {
        "faithfulness": .5, "answer_correctness": .5, "answer_relevancy": .6,
        "citation_precision": 1, "citation_coverage": .5,
    }
    assert review["reference_points"][1]["point"] == "宿主执行工具"


@pytest.mark.parametrize("mutation", [
    lambda v: v["claims"][0].update(text="伪造回答"),
    lambda v: v["claims"][0]["evidence"][0].update(quote="伪造证据"),
    lambda v: v["claims"][0]["evidence"][0].update(source_id=999),
    lambda v: v["claims"][0].update(evidence=[]),
    lambda v: v["claims"][0]["citations"][0].update(id=999),
    lambda v: v["claims"][0].update(citations=[]),
    lambda v: v["claims"][0].update(supported=False),
    lambda v: v["claims"].append(copy.deepcopy(v["claims"][0])),
    lambda v: v["reference_points"].pop(),
    lambda v: v["reference_points"][1].update(index=1),
    lambda v: v["answer_relevancy"].update(score=float("nan")),
    lambda v: v["answer_relevancy"].update(score=True),
    lambda v: v["answer_relevancy"].update(score=1.1),
])
def test_fabricated_excerpts_and_malformed_judgments_are_rejected(judgment, mutation):
    payload, result, value = judgment
    mutation(value)
    with pytest.raises((ValueError, KeyError, TypeError)):
        _validate_and_score(value, payload, result)


def test_no_reference_means_correctness_unavailable_not_zero(judgment):
    payload, result, value = judgment
    payload["reference_points"] = []
    value["reference_points"] = []
    for claim in value["claims"]:
        claim["correct"] = None
    assert _validate_and_score(value, payload, result)["metrics"]["answer_correctness"] is None


def test_refusal_does_not_receive_perfect_faithfulness_or_citation_scores():
    payload = {"answer": "无资料，无法回答。", "answerable": False,
               "contexts": [], "reference_points": []}
    result = {"sections": [{"insufficient": True}]}
    value = {"claims": [], "reference_points": [],
             "answer_relevancy": {"score": 1, "reason": "正确拒答"}}
    metrics = _validate_and_score(value, payload, result)["metrics"]
    assert metrics["faithfulness"] is None and metrics["citation_precision"] is None
    assert metrics["citation_coverage"] is None and metrics["answer_correctness"] is None
    assert metrics["answer_relevancy"] == 1


def test_judge_failure_is_sanitized_and_does_not_raise():
    class BrokenJudge:
        def complete(self, messages):
            raise RuntimeError("private-provider-details")
    quality = evaluate_answer({"question": "x", "answerable": True},
                              {"answer": "x", "sources": []}, BrokenJudge())
    assert quality["status"] == "unavailable"
    assert all(v is None for v in quality["metrics"].values())
    assert "private-provider-details" not in json.dumps(quality)
