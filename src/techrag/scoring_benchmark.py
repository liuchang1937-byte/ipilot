"""Compare production point scoring to reviewed or explicitly pending seed labels."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from .experiments import require_reviewed, write_report
from .interview import QUESTIONING, InterviewEngine, InterviewSession
from .scoring import STATUS_LABELS, rubric_points

SCORING_DATASET = Path(__file__).resolve().parents[2] / "data/evaluation/scoring-v1.json"
SEMANTIC_STATUSES = tuple(s for s in STATUS_LABELS if s != "unverified")


def load_scoring_dataset(path: Path = SCORING_DATASET) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("评分评测集必须包含cases。")
    if not 1 <= len(data["cases"]) <= 500:
        raise ValueError("评分评测案例数需要为1到500。")
    ids = set()
    for case in data["cases"]:
        if not isinstance(case, dict):
            raise ValueError("评分案例必须是对象。")
        if not isinstance(case.get("id"), str) or not case["id"] or case["id"] in ids:
            raise ValueError("评分案例ID必须唯一。")
        ids.add(case["id"])
        if any(
            not isinstance(case.get(k), str) or not case[k].strip()
            for k in ("question", "reference_answer", "scoring_points", "category")
        ):
            raise ValueError("评分案例需要题目、参考答案、评分点和分类。")
        answers = case.get("answers")
        if (
            not isinstance(answers, list)
            or not 1 <= len(answers) <= 3
            or any(not isinstance(a, str) or not a.strip() for a in answers)
        ):
            raise ValueError("评分案例需要1到3次有效回答。")
        expected = case.get("expected_statuses")
        rubric = rubric_points(case["scoring_points"])
        if (
            not isinstance(expected, dict)
            or set(expected) != {p["id"] for p in rubric}
            or any(not isinstance(s, str) or s not in SEMANTIC_STATUSES for s in expected.values())
        ):
            raise ValueError("预期状态必须覆盖评分点且为有效语义状态。")
    return data


def scoring_summary(records: list[dict]) -> dict:
    points = [
        (r["expected_statuses"][p["point_id"]], p["status"])
        for r in records
        for p in r["evaluation"]["point_reviews"]
    ]
    confusion = {s: dict.fromkeys(STATUS_LABELS, 0) for s in SEMANTIC_STATUSES}
    for expected, actual in points:
        confusion[expected][actual] += 1
    valid = [(e, a) for e, a in points if a != "unverified"]
    per_class = {}
    for status in SEMANTIC_STATUSES:
        tp = sum(e == a == status for e, a in points)
        predicted = sum(a == status for _, a in points)
        support = sum(e == status for e, _ in points)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else None
        f1 = 2 * tp / (predicted + support) if support else None
        per_class[status] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    groups: dict[str, list] = {}
    for row in records:
        signature = tuple(p["status"] for p in row["evaluation"]["point_reviews"])
        groups.setdefault(row["case_id"], []).append(signature)
    repeated = [values for values in groups.values() if len(values) > 1]
    stable = [max(Counter(values).values()) / len(values) for values in repeated]
    return {
        "case_observations": len(records),
        "independent_cases": len(groups),
        "point_observations": len(points),
        "verified_points": len(valid),
        "agreement_all_points": sum(e == a for e, a in points) / len(points),
        "agreement_verified_points": (
            sum(e == a for e, a in valid) / len(valid) if valid else None
        ),
        "unverified_rate": 1 - len(valid) / len(points),
        "fallback_cases": sum(r["evaluation"]["scoring_method"] != "model" for r in records),
        "macro_f1": mean(p["f1"] for p in per_class.values() if p["f1"] is not None),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "repeat_modal_agreement": mean(stable) if stable else None,
        "repeat_cases": len(repeated),
        "mean_seconds": mean(r["elapsed_seconds"] for r in records),
    }


def run_scoring_benchmark(
    rag,
    *,
    dataset_path: Path = SCORING_DATASET,
    limit: int | None = None,
    repeats: int = 3,
    reviewed_only: bool = False,
    output_dir: Path = Path("reports/scoring"),
    progress=None,
) -> dict:
    data = load_scoring_dataset(dataset_path)
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError("重复次数需要为1到10。")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(data["cases"])):
        raise ValueError("案例数量超出评分评测集范围。")
    cases = data["cases"][:limit] if limit else data["cases"]
    if reviewed_only:
        require_reviewed(data, cases)
    engine = InterviewEngine(rag)
    records = []
    report_id = uuid.uuid4().hex
    for repeat in range(repeats):
        for case in cases:
            session = InterviewSession(
                id=f"evaluation-{report_id}-{case['id']}",
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
            started = time.perf_counter()
            # Use the exact production prompt, merge policy and quote validator.
            # Sessions are never saved; candidate answers never enter the retrieval corpus.
            evaluation = engine._evaluate(session, case["answers"][-1])
            records.append(
                {
                    "case_id": case["id"],
                    "repeat": repeat + 1,
                    "category": case["category"],
                    "answers": case["answers"],
                    "expected_statuses": case["expected_statuses"],
                    "evaluation": evaluation,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                }
            )
            if progress:
                progress(len(records), len(cases) * repeats, case["id"], "scoring")
            # Checkpoint completed calls so an interrupted long run retains its evidence.
            write_report(
                {"id": report_id, "kind": "scoring_checkpoint", "records": records},
                output_dir / "checkpoints",
                "# 未完成的评分评测\n",
            )
    report = {
        "id": report_id,
        "kind": "scoring_experiment",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": data.get("name", dataset_path.stem),
            "label_status": data.get("label_status", "unreviewed"),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "selected_cases": len(cases),
        },
        "models": {"chat": getattr(rag.chat_model, "chat_model", type(rag.chat_model).__name__)},
        "repeats": repeats,
        "summary": scoring_summary(records),
        "records": records,
        "by_category": {
            c: scoring_summary([r for r in records if r["category"] == c])
            for c in sorted({r["category"] for r in records})
        },
        "notes": [
            "一致率是与所给标注的符合程度，种子标注未复核时不能称为人工评分准确率。",
            "无效原文或降级评分记待核验，并计入全部评分点的一致率分母。",
            "原文存在只证明出处，不能证明语义评价正确。宏F1仅平均有标注支持的类别。",
            "重复一致性是同题多次评价中众数状态向量的比例，不等于判断正确性。",
            "评测调用生产评分逻辑，不保存面试会话，也不修改题库或历史评分。",
        ],
    }
    lines = [
        "# 逐点评分一致性评测",
        "",
        f"- 标注状态：{report['dataset']['label_status']}",
        f"- 独立案例：{len(cases)}；重复次数：{repeats}",
        "",
        "```json",
        json.dumps(report["summary"], ensure_ascii=False, indent=2),
        "```",
        "",
        *[f"- {note}" for note in report["notes"]],
        "",
        "## 逐题检查",
        "",
    ]
    for r in records:
        lines += [
            f"### {r['case_id']} / 第{r['repeat']}次",
            f"- 回答：{r['answers']}",
            f"- 标注：{r['expected_statuses']}",
            "```json",
            json.dumps(r["evaluation"]["point_reviews"], ensure_ascii=False, indent=2),
            "```",
        ]
    write_report(report, output_dir, "\n".join(lines) + "\n")
    return report
