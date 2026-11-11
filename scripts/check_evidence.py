"""Offline data validation and review worksheets; no external model calls."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from techrag.benchmark import load_dataset
from techrag.experiments import EVIDENCE_DATASET, require_reviewed, validate_holdout
from techrag.interview_questions import load_question_directory
from techrag.scoring_benchmark import load_scoring_dataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = load_dataset(EVIDENCE_DATASET)
    validate_holdout(data["cases"])
    questions = {
        f"interview-question://{q.id}": q
        for q in load_question_directory(ROOT / "data/interview_questions")
    }
    missing = {s for c in data["cases"] for s in c["expected_sources"]} - questions.keys()
    if missing:
        raise SystemExit(f"标注来源不存在：{sorted(missing)}")
    scores = load_scoring_dataset()
    for dataset in (data, scores):
        if dataset.get("label_status") == "human_reviewed":
            require_reviewed(dataset, dataset["cases"])
    output = ROOT / "reports/evidence"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "retrieval_cases": len(data["cases"]),
        "split_counts": dict(Counter(c["split"] for c in data["cases"])),
        "retrieval_categories": dict(Counter(c["category"] for c in data["cases"])),
        "independent_groups": len({c["group"] for c in data["cases"]}),
        "source_overlap_between_splits": False,
        "missing_sources": sorted(missing),
        "scoring_cases": len(scores["cases"]),
        "scoring_categories": dict(Counter(c["category"] for c in scores["cases"])),
        "retrieval_label_status": data["label_status"],
        "scoring_label_status": scores["label_status"],
        "reviewed_retrieval_cases": sum(
            c.get("review", {}).get("approved") is True for c in data["cases"]
        ),
        "reviewed_scoring_cases": sum(
            c.get("review", {}).get("approved") is True for c in scores["cases"]
        ),
        "human_review_complete": all(
            d.get("label_status") == "human_reviewed" for d in (data, scores)
        )
        and all(
            c.get("review", {}).get("approved") is True for c in data["cases"] + scores["cases"]
        ),
    }
    (output / "dataset-check.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for filename, dataset in (("retrieval-review.md", data), ("scoring-review.md", scores)):
        lines = [
            f"# {dataset['name']} 人工复核工作表",
            "",
            "在原JSON中修改标注，并填写review.approved、reviewer和reviewed_at。",
            "全部完成后才可将label_status改为human_reviewed；此工作表不会自动批准。",
            "检索需补齐等价来源；评分需区分遗漏与错误，核对等价解法及补充纠错。",
            "",
        ]
        lines += [f"当前标注状态：{dataset['label_status']}。", ""]
        for case in dataset["cases"]:
            lines += [f"## {case['id']} · {case['category']}", "", case["question"], ""]
            if "expected_sources" in case:
                lines += [
                    f"- 分组：{case['split']} / {case['group']}",
                    f"- 可回答：{case['answerable']}",
                    f"- 目标来源：{case['expected_sources']}",
                ]
                for source in case["expected_sources"]:
                    q = questions[source]
                    lines += ["", f"### {source} · {q.text}", "", q.reference_answer]
            else:
                label_name = "种子标签"
                if dataset["label_status"] == "human_reviewed":
                    label_name = "已复核标签"
                lines += [
                    f"- 评分要点：{case['scoring_points']}",
                    f"- 候选人回答：{case['answers']}",
                    f"- {label_name}：{case['expected_statuses']}",
                    "",
                    case["reference_answer"],
                ]
            review = case.get("review", {})
            mark = "x" if review.get("approved") is True else " "
            recorded = bool(
                review.get("approved") is True
                and review.get("reviewer")
                and review.get("reviewed_at")
            )
            lines += [
                "",
                f"- [{mark}] 可回答性或逐点标签合理",
                f"- [{mark}] 来源/参考要点完整且正确",
                f"- [{'x' if recorded else ' '}] 已在JSON填写复核人、日期及备注",
                "",
            ]
            if recorded:
                lines += [
                    f"复核人：{review['reviewer']}；日期：{review['reviewed_at']}。",
                    f"备注：{review.get('notes', '')}",
                    "",
                ]
        (output / filename).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
