"""Analyse only complete repeats; never freeze a selection from partial data."""

from __future__ import annotations

import json
from pathlib import Path

from techrag.benchmark import _summary
from techrag.experiments import paired_comparisons

ROOT = Path(__file__).resolve().parents[1] / "reports/full-evaluation"
VARIANTS = ["hybrid", "hybrid_reranked", "expanded"]


def main():
    data = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))
    expected = {c["id"] for c in data["cases"] if c["split"] == "dev"}
    rows = []
    complete_repeats = []
    for repeat in range(1, 4):
        records = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in (ROOT / "dev" / f"repeat-{repeat}").glob("*/checkpoint/record-*.json")
        ]
        matrix = {(r["case_id"], r["variant"]) for r in records}
        if (
            matrix == {(case, variant) for case in expected for variant in VARIANTS}
            and len(records) == len(expected) * len(VARIANTS)
            and all(not r["error"] and not r["result"]["warnings"] for r in records)
        ):
            complete_repeats.append(repeat)
            rows.extend(r | {"repeat": repeat} for r in records)
    if not rows:
        raise SystemExit("尚无完整、无失败/降级的重复轮次，不给预分析结论。")
    result = {
        "kind": "partial_development_analysis",
        "not_final": True,
        "complete_repeats": complete_repeats,
        "independent_cases": len(expected),
        "summaries": [_summary(v, rows) for v in VARIANTS],
        "comparisons": paired_comparisons(rows),
        "notes": [
            "仅使用完整轮次，各方案具有相同案例与重复数量；未完成的第三轮不参与。",
            "未冻结选型，未查看测试结果，不能替代协议要求的三次开发与三次测试。",
            "纯检索不生成答案，其总流程时间包含拆解、召回与精排，不是问答时延。",
            "数据为项目作者人工复核的合成查询，按标注来源关联组bootstrap，不能泛化为生产表现。",
        ],
    }
    lines = [
        "# 开发集预分析（非最终结果）",
        "",
        f"独立查询{len(expected)}条；仅采用完整轮次{complete_repeats}。",
        "",
        "| 方案 | 完成记录 | 目标来源Recall@3 | MRR | 平均检索流程秒 | P95秒 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['label']} | {row['completed']}/{row['cases']} | {row['recall_at_k']} | "
            f"{row['mrr']} | {row['mean_seconds']} | {row['p95_seconds']} |"
        )
    lines += [
        "",
        "## 配对差值与分组区间",
        "",
        "```json",
        json.dumps(result["comparisons"], ensure_ascii=False, indent=2),
        "```",
        "",
        *[f"- {note}" for note in result["notes"]],
    ]
    (ROOT / "development-partial.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "development-partial.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "complete_repeats": complete_repeats,
                "summaries": [
                    {
                        key: r[key]
                        for key in ("variant", "recall_at_k", "mrr", "mean_seconds", "cases")
                    }
                    for r in result["summaries"]
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
