"""Build a readable final analysis from saved reports, without API calls."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "reports/full-evaluation"
NAMES = {
    "hybrid": "混合基线",
    "hybrid_reranked": "混合加精排",
    "expanded": "混合加MQE/HyDE",
}


def average(rows, key):
    values = [r[key] for r in rows if not r["error"] and r.get(key) is not None]
    return round(mean(values), 6) if values else None


def phase_analysis(report):
    buckets = defaultdict(list)
    for row in report["records"]:
        buckets[(row["variant"], row["category"])].append(row)
    categories = []
    for (variant, category), rows in sorted(buckets.items()):
        categories.append(
            {
                "variant": variant,
                "category": category,
                "independent_cases": len({r["case_id"] for r in rows}),
                "observations": len(rows),
                "answerable_cases": len({r["case_id"] for r in rows if r["answerable"]}),
                "recall_at_k": average(rows, "recall_at_k"),
                "mrr": average(rows, "reciprocal_rank"),
                "mean_seconds": average(rows, "elapsed_seconds"),
                "failures": sum(bool(r["error"]) for r in rows),
            }
        )
    differences = []
    pairs = defaultdict(dict)
    for row in report["records"]:
        if not row["error"] and row["variant"] in {"hybrid", "expanded"}:
            pairs[(row["case_id"], row["repeat"])][row["variant"]] = row
    for (case_id, repeat), pair in sorted(pairs.items()):
        if set(pair) != {"hybrid", "expanded"}:
            continue
        baseline, candidate = pair["hybrid"], pair["expanded"]
        if baseline["recall_at_k"] is None:
            continue
        recall_delta = candidate["recall_at_k"] - baseline["recall_at_k"]
        mrr_delta = candidate["reciprocal_rank"] - baseline["reciprocal_rank"]
        if recall_delta or mrr_delta:
            differences.append(
                {
                    "case_id": case_id,
                    "repeat": repeat,
                    "category": baseline["category"],
                    "question": baseline["question"],
                    "recall_delta": round(recall_delta, 6),
                    "mrr_delta": round(mrr_delta, 6),
                }
            )
    return {"categories": categories, "changed_observations": differences}


def main():
    reports = {}
    hashes = {}
    for phase in ("dev", "test", "quality"):
        path = DIRECTORY / f"{phase}.json"
        reports[phase] = json.loads(path.read_text(encoding="utf-8"))
        hashes[phase] = hashlib.sha256(path.read_bytes()).hexdigest()
    quality = reports["quality"]
    result = {
        "source_report_sha256": hashes,
        "selection": json.loads((DIRECTORY / "selection.json").read_text(encoding="utf-8")),
        "retrieval": {p: phase_analysis(reports[p]) for p in ("dev", "test")},
        "quality_unavailable": [
            {"case_id": r["case_id"], "error": r["error"],
             "evaluation": r["answer_evaluation"]}
            for r in quality["records"]
            if r["error"] or r["answer_evaluation"]["status"] != "completed"
        ],
        "quality_warned_cases": [
            {"case_id": r["case_id"], "warnings": r["result"]["warnings"]}
            for r in quality["records"] if r["result"].get("warnings")
        ],
        "structural_refusals": [
            {"case_id": r["case_id"], "answerable": r["answerable"],
             "correct_refusal": r["correct_refusal"], "false_refusal": r["false_refusal"]}
            for r in quality["records"]
            if r.get("correct_refusal") is not None or r.get("false_refusal")
        ],
        "quality_rubric_diagnostics": {
            "reference_point_counts": {
                str(n): sum(
                    len(r["answer_evaluation"]["reference_points"]) == n
                    for r in quality["records"]
                    if r["answer_evaluation"]["status"] == "completed"
                )
                for n in sorted({
                    len(r["answer_evaluation"]["reference_points"])
                    for r in quality["records"]
                    if r["answer_evaluation"]["status"] == "completed"
                })
            },
            "all_claims_correct_but_zero_coverage": [
                {"case_id": r["case_id"], "question": r["question"],
                 "warnings": r["result"]["warnings"],
                 "point_reviews": r["answer_evaluation"]["reference_points"]}
                for r in quality["records"]
                if r["answer_evaluation"]["status"] == "completed"
                and r.get("answer_correctness") == 0
                and r["answer_evaluation"]["claims"]
                and all(c["correct"] for c in r["answer_evaluation"]["claims"])
            ],
            "note": "参考要点存在完整段落，缺少段落中的某部分会使二元覆盖为0。"
            "例如V2-006的事实均判正确，但因未覆盖参考段落中的示例等内容而得到0分；"
            "该F1反映当前粗粒度评审规则，不能作为真人答案正确率。未事后修改金标准或重新选型。",
        },
    }
    initial_path = DIRECTORY / "quality-first-pass.json"
    if initial_path.exists():
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
        final_rows = {r["case_id"]: r for r in quality["records"]}
        retried = []
        retained = 0
        for row in initial["records"]:
            current = final_rows[row["case_id"]]
            if not row["error"] and row["answer_evaluation"]["status"] == "completed":
                if row != current:
                    raise ValueError("首次有效质量记录发生变化，不能声称只重试无效项。")
                retained += 1
            else:
                retried.append({
                    "case_id": row["case_id"],
                    "initial_status": row["answer_evaluation"]["status"],
                    "final_status": current["answer_evaluation"]["status"],
                    "answer_changed": row["result"]["answer"] != current["result"]["answer"],
                })
        result["quality_recovery"] = {
            "initial_report_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
            "retained_valid_records": retained,
            "initial_judge_completion_rate": retained / len(initial["records"]),
            "initial_summaries": initial["summaries"],
            "retried_cases": retried,
            "note": "重试无效评审项会重新生成该条回答并评审；不是仅对原回答重新打分。"
            "有效低分和引用拦截项不因分数低而重试；本阶段仍是一份每题一次最终记录，不是三轮稳定性实验。",
        }
        attempts = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in (DIRECTORY / "quality").glob("**/checkpoint/attempts/*.json")
        ]
        result["quality_recovery"]["saved_attempts"] = len(attempts)
        result["quality_recovery"]["attempted_judge_calls"] = sum(
            r.get("judge_calls", {}).get("chat_calls", 0) for r in attempts
        )
        result["quality_recovery"]["unavailable_attempts"] = sum(
            r.get("answer_evaluation", {}).get("status") == "unavailable" for r in attempts
        )
    lines = ["# 正式评测结果分析", "", "原始报告及完整指标见[状态汇总](summary.md)。", ""]
    for phase, title in (("dev", "开发集"), ("test", "独立测试集")):
        report = reports[phase]
        lines += [f"## {title}", "",
                  f"{report['independent_cases']}条独立问题，{report['independent_groups']}个来源关联组，三轮重复。",
                  "", "| 方案 | Recall@3 | MRR | 平均检索流程秒 | P95秒 |",
                  "| --- | --- | --- | --- | --- |"]
        for row in report["summaries"]:
            lines.append(f"| {NAMES[row['variant']]} | {row['recall_at_k']} | {row['mrr']} | "
                         f"{row['mean_seconds']} | {row['p95_seconds']} |")
        lines += ["", "### 分类结果", "",
                  "| 方案 | 类别 | 独立题数 | 观测次数 | Recall@3 | MRR | 平均秒 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in result["retrieval"][phase]["categories"]:
            lines.append(f"| {NAMES[row['variant']]} | {row['category']} | "
                         f"{row['independent_cases']} | {row['observations']} | "
                         f"{row['recall_at_k']} | {row['mrr']} | {row['mean_seconds']} |")
    lines += ["", "## 解释与限制", "",
              "- 方案仅按开发集预定规则选择，未用测试集重新调参。",
              "- 测试集37条有标注来源，3条无资料；来源召回和MRR不统计无资料题。",
              "- 测试集两个方案Recall@3均为1，未发现召回提升；MRR配对差值区间含0。",
              "- 测试集排序变化集中在两条术语问题，共5次配对观测，不能据此泛化收益。",
              "- 扩展方案约5.1倍检索流程耗时；当前数据不支持把它包装为整体性能优化。",
              "- 重复轮次不增加独立样本数；区间按来源关联组bootstrap。",
              "- 纯检索三进程负载下的耗时不能与质量阶段单进程问答时延直接比较。",
              "- 回答质量为生成模型的独立评审调用，格式和原文出处已程序校验，"
              "语义判断有同模型偏差。",
              "- 参考要点部分使用完整段落，二元覆盖过粗：有些回答事实全对仍因遗漏段落内容得0分。"
              "当前F1不是真人答案准确率；原始金标准保留，未为提高数字而修改。",
              "- 拒答指标按全部分节insufficient及来源字段判定，不是人工判断的拒答正确率。",
              "- 无资料时的Faithfulness/引用指标可能无有效样本，不将null记为满分。",
              "- 评分集24条、72点、97.2%一致率属于另一项合成回答评测，不能混为本次问答正确率。",
              "", "## 逐题差异与异常", "",
              "下面保留扩展与基线不同的观测、评审缺失和警告。成功重试历史留在checkpoint/attempts，"
              "只将各轮最终有效结果纳入正式统计。", "", "```json",
              json.dumps(
                  {
                      "changed_observations": {
                          p: result["retrieval"][p]["changed_observations"]
                          for p in ("dev", "test")
                      },
                      **{k: result[k] for k in (
                          "quality_unavailable", "quality_warned_cases", "structural_refusals",
                          "quality_rubric_diagnostics",
                      )},
                      "quality_recovery": result.get("quality_recovery"),
                  }, ensure_ascii=False, indent=2,
              ), "```", ""]
    (DIRECTORY / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DIRECTORY / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {DIRECTORY / 'analysis.md'}")


if __name__ == "__main__":
    main()
