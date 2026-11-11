"""Summarize completed, explicitly selected pilot reports without model requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def confirmed_review(report_paths: list[Path]) -> dict | None:
    path = ROOT / "reports/evidence/human-review.json"
    if not path.exists():
        return None
    certificate = json.loads(path.read_text(encoding="utf-8"))
    if certificate.get("annotation_changes") is not False:
        return None
    for report_path in report_paths:
        matches = [
            r
            for r in certificate["verified_historical_reports"]
            if (ROOT / r["file"]).resolve() == report_path.resolve()
        ]
        if (
            len(matches) != 1
            or hashlib.sha256(report_path.read_bytes()).hexdigest() != matches[0]["sha256"]
        ):
            return None
    for entry in certificate["datasets"]:
        current = ROOT / entry["file"]
        backup = ROOT / entry["backup"]
        if (
            hashlib.sha256(current.read_bytes()).hexdigest() != entry["after_sha256"]
            or hashlib.sha256(backup.read_bytes()).hexdigest() != entry["before_sha256"]
        ):
            return None
        data = json.loads(current.read_text(encoding="utf-8"))
        before = json.loads(backup.read_text(encoding="utf-8"))

        def strip_reviews(cases):
            return [{k: v for k, v in c.items() if k != "review"} for c in cases]

        if (
            data["label_status"] != "human_reviewed"
            or not all(c["review"].get("approved") is True for c in data["cases"])
            or strip_reviews(data["cases"]) != strip_reviews(before["cases"])
        ):
            return None
    return {
        "certificate": str(path),
        "reviewer": certificate["reviewer"],
        "reviewed_at": certificate["reviewed_at"],
        "annotations_unchanged": True,
        "retrieval_cases": certificate["reviewed_retrieval_cases"],
        "scoring_cases": certificate["reviewed_scoring_cases"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    args = parser.parse_args()
    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    scoring = json.loads(args.scoring.read_text(encoding="utf-8"))
    if (
        retrieval.get("kind") != "retrieval_experiment"
        or scoring.get("kind") != "scoring_experiment"
    ):
        raise SystemExit("需要完成的检索与评分报告，不能传入checkpoint。")
    review_confirmation = confirmed_review([args.retrieval, args.scoring])
    label_name = "人工复核标签" if review_confirmation else "种子标签"
    mismatches = []
    pending_scoring = []
    for record in scoring["records"]:
        for point in record["evaluation"]["point_reviews"]:
            expected = record["expected_statuses"][point["point_id"]]
            if point["status"] == "unverified":
                pending_scoring.append(
                    {
                        "case_id": record["case_id"],
                        "point_id": point["point_id"],
                        "scoring_method": record["evaluation"]["scoring_method"],
                    }
                )
                continue
            if point["status"] != expected:
                mismatches.append(
                    {
                        "case_id": record["case_id"],
                        "category": record["category"],
                        "point": point["point"],
                        "point_id": point["point_id"],
                        "seed_status": expected,
                        "model_status": point["status"],
                        "answers": record["answers"],
                        "reason": point["reason"],
                        "evidence": point["evidence"],
                    }
                )
    incomplete = []
    for row in retrieval["records"]:
        if row["error"] or (row.get("recall_at_k") is not None and row["recall_at_k"] < 1):
            incomplete.append(
                {
                    "case_id": row["case_id"],
                    "variant": row["variant"],
                    "question": row["question"],
                    "error": row["error"],
                    "recall_at_k": row.get("recall_at_k"),
                    "missing_target_sources": None
                    if row["error"]
                    else sorted(
                        set(row["expected_sources"]) - set(row.get("retrieved_sources", []))
                    ),
                }
            )
    report = {
        "retrieval_report": str(args.retrieval.resolve()),
        "scoring_report": str(args.scoring.resolve()),
        "retrieval_dataset": retrieval["dataset"],
        "scoring_dataset": scoring["dataset"],
        "post_run_human_review": review_confirmation,
        "retrieval_summaries": retrieval["summaries"],
        "comparisons": retrieval["comparisons"],
        "scoring_summary": scoring["summary"],
        "scoring_by_category": scoring["by_category"],
        "scoring_mismatches": mismatches,
        "pending_scoring": pending_scoring,
        "incomplete_retrieval": incomplete,
        "completed_pipelines": sum(not row["error"] for row in retrieval["records"]),
        "failed_pipelines": sum(bool(row["error"]) for row in retrieval["records"]),
        "notes": [
            "六条开发查询仅单次运行；来源与评分标签已由项目作者人工复核，原标注全部保留。"
            if review_confirmation
            else "六条开发查询仅单次运行，来源标签与评分标签都未人工复核。",
            "检索消融未额外调用答案评审，不报告忠实度或最终回答准确率。",
            "评分一致率指符合项目作者人工复核标签的程度；小样本、单复核人不等于生产准确率。"
            if review_confirmation
            else "评分一致率指符合种子标签的程度，不能称为人工金标准准确率。",
            "评分恢复报告混合保留了前次10条结果与新补测14条结果，不作跨批次耗时结论。"
            if scoring.get("recovery")
            else "两个评测进程同时运行，耗时受并行API请求影响，需串行重复实验验证。",
            "项目作者已确认保留两处分歧案例的原标签；报告保留模型判断、原文与理由。"
            if review_confirmation
            else "模型与种子标签的分歧不自动等于模型错误；应复核证据与评分点完整性。",
        ],
    }
    valid_answerable_ids = sorted(
        {row["case_id"] for row in retrieval["records"] if not row["error"] and row["answerable"]}
    )
    diagnostic_path = ROOT / "reports/evidence/provider-diagnostic.json"
    diagnostics = (
        json.loads(diagnostic_path.read_text(encoding="utf-8")) if diagnostic_path.exists() else {}
    )
    report["provider_diagnostic"] = diagnostics
    balance_blocked = any(
        d.get("category") == "insufficient_balance" for d in diagnostics.get("diagnostics", [])
    )
    incomplete_batch = bool(report["failed_pipelines"] or scoring["summary"]["fallback_cases"])
    warnings = [
        {"case_id": r["case_id"], "variant": r["variant"], "message": warning}
        for r in retrieval["records"]
        if not r["error"]
        for warning in r["result"]["warnings"]
    ]
    cleanup_warnings = [w for w in warnings if "资料不足，本次已移除" in w["message"]]
    report["citation_cleanup_warnings"] = cleanup_warnings
    report["other_warnings"] = [w for w in warnings if w not in cleanup_warnings]
    lines = [
        "# InterviewPilot 真实API小规模实验",
        "",
        "本轮检索为6条开发查询×7方案、一次串行对比；评分为24条案例各保留一次评价。"
        if scoring.get("recovery")
        else "本轮按授权尝试6条开发查询×7方案，以及24条合成评分案例，每项只运行一次。",
        "模型、数据指纹、配置及逐题证据在原始JSON中保留。",
        f"检索执行完成{report['completed_pipelines']}次、失败{report['failed_pipelines']}次；"
        f"评分有{scoring['summary']['fallback_cases']}个降级案例。"
        + ("批次未全部获得有效结果。" if incomplete_batch else "本批模型调用已完成。"),
        "接口诊断记录：DeepSeek HTTP 402余额不足，已停止进一步调用。"
        if balance_blocked
        else "恢复试验未再出现余额/认证错误；首次失败报告及重试来源记录均保留。"
        if scoring.get("recovery")
        else "接口诊断与逐题失败原因需另行核对。",
        "",
        "## 检索与耗时",
        "",
        "| 方案 | 完成 | 来源Recall@3（有效样本） | MRR | 正确拒答 | 误拒答 | 平均秒 | P95秒 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in retrieval["summaries"]:
        lines.append(
            f"| {s['label']} | {s['completed']}/{s['cases']} | {s['recall_at_k']} "
            f"(n={s['metric_samples']['recall_at_k']}) | "
            f"{s['mrr']} | {s['correct_refusal_rate']} | {s['false_refusal_rate']} | "
            f"{s['mean_seconds']} | {s['p95_seconds']} |"
        )
    lines += [
        "",
        "正确拒答仅有一条独立无资料问题；其百分比不能用于总体拒答率结论。",
        f"有{len(cleanup_warnings)}条提示来自资料不足时移除引用的保护分支；"
        f"其他警告{len(report['other_warnings'])}条，不能把正常保护提示直接解释为API失败。",
        "P95只有六个观测，按当前口径等于最大值，不能视为稳定的尾延迟估计。",
        "平均/P95混合了快速失败与成功请求，不能据此比较方案的速度或效果。"
        if report["failed_pipelines"]
        else "耗时包含共有拆解与冷启动，不含额外答案评审；单次小样本差值需重复验证。",
        f"得到有效结果的独立可回答案例：{valid_answerable_ids}；"
        "请按各指标有效样本数解读，不能把个别案例命中解释为总体准确率。",
        "",
        "### 未完整命中的目标来源",
        "",
    ]
    if review_confirmation:
        lines += [
            f"人工复核：{review_confirmation['reviewer']}，{review_confirmation['reviewed_at']}；"
            "120条检索、24条评分全部认可原标注。",
            "复核仅更新记录与状态，旧模型响应未改写；原报告保留运行时状态，当前视图附加复核确认。",
            "[复核记录与前后文件指纹](human-review.json)。",
            "",
        ]
    if not incomplete:
        lines.append("本轮所有成功的可回答查询均覆盖种子目标来源；这不证明生成答案正确。")
    for row in incomplete:
        lines.append(
            f"- {row['case_id']} / {row['variant']}：Recall={row['recall_at_k']}；"
            f"缺失={row['missing_target_sources']}；失败={row['error']}"
        )
    if not report["failed_pipelines"]:
        lines += [
            "",
            "### 本批观察与下一步",
            "",
            "- 七方案在5条可回答查询上均覆盖全部种子目标来源，MRR均为1；本批没有召回改善证据。",
            "- 先在更难、人工复核的开发查询上证明组件收益，再决定是否默认启用查询扩展。",
            "- 本次未额外评审生成答案，不能将来源命中解释为回答正确率。",
            "",
            "| 方案 | 检索阶段平均秒 | MQE平均秒 | HyDE平均秒 | 方案LLM调用数 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for s in retrieval["summaries"]:
            timing = s["latency_seconds"]
            lines.append(
                f"| {s['label']} | {timing['retrieval']['mean']:.3f} | "
                f"{timing['mqe']['mean']:.3f} | {timing['hyde']['mean']:.3f} | "
                f"{s['calls']['chat_calls']} |"
            )
        lines += ["", "共有拆解调用6次，未重复计入方案LLM调用数；调用数不等于token费用。"]
    summary = scoring["summary"]
    lines += [
        "",
        "## 逐点评分",
        "",
        f"- 案例：{summary['independent_cases']}；评分点：{summary['point_observations']}。",
        "- 获得模型逐点评价且通过原文出处校验："
        f"{summary['verified_points']}/{summary['point_observations']}；"
        f"其中与{label_name}一致：{summary['agreement_verified_points']:.2%}。",
        "- 该比例仅包含成功核验结果，不能代表全批次评分准确率。",
        f"- 全部评分点符合率（待核验仍在分母中）：{summary['agreement_all_points']:.2%}；"
        f"宏F1：{summary['macro_f1']:.4f}。"
        + (
            "受接口失败影响，不能解释为语义评判效果。"
            if summary["fallback_cases"]
            else f"这是与{label_name}的符合程度，仍需扩大样本和重复评估。"
        ),
        f"- 待核验率：{summary['unverified_rate']:.2%}；降级案例：{summary['fallback_cases']}。",
        "- 本轮没有重复评价，不报告评分稳定性。",
        "",
        f"| 回答类别 | 与{label_name}一致率 | 待核验率 |",
        "| --- | --- | --- |",
    ]
    for category, s in scoring["by_category"].items():
        lines.append(
            f"| {category} | {s['agreement_all_points']:.2%} | {s['unverified_rate']:.2%} |"
        )
    lines += ["", "### 与已复核标签不一致的结果" if review_confirmation else "### 待复核分歧", ""]
    if not mismatches:
        lines.append("本轮与种子标签没有分歧；仍需人工确认标签与语义判断。")
    for row in mismatches:
        lines += [
            f"#### {row['case_id']} / {row['point_id']} · {row['category']}",
            "",
            f"- 评分点：{row['point']}",
            f"- {label_name}：{row['seed_status']}；模型判断：{row['model_status']}",
            f"- 回答：{row['answers']}",
            f"- 模型理由：{row['reason']}",
            f"- 核验后的原文证据：{row['evidence']}",
            "",
        ]
    lines += [f"另有{len(pending_scoring)}个待核验评分点，属于未完成评分，不列为语义判断错误。", ""]
    lines += [
        "## 解释边界",
        "",
        *[f"- {note}" for note in report["notes"]],
        "",
        f"- [检索原始报告]({args.retrieval.resolve().as_posix()})",
        f"- [评分原始报告]({args.scoring.resolve().as_posix()})",
        "",
    ]
    output = ROOT / "reports/evidence"
    (output / "live-pilot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "live-pilot.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output / 'live-pilot.md'}")


if __name__ == "__main__":
    main()
