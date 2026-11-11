"""Summarize saved study progress and completed phases without model requests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "reports/full-evaluation"
METRIC_LABELS = {
    "faithfulness": "事实受资料支持比例",
    "answer_correctness": "事实精确率与参考要点覆盖F1（自动）",
    "answer_relevancy": "回答相关性（模型评分）",
    "citation_precision": "引用支持比例",
    "citation_coverage": "事实引用覆盖率",
    "correct_refusal": "无资料题资料不足状态比例",
    "false_refusal": "可回答题全部分节资料不足比例",
}


def main():
    progress = {}
    completed = {}
    for phase, jobs, pipelines in (("dev", 240, 720), ("test", 120, 240), ("quality", 40, 40)):
        folder = DIRECTORY / phase
        records = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in folder.glob("**/checkpoint/record-*.json")
        ]
        progress[phase] = {
            "completed_case_jobs": len(list(folder.glob("**/complete.json"))),
            "planned_case_jobs": jobs,
            "saved_pipelines": len(records),
            "planned_pipelines": pipelines,
            "failed_pipelines": sum(bool(r["error"]) for r in records),
            "warning_pipelines": sum(bool(r.get("result", {}).get("warnings")) for r in records),
            "completed_answer_judges": sum(
                r.get("answer_evaluation", {}).get("status") == "completed" for r in records
            ),
            "unavailable_answer_judges": sum(
                r.get("answer_evaluation", {}).get("status") == "unavailable" for r in records
            ),
        }
        report_path = DIRECTORY / f"{phase}.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            completed[phase] = {
                k: report[k]
                for k in (
                    "independent_cases",
                    "independent_groups",
                    "repeats",
                    "evaluation_mode",
                    "summaries",
                    "comparisons",
                )
            }
    result = {
        "progress": progress,
        "completed_phases": completed,
        "all_phases_saved": len(completed) == 3,
        "all_pipelines_completed": len(completed) == 3
        and all(
            row["completed"] == row["cases"]
            for phase in completed.values()
            for row in phase["summaries"]
        ),
        "all_answer_judges_completed": "quality" in completed
        and all(
            row["judged_cases"] == row["cases"]
            for row in completed["quality"]["summaries"]
        ),
    }
    selection = DIRECTORY / "selection.json"
    if selection.exists():
        result["frozen_selection"] = json.loads(selection.read_text(encoding="utf-8"))
    analysis_path = DIRECTORY / "analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        recovery = analysis.get("quality_recovery")
        if recovery:
            result["quality_recovery"] = {k: recovery[k] for k in (
                "retained_valid_records", "initial_judge_completion_rate", "saved_attempts",
                "attempted_judge_calls", "unavailable_attempts", "retried_cases", "note",
            )}
    lines = [
        "# 完整评测状态与结果",
        "",
        "| 阶段 | 案例轮次完成 | 已保存方案结果 | 失败 | 带提示 | 有效/无效评审 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for phase, value in progress.items():
        lines.append(
            f"| {phase} | {value['completed_case_jobs']}/{value['planned_case_jobs']} | "
            f"{value['saved_pipelines']}/{value['planned_pipelines']} | "
            f"{value['failed_pipelines']} | {value['warning_pipelines']} | "
            f"{value['completed_answer_judges']}/{value['unavailable_answer_judges']} |"
        )
    for phase, report in completed.items():
        lines += [
            "",
            f"## {phase}",
            "",
            f"独立问题{report['independent_cases']}，来源关联组{report['independent_groups']}，"
            f"重复{report['repeats']}，模式{report['evaluation_mode']}。",
            "",
            "| 方案 | Recall@3 | MRR | 平均流程秒 | P95秒 | 完成 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in report["summaries"]:
            lines.append(
                f"| {row['label']} | {row['recall_at_k']} | {row['mrr']} | "
                f"{row['mean_seconds']} | {row['p95_seconds']} | "
                f"{row['completed']}/{row['cases']} |"
            )
        if phase == "quality":
            lines += ["", "自动回答质量评审（独立调用，默认与生成使用同一模型）：", ""]
            for row in report["summaries"]:
                lines.append(
                    f"评审完成{row['judged_cases']}/{row['cases']}，"
                    f"评审失败{row['judge_failures']}。"
                )
                lines += ["", "| 指标 | 均值 | 有效样本 |", "| --- | --- | --- |"]
                for metric, value in row["answer_quality"].items():
                    lines.append(
                        f"| {METRIC_LABELS[metric]} | {value} | {row['metric_samples'][metric]} |"
                    )
                for metric in ("correct_refusal", "false_refusal"):
                    lines.append(
                        f"| {METRIC_LABELS[metric]} | {row[metric + '_rate']} | "
                        f"{row['metric_samples'][metric]} |"
                    )
            recovery = result.get("quality_recovery")
            if recovery:
                lines += [
                    "",
                    f"首轮有效{recovery['retained_valid_records']}/40；累计"
                    f"{recovery['attempted_judge_calls']}次评审尝试，其中"
                    f"{recovery['unavailable_attempts']}次未取得有效评审。"
                    "上表为断点恢复后的最终记录，不表示单次请求成功率100%。",
                    "只重试无效项，原35条有效记录经比对完全一致；"
                    "重试重新生成该条回答。低分和引用拦截项保留，详见[恢复分析](analysis.md)。",
                ]
        lines += [
            "",
            "配对差值和分组置信区间：",
            "",
            "```json",
            json.dumps(report["comparisons"], ensure_ascii=False, indent=2),
            "```",
        ]
    lines += [
        "",
        "## 统计口径",
        "",
        "- 未保存整阶段报告时只展示进度，不给最终选型或效果结论。",
        "- 检索阶段不生成答案，其流程耗时不能称为问答时延，也不评价语义拒答。",
        "- retrieval计时字段对应召回阶段，精排另计；总流程包含拆解、召回和精排。",
        "- 检索耗时在三个并行案例进程负载下测得；质量阶段为单进程。",
        "- 重复次数不增加独立样本数量，差值区间按来源关联组bootstrap。",
        "- Recall@3按各子问题Top 3目标来源合并命中，复合题MRR取最佳子问题倒数排名，"
        "不是整条复合问题的全局Top 3排名口径。",
        "- 原标注人工复核已完成；生成答案的质量评审为同模型独立调用的自动辅助评审。",
        "- 程序校验结构和原文出处，不能证明语义正确；质量指标不标为人工准确率。",
        "- 质量指标先按每条回答计算，再对有有效值的回答取均值；有效样本列为回答数。",
        "- 本数据集部分参考要点是完整段落，二元覆盖判断较粗；其F1不能解释为真人答案准确率。",
        "",
    ]
    (DIRECTORY / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DIRECTORY / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(progress, ensure_ascii=False))


if __name__ == "__main__":
    main()
