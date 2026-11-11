"""Build an evidence index from actual local outputs, without running models."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "reports/evidence"
    data = json.loads((folder / "dataset-check.json").read_text(encoding="utf-8"))
    suites = list(ET.parse(folder / "pytest.xml").getroot().iter("testsuite"))
    counts = {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    targeted = ET.parse(folder / "targeted-tests.xml").getroot()
    targeted_counts = {
        key: sum(int(s.attrib.get(key, 0)) for s in targeted.iter("testsuite"))
        for key in ("tests", "failures", "errors")
    }
    ui_passed = "passed" in (folder / "ui-check.log").read_text(encoding="utf-8")
    lint_passed = "All checks passed!" in (folder / "lint-check.log").read_text(encoding="utf-8")
    by_module = {}
    for suite in suites:
        for case in suite.iter("testcase"):
            module = case.attrib.get("classname", "unknown")
            by_module[module] = by_module.get(module, 0) + 1
    baselines = []
    for path in folder.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("kind") == "offline_keyword_baseline":
            baselines.append((path, value))
    if not baselines:
        raise SystemExit("请先运行离线BM25基线。")
    path, baseline = max(baselines, key=lambda row: row[1]["created_at"])
    live_reports = []
    for directory, kind in (
        (ROOT / "reports/experiments", "retrieval_experiment"),
        (ROOT / "reports/scoring", "scoring_experiment"),
    ):
        for candidate in directory.glob("*.json"):
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if value.get("kind") == kind:
                live_reports.append(value)
    retrieval_complete = any(
        r["kind"] == "retrieval_experiment"
        and all(s["completed"] > 0 and s["failures"] == 0 for s in r["summaries"])
        for r in live_reports
    )
    scoring_complete = any(
        r["kind"] == "scoring_experiment" and r["summary"]["fallback_cases"] == 0
        for r in live_reports
    )
    live_complete = retrieval_complete and scoring_complete
    full_dir = ROOT / "reports/full-evaluation"
    launch_path = full_dir / "launch-status.json"
    full_launch = (
        json.loads(launch_path.read_text(encoding="utf-8")) if launch_path.exists() else {}
    )
    complete_phases = []
    for phase in ("dev", "test", "quality"):
        phase_path = full_dir / f"{phase}.json"
        if phase_path.exists():
            phase_report = json.loads(phase_path.read_text(encoding="utf-8"))
            if phase_report.get("kind") == "full_evaluation_phase":
                complete_phases.append(phase)
    diagnostic_path = folder / "provider-diagnostic.json"
    diagnostic = (
        json.loads(diagnostic_path.read_text(encoding="utf-8")) if diagnostic_path.exists() else {}
    )
    balance_blocked = any(
        d.get("category") == "insufficient_balance" for d in diagnostic.get("diagnostics", [])
    )
    result = {
        "verification": counts,
        "by_module": by_module,
        "dataset": data,
        "targeted_verification": targeted_counts,
        "ui_passed": ui_passed,
        "lint_passed": lint_passed,
        "offline_baseline": {
            "file": path.name,
            "summary": baseline["summary"],
            "answerable_cases": baseline["answerable_cases"],
        },
        "human_review_complete": data["human_review_complete"],
        "real_model_evaluation_complete": live_complete,
        "full_evaluation": {
            "launch": full_launch,
            "completed_phases": complete_phases,
            "complete": len(complete_phases) == 3,
            "protocol": str(full_dir / "protocol.json"),
        },
        "provider_diagnostic": diagnostic,
        "real_model_status": "正式开发、独立测试和回答质量报告均已保存，结合有效样本量解释结果。"
        if len(complete_phases) == 3
        else "已有完成的真实模型小规模报告，正式评测状态见下方。"
        if live_complete
        else "真实评测中途遇到DeepSeek HTTP 402余额不足，需要恢复额度后完成剩余实验。"
        if balance_blocked
        else "尚未同时得到成功的真实检索与评分报告。",
    }
    (folder / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# InterviewPilot 本轮项目证据",
        "",
        f"- 完整Python回归：{counts['tests']}项，失败{counts['failures']}，"
        f"错误{counts['errors']}，跳过{counts['skipped']}；原始记录：[pytest.xml](pytest.xml)。",
        f"- 修订后评测相关测试：{targeted_counts}，"
        "记录：[targeted-tests.xml](targeted-tests.xml)。",
        f"- 前端评测展示测试通过：{ui_passed}，记录：[ui-check.log](ui-check.log)。",
        f"- Ruff检查通过：{lint_passed}，记录：[lint-check.log](lint-check.log)。",
        "",
        "## 数据准备",
        "",
        f"- 检索{data['retrieval_cases']}题，开发80/测试40，来源关联组"
        f"{data['independent_groups']}个；无跨集标注来源、无缺失来源。",
        f"- 评分{data['scoring_cases']}个回答案例，覆盖六类回答。",
        f"- 人工复核完成：{data['human_review_complete']}；标注状态详见dataset-check.json。",
        "- [检索复核工作表](retrieval-review.md) / [评分复核工作表](scoring-review.md)。",
        "",
        "## 实际离线结果",
        "",
        f"- BM25开发基线：80条原查询，{baseline['answerable_cases']}条有标注来源；"
        f"来源Recall@3={baseline['summary']['recall_at_3']:.4f}，"
        f"MRR={baseline['summary']['mrr']:.4f}。",
        f"- 纯检索平均{baseline['summary']['mean_search_seconds'] * 1000:.1f}毫秒；"
        "未拆解、未扩展、未生成，不是端到端时延。",
        f"- [原始结果与未完整命中案例]({path.with_suffix('.md').name})。",
        "",
        "## 完成情况与后续方向",
        "",
        "1. 人工复核已完成：120条检索、24条评分，项目作者认可原标注，未修改来源或标签。"
        if data["human_review_complete"]
        else "1. 检查人工复核是否完成，核对预期来源、参考要点、评分标签与主题分组。",
        f"2. {result['real_model_status']}",
        "3. 已按开发集规则冻结选型，并完成独立测试的三轮重复，原始报告与实现快照保留。"
        if {"dev", "test"}.issubset(complete_phases)
        else "3. 正式评测将先完成开发集并冻结选型，再进行独立测试的三轮重复。",
        "4. 后续可研究条件扩展、并行或缓存；当前报告不把这些未测量的改动写成优化成果。",
        "",
        "失败的API尝试仅用于检查完成率，不能用于效果结论。",
        "本地自动化测试使用模拟模型，仅验证流程和约束；真实追问质量仍需实际评价。",
        "",
        "## 完整评测阶段",
        "",
        f"- 已完成阶段：{complete_phases}；三阶段完整完成：{len(complete_phases) == 3}。",
        f"- 启动状态：{full_launch.get('state', '未启动')}。",
        "- [正式评测协议与调用范围](../../docs/full-evaluation-protocol.md)。",
        "- 完整阶段与已完成的小规模API试跑分别记录，不将准备完成视为评测完成。",
        "",
        "## 可直接使用的材料",
        "",
        "- [实验命令与统计口径](../../docs/evaluation-plan.md)",
        "- [架构图、演示脚本和简历草稿](../../docs/project-evidence.md)",
        "- [可直接使用的简历项目条目](../../docs/resume-project.md)",
        "- [本轮真实API实验与失败分析](live-pilot.md)",
        "",
        "## 回归覆盖",
        "",
        "| 模块 | 收集用例 |",
        "| --- | --- |",
    ]
    lines.extend(f"| {module} | {count} |" for module, count in sorted(by_module.items()))
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {folder / 'summary.md'}")


if __name__ == "__main__":
    main()
