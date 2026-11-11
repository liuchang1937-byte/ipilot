"""Retry only failed scoring cases and preserve each retained attempt's provenance."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from diagnose_scoring_provider import DiagnosticProvider

from techrag import TechRAG
from techrag.experiments import write_report
from techrag.providers import DashScopeDeepSeekProvider
from techrag.scoring_benchmark import SCORING_DATASET, run_scoring_benchmark, scoring_summary

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "reports/scoring/c00a40431f994623af5b105b2c9007bf.json"


def main():
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    data = json.loads(SCORING_DATASET.read_text(encoding="utf-8"))
    if hashlib.sha256(SCORING_DATASET.read_bytes()).hexdigest() != original["dataset"]["sha256"]:
        raise SystemExit("评分标注已改变，不能合并原试验；请独立重新评估。")
    failed_ids = {
        r["case_id"] for r in original["records"] if r["evaluation"]["scoring_method"] != "model"
    }
    if not failed_ids:
        raise SystemExit("原报告没有需要补跑的降级案例。")
    work = ROOT / "reports/scoring" / f"recovery-{uuid.uuid4().hex}"
    work.mkdir(parents=True)
    subset = work / "retry-dataset.json"
    subset.write_text(
        json.dumps(
            data | {"cases": [c for c in data["cases"] if c["id"] in failed_ids]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    provider = DiagnosticProvider(DashScopeDeepSeekProvider())

    def progress(done, total, case, _variant):
        print(f"[{done}/{total}] {case} scoring-recovery", flush=True)
        if provider.diagnostics and provider.diagnostics[-1]["category"] in {
            "insufficient_balance",
            "authentication",
        }:
            write_report(
                {"id": work.name, "diagnostics": provider.diagnostics},
                work,
                "# 接口仍不可用，已停止后续重试\n",
            )
            raise RuntimeError("接口仍有余额/认证问题，已停止继续调用。")

    with TechRAG(_chat_model=provider, _embedder=provider) as rag:
        retry = run_scoring_benchmark(
            rag, dataset_path=subset, repeats=1, output_dir=work, progress=progress
        )
    if retry["models"] != original["models"]:
        raise SystemExit("评分模型不同，保留新结果但不合并旧报告。")
    attempts = {r["case_id"]: r | {"source_report_id": original["id"]} for r in original["records"]}
    for record in retry["records"]:
        if record["case_id"] not in failed_ids:
            raise ValueError("重试结果超出授权的失败案例范围。")
        attempts[record["case_id"]] = record | {"source_report_id": retry["id"]}
    records = [attempts[c["id"]] for c in data["cases"]]
    report = original | {
        "id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "records": records,
        "summary": scoring_summary(records),
        "by_category": {
            c: scoring_summary([r for r in records if r["category"] == c])
            for c in sorted({r["category"] for r in records})
        },
        "recovery": {
            "original_report": str(ORIGINAL),
            "retry_report": str(work / (retry["id"] + ".json")),
            "retained_case_ids": sorted(set(attempts) - failed_ids),
            "retried_case_ids": sorted(failed_ids),
        },
        "notes": original["notes"]
        + [
            "保留原10条模型结果，仅重试原14条接口降级案例；旧失败与重试报告均保留。",
            "每个案例只保留一次有效尝试用于统计，不将跨批次补测当作重复稳定性实验。",
            "原10条曾与检索进程并行，恢复14条串行；混合批次耗时不作性能结论。",
        ],
    }
    lines = [
        "# 逐点评分恢复后的完整报告",
        "",
        "```json",
        json.dumps(report["summary"], ensure_ascii=False, indent=2),
        "```",
        "",
        *[f"- {note}" for note in report["notes"]],
        "",
        "## 逐题证据",
        "",
    ]
    for record in records:
        lines += [
            f"### {record['case_id']}",
            f"- 原始报告：{record['source_report_id']}",
            f"- 回答：{record['answers']}",
            f"- 种子标签：{record['expected_statuses']}",
            "```json",
            json.dumps(record["evaluation"]["point_reviews"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    path = write_report(report, ROOT / "reports/scoring", "\n".join(lines))
    diagnostic = {
        "scoring_method": "model" if not report["summary"]["fallback_cases"] else "fallback",
        "diagnostics": provider.diagnostics,
        "recovery_report": str(path),
        "checked_at": datetime.now(UTC).isoformat(),
    }
    (ROOT / "reports/evidence/provider-diagnostic.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
