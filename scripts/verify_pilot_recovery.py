"""Audit a merged scoring report's case coverage, retained attempts and excerpts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    merged = json.loads(args.report.read_text(encoding="utf-8"))
    provenance = merged["recovery"]
    original = json.loads(Path(provenance["original_report"]).read_text(encoding="utf-8"))
    retry = json.loads(Path(provenance["retry_report"]).read_text(encoding="utf-8"))
    old_rows = {r["case_id"]: r for r in original["records"]}
    new_rows = {r["case_id"]: r for r in retry["records"]}
    rows = {r["case_id"]: r for r in merged["records"]}
    if len(rows) != len(merged["records"]) or set(rows) != set(old_rows):
        raise ValueError("合并报告包含缺失或重复案例。")
    if set(new_rows) != set(provenance["retried_case_ids"]):
        raise ValueError("重试报告的案例集合与重试清单不符。")
    if merged["models"] != original["models"] or merged["models"] != retry["models"]:
        raise ValueError("报告模型不一致。")
    for case_id, record in rows.items():
        expected = new_rows[case_id] if case_id in new_rows else old_rows[case_id]
        if {k: v for k, v in record.items() if k != "source_report_id"} != expected:
            raise ValueError("合并时改变了原始评分记录。")
        for point in record["evaluation"]["point_reviews"]:
            for evidence in point["evidence"]:
                if evidence["quote"] not in record["answers"][evidence["answer_index"] - 1]:
                    raise ValueError("证据原文与对应回答不匹配。")
    result = {
        "report": str(args.report.resolve()),
        "independent_cases": len(rows),
        "retained_cases": len(provenance["retained_case_ids"]),
        "retried_cases": len(provenance["retried_case_ids"]),
        "original_records_unchanged": True,
        "unique_complete_case_coverage": True,
        "models_match": True,
        "all_evidence_excerpts_match": True,
        "fallback_cases": merged["summary"]["fallback_cases"],
    }
    path = ROOT / "reports/evidence/scoring-recovery-check.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
