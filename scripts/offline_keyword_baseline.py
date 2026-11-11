"""Measure real BM25 recall on the dev seed set without any model requests."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from techrag import TechRAG, TechRAGConfig
from techrag.benchmark import load_dataset
from techrag.experiments import EVIDENCE_DATASET, select_cases, write_report

ROOT = Path(__file__).resolve().parents[1]


class NoExternalCalls:
    def embed(self, _texts):
        raise AssertionError("离线基线不允许调用Embedding。")

    def complete(self, _messages):
        raise AssertionError("离线基线不允许调用LLM。")


def main():
    data = load_dataset(EVIDENCE_DATASET)
    cases = select_cases(data, "dev", None, 42)
    records = []
    provider = NoExternalCalls()
    with TechRAG(
        TechRAGConfig(db_path=ROOT / "data/chroma"), _embedder=provider, _chat_model=provider
    ) as rag:
        corpus = sorted((str(r[5]), str(r[2]), str(r[6])) for r in rag.database.chunk_rows())
        available = {row[0] for row in corpus}
        missing = {s for c in cases for s in c["expected_sources"]} - available
        if missing:
            raise SystemExit(f"未入库的标注来源：{sorted(missing)}")
        for case in cases:
            started = time.perf_counter()
            results = rag.search(
                case["question"],
                mode="keyword",
                top_k=3,
                enable_mqe=False,
                enable_hyde=False,
                enable_rerank=False,
            )
            elapsed = time.perf_counter() - started
            sources = list(dict.fromkeys(r.source for r in results))
            expected = set(case["expected_sources"])
            ranks = [i for i, source in enumerate(sources, 1) if source in expected]
            records.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "expected_sources": sorted(expected),
                    "sources": sources,
                    "elapsed_seconds": elapsed,
                    "recall_at_3": len(expected.intersection(sources)) / len(expected)
                    if expected
                    else None,
                    "reciprocal_rank": 1 / min(ranks) if ranks else 0 if expected else None,
                }
            )
    answerable = [row for row in records if row["expected_sources"]]
    report = {
        "id": uuid.uuid4().hex,
        "kind": "offline_keyword_baseline",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": data["name"],
            "label_status": data["label_status"],
            "split": "dev",
            "selected_cases": len(cases),
            "sha256": hashlib.sha256(EVIDENCE_DATASET.read_bytes()).hexdigest(),
        },
        "corpus_sha256": hashlib.sha256(
            json.dumps(corpus, ensure_ascii=False).encode()
        ).hexdigest(),
        "models_called": 0,
        "answerable_cases": len(answerable),
        "summary": {
            "recall_at_3": mean(r["recall_at_3"] for r in answerable),
            "mrr": mean(r["reciprocal_rank"] for r in answerable),
            "mean_search_seconds": mean(r["elapsed_seconds"] for r in records),
        },
        "records": records,
        "notes": [
            "真实索引上的原查询BM25检索，未拆解，整题只取3个完整父块。",
            "未运行生成、扩展、向量或拒答，不报告回答质量、拒答率或端到端时延。",
            "复合题指标不同于按子问题各取3条的线上流程，不能直接与其宣称提升。",
            "开发集与种子来源标注结果，不是最终测试集或人工金标准。",
        ],
    }
    lines = [
        "# 离线BM25开发基线",
        "",
        "```json",
        json.dumps(
            {k: report[k] for k in ("dataset", "models_called", "answerable_cases", "summary")},
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        *[f"- {note}" for note in report["notes"]],
        "",
        "## 未完整命中案例",
        "",
    ]
    for row in answerable:
        if row["recall_at_3"] < 1:
            lines += [
                f"- {row['id']}：{row['question']}；Recall={row['recall_at_3']}；"
                f"目标={row['expected_sources']}；实际={row['sources']}"
            ]
    path = write_report(report, ROOT / "reports/evidence", "\n".join(lines) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
