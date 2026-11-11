"""Repeatable ablations, grouped holdouts and paired, case-level comparisons."""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from .benchmark import VARIANTS, _summary, load_dataset, run_benchmark

EVIDENCE_DATASET = Path(__file__).resolve().parents[2] / "data/evaluation/knowledge-v2.json"
COMPARISONS = (
    ("vector", "hybrid"),
    ("hybrid", "mqe"),
    ("hybrid", "hyde"),
    ("hybrid", "hybrid_reranked"),
    ("hybrid", "expanded"),
    ("expanded", "reranked"),
)


def validate_holdout(cases: list[dict]) -> None:
    """Equivalent query groups and labelled evidence cannot cross the holdout."""
    groups: dict[str, set] = defaultdict(set)
    sources: dict[str, set] = defaultdict(set)
    for case in cases:
        split = case.get("split")
        if split not in {"dev", "test"}:
            raise ValueError("分组实验要求每个案例填写 dev/test split。")
        groups[case.get("group", case["id"])].add(split)
        for source in case["expected_sources"]:
            sources[source].add(split)
    if any(len(v) > 1 for v in groups.values()) or any(len(v) > 1 for v in sources.values()):
        raise ValueError("开发集与测试集存在相同分组或标注来源，请按知识主题重新分组。")


def select_cases(data: dict, split: str | None, limit: int | None, seed: int) -> list[dict]:
    if split not in {None, "dev", "test"}:
        raise ValueError("split 只能是 dev/test。")
    if split or any("split" in c for c in data["cases"]):
        validate_holdout(data["cases"])
    cases = [c for c in data["cases"] if split is None or c.get("split") == split]
    if not cases:
        raise ValueError("所选分组没有案例。")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(cases)):
        raise ValueError("案例数量超出所选分组范围。")
    # Round-robin categories instead of taking only the first easy questions.
    buckets: dict[str, list] = defaultdict(list)
    rng = random.Random(seed)
    for case in cases:
        buckets[case.get("category", "other")].append(case)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while any(buckets.values()):
        for category in sorted(buckets):
            if buckets[category]:
                selected.append(buckets[category].pop())
    return selected[:limit] if limit else selected


def require_reviewed(data: dict, cases: list[dict]) -> None:
    if data.get("label_status") != "human_reviewed":
        raise ValueError("正式实验需要 human_reviewed 标注状态。")
    for case in cases:
        review = case.get("review", {})
        if (
            not isinstance(review, dict)
            or review.get("approved") is not True
            or not isinstance(review.get("reviewer"), str)
            or not review["reviewer"].strip()
            or not isinstance(review.get("reviewed_at"), str)
            or not review["reviewed_at"].strip()
        ):
            raise ValueError(f"案例 {case['id']} 缺少人工复核记录。")
        try:
            datetime.fromisoformat(review["reviewed_at"])
        except ValueError as error:
            raise ValueError(f"案例 {case['id']} 的复核日期需要为ISO日期或时间。") from error


def paired_comparisons(records: list[dict], *, seed: int = 42) -> list[dict]:
    """Average repeats per case; bootstrap knowledge groups, never repeated rows."""
    metrics = (
        "recall_at_k",
        "reciprocal_rank",
        "faithfulness",
        "answer_correctness",
        "correct_refusal",
        "false_refusal",
        "elapsed_seconds",
    )
    output = []
    for baseline, candidate in COMPARISONS:
        for metric in metrics:
            pairs: dict[tuple, dict] = defaultdict(dict)
            for row in records:
                if row["variant"] in {baseline, candidate} and not row["error"]:
                    value = row.get(metric)
                    if value is not None:
                        pairs[(row["case_id"], row.get("repeat", 1))][row["variant"]] = value
            by_case: dict[str, list] = defaultdict(list)
            for (case_id, _repeat), pair in pairs.items():
                if baseline in pair and candidate in pair:
                    by_case[case_id].append(pair[candidate] - pair[baseline])
            if not by_case:
                continue
            case_groups = {r["case_id"]: r.get("group") or r["case_id"] for r in records}
            by_group: dict[str, list] = defaultdict(list)
            deltas = {case: mean(values) for case, values in by_case.items()}
            for case, delta in deltas.items():
                by_group[case_groups[case]].append(delta)
            groups = sorted(by_group)
            interval = None
            if len(groups) >= 10:
                rng = random.Random(seed)
                samples = sorted(
                    mean(v for group in rng.choices(groups, k=len(groups)) for v in by_group[group])
                    for _ in range(1000)
                )
                interval = [round(samples[24], 6), round(samples[974], 6)]
            output.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "delta": round(mean(deltas.values()), 6),
                    "paired_cases": len(deltas),
                    "paired_observations": sum(len(values) for values in by_case.values()),
                    "independent_groups": len(groups),
                    "bootstrap_95_ci": interval,
                    "direction": "lower"
                    if metric in {"elapsed_seconds", "false_refusal"}
                    else "higher",
                }
            )
    return output


def write_report(report: dict, output_dir: Path, markdown: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report['id']}.json"
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    path.with_suffix(".md").write_text(markdown, encoding="utf-8")
    return path


def run_experiment(
    rag,
    *,
    dataset_path: Path = EVIDENCE_DATASET,
    variants=None,
    split: str | None = "dev",
    limit: int | None = None,
    repeats: int = 3,
    seed: int = 42,
    reviewed_only: bool = False,
    answer_judge: bool = True,
    output_dir: Path = Path("reports/experiments"),
    service=None,
    progress=None,
    judge=None,
) -> dict:
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError("重复次数需要为1到10。")
    data = load_dataset(dataset_path)
    cases = select_cases(data, split, limit, seed)
    if reviewed_only:
        require_reviewed(data, cases)
    variants = list(VARIANTS) if variants is None else list(variants)
    if (
        not variants
        or len(set(variants)) != len(variants)
        or any(v not in VARIANTS for v in variants)
    ):
        raise ValueError("请选择有效且不重复的检索方案。")
    experiment_id = uuid.uuid4().hex
    work = output_dir / experiment_id
    work.mkdir(parents=True, exist_ok=True)
    selected = work / "selected-dataset.json"
    selected.write_text(
        json.dumps(data | {"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reports = []
    records = []
    for repeat in range(repeats):
        # Offsetting each run's initial variant balances first-run cold starts.
        order = variants[repeat % len(variants) :] + variants[: repeat % len(variants)]
        callback = None
        if progress:

            def callback(done, total, case, variant, run=repeat):
                progress(run * total + done, repeats * total, case, variant)

        report = run_benchmark(
            rag,
            dataset_path=selected,
            variants=order,
            service=service,
            output_dir=work,
            answer_judge=answer_judge,
            judge=judge,
            progress=callback,
        )
        reports.append(report)
        records.extend(row | {"repeat": repeat + 1} for row in report["records"])
    if len({r["corpus_sha256"] for r in reports}) != 1:
        raise ValueError("重复实验期间题库发生变化，各轮报告已保留，但不生成合并结论。")
    summaries = [_summary(variant, records) for variant in variants]
    categories = sorted({row["category"] for row in records})
    reviewed = data.get("label_status") == "human_reviewed"
    if reviewed:
        try:
            require_reviewed(data, cases)
        except ValueError:
            reviewed = False
    result = {
        "id": experiment_id,
        "kind": "retrieval_experiment",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": reports[0]["dataset"]
        | {
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "selected_sha256": reports[0]["dataset"]["sha256"],
            "total_cases": len(data["cases"]),
            "split": split,
            "seed": seed,
        },
        "repeats": repeats,
        "models": reports[0]["models"],
        "config": reports[0]["config"],
        "options": reports[0]["options"],
        "corpus_sha256": reports[0]["corpus_sha256"],
        "runs": [
            {"id": r["id"], "summaries": r["summaries"], "shared_plans": r["shared_plans"]}
            for r in reports
        ],
        "summaries": summaries,
        "records": records,
        "by_category": {
            c: [_summary(v, [r for r in records if r["category"] == c]) for v in variants]
            for c in categories
        },
        "comparisons": paired_comparisons(records, seed=seed),
        "claim_ready": bool(
            reviewed
            and split == "test"
            and repeats >= 3
            and len({c.get("group", c["id"]) for c in cases}) >= 10
            and all(not r["error"] and not r["result"]["warnings"] for r in records)
        ),
        "notes": reports[0]["notes"]
        + [
            "重复次数不增加独立问题样本量。差值为候选减基线，仅比较双方成功的相同案例与轮次。",
            "先对每题的配对差值求重复均值，再按知识分组进行bootstrap；少于10组不报告区间。",
            "失败在端到端Recall中记0；成功样本效果均值需与完成率一起看。",
            "claim_ready只是复核/测试集/重复次数/样本/降级条件检查，不证明提升或统计显著性。",
            "测试集用于参数冻结后的最终评估，不能反复查看后继续针对它调参。",
        ],
    }
    write_report(result, output_dir, experiment_markdown(result))
    return result


def experiment_markdown(report: dict) -> str:
    lines = [
        "# 检索消融实验",
        "",
        f"- 数据集：{report['dataset']['name']}",
        f"- 标注状态：{report['dataset']['label_status']}；分组：{report['dataset']['split']}",
        f"- 独立案例：{report['dataset']['selected_cases']}；重复次数：{report['repeats']}",
        f"- 简历证据前置条件通过：{report['claim_ready']}（不等于证明提升）",
        "",
        "| 方案 | 完成 | Recall@3 | 端到端Recall | MRR | 正确拒答 | 平均秒 | P95秒 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in report["summaries"]:
        lines.append(
            f"| {s['label']} | {s['completed']}/{s['cases']} | {s['recall_at_k']} | "
            f"{s['end_to_end_recall_at_k']} | {s['mrr']} | {s['correct_refusal_rate']} | "
            f"{s['mean_seconds']} | {s['p95_seconds']} |"
        )
    lines += [
        "",
        "## 配对差值：候选减基线",
        "",
        "| 基线 → 候选 | 指标 | 差值 | 配对案例/组 | 95%区间 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in report["comparisons"]:
        lines.append(
            f"| {c['baseline']} → {c['candidate']} | {c['metric']} | {c['delta']} | "
            f"{c['paired_cases']}/{c['independent_groups']} | {c['bootstrap_95_ci']} |"
        )
    lines += ["", "## 分类表现", ""]
    for category, summaries in report["by_category"].items():
        lines.append(f"### {category}")
        for s in summaries:
            lines.append(
                f"- {s['label']}：Recall={s['recall_at_k']}，"
                f"拒答={s['correct_refusal_rate']}，失败={s['failures']}"
            )
    lines += ["", "## 解释边界", "", *[f"- {note}" for note in report["notes"]]]
    return "\n".join(lines) + "\n"
