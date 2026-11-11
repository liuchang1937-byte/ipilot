"""Versioned, source-labelled end-to-end knowledge retrieval comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .answer_evaluation import ANSWER_METRICS, evaluate_answer
from .knowledge import KnowledgeOptions, KnowledgeService
from .rag import TechRAG
from .timing import collect_timings, timed_stage

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data/evaluation/knowledge-v1.json"
VARIANTS = {
    "vector": ("向量检索", KnowledgeOptions(mqe=False, hyde=False, rerank=False, mode="vector")),
    "hybrid": ("混合检索", KnowledgeOptions(mqe=False, hyde=False, rerank=False)),
    "mqe": ("混合 + MQE", KnowledgeOptions(mqe=True, hyde=False, rerank=False)),
    "hyde": ("混合 + HyDE", KnowledgeOptions(mqe=False, hyde=True, rerank=False)),
    "hybrid_reranked": ("混合 + 精排", KnowledgeOptions(mqe=False, hyde=False, rerank=True)),
    "expanded": ("混合 + MQE/HyDE", KnowledgeOptions(rerank=False)),
    "reranked": ("混合 + MQE/HyDE + 精排", KnowledgeOptions()),
}
DEFAULT_VARIANTS = ("vector", "hybrid", "expanded", "reranked")
TIMING_STAGES = (
    "decomposition",
    "retrieval",
    "mqe",
    "hyde",
    "embedding",
    "vector_search",
    "keyword_search",
    "rerank",
    "generation",
    "pipeline",
    "total",
    "judge",
)


def load_dataset(path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("评测集必须包含 cases 数组。")
    ids = set()
    if not 1 <= len(data["cases"]) <= 500:
        raise ValueError("评测集需要包含 1 到 500 个案例。")
    for case in data["cases"]:
        if not isinstance(case, dict):
            raise ValueError("评测案例需要为对象。")
        case_id, question = case.get("id"), case.get("question")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in ids
            or not isinstance(question, str)
            or not question.strip()
            or len(question) > 4000
        ):
            raise ValueError("评测案例 ID 必须唯一且问题不能为空。")
        ids.add(case_id)
        if "split" in case and (
            not isinstance(case["split"], str) or case["split"] not in {"dev", "test"}
        ):
            raise ValueError("案例 split 只能是 dev 或 test。")
        if "category" in case and (
            not isinstance(case["category"], str) or not case["category"].strip()
        ):
            raise ValueError("案例 category 必须是非空字符串。")
        if "group" in case and (not isinstance(case["group"], str) or not case["group"].strip()):
            raise ValueError("案例 group 必须是非空字符串。")
        sources, domains = case.get("expected_sources"), case.get("domains", [])
        if (
            not isinstance(sources, list)
            or any(not isinstance(s, str) or not s for s in sources)
            or len(sources) != len(set(sources))
            or not isinstance(domains, list)
            or any(not isinstance(d, str) or not d for d in domains)
            or type(case.get("answerable")) is not bool
            or case["answerable"] != bool(sources)
        ):
            raise ValueError("每个案例需要有效来源、方向和 answerable 标记。")
        points = case.get("reference_points", [])
        answer = case.get("reference_answer")
        grades = case.get("source_relevance", {})
        if (
            not isinstance(points, list)
            or len(points) > 30
            or any(not isinstance(p, str) or not p.strip() or len(p) > 4000 for p in points)
            or (
                answer is not None
                and (not isinstance(answer, str) or not answer.strip() or len(answer) > 12000)
            )
            or not isinstance(grades, dict)
            or any(
                s not in sources or type(g) is not int or not 1 <= g <= 3 for s, g in grades.items()
            )
        ):
            raise ValueError("参考答案、评分点或来源相关程度无效（相关程度为1到3）。")
    return data


class BenchmarkBlockedError(ValueError):
    """Permanent provider failure; existing checkpoint results are retained."""


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _checkpoint_key(case_id: str, variant: str) -> str:
    return hashlib.sha256((case_id + "\0" + variant).encode()).hexdigest()


class _Counted:
    def __init__(self, provider, counts: dict[str, int], errors=None):
        self.provider, self.counts = provider, counts
        self.errors = errors if errors is not None else []

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def _call(self, name, *args):
        try:
            return getattr(self.provider, name)(*args)
        except Exception as error:
            status = getattr(error, "status_code", None)
            self.errors.append(
                {"type": type(error).__name__, "status": status if type(status) is int else None}
            )
            raise

    @timed_stage("embedding")
    def embed(self, texts):
        self.counts["embedding_calls"] += 1
        self.counts["embedding_texts"] += len(texts)
        return self._call("embed", texts)

    def complete(self, messages):
        self.counts["chat_calls"] += 1
        return self._call("complete", messages)

    @timed_stage("rerank")
    def rank(self, query, documents):
        self.counts["rerank_calls"] += 1
        self.counts["rerank_pairs"] += len(documents)
        return self._call("rank", query, documents)


def _metrics(case, result):
    k = result.get("options", {}).get("top_k", 3)
    lists = [
        list(dict.fromkeys(s.get("trace", {}).get("selected_sources", [])))[:k]
        for s in result["sections"]
        if s.get("kind", "knowledge") == "knowledge"
    ]
    ranked = list(dict.fromkeys(source for sources in lists for source in sources))
    expected = set(case["expected_sources"])
    hits = expected.intersection(ranked)
    rank = next((i for i, source in enumerate(ranked, 1) if source in expected), None)
    # Best reciprocal rank across subquestions; never flatten incomparable rankings.
    rr = max(
        (1 / i for sources in lists for i, source in enumerate(sources, 1) if source in expected),
        default=0.0,
    )
    refused = bool(result["sections"]) and all(s["insufficient"] for s in result["sections"])
    grades = {s: case.get("source_relevance", {}).get(s, 1) for s in expected}
    ideal = sum(
        (2**g - 1) / math.log2(i + 2)
        for i, g in enumerate(sorted(grades.values(), reverse=True)[:k])
    )
    precision = mean(sum(s in expected for s in sources) / k for sources in lists) if lists else 0
    ndcg = (
        mean(
            sum((2 ** grades.get(s, 0) - 1) / math.log2(i + 2) for i, s in enumerate(sources))
            / ideal
            for sources in lists
        )
        if lists and ideal
        else 0
    )
    return {
        "top_k": k,
        "retrieved_sources": ranked,
        "recall_at_k": len(hits) / len(expected) if expected else None,
        "precision_at_k": precision if expected else None,
        "ndcg_at_k": ndcg if expected else None,
        "hit": bool(rank) if expected else None,
        "reciprocal_rank": rr if expected else None,
        "all_sources_found": hits == expected if expected else None,
        "correct_refusal": refused and not result["sources"] if not expected else None,
        "false_refusal": refused if expected else None,
    }


def run_benchmark(
    rag: TechRAG,
    *,
    dataset_path: Path = DEFAULT_DATASET,
    variants: list[str] | None = None,
    limit: int | None = None,
    output_dir: Path = Path("reports/evaluations"),
    service: KnowledgeService | None = None,
    progress=None,
    answer_judge: bool = True,
    judge=None,
    retrieval_only: bool = False,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    data = load_dataset(dataset_path)
    answer_judge = answer_judge and not retrieval_only
    variants = list(DEFAULT_VARIANTS) if variants is None else variants
    if (
        not variants
        or len(set(variants)) != len(variants)
        or any(v not in VARIANTS for v in variants)
    ):
        raise ValueError("请选择有效且不重复的检索方案。")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(data["cases"])):
        raise ValueError("案例数量超出评测集范围。")
    cases = data["cases"][:limit] if limit else data["cases"]
    corpus = sorted((str(row[5]), str(row[2]), str(row[6])) for row in rag.database.chunk_rows())
    available = {source for source, _, _ in corpus}
    missing = sorted({s for c in cases for s in c["expected_sources"]} - available)
    if missing:
        raise ValueError("评测标注来源尚未入库：" + "、".join(missing))
    embedder, chat_model = rag.embedder, rag.chat_model
    service = service or KnowledgeService(rag)
    reranker = service.reranker
    corpus_hash = hashlib.sha256(json.dumps(corpus, ensure_ascii=False).encode()).hexdigest()
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    config = asdict(rag.config) | {
        "db_path": str(rag.config.db_path),
        "reranker_dir": str(rag.config.reranker_dir),
    }
    models = {
        "chat": getattr(chat_model, "chat_model", type(chat_model).__name__),
        "embedding": getattr(embedder, "embedding_model", type(embedder).__name__),
        "reranker": type(reranker).__name__,
    }
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        files = [
            Path(__file__).with_name(name)
            for name in (
                "benchmark.py",
                "knowledge.py",
                "rag.py",
                "search.py",
                "providers.py",
                "answer_evaluation.py",
                "experiments.py",
            )
        ]
        manifest = {
            "dataset_sha256": dataset_hash,
            "corpus_sha256": corpus_hash,
            "models": models,
            "config": config,
            "variants": variants,
            "options": {v: asdict(VARIANTS[v][1]) for v in variants},
            "limit": limit,
            "answer_judge": answer_judge,
            "judge_identity": (
                getattr(judge or chat_model, "chat_model", type(judge or chat_model).__name__)
                if answer_judge
                else None
            ),
            "retrieval_only": retrieval_only,
            "code_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest(),
        }
        metadata = checkpoint_dir / "manifest.json"
        if metadata.exists():
            if json.loads(metadata.read_text(encoding="utf-8")) != manifest:
                raise ValueError("断点的数据、题库、模型、配置或代码已改变，不能混合续跑。")
        else:
            _atomic_json(metadata, manifest)
    records = []
    plans = []
    errors = []
    reused_records = 0
    total = len(cases) * len(variants)
    benchmark_started = time.perf_counter()
    judge = judge or chat_model

    def stop_on_permanent_error():
        blocked = next((e for e in errors if e["status"] in {401, 402, 403}), None)
        if blocked:
            reason = "模型API余额不足" if blocked["status"] == 402 else "模型API认证或权限失败"
            if checkpoint_dir is not None:
                _atomic_json(
                    checkpoint_dir / "blocked.json",
                    {
                        "reason": reason,
                        "status": blocked["status"],
                        "completed_records": len(records),
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                )
            raise BenchmarkBlockedError(reason + "，已停止后续调用；已保存的结果可断点续跑。")

    try:
        for number, case in enumerate(cases):
            plan_path = (
                checkpoint_dir / ("plan-" + _checkpoint_key(case["id"], "") + ".json")
                if checkpoint_dir is not None
                else None
            )
            cached_plan = (
                json.loads(plan_path.read_text(encoding="utf-8"))
                if plan_path is not None and plan_path.exists()
                else None
            )
            if cached_plan is not None and not cached_plan["warnings"]:
                plan_record = cached_plan
                plan = [(q["question"], q["kind"]) for q in plan_record["subquestions"]]
                plan_warnings = plan_record["warnings"]
                plan_seconds = plan_record["elapsed_seconds"]
            else:
                plan_counts = {"chat_calls": 0}
                rag.chat_model = _Counted(chat_model, plan_counts, errors)
                plan_started = time.perf_counter()
                plan_warnings = []
                plan = service._plan(case["question"], True, False, plan_warnings)
                plan_seconds = time.perf_counter() - plan_started
                stop_on_permanent_error()
                plan_record = {
                    "case_id": case["id"],
                    "subquestions": [{"question": q, "kind": k} for q, k in plan],
                    "warnings": plan_warnings,
                    "calls": plan_counts,
                    "elapsed_seconds": plan_seconds,
                }
                if plan_path is not None:
                    _atomic_json(plan_path, plan_record)
            plans.append(plan_record)
            # Rotate order to spread warm-up effects across variants.
            order = variants[number % len(variants) :] + variants[: number % len(variants)]
            for variant in order:
                record_path = (
                    checkpoint_dir / ("record-" + _checkpoint_key(case["id"], variant) + ".json")
                    if checkpoint_dir is not None
                    else None
                )
                if record_path is not None and record_path.exists():
                    saved = json.loads(record_path.read_text(encoding="utf-8"))
                    if saved.get("case_id") != case["id"] or saved.get("variant") != variant:
                        raise ValueError("断点记录的案例或方案不匹配。")
                    reusable = not saved["error"]
                    if answer_judge:
                        reusable = reusable and saved["answer_evaluation"]["status"] == "completed"
                    if retrieval_only:
                        reusable = reusable and not saved.get("result", {}).get("warnings")
                    if reusable:
                        records.append(saved)
                        reused_records += 1
                        if progress:
                            progress(len(records), total, case["id"], variant)
                        continue
                counts = dict.fromkeys(
                    [
                        "embedding_calls",
                        "embedding_texts",
                        "chat_calls",
                        "rerank_calls",
                        "rerank_pairs",
                    ],
                    0,
                )
                rag.embedder, rag.chat_model = (
                    _Counted(embedder, counts, errors),
                    _Counted(chat_model, counts, errors),
                )
                service.reranker = _Counted(reranker, counts, errors)
                started = time.perf_counter()
                row = {
                    "case_id": case["id"],
                    "question": case["question"],
                    "category": case.get("category", "other"),
                    "split": case.get("split"),
                    "group": case.get("group", case["id"]),
                    "variant": variant,
                    "expected_sources": case["expected_sources"],
                    "answerable": case["answerable"],
                    "reference_points": case.get("reference_points", []),
                    "reference_answer": case.get("reference_answer"),
                    "source_relevance": case.get("source_relevance", {}),
                }
                try:
                    with collect_timings() as timings:
                        kwargs = {"retrieval_only": True} if retrieval_only else {}
                        result = service.answer(
                            case["question"],
                            domains=tuple(case.get("domains", [])),
                            options=VARIANTS[variant][1],
                            prepared_plan=plan,
                            plan_warnings=plan_warnings,
                            **kwargs,
                        )
                    row.update(_metrics(case, result))
                    if retrieval_only:
                        row.update(correct_refusal=None, false_refusal=None)
                    row.update({"result": result, "error": None})
                except Exception:
                    # Provider exception text can contain secrets; keep a bounded public message.
                    row.update({"error": "本案例处理失败，未计入效果均值；失败率单独统计。"})
                pipeline_seconds = time.perf_counter() - started
                timing_values = {stage: timings.get(stage, 0.0) for stage in TIMING_STAGES}
                timing_values.update(
                    decomposition=plan_seconds,
                    pipeline=pipeline_seconds,
                    total=pipeline_seconds + plan_seconds,
                )
                quality = {
                    "status": "disabled" if not answer_judge else "unavailable",
                    "reason": ("已关闭答案评审。" if not answer_judge else "问答失败，未评审。"),
                    "metrics": dict.fromkeys(ANSWER_METRICS),
                }
                judge_counts = {"chat_calls": 0}
                if answer_judge and not row["error"]:
                    stop_on_permanent_error()
                    judge_started = time.perf_counter()
                    quality = evaluate_answer(case, result, _Counted(judge, judge_counts, errors))
                    timing_values["judge"] = time.perf_counter() - judge_started
                row.update(quality["metrics"])
                row.update(
                    {
                        "elapsed_seconds": round(pipeline_seconds + plan_seconds, 3),
                        "timings_seconds": {k: round(v, 6) for k, v in timing_values.items()},
                        "calls": dict(counts),
                        "answer_evaluation": quality,
                        "judge_calls": judge_counts,
                    }
                )
                records.append(row)
                if record_path is not None:
                    attempts = checkpoint_dir / "attempts"
                    attempts.mkdir(exist_ok=True)
                    _atomic_json(attempts / (uuid.uuid4().hex + ".json"), row)
                    _atomic_json(record_path, row)
                stop_on_permanent_error()
                if progress:
                    progress(len(records), total, case["id"], variant)
    finally:
        rag.embedder, rag.chat_model, service.reranker = embedder, chat_model, reranker
    report = {
        "schema_version": 2,
        "evaluation_mode": "retrieval_only" if retrieval_only else "end_to_end",
        "id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": data.get("name", dataset_path.stem),
            "label_status": data.get("label_status", "unreviewed"),
            "sha256": dataset_hash,
            "total_cases": len(data["cases"]),
            "selected_cases": len(cases),
        },
        "corpus_sha256": corpus_hash,
        "models": models,
        "config": config,
        "resume": {
            "reused_records": reused_records,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        },
        "options": {v: asdict(VARIANTS[v][1]) for v in variants},
        "min_rerank_score": rag.config.knowledge_min_rerank_score,
        "summaries": [_summary(v, records) for v in variants],
        "records": records,
        "shared_plans": plans,
        "answer_judge": {
            "enabled": answer_judge,
            "provider": type(judge).__name__,
            "uses_answer_model": judge is chat_model,
        },
        "benchmark_wall_seconds": round(time.perf_counter() - benchmark_started, 3),
        "notes": [
            "标注状态记录于dataset.label_status；人工复核不等于生产数据或独立多复核人基准。",
            "每个子问题 K=3；复合问题 Recall 统计全部标注来源覆盖率。"
            "MRR 为首个相关来源的最佳子问题倒数排名。",
            "Precision@K 固定以K为分母，缺少结果记为未命中；对子问题求平均。"
            "nDCG@K 对子问题求平均，使用整题来源标注，复合题不等同于逐子问标注的标准nDCG。"
            "默认二元相关性；source_relevance 可指定1到3级相关程度。",
            "标注仅列目标来源，未穷举全部等价证据；Precision和nDCG可能低估实际相关性。",
            "失败案例不计效果均值；请结合完成率、降级率和逐题结果判断。",
            "耗时包含查询拆解、扩展、检索、精排和生成，以及首次模型加载；方案执行顺序轮换。",
            "每题拆解只执行一次，所有方案复用同一子问题计划以控制变量。"
            "共有拆解耗时计入各方案耗时；共有调用单独记录，不重复计入方案调用量。",
            "分阶段耗时单位为秒，retrieval包含MQE、HyDE、Embedding、向量/BM25检索及融合，"
            "这些子阶段不能再与retrieval相加。total=共有拆解+pipeline，不含答案评审、"
            "HTTP排队或报告写盘；本接口非流式，不记录首Token延迟。judge耗时与调用单列。",
            "Faithfulness=有资料支持的事实数/事实数；引用准确性按事实—引用对统计，"
            "引用覆盖率=带引用的事实数/事实数。拒答无事实时这些指标为未评估。",
            "答案正确性为正确事实精确率与参考要点覆盖率的F1；无参考答案/要点时不评。"
            "相关性由评审模型按0到1打分；模型评审与生成分开调用，默认使用同一模型。"
            "程序校验连续原文、证据出处和结构，语义判断、事实拆分完整性仍需人工复核。"
            "评审失败不影响检索结果，不计入质量均值；每项指标单列有效样本数。",
            "调用量为应用层调用次数，不是 token 用量或供应商内部重试次数。"
            "引用存在不等于结论受支持，答案正确性仍需复核。",
            "断点复用相同输入、配置、代码与题库的已成功结果，失败项重新执行；"
            "各记录保留原耗时与调用量，benchmark_wall_seconds仅为当前启动的墙钟时间。",
            "retrieval_only只运行真实检索链路，未生成答案，不评拒答或回答质量；"
            "此模式下total是检索流程耗时，不能当作问答时延。"
            if retrieval_only
            else "end_to_end运行生成及引用校验，额外答案评审独立计量。",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{report['id']}.json"
    temporary = destination.with_suffix(".pending")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    destination.with_suffix(".md").write_text(benchmark_markdown(report), encoding="utf-8")
    return report


def _summary(variant, records):
    rows = [r for r in records if r["variant"] == variant]
    successful = [r for r in rows if not r["error"]]

    def average(key):
        values = [r[key] for r in successful if r.get(key) is not None]
        return round(mean(values), 4) if values else None

    times = sorted(r["elapsed_seconds"] for r in rows)
    quality_rows = [
        r for r in successful if r.get("answer_evaluation", {}).get("status") == "completed"
    ]

    def latency(stage):
        values = sorted(
            r["timings_seconds"][stage]
            for r in rows
            if stage != "judge" or r["judge_calls"]["chat_calls"]
        )
        if not values:
            return {"mean": None, "p50": None, "p95": None, "max": None, "samples": 0}
        return {
            "mean": round(mean(values), 6),
            "p50": values[max(0, math.ceil(len(values) * 0.5) - 1)],
            "p95": values[max(0, math.ceil(len(values) * 0.95) - 1)],
            "max": max(values),
            "samples": len(values),
        }

    metric_names = (
        "recall_at_k",
        "precision_at_k",
        "ndcg_at_k",
        "hit",
        "reciprocal_rank",
        "correct_refusal",
        "false_refusal",
        *ANSWER_METRICS,
    )
    return {
        "variant": variant,
        "label": VARIANTS[variant][0],
        "cases": len(rows),
        "completed": len(successful),
        "failures": len(rows) - len(successful),
        "recall_at_k": average("recall_at_k"),
        "precision_at_k": average("precision_at_k"),
        "ndcg_at_k": average("ndcg_at_k"),
        "hit_rate": average("hit"),
        "mrr": average("reciprocal_rank"),
        "all_sources_rate": average("all_sources_found"),
        "correct_refusal_rate": average("correct_refusal"),
        "false_refusal_rate": average("false_refusal"),
        "mean_seconds": round(mean(times), 3),
        "completion_rate": len(successful) / len(rows),
        "end_to_end_recall_at_k": (
            mean(r.get("recall_at_k") or 0 for r in rows if r["answerable"])
            if any(r["answerable"] for r in rows)
            else None
        ),
        "p95_seconds": times[max(0, math.ceil(len(times) * 0.95) - 1)],
        "latency_seconds": {stage: latency(stage) for stage in TIMING_STAGES},
        "metric_samples": {
            key: sum(r.get(key) is not None for r in successful) for key in metric_names
        },
        "answer_quality": {key: average(key) for key in ANSWER_METRICS},
        "judged_cases": len(quality_rows),
        "judge_failures": sum(
            r.get("answer_evaluation", {}).get("status") == "unavailable" for r in successful
        ),
        "judge_calls": sum(r.get("judge_calls", {}).get("chat_calls", 0) for r in rows),
        "degraded_cases": sum(bool(r["result"]["warnings"]) for r in successful),
        "reranked_cases": sum(
            any(s["trace"].get("reranked") for s in r["result"]["sections"]) for r in successful
        ),
        "calls": {k: sum(r["calls"][k] for r in rows) for k in rows[0]["calls"]},
    }


def benchmark_markdown(report):
    lines = [
        f"# 检索评测 {report['id']}",
        "",
        f"- 时间：{report['created_at']}",
        f"- 数据集：{report['dataset']['name']}（{report['dataset']['label_status']}）",
        f"- 题库指纹：{report['corpus_sha256']}",
        "",
        "## 检索质量",
        "",
        "| 方案 | 完成 | Recall@3 | Precision@3 | Hit Rate@3 | MRR | nDCG@3 | 正确拒答 | 误拒答 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in report["summaries"]:
        lines.append(
            f"| {s['label']} | {s['completed']}/{s['cases']} | {s['recall_at_k']} | "
            f"{s['precision_at_k']} | {s['hit_rate']} | {s['mrr']} | {s['ndcg_at_k']} | "
            f"{s['correct_refusal_rate']} | {s['false_refusal_rate']} |"
        )
    lines += [
        "",
        "## 回答质量",
        "",
        "分数后括号为有效样本数；None为未评估。",
        "",
        "| 方案 | 忠实度 | 正确性F1 | 相关性 | 引用准确性 | 引用覆盖 | 评审完成/失败 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in report["summaries"]:
        values = [f"{s['answer_quality'][k]} ({s['metric_samples'][k]})" for k in ANSWER_METRICS]
        lines.append(
            f"| {s['label']} | "
            + " | ".join(values)
            + f" | {s['judged_cases']}/{s['judge_failures']} |"
        )
    lines += [
        "",
        "## 具体时延（秒）",
        "",
        "| 方案 | 平均 | P50 | P95 | 最大 | 评审平均（另计） |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in report["summaries"]:
        t = s["latency_seconds"]["total"]
        lines.append(
            f"| {s['label']} | {t['mean']} | {t['p50']} | {t['p95']} | {t['max']} | "
            f"{s['latency_seconds']['judge']['mean']} |"
        )
    lines += [
        "",
        "| 阶段（平均/P95） | " + " | ".join(s["label"] for s in report["summaries"]) + " |",
        "| --- | " + " | ".join("---" for _ in report["summaries"]) + " |",
    ]
    for stage in TIMING_STAGES:
        values = [
            f"{s['latency_seconds'][stage]['mean']}/{s['latency_seconds'][stage]['p95']}"
            for s in report["summaries"]
        ]
        lines.append(f"| {stage} | " + " | ".join(values) + " |")
    lines += ["", *[f"- {note}" for note in report["notes"]], "", "## 逐题结果", ""]
    for row in report["records"]:
        retrieval_values = [
            row.get(k)
            for k in (
                "recall_at_k",
                "precision_at_k",
                "hit",
                "ndcg_at_k",
                "reciprocal_rank",
            )
        ]
        lines.extend(
            [
                f"### {row['case_id']} / {row['variant']}",
                row["question"],
                f"- 耗时：{row['elapsed_seconds']} 秒；调用：{row['calls']}",
                f"- 分阶段秒数：{row['timings_seconds']}",
                f"- 答案评审调用：{row['judge_calls']}",
                f"- Recall/Precision/Hit/nDCG/RR：{retrieval_values}",
                f"- 标注来源：{', '.join(row['expected_sources']) or '无，应拒答'}",
                f"- 检索来源：{', '.join(row.get('retrieved_sources', [])) or '无'}",
                row["error"] or row["result"]["answer"],
                "",
                "```json",
                json.dumps(row["answer_evaluation"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)
