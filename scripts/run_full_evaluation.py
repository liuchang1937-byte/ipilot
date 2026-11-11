"""Resumable grouped retrieval study, frozen holdout and separate answer evaluation."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from techrag import TechRAG, TechRAGConfig
from techrag.benchmark import (
    BenchmarkBlockedError,
    _atomic_json,
    _summary,
    load_dataset,
    run_benchmark,
)
from techrag.experiments import (
    EVIDENCE_DATASET,
    paired_comparisons,
    require_reviewed,
    select_cases,
    write_report,
)
from techrag.knowledge import KnowledgeService
from techrag.providers import CHAT_MODEL, EMBEDDING_MODEL, OnnxCrossEncoder

ROOT = Path(__file__).resolve().parents[1]
DEV_VARIANTS = ["hybrid", "hybrid_reranked", "expanded"]
_RAG = None
_SERVICE = None


@contextmanager
def runner_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "runner.lock").open("a+b")
    try:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("该目录已有正在运行的评测，可读取live.log查看进度。") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _worker_init(config):
    global _RAG, _SERVICE
    config = dict(config)
    for key in ("db_path", "reranker_dir"):
        config[key] = Path(config[key])
    _RAG = TechRAG(TechRAGConfig(**config))
    _SERVICE = KnowledgeService(_RAG)
    atexit.register(_RAG.close)


def _run_case(task):
    directory = Path(task["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    dataset = directory / "case.json"
    expected = task["dataset"] | {"cases": [task["case"]]}
    if dataset.exists():
        if json.loads(dataset.read_text(encoding="utf-8")) != expected:
            raise ValueError("已保存的案例数据改变，不能续跑。")
    else:
        _atomic_json(dataset, expected)
    report = run_benchmark(
        _RAG,
        dataset_path=dataset,
        variants=task["variants"],
        service=_SERVICE,
        output_dir=directory,
        retrieval_only=task["retrieval_only"],
        answer_judge=task["answer_judge"],
        checkpoint_dir=directory / "checkpoint",
    )
    _atomic_json(directory / "complete.json", report)
    return {
        "case_id": task["case"]["id"],
        "repeat": task["repeat"],
        "report_path": str(directory / "complete.json"),
    }


def _source_hash():
    files = sorted((ROOT / "src/techrag").glob("*.py")) + [Path(__file__).resolve()]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def _protocol(directory, workers):
    data = load_dataset(EVIDENCE_DATASET)
    require_reviewed(data, data["cases"])
    select_cases(data, "dev", None, 42)
    config = TechRAGConfig(
        db_path=ROOT / "data/chroma", reranker_dir=ROOT / "models/mmarco-cross-encoder"
    )
    with TechRAG(config) as rag:
        corpus = sorted((str(r[5]), str(r[2]), str(r[6])) for r in rag.database.chunk_rows())
    available = {r[0] for r in corpus}
    missing = {s for c in data["cases"] for s in c["expected_sources"]} - available
    if missing:
        raise ValueError("标注来源缺失：" + ", ".join(sorted(missing)))
    assets = {
        name: hashlib.sha256((config.reranker_dir / name).read_bytes()).hexdigest()
        for name in OnnxCrossEncoder.files
    }
    config_data = asdict(config) | {
        "db_path": str(config.db_path),
        "reranker_dir": str(config.reranker_dir),
    }
    expected = {
        "dataset_sha256": hashlib.sha256(EVIDENCE_DATASET.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(
            json.dumps(corpus, ensure_ascii=False).encode()
        ).hexdigest(),
        "code_sha256": _source_hash(),
        "config": config_data,
        "reranker_assets": assets,
        "models": {"chat": CHAT_MODEL, "embedding": EMBEDDING_MODEL},
        "dev_variants": DEV_VARIANTS,
        "retrieval_repeats": 3,
        "seed": 42,
        "retrieval_workers": workers,
        "quality_workers": 1,
        "selection_rule": "先完整性，再Recall、MRR；相同效果选择较低检索耗时；仅在开发集选择。",
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            has_runs = any(directory.glob("*/repeat-*/*/case.json")) or any(
                (directory / name).exists()
                for name in ("dev.json", "test.json", "quality.json", "selection.json")
            )
            if has_runs:
                raise ValueError("实验协议、代码、数据或索引已改变，请使用新的实验目录。")
            # A prepared protocol with no executed jobs can be updated before launch.
            _atomic_json(path, expected)
            _atomic_json(directory / "dataset.json", data)
    else:
        _atomic_json(path, expected)
        _atomic_json(directory / "dataset.json", data)
    return expected, data


def _merge_phase(directory, phase, tasks, protocol, elapsed):
    if _source_hash() != protocol["code_sha256"]:
        raise ValueError("实验期间代码改变，不合并不同实现的结果。")
    records = []
    reports = []
    for task in tasks:
        path = Path(task["directory"]) / "complete.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["corpus_sha256"] != protocol["corpus_sha256"]:
            raise ValueError("实验期间题库发生变化，不合并结果。")
        if (
            report["models"]["chat"] != protocol["models"]["chat"]
            or report["models"]["embedding"] != protocol["models"]["embedding"]
            or report["config"] != protocol["config"]
        ):
            raise ValueError("案例结果的模型或配置与冻结协议不同。")
        reports.append({"case_id": task["case"]["id"], "repeat": task["repeat"], "file": str(path)})
        records.extend(r | {"repeat": task["repeat"]} for r in report["records"])
    variants = tasks[0]["variants"]
    variants = [v for v in DEV_VARIANTS if v in variants]
    result = {
        "id": phase,
        "kind": "full_evaluation_phase",
        "created_at": datetime.now(UTC).isoformat(),
        "phase": phase,
        "protocol": str(directory / "protocol.json"),
        "dataset_sha256": protocol["dataset_sha256"],
        "corpus_sha256": protocol["corpus_sha256"],
        "independent_cases": len({r["case_id"] for r in records}),
        "independent_groups": len({r["group"] for r in records}),
        "repeats": 1 if phase == "quality" else 3,
        "evaluation_mode": "end_to_end" if phase == "quality" else "retrieval_only",
        "summaries": [_summary(v, records) for v in variants],
        "comparisons": paired_comparisons(records),
        "records": records,
        "runs": reports,
        "current_attempt_wall_seconds": elapsed,
        "notes": [
            "检索阶段三进程并行，每个案例内各方案串行并轮换顺序；耗时在此并发负载下测得。",
            "纯检索不生成答案，不评价正确拒答或回答质量，也不将其耗时称为问答时延。",
            "质量阶段单进程、独立40条测试问题、各一次生成与评审，不能当作三次稳定性实验。",
            "重复记录先按案例汇总，配对区间按来源关联组bootstrap，避免重复样本扩大样本量。",
            "断点保留原输入、调用量、耗时和成功响应，当前墙钟时间只代表本次启动。",
        ],
    }
    lines = [
        f"# 完整评测：{phase}",
        "",
        f"- 独立问题：{result['independent_cases']}；"
        f"分组：{result['independent_groups']}；重复：{result['repeats']}。",
        f"- 模式：{result['evaluation_mode']}。",
        "",
        "| 方案 | 完成 | Recall@3 | MRR | 平均流程秒 | P95秒 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in result["summaries"]:
        lines.append(
            f"| {s['label']} | {s['completed']}/{s['cases']} | {s['recall_at_k']} | "
            f"{s['mrr']} | {s['mean_seconds']} | {s['p95_seconds']} |"
        )
    lines += [
        "",
        "## 配对比较",
        "",
        "```json",
        json.dumps(result["comparisons"], ensure_ascii=False, indent=2),
        "```",
        "",
        *[f"- {note}" for note in result["notes"]],
    ]
    write_report(result, directory, "\n".join(lines) + "\n")
    return result


def _freeze_selection(directory, dev, protocol):
    if any(s["failures"] or s["degraded_cases"] for s in dev["summaries"]):
        raise ValueError("开发集仍有失败或检索降级，不能冻结选型；可续跑失败/降级项。")

    def key(row):
        return (
            -row["completion_rate"],
            -row["recall_at_k"],
            -row["mrr"],
            row["latency_seconds"]["retrieval"]["mean"],
        )

    winner = min(dev["summaries"], key=key)["variant"]
    variants = ["hybrid"] + ([winner] if winner != "hybrid" else ["hybrid_reranked"])
    value = {
        "selected_variant": winner,
        "holdout_variants": variants,
        "development_report": str(directory / "dev.json"),
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "rule": protocol["selection_rule"],
        "frozen_before_test": True,
    }
    path = directory / "selection.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError("已有冻结选型与当前开发结果不同，不能按测试结果重新选型。")
    _atomic_json(path, value)
    return value


def execute_study(args):
    protocol, data = _protocol(args.output_dir, args.workers)
    phases = ["dev", "test", "quality"] if args.phase == "all" else [args.phase]
    for phase in phases:
        started = time.perf_counter()
        if phase != "dev":
            dev = json.loads((args.output_dir / "dev.json").read_text(encoding="utf-8"))
            selection = _freeze_selection(args.output_dir, dev, protocol)
        variants = (
            DEV_VARIANTS
            if phase == "dev"
            else (
                selection["holdout_variants"]
                if phase == "test"
                else [selection["selected_variant"]]
            )
        )
        cases = select_cases(data, "dev" if phase == "dev" else "test", None, 42)
        tasks = []
        for repeat in range(1, 2 if phase == "quality" else 4):
            shuffled = list(cases)
            random.Random(42 + repeat).shuffle(shuffled)
            for number, case in enumerate(shuffled):
                offset = (number + repeat - 1) % len(variants)
                order = variants[offset:] + variants[:offset]
                key = hashlib.sha256(case["id"].encode()).hexdigest()[:16]
                tasks.append(
                    {
                        "dataset": data,
                        "case": case,
                        "repeat": repeat,
                        "variants": order,
                        "retrieval_only": phase != "quality",
                        "answer_judge": phase == "quality",
                        "directory": str(args.output_dir / phase / f"repeat-{repeat}" / key),
                    }
                )
        print(
            f"Starting {phase}: {len(cases)} cases, {len(variants)} variants, {len(tasks)} jobs",
            flush=True,
        )
        workers = 1 if phase == "quality" else args.workers
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init, initargs=(protocol["config"],)
        ) as executor:
            futures = {executor.submit(_run_case, t): t for t in tasks}
            done = 0
            try:
                for future in as_completed(futures):
                    result = future.result()
                    done += 1
                    print(
                        f"[{phase} {done}/{len(tasks)}] {result['case_id']} "
                        f"repeat={result['repeat']}",
                        flush=True,
                    )
            except BenchmarkBlockedError:
                for future in futures:
                    future.cancel()
                raise
        result = _merge_phase(
            args.output_dir, phase, tasks, protocol, time.perf_counter() - started
        )
        if phase == "dev":
            _freeze_selection(args.output_dir, result, protocol)
        print(f"Saved: {args.output_dir / (phase + '.json')}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("dev", "test", "quality", "all"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/full-evaluation")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    with runner_lock(args.output_dir):
        execute_study(args)


if __name__ == "__main__":
    main()
