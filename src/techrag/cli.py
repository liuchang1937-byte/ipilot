from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import DEFAULT_DATASET, VARIANTS, run_benchmark
from .config import TechRAGConfig
from .evaluation import evaluate, generate_evals
from .experiments import EVIDENCE_DATASET, run_experiment
from .interview import COMPLETED, InterviewEngine, InterviewStep
from .interview_questions import index_question_directory
from .migration import import_legacy_snapshot
from .providers import BGEReranker, OnnxCrossEncoder
from .rag import TechRAG
from .scoring_benchmark import SCORING_DATASET, run_scoring_benchmark


def _json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="techrag", description="Bilingual RAG with Chroma")
    parser.add_argument(
        "--db", "--chroma-path", type=Path, default=Path("data/chroma"),
        help="Chroma persistence directory",
    )
    parser.add_argument("--self-query", action="store_true")
    parser.add_argument("--mqe", action="store_true", help="enable multi-query expansion")
    parser.add_argument(
        "--reranker",
        choices=("bge",),
        help="optional second-stage reranker (downloads the model on first use)",
    )
    parser.add_argument(
        "--hyde", action="store_true", help="enable hypothetical-document expansion"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="initialize Chroma collections")
    subparsers.add_parser("prepare-reranker", help="download and verify the local Cross Encoder")
    web = subparsers.add_parser("web", help="serve the interview website")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    migrate = subparsers.add_parser("migrate", help="import a legacy JSON export into Chroma")
    migrate.add_argument("snapshot", type=Path)
    add = subparsers.add_parser("add", help="convert files to Markdown and index them")
    add.add_argument("paths", nargs="+", type=Path)
    add.add_argument("--metadata", type=_json_object, help="document metadata as a JSON object")
    search = subparsers.add_parser("search", help="retrieve relevant chunks")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--mode", choices=("vector", "keyword", "hybrid"), default="hybrid")
    search.add_argument("--filter", type=_json_object, help="metadata filter as a JSON object")
    ask = subparsers.add_parser("ask", help="retrieve context and generate an answer")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=5)
    ask.add_argument("--stream", action="store_true")
    ask.add_argument("--filter", type=_json_object, help="metadata filter as a JSON object")
    delete = subparsers.add_parser("delete", help="delete documents by source")
    delete.add_argument("sources", nargs="*")
    delete.add_argument("--filter", type=_json_object, help="delete by metadata JSON object")
    subparsers.add_parser("stats", help="show document and chunk counts")
    evaluation = subparsers.add_parser("eval", help="generate or run evaluation cases")
    evaluation.add_argument("--generate", type=int, default=0, metavar="N")
    evaluation.add_argument("--answers", action="store_true")
    evaluation.add_argument(
        "--compare", action="store_true", help="compare knowledge retrieval pipelines",
    )
    evaluation.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    evaluation.add_argument("--variant", action="append", choices=tuple(VARIANTS), dest="variants")
    evaluation.add_argument("--limit", type=int, help="run the first N labelled cases")
    evaluation.add_argument("--output-dir", type=Path, default=Path("reports/evaluations"))
    evaluation.add_argument("--no-answer-judge", action="store_true",
                            help="skip separate answer quality judging")
    evaluation.add_argument(
        "--ablation", action="store_true", help="run grouped component ablations"
    )
    evaluation.add_argument("--repeat", type=int, default=3, help="ablation repetitions (1-10)")
    evaluation.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--require-reviewed", action="store_true")

    score_eval = subparsers.add_parser("score-eval", help="evaluate production point scoring")
    score_eval.add_argument("--dataset", type=Path, default=SCORING_DATASET)
    score_eval.add_argument("--limit", type=int)
    score_eval.add_argument("--repeat", type=int, default=3)
    score_eval.add_argument("--require-reviewed", action="store_true")
    score_eval.add_argument("--output-dir", type=Path, default=Path("reports/scoring"))

    questions = subparsers.add_parser("questions", help="manage the interview question bank")
    question_commands = questions.add_subparsers(dest="questions_command", required=True)
    question_import = question_commands.add_parser(
        "import", help="validate and index structured Markdown questions"
    )
    question_import.add_argument("path", type=Path)
    question_commands.add_parser("domains", help="list indexed interview domains")

    interview = subparsers.add_parser("interview", help="run an Agent-driven mock interview")
    interview_commands = interview.add_subparsers(dest="interview_command", required=True)
    interview_start = interview_commands.add_parser("start", help="start an interview")
    interview_start.add_argument("--mode", choices=("technical", "behavioral"), default="technical")
    interview_start.add_argument("--rounds", choices=(5, 10, 15), type=int, default=5)
    interview_start.add_argument(
        "--difficulty-policy", choices=("adaptive", "locked"), default="adaptive"
    )
    interview_start.add_argument("--difficulty", choices=(1, 2, 3), type=int, default=2)
    interview_start.add_argument(
        "--domain", action="append", dest="domains", help="limit interview domains (repeatable)"
    )
    interview_start.add_argument(
        "--topic", action="append", dest="topics", help="limit knowledge points (repeatable)"
    )
    interview_resume = interview_commands.add_parser("resume", help="resume a saved interview")
    interview_resume.add_argument("session_id")
    interview_report = interview_commands.add_parser("report", help="print or export a report")
    interview_report.add_argument("session_id")
    interview_report.add_argument("--output", type=Path, help="write the Markdown report")
    interview_report.add_argument("--pdf", type=Path, help="regenerate the PDF at this path")
    interview_delete = interview_commands.add_parser(
        "delete", help="delete a session and its structured history"
    )
    interview_delete.add_argument("session_id")
    interview_delete.add_argument(
        "--keep-pdf", action="store_true", help="retain the generated PDF file"
    )
    return parser


def _run_interview(engine: InterviewEngine, step: InterviewStep) -> None:
    current = step
    print("可以直接作答；暂时没思路可说“不知道”，也可以请求解释问题或说“跳过”。")
    while True:
        print(f"\nSession: {current.session_id}")
        if current.phase == COMPLETED:
            if current.interviewer_message:
                print(f"面试官：{current.interviewer_message}")
            print("\n" + (current.report or engine.report(current.session_id)))
            if current.report_path:
                print(f"\nPDF report: {current.report_path}")
            return
        label = {
            "hint": "本题提示", "followup": "本题追问", "clarification": "问题解释",
        }.get(current.decision, {
            "self_introduction": "自我介绍", "project_deep_dive": "项目深挖",
            "questioning": "正式问答", "candidate_questions": "候选人反问",
        }.get(current.phase, current.phase))
        print(f"[{label}] 面试官：{current.interviewer_message}")
        try:
            answer = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n面试已暂停。使用 `techrag interview resume {current.session_id}` 继续。")
            return
        if not answer:
            print("回答不能为空，请重新输入。")
            continue
        current = engine.submit(current.session_id, answer)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "eval" and args.require_reviewed and not args.ablation:
        raise SystemExit("--require-reviewed requires --ablation")
    config = TechRAGConfig(
        db_path=args.db,
        self_query=args.self_query,
        enable_mqe=args.mqe,
        enable_hyde=args.hyde,
    )
    if args.command == "prepare-reranker":
        OnnxCrossEncoder(config.reranker_dir).prepare()
        print(f"Cross Encoder ready: {config.reranker_dir}")
        return 0
    if args.command == "web":
        import uvicorn

        from .web import create_app

        uvicorn.run(create_app(config), host=args.host, port=args.port)
        return 0
    reranker = BGEReranker() if args.reranker == "bge" else None
    with TechRAG(config, reranker=reranker) as rag:
        if args.command == "init":
            print(f"Initialized {args.db}")
        elif args.command == "migrate":
            print(json.dumps(import_legacy_snapshot(rag.database, args.snapshot)))
        elif args.command == "add":
            for path in args.paths:
                document_id = rag.add_file(path, metadata=args.metadata)
                print(f"Indexed {path} (document {document_id})")
        elif args.command == "search":
            results = rag.search(
                args.query,
                top_k=args.k,
                mode=args.mode,
                metadata_filter=args.filter,
            )
            for index, result in enumerate(results, start=1):
                print(f"[{index}] {result.source} score={result.score:.4f}\n{result.content}\n")
        elif args.command == "ask":
            if args.stream:
                for token in rag.stream(
                    args.question, top_k=args.k, metadata_filter=args.filter
                ):
                    print(token, end="", flush=True)
                print()
            else:
                answer = rag.ask(
                    args.question, top_k=args.k, metadata_filter=args.filter
                )
                print(answer.text)
                if answer.sources:
                    print("\nSources:")
                    for index, source in enumerate(answer.sources, start=1):
                        print(f"[{index}] {source.source}")
        elif args.command == "delete":
            if not args.sources and not args.filter:
                raise SystemExit("delete requires at least one source or --filter")
            for source in args.sources:
                status = "Deleted" if rag.delete(source) else "Not found"
                print(f"{status}: {source}")
            if args.filter:
                count = rag.delete_by_metadata(args.filter)
                print(f"Deleted {count} document(s) by metadata")
        elif args.command == "stats":
            documents, chunks = rag.stats()
            print(f"documents={documents} chunks={chunks}")
        elif args.command == "score-eval":
            report = run_scoring_benchmark(
                rag, dataset_path=args.dataset, limit=args.limit, repeats=args.repeat,
                reviewed_only=args.require_reviewed, output_dir=args.output_dir,
                progress=lambda done, total, case, variant: print(
                    f"[{done}/{total}] {case} {variant}", flush=True),
            )
            print(json.dumps(report["summary"], ensure_ascii=False))
            print(f"Saved: {(args.output_dir / (report['id'] + '.json')).resolve()}")
        elif args.command == "eval":
            if args.compare or args.ablation:
                if args.generate or args.answers:
                    raise SystemExit("--compare cannot be combined with --generate/--answers")
                options = dict(variants=args.variants, limit=args.limit,
                               output_dir=args.output_dir, answer_judge=not args.no_answer_judge,
                               progress=lambda done, total, case, variant: print(
                                   f"[{done}/{total}] {case} {variant}", flush=True))
                if args.ablation:
                    dataset = EVIDENCE_DATASET if args.dataset == DEFAULT_DATASET else args.dataset
                    report = run_experiment(
                        rag, dataset_path=dataset, repeats=args.repeat, split=args.split,
                        seed=args.seed, reviewed_only=args.require_reviewed, **options,
                    )
                else:
                    report = run_benchmark(rag, dataset_path=args.dataset, **options)
                for row in report["summaries"]:
                    print(json.dumps(row, ensure_ascii=False))
                print(f"Saved: {(args.output_dir / (report['id'] + '.json')).resolve()}")
                return 0
            if args.generate:
                generated = generate_evals(rag, num_evals=args.generate)
                print(f"Generated {len(generated)} evaluation cases")
            report = evaluate(rag, evaluate_answers=args.answers)
            print(
                f"cases={report.cases} recall@k={report.recall_at_k:.3f} "
                f"mrr={report.mean_reciprocal_rank:.3f}"
            )
        elif args.command == "questions":
            if args.questions_command == "import":
                count = index_question_directory(rag, args.path)
                print(f"Imported {count} interview questions from {args.path}")
            elif args.questions_command == "domains":
                counts: dict[tuple[str, str], int] = {}
                for _, metadata in rag.database.document_rows():
                    if metadata.get("collection") == "interview-question":
                        key = (str(metadata.get("interview_type")), str(metadata.get("domain")))
                        counts[key] = counts.get(key, 0) + 1
                for (mode, domain), count in sorted(counts.items()):
                    print(f"{mode}\t{domain}\t{count}")
        elif args.command == "interview":
            engine = InterviewEngine(rag)
            if args.interview_command == "start":
                step = engine.start(
                    mode=args.mode,
                    rounds=args.rounds,
                    difficulty_policy=args.difficulty_policy,
                    difficulty=args.difficulty,
                    domains=tuple(args.domains) if args.domains else None,
                    topics=tuple(args.topics or ()),
                )
                _run_interview(engine, step)
            elif args.interview_command == "resume":
                _run_interview(engine, engine.current(args.session_id))
            elif args.interview_command == "report":
                content = engine.report(args.session_id)
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(content, encoding="utf-8")
                    print(f"Report written to {args.output}")
                else:
                    print(content)
                if args.pdf:
                    path = engine.export_report(args.session_id, args.pdf, markdown=content)
                    print(f"PDF report written to {path}")
            elif args.interview_command == "delete":
                deleted = engine.delete(args.session_id, keep_pdf=args.keep_pdf)
                status = "Deleted" if deleted else "Not found"
                print(f"{status}: interview session {args.session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
