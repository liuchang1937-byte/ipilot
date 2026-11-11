from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from threading import RLock
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .benchmark import DEFAULT_DATASET, DEFAULT_VARIANTS, VARIANTS, load_dataset, run_benchmark
from .config import TechRAGConfig
from .interview import COMPLETED, InterviewEngine, InterviewSession, InterviewStep
from .knowledge import KnowledgeOptions
from .question_editor import DISPLAY_NAMES, EditorError, QuestionEditor
from .rag import TechRAG

_PUBLIC = Path(__file__).with_name("public")
_NAMES = DISPLAY_NAMES


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["technical", "behavioral"] = "technical"
    rounds: Literal[5, 10, 15] = 5
    difficulty: Literal[1, 2, 3] = 2
    difficulty_policy: Literal["adaptive", "locked"] = "adaptive"
    domains: list[str] = Field(min_length=1, max_length=30)
    topics: list[str] = Field(default_factory=list, max_length=100)


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=12000)
    expected_turn_index: int = Field(ge=0)


class FinishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_turn_index: int = Field(ge=0)


class KnowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    domains: list[str] = Field(default_factory=list, max_length=30)
    decompose: bool = True
    mqe: bool = True
    hyde: bool = True
    rerank: bool = True
    top_k: int = Field(default=3, ge=1, le=5)


class BankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=60)
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$", default_factory=lambda: uuid4().hex)


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Literal[6, 20] = 6
    answer_judge: bool = True
    variants: list[Literal["vector", "hybrid", "mqe", "hyde", "hybrid_reranked",
                           "expanded", "reranked"]] = Field(
        default_factory=lambda: list(DEFAULT_VARIANTS), min_length=1, max_length=7,
    )


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=20000)
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$", default_factory=lambda: uuid4().hex)


class EditQuestionRequest(QuestionRequest):
    expected_revision: str = Field(min_length=1, max_length=64)


def _summary(session: InterviewSession) -> dict[str, object]:
    scores = [mean(values) for values in session.dimension_totals.values() if values]
    return {
        "id": session.id, "mode": session.mode, "phase": session.phase,
        "domains": list(session.domains), "topics": list(session.topics),
        "target_rounds": session.target_rounds, "completed_rounds": session.completed_rounds,
        "difficulty": session.difficulty, "difficulty_policy": session.difficulty_policy,
        "turn_index": session.turn_index, "current_domain": session.current_domain,
        "created_at": session.created_at,
        "current_knowledge_points": list(session.current_knowledge_points),
        "blocked_knowledge_points": list(session.blocked_knowledge_points),
        "score": round(mean(scores), 1) if scores else None,
    }


def _snapshot(engine: InterviewEngine, step: InterviewStep) -> dict[str, object]:
    session = engine._load(step.session_id)
    turns = engine.store.turns(session.id)
    messages = []
    for index, turn in enumerate(turns):
        messages.append({"role": "assistant", "content": turn["question"],
                         "phase": turn["phase"], "kind": "question"})
        messages.append({"role": "user", "content": turn["answer"],
                         "phase": turn["phase"], "kind": "answer"})
        decision = turn.get("decision") or {}
        reply = decision.get("interviewer_reply")
        if (reply and index + 1 < len(turns)
                and reply != turns[index + 1]["question"]):
            messages.append({"role": "assistant", "content": reply,
                             "phase": turn["phase"], "kind": "reply"})
    if step.interviewer_message:
        messages.append({"role": "assistant", "content": step.interviewer_message,
                         "phase": step.phase, "kind": step.decision})
    return {"session": _summary(session), "step": asdict(step), "messages": messages}


def create_app(
    config: TechRAGConfig | None = None, *, engine: InterviewEngine | None = None,
    report_dir: str | Path = "reports",
    question_dir: str | Path = "data/online_question_banks",
    evaluation_dir: str | Path = "reports/evaluations",
) -> FastAPI:
    """Serve a local interview UI and its API using one persistent engine."""
    lock = RLock()
    evaluation_dir = Path(evaluation_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rag = None
        if engine is None:
            rag = TechRAG(config or TechRAGConfig())
            app.state.engine = InterviewEngine(rag, report_dir=report_dir)
        else:
            app.state.engine = engine
        app.state.editor = QuestionEditor(app.state.engine.rag, root=question_dir)
        try:
            yield
        finally:
            if rag is not None:
                rag.close()

    app = FastAPI(title="InterviewPilot", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=_PUBLIC / "static"), name="static")

    def get_engine() -> InterviewEngine:
        return app.state.engine

    def load(session_id: str) -> InterviewSession:
        try:
            return get_engine()._load(session_id)
        except ValueError as error:
            raise HTTPException(404, "这场面试不存在，请返回面试记录重新选择。") from error

    def check_revision(session: InterviewSession, expected: int) -> None:
        if session.turn_index != expected:
            raise HTTPException(409, "面试进度已更新，请刷新对话后继续。")

    def mutate(session_id: str, action: Callable[[], InterviewStep]):
        active = get_engine()
        before = active.store.load_session_data(session_id)
        before_turns = {turn["turn_index"] for turn in active.store.turns(session_id)}
        try:
            return _snapshot(active, action())
        except Exception:
            # Model retrieval can fail after a turn has been written. Restore
            # this request's local state so a retry cannot duplicate that turn.
            for turn in active.store.turns(session_id):
                if turn["turn_index"] not in before_turns:
                    active.rag.database.delete_records(
                        "turn", session_id=session_id, turn_index=turn["turn_index"],
                    )
            active.rag.database.put_record("session", session_id, before, session_id=session_id)
            raise

    @app.exception_handler(ValueError)
    async def invalid_value(_request: Request, error: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.exception_handler(EditorError)
    async def editor_error(_request: Request, error: EditorError):
        return JSONResponse(status_code=error.status, content={"detail": str(error)})

    @app.exception_handler(Exception)
    async def failed_request(_request: Request, _error: Exception):
        # Never expose provider credentials or internal model exception text.
        return JSONResponse(status_code=500, content={
            "detail": "本次处理失败，请刷新对话确认进度后重试。已保存的面试记录会保留。",
        })

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_PUBLIC / "index.html")

    @app.get("/api/evaluations")
    def evaluations():
        with lock:
            dataset = load_dataset(DEFAULT_DATASET)
            reports = []
            for path in evaluation_dir.glob("*.json"):
                if len(path.stem) != 32 or any(c not in "0123456789abcdef" for c in path.stem):
                    continue
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    keys = ("id", "created_at", "dataset", "summaries")
                    reports.append({k: value[k] for k in keys})
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            return {"dataset": {k: dataset.get(k) for k in ("name", "label_status", "description")},
                    "case_count": len(dataset["cases"]),
                    "variants": [{"id": k, "label": v[0], "default": k in DEFAULT_VARIANTS}
                                 for k, v in VARIANTS.items()],
                    "reports": sorted(reports, key=lambda r: r["created_at"], reverse=True)[:30]}

    @app.post("/api/evaluations/run")
    def evaluation_run(body: BenchmarkRequest):
        with lock:
            return run_benchmark(get_engine().rag, variants=body.variants,
                                 limit=body.limit, output_dir=evaluation_dir,
                                 answer_judge=body.answer_judge)

    @app.get("/api/evaluations/{report_id}")
    def evaluation_result(report_id: str):
        if len(report_id) != 32 or any(c not in "0123456789abcdef" for c in report_id):
            raise HTTPException(404, "评测记录不存在。")
        path = evaluation_dir / f"{report_id}.json"
        if not path.is_file():
            raise HTTPException(404, "评测记录不存在。")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/evaluations/{report_id}/download")
    def evaluation_download(report_id: str):
        evaluation_result(report_id)
        return FileResponse(evaluation_dir / f"{report_id}.json", media_type="application/json",
                            filename=f"evaluation-{report_id}.json")

    @app.get("/api/banks")
    def banks():
        with lock:
            names = {b["key"]: b["name"] for b in app.state.editor.banks()}
            groups: dict[tuple[str, str], dict[str, object]] = {}
            for mode in ("technical", "behavioral"):
                for question in get_engine()._question_bank(mode):
                    key = (mode, question.domain)
                    group = groups.setdefault(key, {
                        "id": question.domain, "name": names[f"{mode}:{question.domain}"],
                        "mode": mode, "count": 0, "topics": set(),
                        "topic_questions": {}, "difficulty_counts": [0, 0, 0],
                    })
                    group["count"] += 1
                    group["topics"].update(question.routing_points)
                    for point in question.routing_points:
                        group["topic_questions"].setdefault(point, []).append(question.id)
                    group["difficulty_counts"][question.difficulty - 1] += 1
            values = []
            for group in groups.values():
                values.append({**group, "topics": sorted(group["topics"])})
            return {"banks": values, "total_questions": sum(g["count"] for g in values)}

    @app.get("/api/library/banks")
    def library_banks():
        with lock:
            return {"banks": app.state.editor.banks()}

    @app.post("/api/library/banks", status_code=201)
    def library_create(body: BankRequest):
        with lock:
            return app.state.editor.create_bank(body.name, body.request_id)

    @app.get("/api/library/banks/{bank_key}/questions")
    def library_questions(bank_key: str, search: str = Query(default="", max_length=200),
                          offset: int = Query(default=0, ge=0),
                          limit: int = Query(default=30, ge=1, le=100)):
        with lock:
            return app.state.editor.questions(bank_key, search, offset, limit)

    @app.post("/api/library/banks/{bank_key}/questions", status_code=201)
    def library_add(bank_key: str, body: QuestionRequest):
        with lock:
            return app.state.editor.save(bank_key, body.question, body.answer, body.request_id)

    @app.get("/api/library/questions/{question_id}")
    def library_question(question_id: str):
        with lock:
            return app.state.editor.get(question_id)

    @app.put("/api/library/questions/{question_id}")
    def library_edit(question_id: str, body: EditQuestionRequest):
        with lock:
            editor = app.state.editor
            current = editor.get(question_id)
            return editor.save(current["bank_key"], body.question, body.answer, body.request_id,
                               question_id=question_id, expected_revision=body.expected_revision)

    @app.get("/api/library/banks/{bank_key}/export")
    def library_export(bank_key: str):
        from .question_editor import _markdown

        with lock:
            editor = app.state.editor
            bank = editor.bank(bank_key)
            questions = [q for q, _, _ in editor._rows().values()
                         if f"{q.interview_type}:{q.domain}" == bank_key]
            text = (f"---\ndomain: {bank['domain']}\ninterview_type: {bank['mode']}\n---\n\n"
                    + "\n\n".join(_markdown(q).split("---\n", 2)[-1] for q in questions))
            return Response(text, media_type="text/markdown; charset=utf-8", headers={
                "Content-Disposition": "attachment; filename=question-bank.md",
            })

    @app.get("/api/sessions")
    def sessions():
        with lock:
            records = get_engine().store.database.list_records("session")
            values = [_summary(InterviewSession.from_data(record)) for record in records]
            return {"sessions": sorted(values, key=lambda value: value["created_at"], reverse=True)}

    @app.get("/api/knowledge")
    def knowledge_info():
        with lock:
            active = get_engine()
            documents, chunks = active.rag.stats()
            domains = sorted({str(metadata["domain"])
                              for _, metadata in active.rag.database.document_rows()
                              if metadata.get("domain")})
            names = {b["domain"]: b["name"] for b in app.state.editor.banks()}
            reranker = active.knowledge.reranker
            return {"documents": documents, "parent_chunks": chunks,
                    "domains": [{"id": d, "name": names.get(d, _NAMES.get(d, d))} for d in domains],
                    "reranker_ready": bool(getattr(reranker, "ready", True)),
                    "reranker_model": getattr(reranker, "repository", type(reranker).__name__)}

    @app.get("/api/knowledge/history")
    def knowledge_history():
        with lock:
            records = get_engine().rag.database.list_records("knowledge-answer")
            return {"items": [{"id": r["id"], "question": r["question"],
                               "created_at": r["created_at"]}
                              for r in sorted(records, key=lambda r: r["created_at"],
                                              reverse=True)[:50]]}

    @app.post("/api/knowledge/ask")
    def knowledge_ask(body: KnowledgeRequest):
        with lock:
            active = get_engine()
            available = {metadata.get("domain")
                         for _, metadata in active.rag.database.document_rows()}
            if any(domain not in available for domain in body.domains):
                raise HTTPException(400, "所选知识库不存在，请刷新页面后重试。")
            result = active.knowledge.answer(
                body.question, domains=tuple(body.domains), options=KnowledgeOptions(
                    decompose=body.decompose, mqe=body.mqe, hyde=body.hyde,
                    rerank=body.rerank, top_k=body.top_k,
                ),
            )
            result.update({"id": uuid4().hex, "created_at": datetime.now(UTC).isoformat(),
                           "domains": body.domains})
            active.rag.database.put_record("knowledge-answer", result["id"], result)
            return result

    @app.get("/api/knowledge/answers/{answer_id}")
    def knowledge_answer(answer_id: str):
        with lock:
            record = get_engine().rag.database.get_record("knowledge-answer", answer_id)
            if record is None:
                raise HTTPException(404, "这条知识库问答不存在。")
            return record

    @app.post("/api/sessions", status_code=201)
    def start(body: StartRequest):
        with lock:
            active = get_engine()
            # Reject an empty intersection before creating a misleading interview.
            eligible = [q for q in active._question_bank(body.mode)
                        if q.domain in body.domains and (
                            not body.topics or set(body.topics).intersection(
                                (*q.routing_points, *q.tags)
                            )
                        )]
            if not eligible:
                raise HTTPException(400, "当前选择没有可用题目，请调整题库或知识点范围。")
            step = active.start(
                mode=body.mode, rounds=body.rounds, difficulty=body.difficulty,
                difficulty_policy=body.difficulty_policy,
                domains=tuple(body.domains), topics=tuple(body.topics),
            )
            return _snapshot(active, step)

    @app.get("/api/sessions/{session_id}")
    def current(session_id: str):
        with lock:
            load(session_id)
            return _snapshot(get_engine(), get_engine().current(session_id))

    @app.post("/api/sessions/{session_id}/answers")
    def answer(session_id: str, body: AnswerRequest):
        with lock:
            session = load(session_id)
            check_revision(session, body.expected_turn_index)
            if session.phase == COMPLETED:
                raise HTTPException(409, "本场面试已结束，可以查看报告或开始新的面试。")
            return mutate(session_id, lambda: get_engine().submit(session_id, body.answer))

    @app.post("/api/sessions/{session_id}/finish")
    def finish(session_id: str, body: FinishRequest):
        with lock:
            check_revision(load(session_id), body.expected_turn_index)
            return mutate(session_id, lambda: get_engine().finish(session_id))

    @app.get("/api/sessions/{session_id}/report")
    def report(session_id: str):
        with lock:
            session = load(session_id)
            active = get_engine()
            registered = active.store.report_data(session_id) or {}
            markdown = registered.get("markdown_content") or active.report(session_id)
            pdf = registered.get("pdf_path")
            pdf_ready = bool(pdf and Path(str(pdf)).is_file())
            reviews_by_question = {}
            for turn in active.store.turns(session_id):
                if turn["evaluation"]:
                    value = turn["evaluation"]
                    key = str(turn["question_id"]).split(":")[0]
                    review = reviews_by_question.setdefault(key, {
                        "question": turn["question"], "answers": [],
                    })
                    review["answers"].append(turn["answer"])
                    review.update({
                        "domain": turn["domain"], "score": value.get("overall"),
                        "feedback": value.get("feedback", ""),
                        "missing_points": value.get("missing_points", []),
                        "incorrect_points": value.get("incorrect_points", []),
                        "point_reviews": value.get("point_reviews", []),
                        "point_review_complete": value.get("point_review_complete", False),
                        "scoring_method": value.get("scoring_method", "legacy"),
                        "pending": (turn.get("decision") or {}).get("round_complete") is False,
                    })
            reviews = [{**review, "answer": "\n\n".join(review["answers"])}
                       for review in reviews_by_question.values()]
            return {
                "session": _summary(session), "markdown": markdown, "reviews": reviews,
                "dimensions": {key: round(mean(values), 1) for key, values in
                               session.dimension_totals.items() if values},
                "domain_scores": {key: round(mean(values), 1) for key, values in
                                  session.domain_scores.items() if values},
                "pdf_ready": pdf_ready,
                "pdf_url": f"/api/sessions/{session_id}/report.pdf" if pdf_ready else None,
                "markdown_url": f"/api/sessions/{session_id}/report.md",
            }

    @app.get("/api/sessions/{session_id}/report.md")
    def markdown_download(session_id: str):
        with lock:
            value = report(session_id)
            return Response(value["markdown"], media_type="text/markdown; charset=utf-8", headers={
                "Content-Disposition": f'attachment; filename="interview-{session_id}.md"',
            })

    @app.get("/api/sessions/{session_id}/report.pdf")
    def pdf_download(session_id: str):
        with lock:
            load(session_id)
            registered = get_engine().store.report_data(session_id) or {}
            path = registered.get("pdf_path")
            if not path or not Path(str(path)).is_file():
                raise HTTPException(404, "PDF 尚未生成，可先下载 Markdown 报告。")
            return FileResponse(path, media_type="application/pdf",
                                filename=f"interview-{session_id}.pdf")

    return app
