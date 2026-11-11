from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Literal

from .interview_questions import InterviewQuestion, question_from_content
from .interview_store import InterviewStore
from .knowledge import KnowledgeService
from .rag import TechRAG
from .report_pdf import export_report_pdf
from .scoring import point_reviews, rubric_points

InterviewMode = Literal["technical", "behavioral"]
DifficultyPolicy = Literal["adaptive", "locked"]

SELF_INTRODUCTION = "self_introduction"
PROJECT_DEEP_DIVE = "project_deep_dive"
QUESTIONING = "questioning"
CANDIDATE_QUESTIONS = "candidate_questions"
COMPLETED = "completed"

_DIMENSIONS = ("knowledge", "reasoning", "depth", "communication", "adaptability")
_DIMENSION_LABELS = {
    "knowledge": "知识与经验",
    "reasoning": "分析与推理",
    "depth": "回答深度",
    "communication": "表达沟通",
    "adaptability": "应变与反思",
}


@dataclass(slots=True)
class InterviewSession:
    id: str
    mode: str
    phase: str
    target_rounds: int
    difficulty_policy: str
    difficulty: int
    domains: tuple[str, ...]
    completed_rounds: int = 0
    turn_index: int = 0
    current_question_id: str | None = None
    current_question: str = ""
    current_domain: str | None = None
    current_reference_answer: str = ""
    current_scoring_points: str = ""
    current_sources: tuple[str, ...] = ()
    current_followups: int = 0
    asked_question_ids: tuple[str, ...] = ()
    introduction: str = ""
    dimension_totals: dict[str, list[float]] = field(default_factory=dict)
    domain_scores: dict[str, list[float]] = field(default_factory=dict)
    base_question: str = ""
    question_answers: list[str] = field(default_factory=list)
    current_clarifications: int = 0
    interviewer_reply: str = ""
    turn_kind: str = "question"
    topics: tuple[str, ...] = ()
    current_knowledge_points: tuple[str, ...] = ()
    blocked_knowledge_points: tuple[str, ...] = ()
    next_action: str = "new_topic"
    next_query: str = ""
    related_rounds: int = 0
    created_at: str = ""

    @classmethod
    def from_data(cls, data: dict[str, object]) -> InterviewSession:
        values = dict(data)
        for key in (
            "domains", "current_sources", "asked_question_ids", "topics",
            "current_knowledge_points", "blocked_knowledge_points",
        ):
            values[key] = tuple(values.get(key, ()))
        values["dimension_totals"] = {
            str(key): [float(item) for item in value]
            for key, value in dict(values.get("dimension_totals", {})).items()
        }
        values["domain_scores"] = {
            str(key): [float(item) for item in value]
            for key, value in dict(values.get("domain_scores", {})).items()
        }
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class InterviewStep:
    session_id: str
    phase: str
    interviewer_message: str
    decision: str
    report: str | None = None
    report_path: str | None = None


class InterviewEngine:
    def __init__(self, rag: TechRAG, *, report_dir: str | Path = "reports") -> None:
        self.rag = rag
        self.store = InterviewStore(rag.database)
        self.knowledge = KnowledgeService(rag)
        self.report_dir = Path(report_dir)

    def start(
        self,
        *,
        mode: InterviewMode = "technical",
        rounds: int = 5,
        difficulty_policy: DifficultyPolicy = "adaptive",
        difficulty: int = 2,
        domains: tuple[str, ...] | None = None,
        topics: tuple[str, ...] = (),
    ) -> InterviewStep:
        if rounds not in {5, 10, 15}:
            raise ValueError("rounds must be one of 5, 10, or 15")
        if difficulty not in {1, 2, 3}:
            raise ValueError("difficulty must be 1, 2, or 3")
        available = {
            question.domain for question in self._question_bank(mode)
        }
        selected_domains = tuple(dict.fromkeys(domains or sorted(available)))
        if domains and set(domains) - available:
            missing_domains = sorted(set(domains) - available)
            raise ValueError(f"no indexed questions for domains: {missing_domains}")
        session = InterviewSession(
            id=uuid.uuid4().hex,
            mode=mode,
            phase=SELF_INTRODUCTION,
            target_rounds=rounds,
            difficulty_policy=difficulty_policy,
            difficulty=difficulty,
            domains=tuple(selected_domains),
            topics=topics,
            created_at=datetime.now(UTC).isoformat(),
            current_question_id="opening-self-introduction",
            current_question="请先用一到两分钟做自我介绍，并重点说明与你目标岗位相关的项目经历。",
        )
        self.store.save_session(session)
        return self._step(session, "start")

    def submit(self, session_id: str, answer: str) -> InterviewStep:
        session = self._load(session_id)
        cleaned = answer.strip()
        if not cleaned:
            raise ValueError("answer cannot be empty")
        if session.phase == COMPLETED:
            raise ValueError("interview is already complete")

        if session.phase == SELF_INTRODUCTION:
            self._record(session, cleaned, None, {"action": "project_deep_dive"})
            session.introduction = cleaned
            session.phase = PROJECT_DEEP_DIVE
            session.current_question_id = "project-deep-dive-1"
            session.current_question = self._project_question(cleaned)
            session.current_domain = "project"
            session.current_reference_answer = ""
            session.current_scoring_points = "说明个人职责、技术选择、实现细节和可验证结果。"
            session.current_sources = ()
            session.interviewer_reply = "谢谢你的介绍，我们先聊聊你刚才提到的项目。"
            self.store.save_session(session)
            return self._step(session, "project_deep_dive")

        if session.phase == PROJECT_DEEP_DIVE:
            reply = self._conversational_response(session, cleaned, "transition")
            session.next_query = (
                f"自我介绍：{session.introduction[:1000]}\n项目回答：{cleaned[:1200]}"
            )
            self._record(session, cleaned, None, {"action": "enter_questioning"})
            session.phase = QUESTIONING
            session.current_followups = 0
            question = self._next_question(session)
            if question is None:
                self._end_questioning(session, exhausted=True)
            else:
                self._set_question(session, question)
            session.interviewer_reply = reply
            self.store.save_session(session)
            return self._step(session, "retrieve_question")

        if session.phase == CANDIDATE_QUESTIONS:
            result = self.knowledge.answer(cleaned, interview=True)
            evidence = result["sources"]
            session.current_sources = tuple(
                f"[{source['id']}] {source['title']} ({source['source']})" for source in evidence
            )
            session.interviewer_reply = result["answer"]
            if evidence:
                session.interviewer_reply += "\n\n参考资料：\n" + "\n".join(session.current_sources)
            if result["warnings"]:
                session.interviewer_reply += "\n\n" + "\n".join(result["warnings"])
            session.interviewer_reply += "\n\n本场模拟面试结束，感谢你的参与。"
            self._record(session, cleaned, None, {
                "action": "finish", "knowledge": {
                    "sources": [{key: source[key] for key in ("id", "title", "source")}
                                for source in evidence],
                    "questions": [section["question"] for section in result["sections"]],
                    "warnings": result["warnings"],
                },
            })
            session.phase = COMPLETED
            session.current_question_id = None
            session.current_question = ""
            self.store.save_session(session)
            report = self.report(session.id)
            try:
                report_path = self.export_report(session.id, markdown=report)
            except Exception as error:
                self.store.save_report(
                    session_id=session.id,
                    markdown_content=report,
                    pdf_path=None,
                    checksum=None,
                    status="failed",
                    error_message=str(error),
                )
                return self._step(session, "finish_pdf_failed", report=report)
            self._clear_transient_state(session)
            self.store.save_session(session)
            return self._step(
                session,
                "finish",
                report=report,
                report_path=str(report_path),
            )

        # Older saved sessions have no base-question or conversation fields.
        if not session.base_question:
            session.base_question = session.current_question
        if not session.current_knowledge_points:
            question_id = (session.current_question_id or "").split(":")[0]
            question = next((q for q in self._question_bank(session.mode)
                             if q.id == question_id), None)
            if question is not None:
                session.current_knowledge_points = question.routing_points
        intent = _answer_intent(cleaned)
        if intent == "clarification":
            session.interviewer_reply = self._conversational_response(
                session, cleaned, "clarification"
            ) if session.current_clarifications < 2 else (
                "我们先按你对问题的理解作答；如果暂时没有思路，也可以说跳过。"
            )
            session.current_clarifications += 1
            session.turn_kind = "clarification"
            self._record(session, cleaned, None, {"action": "clarification"})
            self.store.save_session(session)
            return self._step(session, "clarification")

        session.question_answers.append(cleaned)
        has_prior_answer = any(
            _answer_intent(item) == "answer" for item in session.question_answers[:-1]
        )
        if intent == "skip" or (intent == "unknown" and not has_prior_answer):
            evaluation = {
                "scores": dict.fromkeys(_DIMENSIONS, 0.0),
                "overall": 0.0,
                "missing_points": ["未能回答当前问题"],
                "incorrect_points": [],
                "feedback": "候选人明确表示不会或选择跳过。",
                "scoring_method": "explicit_skip",
                "point_reviews": point_reviews(
                    [{"point_id": p["id"], "status": "not_addressed", "evidence": [],
                      "reason": "候选人未作答，明确表示不会或跳过。"}
                     for p in rubric_points(session.current_scoring_points)]
                    if not has_prior_answer else [],
                    rubric_points(session.current_scoring_points), session.question_answers,
                    unavailable_reason="候选人结束本题，已有回答的逐点掌握情况待核验。",
                ),
            }
        else:
            evaluation = self._evaluate(session, cleaned)
        overall = float(evaluation["overall"])

        needs_followup = overall < 60 or evaluation.get("needs_followup") is True
        unmastered = _point_keys(_string_list(evaluation.get("unmastered_points")))
        if (intent == "answer" and needs_followup and session.current_followups < 1
                and not unmastered.intersection(_point_keys(session.current_knowledge_points))):
            reply = self._follow_up(session, evaluation, cleaned)
            self._record(session, cleaned, evaluation, {
                "action": "followup", "reason": "probe answer",
                "score": overall,
                "round_complete": False,
                "interviewer_reply": reply,
            })
            session.current_followups += 1
            session.current_question_id = f"{session.current_question_id}:followup"
            session.current_question = reply
            session.interviewer_reply = ""
            session.turn_kind = "followup"
            self.store.save_session(session)
            return self._step(session, session.turn_kind)

        if intent == "skip":
            reply = "好的，这题先跳过，我会把它记为待复习的知识点。"
        elif intent == "unknown":
            reply = "没关系，这题先到这里，我会在复盘中列出需要补充的知识点。"
        else:
            reply = str(evaluation.get("interviewer_reply") or evaluation["feedback"])
        session.interviewer_reply = reply
        self._plan_next_question(session, evaluation, intent)
        self._accumulate(session, evaluation, overall)
        session.completed_rounds += 1

        if session.completed_rounds >= session.target_rounds:
            decision = {
                "action": "candidate_questions", "reason": "target rounds reached",
                "round_complete": True, "interviewer_reply": reply,
            }
            self._record(session, cleaned, evaluation, decision)
            self._end_questioning(session)
            self.store.save_session(session)
            return self._step(session, "candidate_questions")

        previous = session.difficulty
        if session.difficulty_policy == "adaptive":
            if overall >= 80:
                session.difficulty = min(3, session.difficulty + 1)
            elif overall < 50:
                session.difficulty = max(1, session.difficulty - 1)
        question = self._next_question(session)
        action = session.next_action if question is not None else "scope_exhausted"
        decision = {
            "action": action, "score": overall, "difficulty": session.difficulty,
            "previous_difficulty": previous, "retrieval_query": session.next_query,
            "blocked_knowledge_points": list(session.blocked_knowledge_points),
            "round_complete": True, "interviewer_reply": reply,
            "next_question_id": question.id if question is not None else None,
        }
        self._record(session, cleaned, evaluation, decision)
        session.current_followups = 0
        if question is None:
            self._end_questioning(session, exhausted=True)
            action = "scope_exhausted"
        else:
            self._set_question(session, question)
        self.store.save_session(session)
        return self._step(session, action)

    def finish(self, session_id: str) -> InterviewStep:
        """End early, retaining completed answers and the latest scored follow-up."""
        session = self._load(session_id)
        if session.phase == COMPLETED:
            stored = self.store.report_data(session_id) or {}
            return self._step(
                session, "finish", report=stored.get("markdown_content"),
                report_path=stored.get("pdf_path"),
            )
        if session.phase == QUESTIONING and session.question_answers:
            turns = self.store.turns(session_id)
            pending = next((turn for turn in reversed(turns)
                            if turn.get("evaluation") is not None), None)
            if pending and (pending.get("decision") or {}).get("round_complete") is False:
                evaluation = pending["evaluation"]
                self._accumulate(session, evaluation, float(evaluation["overall"]))
                session.completed_rounds += 1
        session.phase = COMPLETED
        session.current_question_id = None
        session.current_question = ""
        session.interviewer_reply = "本场面试已结束，已作答的内容会保留在复盘报告中。"
        self.store.save_session(session)
        report = self.report(session_id)
        try:
            path = self.export_report(session_id, markdown=report)
        except Exception:
            return self._step(session, "finish_pdf_failed", report=report)
        self._clear_transient_state(session)
        self.store.save_session(session)
        return self._step(session, "finish", report=report, report_path=str(path))

    def current(self, session_id: str) -> InterviewStep:
        """Return the persisted interview position without advancing the session."""

        session = self._load(session_id)
        report = self.store.report_data(session_id)
        return self._step(
            session,
            "resume",
            report_path=str(report["pdf_path"]) if report and report.get("pdf_path") else None,
        )

    def report(self, session_id: str) -> str:
        session = self._load(session_id)
        turns = self.store.turns(session_id)
        averages = {
            dimension: round(mean(values), 1) if values else 0.0
            for dimension, values in session.dimension_totals.items()
        }
        for dimension in _DIMENSIONS:
            averages.setdefault(dimension, 0.0)
        domain_averages = {
            domain: round(mean(values), 1) for domain, values in session.domain_scores.items()
        }
        weak_domains = [
            domain for domain, _ in sorted(domain_averages.items(), key=lambda item: item[1])[:3]
        ]
        recommendations = self._recommend(session, weak_domains)
        lines = [
            f"# 面试报告：{session.id}",
            "",
            f"- 模式：{session.mode}",
            f"- 状态：{session.phase}",
            f"- 已完成正式轮次：{session.completed_rounds}/{session.target_rounds}",
            f"- 面试范围：{'、'.join(session.domains)}",
            f"- 限定知识点：{'、'.join(session.topics) or '未限定'}",
            f"- 本场跳过的知识点：{'、'.join(session.blocked_knowledge_points) or '暂无'}",
            "",
            "## 五维能力",
            "",
            *[
                f"- {_DIMENSION_LABELS[dimension]}：{averages[dimension]:.1f}"
                for dimension in _DIMENSIONS
            ],
            "",
            "## 知识域表现",
            "",
            *(
                [f"- {domain}：{score:.1f}" for domain, score in domain_averages.items()]
                or ["- 暂无正式题评分"]
            ),
            "",
            "## 短板域",
            "",
            *( [f"- {domain}" for domain in weak_domains] or ["- 暂无"] ),
            "",
            "## 逐题复盘",
            "",
        ]
        for turn in turns:
            evaluation = turn["evaluation"] or {}
            lines.extend(
                [
                    f"### 第 {int(turn['turn_index']) + 1} 次对话｜{turn['question']}",
                    "",
                    f"- 回答：{turn['answer']}",
                    f"- 得分：{evaluation.get('overall', '未评分')}",
                    f"- 反馈：{evaluation.get('feedback', '流程环节')}",
                    f"- 决策：{json.dumps(turn['decision'], ensure_ascii=False)}",
                    f"- 证据：{', '.join(turn['sources']) or '无'}",
                    "",
                ]
            )
            if evaluation.get("point_reviews"):
                lines.append("逐项评分依据：")
                for point in evaluation["point_reviews"]:
                    lines.append(f"- {point['point_id']}｜{point['label']}：{point['point']}")
                    lines.append(f"  判断说明：{point['reason']}")
                    for evidence in point["evidence"]:
                        lines.append(
                            f"  第 {evidence['answer_index']} 次回答原文：{evidence['quote']}"
                        )
                lines.append("")
        lines.extend(["## 学习推荐", "", *(recommendations or ["- 暂无推荐"]), ""])
        return "\n".join(lines)

    def export_report(
        self,
        session_id: str,
        output_path: str | Path | None = None,
        *,
        markdown: str | None = None,
    ) -> Path:
        """Generate and register a PDF report without deleting review history."""

        self._load(session_id)
        content = markdown or self.report(session_id)
        destination = Path(output_path) if output_path else (
            self.report_dir / f"interview-{session_id}.pdf"
        )
        try:
            path = export_report_pdf(content, destination)
            checksum = sha256(path.read_bytes()).hexdigest()
            self.store.save_report(
                session_id=session_id,
                markdown_content=content,
                pdf_path=str(path),
                checksum=checksum,
                status="ready",
            )
            return path
        except Exception as error:
            self.store.save_report(
                session_id=session_id,
                markdown_content=content,
                pdf_path=None,
                checksum=None,
                status="failed",
                error_message=str(error),
            )
            raise

    def delete(self, session_id: str, *, keep_pdf: bool = False) -> bool:
        """Delete a session and its structured history, optionally retaining its PDF."""

        report = self.store.report_data(session_id)
        if report is None and self.store.load_session_data(session_id) is None:
            return False
        if not keep_pdf and report and report.get("pdf_path"):
            path = Path(str(report["pdf_path"]))
            if path.is_file():
                path.unlink()
        return self.store.delete_session(session_id)

    def _load(self, session_id: str) -> InterviewSession:
        data = self.store.load_session_data(session_id)
        if not data:
            raise ValueError(f"interview session not found: {session_id}")
        return InterviewSession.from_data(data)

    def _step(
        self,
        session: InterviewSession,
        decision: str,
        *,
        report: str | None = None,
        report_path: str | None = None,
    ) -> InterviewStep:
        return InterviewStep(
            session_id=session.id,
            phase=session.phase,
            interviewer_message="\n".join(
                part for part in (session.interviewer_reply, session.current_question) if part
            ),
            decision=session.turn_kind if decision == "resume" else decision,
            report=report,
            report_path=report_path,
        )

    @staticmethod
    def _clear_transient_state(session: InterviewSession) -> None:
        """Remove duplicated runtime scratch data only after PDF export succeeds."""

        session.current_reference_answer = ""
        session.current_scoring_points = ""
        session.current_sources = ()
        session.current_followups = 0
        session.asked_question_ids = ()
        session.introduction = ""
        session.base_question = ""
        session.question_answers = []
        session.current_clarifications = 0
        session.current_knowledge_points = ()
        session.next_query = ""

    def _record(
        self,
        session: InterviewSession,
        answer: str,
        evaluation: dict[str, object] | None,
        decision: dict[str, object],
    ) -> None:
        self.store.append_turn(
            session_id=session.id,
            turn_index=session.turn_index,
            phase=session.phase,
            domain=session.current_domain,
            question_id=session.current_question_id,
            question=session.current_question,
            answer=answer,
            evaluation=evaluation,
            decision=decision,
            sources=session.current_sources,
        )
        session.turn_index += 1

    def _project_question(self, introduction: str) -> str:
        prompt = (
            "你是计算机校招面试官。根据自我介绍生成一道项目深挖题。"
            "只输出问题，不要解释；重点追问个人职责、技术选择或量化结果。\n"
            f"自我介绍：{introduction}"
        )
        try:
            value = self.rag.chat_model.complete([{"role": "user", "content": prompt}]).strip()
            return value or "请选择一个你最熟悉的项目，说明你的职责和最关键的技术决策。"
        except Exception:
            return "请选择一个你最熟悉的项目，说明你的职责和最关键的技术决策。"

    def _question_bank(self, mode: str) -> list[InterviewQuestion]:
        questions = []
        for row in self.rag.database.chunk_rows():
            metadata = json.loads(str(row[6]))
            if (metadata.get("collection") == "interview-question"
                    and metadata.get("interview_type") == mode):
                questions.append(question_from_content(str(row[2]), metadata))
        return questions

    def _next_question(self, session: InterviewSession) -> InterviewQuestion | None:
        blocked = _point_keys(session.blocked_knowledge_points)
        current = _point_keys(session.current_knowledge_points)
        bank = self._question_bank(session.mode)
        # Propagate unavailable prerequisites through the bank, including across
        # domains, without reporting inferred dependent points as candidate failures.
        while True:
            unavailable = blocked | {
                point for question in bank
                if blocked.intersection(_point_keys(question.prerequisites))
                for point in _point_keys(question.routing_points)
            }
            if unavailable == blocked:
                break
            blocked = unavailable
        candidates = [
            question for question in bank
            if question.domain in session.domains
            and question.id not in session.asked_question_ids
            and not blocked.intersection(_point_keys(
                (*question.routing_points, *question.prerequisites)
            ))
            and (not session.topics or _point_keys(session.topics).intersection(
                _point_keys((*question.routing_points, *question.tags))
            ))
        ]
        if not candidates:
            return None
        # Decide topic intent before difficulty; never return to a blocked point.
        related = [q for q in candidates if current.intersection(
            _point_keys((*q.routing_points, *q.prerequisites))
        )]
        fresh = [q for q in candidates if not current.intersection(
            _point_keys((*q.routing_points, *q.prerequisites))
        )]
        if session.next_action == "extend" and related:
            candidates = related
        elif session.next_action == "new_topic" and fresh:
            candidates = fresh
        else:
            session.next_action = "new_topic"
            session.related_rounds = 0
            if session.completed_rounds:
                session.next_query = "校招面试新话题，考察尚未覆盖的知识点。"
        at_level = [q for q in candidates if q.difficulty == session.difficulty]
        if at_level:
            candidates = at_level
        # The candidate has authorized using interview text for query embeddings.
        # Scope and exclusions remain hard filters regardless of the query content.
        context = session.next_query or f"自我介绍：{session.introduction[:1000]}"
        domains = tuple(dict.fromkeys(q.domain for q in candidates))
        query = (
            f"{' '.join(domains)} 校招面试题，难度 {session.difficulty}，"
            + ("相关应用与工程权衡" if session.next_action == "extend" else "考察新的知识点")
            + f"\n{context}"
        )
        session.next_query = query
        # Filter eligible IDs before retrieval, so used/blocked top hits cannot
        # crowd out the remaining bank or force a query outside the chosen scope.
        results = self.rag.search(query, top_k=10, metadata_filter={
            "collection": "interview-question", "interview_type": session.mode,
            "question_id": [q.id for q in candidates],
        })
        eligible = {q.id: q for q in candidates}
        for result in results:
            question = question_from_content(result.content, result.metadata)
            if question.id in eligible:
                return question
        return None

    def _plan_next_question(
        self, session: InterviewSession, evaluation: dict[str, object], intent: str,
    ) -> None:
        points = session.current_knowledge_points or (session.base_question,)
        reported = _point_keys(_string_list(evaluation.get("unmastered_points")))
        blocked = list(session.blocked_knowledge_points)
        blocked.extend(point for point in points if _point_keys((point,)) & reported)
        if intent in {"unknown", "skip"}:
            blocked.extend(points)
        session.blocked_knowledge_points = tuple(dict.fromkeys(blocked))
        can_extend = (
            intent == "answer" and float(evaluation["overall"]) >= 60
            and not _point_keys(points).intersection(_point_keys(blocked))
            and session.related_rounds < 2
        )
        requested = evaluation.get("next_action")
        extend = can_extend and (
            requested == "extend"
            or (requested != "new_topic" and float(evaluation["overall"]) >= 70)
        )
        session.next_action = "extend" if extend else "new_topic"
        session.related_rounds = session.related_rounds + 1 if extend else 0
        if extend:
            focus = str(evaluation.get("next_query") or "、".join(points))[:400]
            answers = "\n".join(answer[:600] for answer in session.question_answers)
            session.next_query = (
                f"围绕 {focus} 的相关应用与工程权衡。\n"
                f"上一题：{session.base_question[:400]}\n候选人本题回答：{answers[:1200]}"
            )
        else:
            session.next_query = "校招面试新话题，考察尚未覆盖的知识点、原理理解和实践能力。"

    @staticmethod
    def _end_questioning(session: InterviewSession, *, exhausted: bool = False) -> None:
        session.phase = CANDIDATE_QUESTIONS
        session.current_question_id = "candidate-questions"
        session.current_question = (
            "本场范围内没有其他未问且适合继续考察的题目，正式问答提前结束。"
            if exhausted else "本场面试的正式问题到这里。"
        ) + "你有什么想向面试官了解的吗？"
        session.current_domain = None
        session.current_reference_answer = ""
        session.current_scoring_points = ""
        session.current_knowledge_points = ()
        session.current_sources = ()
        session.current_followups = 0
        session.turn_kind = "question"

    @staticmethod
    def _set_question(session: InterviewSession, question: InterviewQuestion) -> None:
        session.current_question_id = question.id
        session.current_question = question.text
        session.current_domain = question.domain
        session.current_knowledge_points = question.routing_points
        session.current_reference_answer = question.reference_answer
        session.current_scoring_points = question.scoring_points
        session.current_sources = (question.source_path,)
        session.asked_question_ids = (*session.asked_question_ids, question.id)
        session.base_question = question.text
        session.question_answers = []
        session.current_followups = 0
        session.current_clarifications = 0
        session.turn_kind = "question"

    def _evaluate(self, session: InterviewSession, answer: str) -> dict[str, object]:
        rubric = rubric_points(session.current_scoring_points)
        answers = list(session.question_answers) or [answer]
        numbered_answers = [{"answer_index": i, "answer": a} for i, a in enumerate(answers, 1)]
        prompt = (
            "你是严格但公平的计算机面试评分官。只返回JSON对象，不要Markdown。"
            "五个维度均为0到100：knowledge, reasoning, depth, communication, adaptability。"
            "还需返回overall、missing_points字符串数组、incorrect_points字符串数组、"
            "feedback字符串，以及interviewer_reply字符串、needs_followup布尔值。"
            "另外返回unmastered_points数组，仅列候选人明确不会的当前知识点；"
            "遗漏或表述不全不等于不会。返回next_action（extend或new_topic）及next_query。"
            "理解基础且能继续讨论时引申，否则换话题；next_query应承接候选人的具体回答。"
            "评分必须依据参考答案和评分要点，合并候选人本题的回答与补充回答评分。"
            "参考答案不是唯一正确解法：候选人给出合理的等价原理、实现或例子也应认可。"
            "不因措辞、顺序或没有背出扩展细节而扣分。遗漏不等于答错，也不等于明确不会。"
            "额外返回point_reviews数组，覆盖下面每个评分点ID恰好一次，不得新增评分点。"
            "每项字段point_id、status、reason、evidence。status只能是mastered/partial/"
            "incorrect/not_addressed/unverified。reason解释判断与认可的等价解法。"
            "evidence为数组，每项answer_index是从1开始的本题回答编号，quote必须是该次"
            "回答的连续原文，不得改写或引用参考答案。掌握、部分掌握或错误必须有回答原文支持，"
            "未涉及的evidence为空；不能可靠判断则unverified，不编造依据。"
            "interviewer_reply用一两句话直接回应候选人具体说了什么，不报分数、"
            "不空泛夸奖、不背诵参考答案。若有需要核实的关键点，needs_followup为true。\n"
            f"原题：{session.base_question or session.current_question}\n"
            f"当前追问：{session.current_question}\n"
            f"参考答案：{session.current_reference_answer}\n"
            f"评分要点：{session.current_scoring_points}\n"
            f"逐项评分点：{json.dumps(rubric, ensure_ascii=False)}\n"
            f"编号回答：{json.dumps(numbered_answers, ensure_ascii=False)}\n"
            f"当前知识点：{json.dumps(session.current_knowledge_points, ensure_ascii=False)}\n"
            f"本场方向范围：{session.domains}；知识点范围：{session.topics}\n"
            f"本题回答历史：{json.dumps(session.question_answers, ensure_ascii=False)}\n"
            f"候选人最新回答：{answer}"
        )
        try:
            raw = self.rag.chat_model.complete([{"role": "user", "content": prompt}])
            parsed = _json_object(raw)
            result = _normalize_evaluation(parsed)
            result["scoring_method"] = "model"
            result["point_reviews"] = point_reviews(parsed.get("point_reviews"), rubric, answers)
            result["point_review_complete"] = bool(rubric) and all(
                p["status"] != "unverified" for p in result["point_reviews"]
            )
            return result
        except Exception:
            result = _fallback_evaluation(
                "\n".join(session.question_answers) or answer, session.current_reference_answer
            )
            result["point_reviews"] = point_reviews(
                [], rubric, answers, unavailable_reason="评分服务不可用，未进行语义逐点评价。",
            )
            result["point_review_complete"] = False
            return result

    def _follow_up(
        self, session: InterviewSession, evaluation: dict[str, object], answer: str,
        *, hint: bool = False,
    ) -> str:
        prompt = (
            "你是正在与候选人对话的计算机面试官，请生成澄清追问。"
            "先用一句短话回应候选人的具体回答，再只问一个简短问题，总长不超过100字。"
            "不要把所有评分要点堆成问题，不要泄露完整答案，不要要求同时回答多个环节。"
            + ("候选人表示不知道：承认暂时没思路，提供一个简单场景或小提示，"
               "从比原题更基础的角度引导。\n" if hint else
               "承接候选人已经说出的概念，选择一个关键缺口往下问。\n")
            + f"原题：{session.base_question}\n"
            f"参考答案：{session.current_reference_answer}\n"
            f"候选人最新回答：{answer}\n"
            f"本题回答历史：{json.dumps(session.question_answers, ensure_ascii=False)}\n"
            f"缺失点：{evaluation.get('missing_points', [])}\n"
            f"错误点：{evaluation.get('incorrect_points', [])}\n"
            f"评分要点：{session.current_scoring_points}"
        )
        try:
            value = self.rag.chat_model.complete([{"role": "user", "content": prompt}]).strip()
            if (value and value not in {"[]", "{}"} and len(value) <= 150
                    and value.count("？") + value.count("?") <= 1):
                return value
        except Exception:
            pass
        if hint:
            return "没关系，我们从简单的角度想一想：你接触过与这道题相关的使用场景吗？"
        return f"你提到“{answer[:40]}”，能举一个具体的例子吗？"

    def _conversational_response(
        self, session: InterviewSession, answer: str, intent: str
    ) -> str:
        instructions = {
            "clarification": "候选人在请求解释问题。用简单语言解释他不明白的部分，"
            "不要透露参考答案，不换题，不评价这次请求。",
            "transition": "用一句话回应候选人的项目回答，然后自然过渡到正式问答，"
            "不提新问题，不编造候选人没说过的经历。",
            "candidate_questions": "回答候选人的反问；你是模拟面试官，没有真实公司的内部信息，"
            "不能编造团队或招聘承诺。对技术或通用问题直接作答，最后简短结束面试。",
        }
        prompt = (
            f"你是与候选人交流的面试官。{instructions[intent]}回复不超过150字。\n"
            f"原题：{session.base_question}\n"
            f"当前问题：{session.current_question}\n"
            f"自我介绍：{session.introduction}\n"
            f"本题回答历史：{json.dumps(session.question_answers, ensure_ascii=False)}\n"
            f"候选人最新发言：{answer}"
        )
        try:
            value = self.rag.chat_model.complete([{"role": "user", "content": prompt}]).strip()
            if value and value not in {"[]", "{}"}:
                return value
        except Exception:
            pass
        return {
            "clarification": "我想了解你对这个概念的理解，可以先从熟悉的例子讲起。",
            "transition": "谢谢你补充的项目经历，接下来我们聊几个相关的基础问题。",
            "candidate_questions": "你的反问已记录；目前无法生成可靠的答复。本场模拟面试结束。",
        }[intent]

    @staticmethod
    def _accumulate(
        session: InterviewSession, evaluation: dict[str, object], overall: float
    ) -> None:
        scores = dict(evaluation.get("scores", {}))
        for dimension in _DIMENSIONS:
            session.dimension_totals.setdefault(dimension, []).append(
                float(scores.get(dimension, overall))
            )
        if session.current_domain:
            session.domain_scores.setdefault(session.current_domain, []).append(overall)

    def _recommend(self, session: InterviewSession, weak_domains: list[str]) -> list[str]:
        recommendations: list[str] = []
        for domain in weak_domains:
            results = self.rag.search(
                f"{domain} 基础学习与复习题",
                top_k=2,
                metadata_filter={"collection": "interview-question", "domain": domain},
            )
            for result in results:
                try:
                    question = question_from_content(result.content, result.metadata)
                except ValueError:
                    continue
                recommendations.append(f"- [{domain}] 复习并重新回答：{question.text}")
        return list(dict.fromkeys(recommendations))


def _json_object(value: str) -> dict[str, object]:
    match = re.search(r"\{.*\}", value, re.DOTALL)
    if not match:
        raise ValueError("model did not return a JSON object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("evaluation must be an object")
    return parsed


def _normalize_evaluation(value: dict[str, object]) -> dict[str, object]:
    raw_scores = dict(value.get("scores", {}))
    scores = {
        dimension: _bounded_score(raw_scores.get(dimension, value.get("overall", 0)))
        for dimension in _DIMENSIONS
    }
    overall = _bounded_score(value.get("overall", mean(scores.values())))
    return {
        "scores": scores,
        "overall": overall,
        "missing_points": _string_list(value.get("missing_points")),
        "incorrect_points": _string_list(value.get("incorrect_points")),
        "feedback": str(value.get("feedback", "")),
        "interviewer_reply": str(value.get("interviewer_reply", "")),
        "needs_followup": value.get("needs_followup") is True,
        "unmastered_points": _string_list(value.get("unmastered_points")),
        "next_action": value.get("next_action")
        if value.get("next_action") in {"extend", "new_topic"} else "",
        "next_query": str(value.get("next_query") or "")[:400],
    }


def _fallback_evaluation(answer: str, reference: str) -> dict[str, object]:
    answer_terms = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", answer.lower()))
    reference_terms = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", reference.lower()))
    coverage = len(answer_terms & reference_terms) / max(1, len(reference_terms))
    overall = round(min(85.0, max(20.0, 20.0 + coverage * 130.0)), 1)
    scores = {dimension: overall for dimension in _DIMENSIONS}
    return {
        "scores": scores,
        "overall": overall,
        "missing_points": ["模型评分不可用，已使用词项覆盖率降级评分"],
        "incorrect_points": [],
        "feedback": "当前结果为降级评分，正式报告建议使用模型评分。",
        "scoring_method": "fallback",
        "interviewer_reply": "评分服务暂时不可用，这题的评价会在报告中标注为降级评分。",
    }


def _bounded_score(value: object) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _point_keys(points: tuple[str, ...] | list[str]) -> set[str]:
    return {point.strip().casefold() for point in points if point.strip()}


def _answer_intent(answer: str) -> str:
    normalized = re.sub(r"[\s，。！？,.!?]", "", answer).lower()
    if normalized in {"跳过", "跳过这题", "下一题", "换一题", "skip", "pass"}:
        return "skip"
    if normalized in {
        "不知道", "我不知道", "不清楚", "不会", "我不会", "没思路", "不知道怎么回答",
        "不太清楚", "没学过", "不了解", "不知道啊", "完全不知道", "idontknow",
    }:
        return "unknown"
    if re.fullmatch(
        r"(这题|这个|这方面|这个问题)?(我)?(不太会|不太了解|不知道|不会|没思路)(啊|呀|呢)?",
        normalized,
    ):
        return "unknown"
    if normalized in {"什么意思", "没明白", "解释一下", "给点提示", "能给点提示吗"}:
        return "clarification"
    if re.search(r"(没听懂|没看懂|不理解题意|题目.*什么意思|问题.*什么意思|"
                 r"解释一下.{0,8}(问题|题目)|(?:能否|能不能|可以|请|麻烦).{0,5}(解释|提示)|"
                 r"能.{0,5}(解释|提示).{0,6}[吗么？?]|给.{0,3}提示|再说一遍)", answer):
        return "clarification"
    return "answer"


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value else []
