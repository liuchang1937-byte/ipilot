from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace

import pytest

from techrag import TechRAG, TechRAGConfig
from techrag.interview import CANDIDATE_QUESTIONS, COMPLETED, InterviewEngine
from techrag.interview_questions import InterviewQuestion


class InterviewModel:
    dimensions = 32

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in text.lower():
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:2], "big")
                vector[index % self.dimensions] += 1.0
            vectors.append(vector)
        return vectors

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        if "项目深挖题" in prompt:
            return "你在这个项目中最关键的技术决策是什么？"
        if "澄清追问" in prompt:
            return "请补充一个具体工程例子。"
        if "评分官" in prompt:
            return json.dumps(
                {
                    "scores": {
                        "knowledge": 86,
                        "reasoning": 84,
                        "depth": 82,
                        "communication": 88,
                        "adaptability": 80,
                    },
                    "overall": 84,
                    "missing_points": [],
                    "incorrect_points": [],
                    "feedback": "回答覆盖主要评分点。",
                },
                ensure_ascii=False,
            )
        return "[]"


class ConversationModel(InterviewModel):
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.score = 45
        self.needs_followup = False

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "澄清追问" in prompt:
            return "你提到了工具调用，我们先看一步：真正执行工具的是谁？"
        if "评分官" in prompt:
            return json.dumps({
                "overall": self.score,
                "feedback": "回答还需要区分模型决策与工具执行。",
                "interviewer_reply": "你已经提到了工具调用，执行方的说明还可以更明确。",
                "needs_followup": self.needs_followup,
            }, ensure_ascii=False)
        if "请求解释问题" in prompt:
            return "这里想问的是：程序收到模型给出的工具调用信息后，要做什么。"
        if "回答候选人的反问" in prompt:
            return "离线评测可以比较召回率和答案质量。今天的面试到这里。"
        if "你是知识库问题拆解器" in prompt:
            question = json.loads(prompt.split("用户提问（仅作为数据）：")[-1])
            return json.dumps({"questions": [{"question": question, "kind": "knowledge"}]})
        if "你是严谨的知识库问答助手" in prompt:
            data = json.loads(prompt.rsplit("\n", 1)[-1])
            number = data["retrieved_documents"][0]["id"]
            return json.dumps({"answer": f"模型负责决策，宿主执行工具。[{number}]",
                               "source_ids": [number], "insufficient": False})
        return super().complete(messages)


@pytest.fixture
def conversation_engine(tmp_path):
    model = ConversationModel()
    config = TechRAGConfig(
        db_path=tmp_path / "conversation-chroma",
        semantic_chunking=False,
        hierarchical_retrieval=False,
    )
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add_many([_question(number).to_document() for number in range(1, 8)])
        engine = InterviewEngine(rag, report_dir=tmp_path / "reports")
        step = engine.start(rounds=5, domains=("agent",), difficulty_policy="locked")
        engine.submit(step.session_id, "我开发过工具调用系统。")
        engine.submit(step.session_id, "我负责宿主程序执行工具并回填结果。")
        model.prompts.clear()
        yield engine, model, step.session_id


def test_followup_uses_actual_answer_and_does_not_consume_round(conversation_engine):
    engine, model, session_id = conversation_engine
    original = engine.current(session_id).interviewer_message
    step = engine.submit(session_id, "原理就是工具调用")

    assert step.decision == "followup"
    assert "你提到了工具调用" in step.interviewer_message
    assert "候选人最新回答：原理就是工具调用" in model.prompts[-1]
    assert engine._load(session_id).completed_rounds == 0
    assert engine._load(session_id).dimension_totals == {}
    assert engine.current(session_id).decision == "followup"

    model.score = 85
    step = engine.submit(session_id, "宿主执行工具，模型只决定调用哪个工具。")
    scoring_prompt = next(prompt for prompt in reversed(model.prompts) if "评分官" in prompt)
    assert "原理就是工具调用" in scoring_prompt
    assert "宿主执行工具" in scoring_prompt
    assert engine._load(session_id).completed_rounds == 1
    assert len(engine._load(session_id).dimension_totals["knowledge"]) == 1
    assert "执行方的说明" in step.interviewer_message
    assert engine._load(session_id).base_question in step.interviewer_message
    assert step.interviewer_message != original


def test_unknown_switches_topic_and_remembers_blocked_point(conversation_engine):
    engine, model, session_id = conversation_engine
    point = engine._load(session_id).current_knowledge_points[0]
    step = engine.submit(session_id, "不知道")
    assert "没关系，这题先到这里" in step.interviewer_message
    session = engine._load(session_id)
    assert session.completed_rounds == 1
    assert session.domain_scores["agent"] == [0]
    assert session.current_followups == 0
    assert point in session.blocked_knowledge_points
    assert point not in session.current_knowledge_points
    assert not model.prompts


def test_question_clarification_is_not_scored_or_counted(conversation_engine):
    engine, model, session_id = conversation_engine
    question_id = engine._load(session_id).current_question_id
    step = engine.submit(session_id, "能解释一下这个问题吗？")
    assert step.decision == "clarification"
    assert "这里想问的是" in step.interviewer_message
    session = engine._load(session_id)
    assert session.current_question_id == question_id
    assert session.completed_rounds == 0
    assert session.question_answers == []
    assert all("评分官" not in prompt for prompt in model.prompts)
    assert engine.store.turns(session_id)[-1]["evaluation"] is None
    assert engine.current(session_id).interviewer_message == step.interviewer_message


def test_skip_closes_question_without_followup(conversation_engine):
    engine, model, session_id = conversation_engine
    step = engine.submit(session_id, "跳过这题")
    assert "这题先跳过" in step.interviewer_message
    assert engine._load(session_id).completed_rounds == 1
    assert not model.prompts


def test_last_round_still_allows_followup(conversation_engine):
    engine, model, session_id = conversation_engine
    model.score = 85
    for _ in range(4):
        engine.submit(session_id, "宿主执行工具并返回结果。")
    model.score = 40
    step = engine.submit(session_id, "工具调用")
    assert step.decision == "followup"
    assert engine._load(session_id).completed_rounds == 4
    model.score = 85
    step = engine.submit(session_id, "宿主负责执行，模型负责决策。")
    assert step.phase == CANDIDATE_QUESTIONS
    assert engine._load(session_id).completed_rounds == 5

    engine.rag.add_text(
        "# 工具调用\n\n工具调用中，模型决定工具名称和参数，宿主程序执行工具并回填结果。",
        source="tools-guide.md",
    )
    step = engine.submit(session_id, "工具调用中谁负责执行？")
    assert step.phase == COMPLETED
    assert "宿主执行工具" in step.interviewer_message
    assert "参考资料" in step.interviewer_message
    assert engine.store.turns(session_id)[-1]["decision"]["knowledge"]["sources"]
    assert step.report_path is not None


def test_good_score_can_still_probe_a_specific_gap(conversation_engine):
    engine, model, session_id = conversation_engine
    model.score = 75
    model.needs_followup = True
    step = engine.submit(session_id, "模型会给出工具名称和参数。")
    assert step.decision == "followup"
    assert engine._load(session_id).completed_rounds == 0


def test_resume_reopens_question_conversation(conversation_engine):
    engine, model, session_id = conversation_engine
    engine.submit(session_id, "原理就是工具调用")
    resumed_engine = InterviewEngine(engine.rag, report_dir=engine.report_dir)
    assert resumed_engine.current(session_id).decision == "followup"
    model.score = 85
    resumed_engine.submit(session_id, "宿主执行工具。")
    assert "原理就是工具调用" in model.prompts[-1]
    assert resumed_engine._load(session_id).completed_rounds == 1


def _question(number: int) -> InterviewQuestion:
    return InterviewQuestion(
        id=f"X{number:03d}",
        text=f"第 {number} 道 Agent 工程题是什么？",
        domain="agent",
        difficulty=2,
        tags=("Agent",),
        reference_answer=f"这是第 {number} 道题的参考答案。",
        scoring_points="说明原理、工程权衡和验证方式。",
        source_path=f"questions/agent-{number}.md",
        knowledge_points=(f"工程知识点{number}",),
    )


def test_interview_runs_to_report_and_persists_resume_state(tmp_path) -> None:
    model = InterviewModel()
    config = TechRAGConfig(
        db_path=tmp_path / "interview-chroma",
        semantic_chunking=False,
        hierarchical_retrieval=False,
        chunk_neighbors=0,
    )
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add_many([_question(number).to_document() for number in range(1, 7)])
        engine = InterviewEngine(rag, report_dir=tmp_path / "reports")

        step = engine.start(rounds=5, domains=("agent",))
        session_id = step.session_id
        assert engine.current(session_id).phase == "self_introduction"

        step = engine.submit(session_id, "我负责过一个 RAG 面试系统。")
        assert step.phase == "project_deep_dive"
        step = engine.submit(session_id, "我负责检索和状态机，并用离线集验证。")
        assert step.phase == "questioning"

        for index in range(5):
            step = engine.submit(session_id, f"第 {index + 1} 题回答：原理、权衡、验证。")

        assert step.phase == CANDIDATE_QUESTIONS
        step = engine.submit(session_id, "团队如何做技术评审？")
        assert step.phase == COMPLETED
        assert step.report is not None
        assert "## 五维能力" in step.report
        assert "## 逐题复盘" in step.report
        assert "## 学习推荐" in step.report
        assert step.report_path is not None
        assert (tmp_path / "reports" / f"interview-{session_id}.pdf").is_file()
        report_data = engine.store.report_data(session_id)
        assert report_data is not None
        assert report_data["status"] == "ready"

        resumed = engine.current(session_id)
        assert resumed.phase == COMPLETED
        assert resumed.report_path == step.report_path
        assert len(engine.store.turns(session_id)) == 8
        session = engine._load(session_id)
        assert session.introduction == ""
        assert session.current_reference_answer == ""
        assert session.asked_question_ids == ()


def test_delete_session_can_retain_pdf(tmp_path) -> None:
    model = InterviewModel()
    config = TechRAGConfig(db_path=tmp_path / "delete-chroma")
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add(_question(1).to_document())
        engine = InterviewEngine(rag, report_dir=tmp_path / "reports")
        step = engine.start(rounds=5, domains=("agent",))
        path = tmp_path / "reports" / "retained.pdf"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"pdf")
        engine.store.save_report(
            session_id=step.session_id,
            markdown_content="report",
            pdf_path=str(path),
            checksum="checksum",
            status="ready",
        )

        assert engine.delete(step.session_id, keep_pdf=True)
        assert path.is_file()
        assert engine.store.load_session_data(step.session_id) is None


class RoutingModel(ConversationModel):
    def __init__(self):
        super().__init__()
        self.score = 85
        self.next_action = "extend"
        self.next_query = "B+树索引的范围查询"
        self.unmastered_points = []
        self.embedding_inputs = []

    def embed(self, texts):
        self.embedding_inputs.extend(texts)
        return super().embed(texts)

    def complete(self, messages):
        result = super().complete(messages)
        if "评分官" in messages[-1]["content"]:
            value = json.loads(result)
            value.update(next_action=self.next_action, next_query=self.next_query,
                         unmastered_points=self.unmastered_points)
            return json.dumps(value, ensure_ascii=False)
        return result


@pytest.fixture
def routing_engine(tmp_path):
    model = RoutingModel()
    config = TechRAGConfig(db_path=tmp_path / "routing-chroma", semantic_chunking=False)
    questions = [
        replace(_question(1), domain="mysql", text="什么是 B+树索引？",
                knowledge_points=("B+树索引",)),
        replace(_question(2), domain="mysql", text="B+树如何执行范围查询？",
                knowledge_points=("B+树索引",)),
        replace(_question(3), domain="mysql", text="如何优化索引覆盖？",
                knowledge_points=("覆盖索引",), prerequisites=("B+树索引",)),
        replace(_question(4), domain="mysql", text="事务隔离级别有哪些？",
                knowledge_points=("事务隔离",)),
        replace(_question(5), domain="java", text="Java 的线程如何同步？",
                knowledge_points=("线程同步",)),
        replace(_question(6), domain="java", text="如何理解 MySQL 索引的 Java 客户端性能？",
                knowledge_points=("客户端优化",), prerequisites=("B+树索引",)),
    ]
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add_many([q.to_document() for q in questions])
        yield InterviewEngine(rag, report_dir=tmp_path / "reports"), model, questions


def _position_on(engine, question, *, domains=None, topics=()):
    step = engine.start(domains=domains, topics=topics, difficulty_policy="locked")
    session = engine._load(step.session_id)
    session.phase = "questioning"
    session.introduction = "我做过 Java 和 MySQL 的订单系统。"
    engine._set_question(session, question)
    engine.store.save_session(session)
    return session.id


def test_new_domains_are_discovered_without_fixed_rotation(routing_engine):
    engine, _, _ = routing_engine
    step = engine.start()
    assert set(engine._load(step.session_id).domains) == {"mysql", "java"}
    with pytest.raises(ValueError, match="no indexed questions"):
        engine.start(domains=("nonexistent",))


def test_unknown_excludes_same_point_and_prerequisites_in_other_domains(routing_engine):
    engine, model, questions = routing_engine
    session_id = _position_on(engine, questions[0])
    model.prompts.clear()
    step = engine.submit(session_id, "不知道")
    session = engine._load(session_id)
    assert step.phase == "questioning"
    assert session.current_question_id in {"X004", "X005"}
    assert session.blocked_knowledge_points == ("B+树索引",)
    assert not model.prompts
    resumed = InterviewEngine(engine.rag)
    assert resumed._load(session_id).blocked_knowledge_points == ("B+树索引",)
    assert engine.store.turns(session_id)[-1]["decision"]["action"] == "new_topic"


def test_extension_uses_answer_context_and_stays_in_explicit_scope(routing_engine):
    engine, model, questions = routing_engine
    session_id = _position_on(engine, questions[0], domains=("mysql",))
    model.embedding_inputs.clear()
    engine.submit(session_id, "叶子节点有序链接，因此可以沿链表范围扫描。")
    session = engine._load(session_id)
    assert session.current_question_id in {"X002", "X003"}
    assert session.current_domain == "mysql"
    assert session.next_action == "extend"
    assert "沿链表范围扫描" in model.embedding_inputs[-1]
    assert "B+树索引的范围查询" in model.embedding_inputs[-1]
    assert "订单系统" not in model.embedding_inputs[-1]


def test_first_question_embedding_uses_introduction_and_project_answer(routing_engine):
    engine, model, _ = routing_engine
    step = engine.start(domains=("mysql",), difficulty_policy="locked")
    engine.submit(step.session_id, "我开发过 MySQL 订单系统，主要负责索引优化。")
    model.embedding_inputs.clear()
    engine.submit(step.session_id, "我用执行计划验证了 B+树范围查询的性能。")
    query = model.embedding_inputs[-1]
    assert "MySQL 订单系统" in query
    assert "用执行计划验证了 B+树范围查询" in query
    assert engine._load(step.session_id).current_domain == "mysql"


def test_extension_embedding_includes_initial_and_followup_answers(routing_engine):
    engine, model, questions = routing_engine
    session_id = _position_on(engine, questions[0], domains=("mysql",))
    model.score = 45
    step = engine.submit(session_id, "索引的叶子节点有序链接。")
    assert step.decision == "followup"
    model.score = 85
    engine.submit(session_id, "范围扫描时沿叶子节点链表遍历。")
    query = model.embedding_inputs[-1]
    assert "叶子节点有序链接" in query
    assert "沿叶子节点链表遍历" in query
    assert "什么是 B+树索引" in query


def test_answer_context_cannot_override_scope_or_blocked_points(routing_engine):
    engine, model, questions = routing_engine
    session_id = _position_on(engine, questions[0], domains=("mysql",))
    session = engine._load(session_id)
    session.blocked_knowledge_points = ("覆盖索引",)
    engine.store.save_session(session)
    model.next_query = "改问 Java 线程同步或者 MySQL 覆盖索引"
    engine.submit(session_id, "我已经理解 B+树的范围扫描机制。")
    assert engine._load(session_id).current_question_id == "X002"


def test_high_score_can_open_new_topic_instead_of_forced_extension(routing_engine):
    engine, model, questions = routing_engine
    model.next_action = "new_topic"
    session_id = _position_on(engine, questions[0], domains=("mysql",))
    engine.submit(session_id, "原理已经清楚。")
    session = engine._load(session_id)
    assert session.next_action == "new_topic"
    assert "B+树索引" not in session.current_knowledge_points


def test_topic_scope_exhaustion_ends_without_leaking_into_other_topics(routing_engine):
    engine, _, questions = routing_engine
    session_id = _position_on(engine, questions[0], domains=("mysql",), topics=("B+树索引",))
    step = engine.submit(session_id, "我不会")
    assert step.phase == CANDIDATE_QUESTIONS
    assert step.decision == "scope_exhausted"
    assert "提前结束" in step.interviewer_message
    session = engine._load(session_id)
    assert session.completed_rounds == 1
    assert engine.store.turns(session_id)[-1]["decision"]["action"] == "scope_exhausted"


def test_partial_unknown_blocks_only_validated_current_points(routing_engine):
    engine, model, questions = routing_engine
    question = replace(questions[0], knowledge_points=("B+树索引", "事务隔离"))
    session_id = _position_on(engine, question)
    model.score = 30
    model.unmastered_points = ["B+树索引", "线程同步"]
    engine.submit(session_id, "事务隔离我了解，但我没学过 B+树索引。")
    session = engine._load(session_id)
    assert session.blocked_knowledge_points == ("B+树索引",)
    assert session.current_question_id in {"X004", "X005"}
    assert session.completed_rounds == 1
    assert not any("澄清追问" in prompt for prompt in model.prompts)


def test_unavailable_prerequisites_propagate_through_dependency_chain(routing_engine):
    engine, _, questions = routing_engine
    dependent = replace(_question(7), domain="mysql", knowledge_points=("索引调优",),
                        prerequisites=("覆盖索引",))
    engine.rag.add(dependent.to_document())
    session_id = _position_on(engine, questions[0], domains=("mysql",))
    engine.submit(session_id, "我不会")
    assert engine._load(session_id).current_question_id == "X004"


def test_incomplete_answer_is_not_permanently_blocked(conversation_engine):
    engine, model, session_id = conversation_engine
    point = engine._load(session_id).current_knowledge_points[0]
    model.score = 40
    step = engine.submit(session_id, "我只知道模型负责决策。")
    assert step.decision == "followup"
    engine.submit(session_id, "执行细节还没有说完整。")
    assert point not in engine._load(session_id).blocked_knowledge_points


def test_legacy_session_hydrates_current_points_before_blocking(routing_engine):
    engine, _, questions = routing_engine
    session_id = _position_on(engine, questions[0], domains=("mysql",))
    session = engine._load(session_id)
    session.current_knowledge_points = ()
    engine.store.save_session(session)
    engine.submit(session_id, "不知道")
    resumed = engine._load(session_id)
    assert resumed.blocked_knowledge_points == ("B+树索引",)
    assert resumed.current_question_id == "X004"


def test_used_top_candidates_do_not_hide_remaining_questions(tmp_path):
    model = InterviewModel()
    config = TechRAGConfig(db_path=tmp_path / "remaining-chroma", semantic_chunking=False)
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        questions = [_question(number) for number in range(1, 16)]
        rag.add_many([q.to_document() for q in questions])
        engine = InterviewEngine(rag)
        step = engine.start(domains=("agent",))
        session = engine._load(step.session_id)
        session.asked_question_ids = tuple(q.id for q in questions[:-1])
        assert engine._next_question(session).id == questions[-1].id


def test_repeated_extension_is_bounded(routing_engine):
    engine, _, questions = routing_engine
    session_id = _position_on(engine, questions[0], domains=("mysql",))
    engine.submit(session_id, "我能解释索引。")
    engine.submit(session_id, "我也能解释查询计划。")
    engine.submit(session_id, "我能解释应用与权衡。")
    session = engine._load(session_id)
    assert session.next_action == "new_topic"
    assert session.current_question_id == "X004"
    assert session.related_rounds == 0
