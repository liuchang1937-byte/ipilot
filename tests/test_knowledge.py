import json

import pytest
from test_features import FeatureModel

from techrag import TechRAG, TechRAGConfig
from techrag.knowledge import KnowledgeOptions, KnowledgeService


class KnowledgeModel(FeatureModel):
    def __init__(self):
        self.prompts = []
        self.plan = ["Agent 架构？", "Function Calling 原理？"]
        self.bad_citations = False

    def complete(self, messages):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "知识库问题拆解器" in prompt:
            return json.dumps({"questions": [{"question": q, "kind": "knowledge"}
                                             for q in self.plan]})
        if "Generate three diverse" in prompt:
            query = prompt.split("Query: ")[-1]
            return json.dumps([query + " 原理", query + " 工作机制", query + " 原理"])
        if "hypothetical passage" in prompt:
            return "假设文本标记FAKE_EVIDENCE：Agent 规划工具与 Function Calling 宿主执行。"
        if "严谨的知识库问答助手" in prompt:
            data = json.loads(prompt.rsplit("\n", 1)[-1])
            # The hypothetical text must never leak into the evidence context.
            assert "FAKE_EVIDENCE" not in json.dumps(data)
            number = 999 if self.bad_citations else data["retrieved_documents"][0]["id"]
            return json.dumps({"answer": f"根据真实资料回答。[{number}]",
                               "source_ids": [number], "insufficient": False})
        return "[]"


class RelevantReranker:
    def __init__(self):
        self.calls = []

    def rank(self, query, documents):
        self.calls.append((query, documents))
        needle = "Agent" if "Agent" in query else "Function Calling"
        return [0.9 if needle in document else 0.001 for document in documents]


@pytest.fixture
def knowledge(tmp_path):
    model = KnowledgeModel()
    reranker = RelevantReranker()
    config = TechRAGConfig(db_path=tmp_path / "knowledge", semantic_chunking=False)
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add_text("# Agent 架构\n\nAgent 包括规划、记忆和工具调用。",
                     source="agent.md", metadata={"domain": "agent"})
        rag.add_text("# Function Calling\n\nFunction Calling 中模型决策，宿主执行工具。",
                     source="tools.md", metadata={"domain": "tools"})
        rag.add_text("# 无关资料\n\n今天的天气很好。", source="other.md")
        yield KnowledgeService(rag, reranker=reranker), model, reranker


def test_compound_questions_retrieve_separately_and_cite_complete_parents(knowledge):
    service, _, reranker = knowledge
    result = service.answer("Agent 架构？Function Calling 原理？")
    assert len(result["sections"]) == 2
    assert len(reranker.calls) == 2
    assert {s["source"] for s in result["sources"]} == {"agent.md", "tools.md"}
    assert len({s["id"] for s in result["sources"]}) == 2
    for section in result["sections"]:
        assert section["trace"]["mode"] == "hybrid"
        assert section["trace"]["reranked"]
        assert {q["kind"] for q in section["trace"]["queries"]} == {"original", "mqe", "hyde"}
    assert all(s["content"].startswith("#") for s in result["sources"])
    assert all(s["keyword_score"] > 0 and s["vector_score"] > 0 for s in result["sources"])


def test_scoping_is_applied_to_every_subquestion_and_missing_is_explicit(knowledge):
    service, _, _ = knowledge
    result = service.answer("Agent 架构？Function Calling 原理？", domains=("agent",))
    assert all(s["domain"] == "agent" for s in result["sources"])
    missing = result["sections"][1]
    assert missing["insufficient"] and missing["source_ids"] == []
    assert "没有找到足够相关" in missing["answer"]


def test_optional_expansions_and_reranker_can_be_disabled(knowledge):
    service, model, reranker = knowledge
    result = service.answer("Agent 架构？", options=KnowledgeOptions(
        decompose=False, mqe=False, hyde=False, rerank=False,
    ))
    assert not reranker.calls
    assert len(result["sections"]) == 1
    assert [q["kind"] for q in result["sections"][0]["trace"]["queries"]] == ["original"]
    assert not any("假设" in p or "拆解器" in p for p in model.prompts)


def test_bad_citation_is_not_exposed_as_a_supported_answer(knowledge):
    service, model, _ = knowledge
    model.bad_citations = True
    result = service.answer("Agent 架构？", options=KnowledgeOptions(decompose=False))
    assert result["sources"] == []
    assert result["sections"][0]["insufficient"]
    assert "999" not in result["answer"]
    assert result["warnings"]


def test_insufficient_answer_cannot_keep_unrelated_source_citations(knowledge, monkeypatch):
    service, model, _ = knowledge
    original = model.complete
    def complete(messages):
        if "严谨的知识库问答助手" in messages[-1]["content"]:
            return json.dumps({"answer": "没有找到答案，但看这些无关资料。[1]",
                               "source_ids": [1], "insufficient": True})
        return original(messages)
    monkeypatch.setattr(model, "complete", complete)
    result = service.answer("Agent 架构？", options=KnowledgeOptions(decompose=False))
    assert result["sources"] == []
    assert result["sections"][0]["source_ids"] == []
    assert result["sections"][0]["insufficient"]
    assert "[1]" not in result["answer"]
    assert result["warnings"]


def test_failed_expansions_and_precision_keep_original(knowledge, monkeypatch):
    service, _, _ = knowledge

    def fail(*args, **kwargs):
        raise RuntimeError("secret provider error")

    monkeypatch.setattr(service.rag, "_generate_mqe", fail)
    monkeypatch.setattr(service.rag, "_generate_hyde", fail)
    monkeypatch.setattr(service.reranker, "rank", fail)
    result = service.answer("Agent 架构？", options=KnowledgeOptions(decompose=False))
    assert len(result["warnings"]) == 3
    assert not result["sections"][0]["trace"]["reranked"]
    assert result["sources"]
    assert "secret" not in json.dumps(result)


def test_invalid_plan_keeps_the_entire_compound_question(knowledge):
    service, model, _ = knowledge
    model.plan = [f"问题{i}" for i in range(6)]
    original = "Agent 架构以及 Function Calling 原理？"
    result = service.answer(original)
    assert result["sections"][0]["question"] == original
    assert any("拆解" in warning for warning in result["warnings"])


def test_no_documents_does_not_generate_a_model_memory_answer(knowledge):
    service, model, _ = knowledge
    result = service.answer("Agent 架构？", domains=("absent",))
    assert not result["sources"]
    assert all(section["insufficient"] for section in result["sections"])
    assert not any("严谨的知识库问答助手" in prompt for prompt in model.prompts)


def test_interview_general_company_questions_skip_retrieval(knowledge, monkeypatch):
    service, model, _ = knowledge
    original = model.complete

    def complete(messages):
        if "拆解器" in messages[-1]["content"]:
            return '{"questions":[{"question":"贵公司薪资如何？","kind":"general"}]}'
        return original(messages)

    monkeypatch.setattr(model, "complete", complete)
    monkeypatch.setattr(service.rag, "search", lambda *a, **k: pytest.fail("should not retrieve"))
    result = service.answer("贵公司薪资如何？", interview=True)
    assert result["sources"] == []
    assert "实际招聘方确认" in result["answer"]


def test_input_bounds_reject_blank_or_invalid_top_k(knowledge):
    service, _, _ = knowledge
    with pytest.raises(ValueError):
        service.answer("   ")
    with pytest.raises(ValueError):
        service.answer("x" * 4001)
    with pytest.raises(ValueError):
        service.answer("Agent", options=KnowledgeOptions(top_k=0))
