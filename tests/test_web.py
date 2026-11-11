from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from test_interview import ConversationModel, _question

from techrag import TechRAG, TechRAGConfig
from techrag.interview import InterviewEngine
from techrag.web import create_app


def test_knowledge_api_persists_answers_and_exposes_actual_document_domains(web_client):
    client, engine, _ = web_client

    class PositiveReranker:
        def rank(self, query, documents):
            return [0.8] * len(documents)

    engine.knowledge.reranker = PositiveReranker()
    info = client.get("/api/knowledge").json()
    assert info["documents"] == 7
    assert info["domains"] == [{"id": "agent", "name": "AI Agent"}]
    response = client.post("/api/knowledge/ask", json={
        "question": "工具调用中谁负责执行？", "domains": ["agent"],
        "mqe": False, "hyde": False,
    })
    assert response.status_code == 200
    result = response.json()
    assert result["sources"] and "宿主执行工具" in result["answer"]
    assert result["sections"][0]["trace"]["reranked"]
    assert client.get(f"/api/knowledge/answers/{result['id']}").json() == result
    assert client.get("/api/knowledge/history").json()["items"][0]["id"] == result["id"]
    assert client.get("/api/knowledge/answers/absent").status_code == 404
    # Question history is operational data and must not become retrieval evidence.
    assert engine.rag.stats()[0] == 7


def test_knowledge_api_validates_scope_and_input_before_generation(web_client):
    client, _, model = web_client
    model.prompts.clear()
    for payload, code in [
        ({"question": "Agent", "domains": ["mysql"]}, 400),
        ({"question": " "}, 400),
        ({"question": "x" * 4001}, 422),
        ({"question": "Agent", "top_k": 0}, 422),
        ({"question": "Agent", "unknown": True}, 422),
    ]:
        assert client.post("/api/knowledge/ask", json=payload).status_code == code
    assert model.prompts == []


@pytest.fixture
def web_client(tmp_path):
    model = ConversationModel()
    model.score = 85
    config = TechRAGConfig(db_path=tmp_path / "web-chroma", semantic_chunking=False)
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add_many([_question(number).to_document() for number in range(1, 8)])
        engine = InterviewEngine(rag, report_dir=tmp_path / "reports")
        with TestClient(create_app(engine=engine, evaluation_dir=tmp_path / "evaluations"),
                        raise_server_exceptions=False) as client:
            yield client, engine, model


def _start(client, **settings):
    response = client.post("/api/sessions", json={"domains": ["agent"], **settings})
    assert response.status_code == 201
    return response.json()


def _answer(client, snapshot, answer):
    session = snapshot["session"]
    response = client.post(f"/api/sessions/{session['id']}/answers", json={
        "answer": answer, "expected_turn_index": session["turn_index"],
    })
    assert response.status_code == 200
    return response.json()


def test_website_and_static_assets_are_served(web_client):
    client, _, _ = web_client
    assert "认真练习，从容上场" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/../web.py").status_code == 404


def test_evaluation_api_history_validation_and_safe_report_paths(web_client, monkeypatch):
    client, _, model = web_client
    info = client.get("/api/evaluations").json()
    assert info["case_count"] == 20 and not info["reports"]
    assert len(info["variants"]) == 7
    assert sum(v["default"] for v in info["variants"]) == 4
    assert "检索评测" in client.get("/").text
    for body in [{"limit": 1}, {"variants": []}, {"variants": ["unsupported"]}]:
        assert client.post("/api/evaluations/run", json=body).status_code == 422
    response = client.post("/api/evaluations/run", json={"variants": ["vector", "vector"]})
    assert response.status_code == 400
    assert not model.prompts

    def run(_rag, *, variants, limit, output_dir, answer_judge):
        import json
        assert answer_judge is False
        report = {"id": "a" * 32, "created_at": "2026-10-06T00:00:00+00:00",
                  "dataset": {"selected_cases": limit},
                  "summaries": [{"variant": v} for v in variants]}
        output_dir.mkdir(parents=True)
        (output_dir / f"{report['id']}.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr("techrag.web.run_benchmark", run)
    result = client.post("/api/evaluations/run", json={
        "limit": 6, "variants": ["vector"], "answer_judge": False,
    })
    assert result.status_code == 200
    assert client.get(f"/api/evaluations/{'a' * 32}").json() == result.json()
    download = client.get(f"/api/evaluations/{'a' * 32}/download")
    assert download.json() == result.json()
    assert "attachment" in download.headers["content-disposition"]
    assert client.get("/api/evaluations/missing/download").status_code == 404
    assert client.get("/api/evaluations").json()["reports"][0]["id"] == "a" * 32
    assert client.get("/api/evaluations/missing").status_code == 404
    assert client.get(f"/api/evaluations/{'b' * 32}").status_code == 404


def test_report_api_exposes_point_evidence_without_regrading_history(web_client):
    client, engine, model = web_client
    snapshot = _start(client)
    snapshot = _answer(client, snapshot, "我做过Agent项目")
    snapshot = _answer(client, snapshot, "模型决定调用，宿主执行工具")
    snapshot = _answer(client, snapshot, "模型决定调用，宿主执行工具")
    report = client.get(f"/api/sessions/{snapshot['session']['id']}/report").json()
    # This old fixture model has no point reviews: it must remain explicitly unverified.
    assert report["reviews"][0]["scoring_method"] == "model"
    assert report["reviews"][0]["point_reviews"]
    assert all(p["status"] == "unverified" for p in report["reviews"][0]["point_reviews"])
    before = list(model.prompts)
    client.get(f"/api/sessions/{snapshot['session']['id']}/report")
    assert model.prompts == before


def test_banks_reflect_new_imports_and_exclude_non_questions(web_client):
    client, engine, _ = web_client
    banks = client.get("/api/banks").json()["banks"]
    assert [bank["id"] for bank in banks] == ["agent"]
    assert banks[0]["count"] == 7
    assert banks[0]["topic_questions"]["工程知识点1"] == ["X001"]
    engine.rag.add(replace(_question(8), domain="java").to_document())
    banks = client.get("/api/banks").json()["banks"]
    assert {bank["id"] for bank in banks} == {"agent", "java"}
    assert next(bank for bank in banks if bank["id"] == "java")["name"] == "Java"


def test_scope_validation_and_question_answers_are_not_exposed(web_client):
    client, _, _ = web_client
    assert client.post("/api/sessions", json={"domains": []}).status_code == 422
    assert client.post("/api/sessions", json={"domains": ["mysql"]}).status_code == 400
    assert client.post("/api/sessions", json={
        "domains": ["agent"], "topics": ["not-in-bank"],
    }).status_code == 400
    started = _start(client, topics=["工程知识点1"])
    assert started["session"]["topics"] == ["工程知识点1"]
    assert "参考答案" not in str(started)
    assert "current_reference_answer" not in str(started)
    assert client.get("/api/sessions/unknown").status_code == 404


def test_browser_interview_resume_and_stale_answer_protection(web_client):
    client, engine, _ = web_client
    start = _start(client)
    snapshot = _answer(client, start, "我做过一个工具调用系统。")
    assert snapshot["session"]["phase"] == "project_deep_dive"
    session_id = snapshot["session"]["id"]
    duplicate = client.post(f"/api/sessions/{session_id}/answers", json={
        "answer": "我做过一个工具调用系统。", "expected_turn_index": 0,
    })
    assert duplicate.status_code == 409
    assert len(engine.store.turns(session_id)) == 1
    resumed = client.get(f"/api/sessions/{session_id}").json()
    assert any(m["content"] == "我做过一个工具调用系统。" for m in resumed["messages"])
    snapshot = _answer(client, resumed, "我负责工具执行和结果回填。")
    assert snapshot["session"]["phase"] == "questioning"
    assert snapshot["session"]["current_domain"] == "agent"
    assert len(client.get("/api/sessions").json()["sessions"]) == 1


def test_early_finish_scores_pending_answer_once_and_exports_report(web_client):
    client, engine, model = web_client
    snapshot = _answer(client, _start(client), "我做过 Agent 项目。")
    snapshot = _answer(client, snapshot, "我负责状态机。")
    model.score = 45
    snapshot = _answer(client, snapshot, "模型决定调用工具。")
    assert snapshot["step"]["decision"] == "followup"
    session = snapshot["session"]
    path = f"/api/sessions/{session['id']}/finish"
    body = {"expected_turn_index": session["turn_index"]}
    response = client.post(path, json=body)
    assert response.status_code == 200
    assert response.json()["session"]["phase"] == "completed"
    assert response.json()["session"]["completed_rounds"] == 1
    assert client.post(path, json=body).json()["session"]["completed_rounds"] == 1
    report = client.get(f"/api/sessions/{session['id']}/report").json()
    assert report["session"]["score"] == 45
    assert report["pdf_ready"]
    assert "模型决定调用工具" in report["markdown"]
    assert client.get(report["pdf_url"]).content.startswith(b"%PDF")
    assert "attachment" in client.get(report["markdown_url"]).headers["content-disposition"]
    rejected = client.post(f"/api/sessions/{session['id']}/answers", json={
        "answer": "结束后再回答", "expected_turn_index": session["turn_index"],
    })
    assert rejected.status_code == 409
    assert engine._load(session["id"]).completed_rounds == 1


def test_failed_model_request_does_not_leak_provider_error(web_client, monkeypatch):
    client, engine, _ = web_client
    snapshot = _start(client)

    def fail(*_args):
        raise RuntimeError("SECRET_PROVIDER_KEY")

    monkeypatch.setattr(engine, "submit", fail)
    response = client.post(f"/api/sessions/{snapshot['session']['id']}/answers", json={
        "answer": "测试", "expected_turn_index": 0,
    })
    assert response.status_code == 500
    assert "SECRET_PROVIDER_KEY" not in response.text
    assert "已保存" in response.text


def test_failed_retrieval_rolls_back_partial_turn_and_allows_retry(web_client, monkeypatch):
    client, engine, _ = web_client
    snapshot = _answer(client, _start(client), "这是合成的项目介绍。")
    session_id = snapshot["session"]["id"]
    original = engine._next_question

    def fail(_session):
        raise RuntimeError("embedding temporarily unavailable")

    monkeypatch.setattr(engine, "_next_question", fail)
    body = {"answer": "这是项目回答。", "expected_turn_index": 1}
    assert client.post(f"/api/sessions/{session_id}/answers", json=body).status_code == 500
    assert len(engine.store.turns(session_id)) == 1
    assert engine._load(session_id).turn_index == 1
    monkeypatch.setattr(engine, "_next_question", original)
    retried = client.post(f"/api/sessions/{session_id}/answers", json=body)
    assert retried.status_code == 200
    assert retried.json()["session"]["phase"] == "questioning"
    assert len(engine.store.turns(session_id)) == 2


def test_report_groups_initial_and_followup_answers_as_one_question(web_client):
    client, _, model = web_client
    snapshot = _answer(client, _start(client), "自我介绍。")
    snapshot = _answer(client, snapshot, "项目实现。")
    model.score = 45
    snapshot = _answer(client, snapshot, "模型负责决策。")
    model.score = 85
    snapshot = _answer(client, snapshot, "宿主执行工具并回填结果。")
    report = client.get(f"/api/sessions/{snapshot['session']['id']}/report").json()
    assert len(report["reviews"]) == 1
    assert report["reviews"][0]["score"] == 85
    assert "模型负责决策" in report["reviews"][0]["answer"]
    assert "宿主执行工具" in report["reviews"][0]["answer"]
