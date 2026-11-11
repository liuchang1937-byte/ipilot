import json

from techrag.scoring import point_reviews, rubric_points

pytest_plugins = ["test_interview"]


def test_quotes_must_match_the_numbered_answer_and_cover_all_rubric_points():
    rubric = rubric_points("1. 模型选择工具；2. 宿主执行工具\n3. 结果回填")
    reviews = point_reviews([
        {"point_id": "P1", "status": "mastered", "reason": "准确区分职责。",
         "evidence": [{"answer_index": 1, "quote": "模型选择工具"}]},
        {"point_id": "P2", "status": "mastered", "reason": "伪造证据。",
         "evidence": [{"answer_index": 1, "quote": "宿主执行工具"}]},
        {"point_id": "invented", "status": "mastered", "reason": "额外要求", "evidence": []},
    ], rubric, ["模型选择工具", "宿主执行工具"])
    assert len(reviews) == 3
    assert reviews[0]["status"] == "mastered"
    assert reviews[0]["evidence"][0]["quote"] == "模型选择工具"
    assert [r["status"] for r in reviews[1:]] == ["unverified", "unverified"]
    assert all(not r["evidence"] for r in reviews[1:])


def test_duplicate_points_invalid_status_and_non_quotes_are_not_verified():
    rubric = rubric_points("模型决策；宿主执行；回填结果；安全校验")
    reviews = point_reviews([
        {"point_id": "P1", "status": "mastered", "reason": "a", "evidence": []},
        {"point_id": "P1", "status": "not_addressed", "reason": "b", "evidence": []},
        {"point_id": "P2", "status": [], "reason": "b", "evidence": []},
        {"point_id": "P3", "status": "incorrect", "reason": "b",
         "evidence": [{"answer_index": True, "quote": "调用"}]},
        {"point_id": "P4", "status": "partial", "reason": "b", "evidence": None},
    ], rubric, ["调用"])
    assert all(r["status"] == "unverified" for r in reviews)


def test_omission_is_distinct_from_error_and_equivalent_solutions_have_evidence():
    reviews = point_reviews([
        {"point_id": "P1", "status": "mastered", "reason": "外部锁是合理的等价方案。",
         "evidence": [{"answer_index": 2, "quote": "也可以使用外部锁"}]},
        {"point_id": "P2", "status": "not_addressed", "reason": "未谈到安全发布。",
         "evidence": []},
    ], rubric_points("线程安全方案；安全发布"), ["不用并发容器", "也可以使用外部锁"])
    assert reviews[0]["label"] == "掌握"
    assert reviews[1]["label"] == "未涉及"
    assert not reviews[1]["evidence"]


def test_long_rubrics_keep_the_tail_instead_of_dropping_it():
    points = rubric_points("；".join(f"要点{i}" for i in range(24)))
    assert len(points) == 16 and "要点23" in points[-1]["point"]


def test_model_evaluation_attaches_validated_reviews_and_exports_them(conversation_engine):
    engine, model, session_id = conversation_engine
    session = engine._load(session_id)
    session.current_scoring_points = "模型决策；宿主执行"
    session.question_answers = ["模型决策", "宿主执行"]
    original = model.complete

    def complete(messages):
        if "评分官" in messages[-1]["content"]:
            payload = json.loads(original(messages))
            payload["point_reviews"] = [
                {"point_id": "P1", "status": "mastered", "reason": "职责清晰。",
                 "evidence": [{"answer_index": 1, "quote": "模型决策"}]},
                {"point_id": "P2", "status": "mastered", "reason": "补充回答正确。",
                 "evidence": [{"answer_index": 2, "quote": "宿主执行"}]},
            ]
            return json.dumps(payload)
        return original(messages)

    model.complete = complete
    result = engine._evaluate(session, "宿主执行")
    assert result["point_review_complete"] and result["scoring_method"] == "model"
    engine._record(session, "宿主执行", result, {"action": "new_topic"})
    report = engine.report(session_id)
    assert "P1｜掌握：模型决策" in report and "第 2 次回答原文：宿主执行" in report


def test_fallback_does_not_claim_semantic_point_judgements(conversation_engine):
    engine, model, session_id = conversation_engine
    session = engine._load(session_id)
    def fail(_messages):
        raise RuntimeError("unavailable")
    model.complete = fail
    result = engine._evaluate(session, "测试回答")
    assert result["scoring_method"] == "fallback"
    assert all(p["status"] == "unverified" for p in result["point_reviews"])
