from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from techrag import TechRAG, TechRAGConfig
from techrag.database import ChunkRecord, RepresentationRecord
from techrag.interview import InterviewEngine
from techrag.interview_questions import (
    InterviewQuestion,
    load_question_directory,
    load_question_file,
    question_from_content,
)
from techrag.text import count_tokens, normalize_markdown

QUESTION_DIRECTORY = Path(__file__).parents[1] / "data" / "interview_questions"


def test_bundled_question_bank_is_structured_and_unique() -> None:
    questions = load_question_directory(QUESTION_DIRECTORY)

    assert len(questions) == 98
    assert len({question.id for question in questions}) == 98
    assert {question.interview_type for question in questions} == {"technical", "behavioral"}
    assert all(question.reference_answer for question in questions)
    assert all(question.scoring_points for question in questions)


def test_behavioral_frontmatter_is_preserved_in_document_metadata() -> None:
    question = load_question_file(QUESTION_DIRECTORY / "behavioral.md")[0]

    assert question.interview_type == "behavioral"
    assert question.to_document().metadata["interview_type"] == "behavioral"
    assert question.to_document().metadata["domain"] == "behavioral"


def test_exam_direction_does_not_leak_into_tags():
    question = load_question_file(QUESTION_DIRECTORY / "agent.md")[1]
    assert question.tags == ("架构", "规划", "工具", "记忆")
    document = question.to_document()
    # Old indexed parents may contain the direction immediately after tags.
    content = document.content.replace("\n\n参考答案", "\n- 考察方向：说明核心组件。\n\n参考答案")
    restored = question_from_content(content, document.metadata)
    assert restored.tags == question.tags


def test_knowledge_points_and_prerequisites_round_trip(tmp_path):
    path = tmp_path / "mysql.md"
    path.write_text(
        "---\ndomain: mysql\n---\n\n## M001｜怎样优化覆盖索引？\n\n"
        "- 难度：进阶\n- 知识点：覆盖索引、执行计划\n- 前置知识：B+树索引\n"
        "- 参考答案：分析执行计划。\n- 评分要点：避免回表。\n", encoding="utf-8",
    )
    question = load_question_file(path)[0]
    document = question.to_document()
    restored = question_from_content(document.content, document.metadata)
    assert restored.knowledge_points == ("覆盖索引", "执行计划")
    assert restored.prerequisites == ("B+树索引",)
    assert restored.reference_answer == "分析执行计划。"


def test_routing_metadata_refresh_reuses_vectors_and_persists(tmp_path):
    model = QuestionModel()
    config = TechRAGConfig(db_path=tmp_path / "routing-refresh-chroma")
    question = _long_question()
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        original_id = rag.add(question.to_document())
        child_ids = [row["id"] for row in rag.database.representation_rows()]
        model.embedding_inputs.clear()
        updated = replace(question, knowledge_points=("父子块",), prerequisites=("切块",))
        assert rag.add(updated.to_document()) == original_id
        assert not model.embedding_inputs
        assert child_ids == [row["id"] for row in rag.database.representation_rows()]
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        row = rag.database.chunk_rows()[0]
        restored = question_from_content(str(row[2]), json.loads(str(row[6])))
        assert restored.knowledge_points == ("父子块",)
        assert restored.prerequisites == ("切块",)


class QuestionModel:
    def __init__(self) -> None:
        self.embedding_inputs: list[str] = []
        self.prompts: list[str] = []
        self.score = 84

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedding_inputs.extend(texts)
        vectors = []
        for text in texts:
            vector = [0.0] * 32
            for word in text.lower().split():
                index = int.from_bytes(hashlib.sha256(word.encode()).digest()[:2], "big")
                vector[index % 32] += 1.0
            vectors.append(vector)
        return vectors

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        self.prompts.append(messages[-1]["content"])
        return json.dumps({
            "overall": self.score, "feedback": "覆盖主要评分点", "needs_followup": False,
        }, ensure_ascii=False)


def _long_question(question_id: str = "R999") -> InterviewQuestion:
    return InterviewQuestion(
        id=question_id,
        text="如何为长答案建立检索索引？",
        domain="rag",
        difficulty=2,
        tags=("父子块", "索引"),
        reference_answer=(
            "完整答案开始。\n\n"
            + "\n\n".join(
                f"第{index}段说明：父块保存完整评分上下文，子块负责精确召回。" * 4
                for index in range(8)
            )
            + f"\n\n答案末尾包含 {question_id}_TAIL_MARKER。"
        ),
        scoring_points="保留完整父块。\n\n  - 子块不能跨题。\n  - 返回完整评分依据。",
        source_path="questions.md",
    )


@pytest.mark.parametrize(
    ("markdown", "semantic", "hierarchical"),
    [(True, True, True), (True, False, True), (False, True, True), (False, False, False)],
)
def test_long_question_has_one_parent_and_retrieves_complete_scoring_context(
    tmp_path, markdown, semantic, hierarchical
) -> None:
    model = QuestionModel()
    config = TechRAGConfig(
        db_path=tmp_path / "questions-chroma",
        chunk_size=100,
        chunk_overlap=0,
        child_chunk_size=40,
        child_chunk_overlap=5,
        markdown_chunking=markdown,
        semantic_chunking=semantic,
        hierarchical_retrieval=hierarchical,
    )
    question = _long_question()
    other = _long_question("R998")
    parent = normalize_markdown(question.to_document().content)
    assert count_tokens(parent) > config.chunk_size * 5
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add_many([question.to_document(), other.to_document()])
        assert rag.stats() == (2, 2)
        rows = [(row["content"],) for row in
                rag.database.representation_rows(source=question.to_document().source)]
        assert len(rows) > 2
        for (child,) in rows:
            assert child.startswith(
                f"# {normalize_markdown(question.text)}\n题目编号: {question.id}\n领域: rag"
            )
            assert "R998" not in child
            assert count_tokens(child.split("\n\n", 1)[1]) <= config.child_chunk_size
        # Semantic parent splitting must not embed the complete, oversized answer.
        assert all(count_tokens(text) < config.chunk_size for text in model.embedding_inputs)
        for mode in ("keyword", "vector", "hybrid"):
            results = rag.search(
                "R999_TAIL_MARKER", mode=mode, top_k=10,
                metadata_filter={"question_id": question.id},
            )
            assert len(results) == 1
            assert results[0].content == parent
            restored = question_from_content(results[0].content, results[0].metadata)
            assert restored.reference_answer == normalize_markdown(question.reference_answer)
            assert restored.scoring_points == normalize_markdown(question.scoring_points)


def test_question_update_replaces_children_and_skips_unchanged_import(tmp_path) -> None:
    model = QuestionModel()
    config = TechRAGConfig(
        db_path=tmp_path / "updates-chroma", child_chunk_size=40, child_chunk_overlap=0,
    )
    question = _long_question()
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        first_id = rag.add(question.to_document())
        model.embedding_inputs.clear()
        # Generic parent settings have no effect on question indexing.
        rag.config = replace(config, chunk_size=100, chunk_overlap=0, semantic_chunking=False,
                             markdown_chunking=False, hierarchical_retrieval=False)
        assert rag.add(question.to_document()) == first_id
        assert model.embedding_inputs == []
        updated = replace(question, reference_answer="修订答案包含 NEW_VERSION_MARKER。")
        assert rag.add(updated.to_document()) != first_id
        assert rag.stats() == (1, 1)
        assert rag.search("R999_TAIL_MARKER", mode="keyword") == []
        assert len(rag.search("NEW_VERSION_MARKER", mode="keyword")) == 1
        children_before = {row["id"] for row in rag.database.representation_rows()}
        rag.config = replace(rag.config, child_chunk_size=20)
        assert rag.add(updated.to_document())
        assert rag.stats() == (1, 1)
        children_after = {row["id"] for row in rag.database.representation_rows()}
        assert set(children_before).isdisjoint(children_after)


def test_reimport_migrates_legacy_fragmented_questions(tmp_path) -> None:
    model = QuestionModel()
    config = TechRAGConfig(db_path=tmp_path / "legacy-chroma")
    question = _long_question()
    document = question.to_document()
    content = normalize_markdown(document.content)
    old_signature = json.dumps({
        "version": "hierarchical-v2", "markdown": True, "semantic": True,
        "threshold": 0.45, "size": 700, "overlap": 100,
        "hierarchical_retrieval": True, "child_chunk_size": 180, "child_chunk_overlap": 30,
    }, sort_keys=True)
    checksum = hashlib.sha256(f"{old_signature}\n{content}".encode()).hexdigest()
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        pieces = content.split("评分要点:")
        rag.database.replace_document(
            source=document.source, checksum=checksum, metadata=document.metadata,
            chunks=[
                ChunkRecord(piece, [RepresentationRecord(piece, [1.0] * 32, {})])
                for piece in pieces
            ],
        )
        assert rag.stats() == (1, 2)
        rag.add(document)
        assert rag.stats() == (1, 1)
        assert rag.search("R999_TAIL_MARKER", mode="keyword")[0].content == content


def test_multiline_markdown_answers_reach_interview_grading(tmp_path) -> None:
    path = tmp_path / "rag.md"
    path.write_text(
        "---\ndomain: rag\n---\n\n## R999｜如何检索？\n\n"
        "- 难度：中等\n- 标签：检索\n- 参考答案：先检索子块。\n\n"
        "  然后返回完整父块。\n\n  ```python\n  return parent\n  ```\n\n"
        "- 评分要点：正确区分父子块。\n  - 保留完整答案。\n\n---\n\n"
        "## R998｜如何更新？\n- 难度：中等\n- 参考答案：替换旧索引。\n"
        "- 评分要点：删除旧子块。\n",
        encoding="utf-8",
    )
    questions = load_question_file(path)
    assert len(questions) == 2
    assert "return parent" in questions[0].reference_answer
    assert "- 保留完整答案。" in questions[0].scoring_points
    assert "---" not in questions[0].scoring_points
    assert "如何更新" not in questions[0].scoring_points
    model = QuestionModel()
    model.score = 45
    config = TechRAGConfig(db_path=tmp_path / "grading-chroma")
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        long_question = replace(_long_question(), reference_answer=(
            _long_question().reference_answer + "\n\n" + questions[0].reference_answer
        ))
        rag.add(long_question.to_document())
        engine = InterviewEngine(rag, report_dir=tmp_path / "reports")
        step = engine.start(domains=("rag",), difficulty_policy="locked")
        engine.submit(step.session_id, "我做过检索系统。")
        engine.submit(step.session_id, "我负责索引。")
        session = engine._load(step.session_id)
        assert session.current_question_id == long_question.id
        assert session.current_reference_answer == normalize_markdown(
            long_question.reference_answer
        )
        engine.submit(step.session_id, "我会用子块检索，再返回父块用于评分。")
        scoring_prompt = next(prompt for prompt in reversed(model.prompts) if "评分官" in prompt)
        assert normalize_markdown(long_question.reference_answer) in scoring_prompt
        assert normalize_markdown(long_question.scoring_points) in scoring_prompt
