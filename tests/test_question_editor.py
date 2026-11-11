import json
import uuid
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from test_features import FeatureModel
from test_interview import _question

from techrag import TechRAG, TechRAGConfig
from techrag.interview import InterviewEngine
from techrag.interview_questions import index_question_directory, load_question_file
from techrag.question_editor import EditorError, QuestionEditor
from techrag.web import create_app


class EditorModel(FeatureModel):
    def __init__(self):
        self.prompts = []
        self.embedding_inputs = []
        self.fail_embeddings = False
        self.bank_info = {"domain": "java", "mode": "technical", "description": "Java 基础题库"}
        self.info = {"difficulty": 2, "tags": ["Java", "哈希表"],
                     "knowledge_points": ["哈希表"], "prerequisites": [],
                     "scoring_points": "说明哈希定位、冲突处理和扩容条件。"}

    def embed(self, texts):
        if self.fail_embeddings:
            raise RuntimeError("private provider failure")
        self.embedding_inputs.extend(texts)
        return super().embed(texts)

    def complete(self, messages):
        assert isinstance(messages[-1]["content"], str)
        self.prompts.append(messages[-1]["content"])
        return json.dumps(self.bank_info if "题库目录设计助手" in messages[-1]["content"]
                          else self.info, ensure_ascii=False)


@pytest.fixture
def editor(tmp_path):
    model = EditorModel()
    with TechRAG(TechRAGConfig(db_path=tmp_path / "chroma", semantic_chunking=False),
                 _embedder=model, _chat_model=model) as rag:
        manager = QuestionEditor(rag, tmp_path / "online")
        yield manager, model


def token():
    return uuid.uuid4().hex


def create_question(editor):
    manager, _ = editor
    bank = manager.create_bank("Java", token())
    saved = manager.save(bank["key"], "HashMap 如何定位键？",
                         "通过哈希值定位桶并处理冲突。", token())
    return manager, bank, saved


def test_empty_bank_then_qa_is_indexed_as_one_parent_with_generated_metadata(editor):
    manager, model = editor
    bank = manager.create_bank("Java", token())
    assert bank["count"] == 0 and manager.rag.stats() == (0, 0)
    original = "HashMap 通过哈希定位。\n\n" + "介绍冲突处理和扩容。" * 100
    saved = manager.save(bank["key"], "HashMap 如何定位键？", original, token())
    assert saved["answer"] == original
    assert saved["difficulty"] == 2 and saved["knowledge_points"] == ["哈希表"]
    assert saved["scoring_points"] == model.info["scoring_points"]
    assert manager.rag.stats() == (1, 1)
    assert len(manager.rag.database.representation_rows()) > 1
    assert manager.banks()[0]["count"] == 1
    results = manager.rag.search("HashMap", mode="keyword")
    assert len(results) == 1 and "扩容" in results[0].content
    assert manager.questions(bank["key"], search="HashMap")["total"] == 1


def test_edit_imported_question_keeps_identity_and_original_file(editor, tmp_path):
    manager, model = editor
    original = replace(_question(1), domain="java", reference_answer="旧答案", tags=("Java",))
    source = tmp_path / "original.md"
    source.write_text("原始文件保持不变", encoding="utf-8")
    manager.rag.add(replace(original, source_path=str(source)).to_document())
    before = manager.get(original.id)
    model.info["knowledge_points"] = ["新知识点"]
    saved = manager.save("technical:java", "修改后的题目", "修订后的答案，包含新知识点。", token(),
                         question_id=original.id, expected_revision=before["revision"])
    assert saved["id"] == original.id and saved["revision"] != before["revision"]
    assert source.read_text(encoding="utf-8") == "原始文件保持不变"
    assert manager.rag.stats() == (1, 1)
    exported = load_question_file(manager.root / f"{saved['id']}.md")[0]
    assert exported.reference_answer == saved["answer"]
    assert exported.text == saved["question"]
    assert exported.knowledge_points == ("新知识点",)


def test_answer_markers_and_code_round_trip_without_becoming_fields(editor):
    manager, bank, saved = create_question(editor)
    answer = ("首段。\n\n评分要点：这是答案的一部分\n"
              "## Q999｜这也属于答案\n```python\n    run()\n```")
    saved = manager.save(bank["key"], saved["question"], answer, token(),
                         question_id=saved["id"], expected_revision=saved["revision"])
    assert manager.get(saved["id"])["answer"] == answer
    restored = load_question_file(manager.root / f"{saved['id']}.md")
    assert len(restored) == 1 and restored[0].reference_answer == answer


def test_same_request_is_idempotent_and_stale_edit_is_rejected_before_ai(editor):
    manager, bank, saved = create_question(editor)
    manager, model = editor
    revision = saved["revision"]
    model.prompts.clear()
    request_id = token()
    updated = manager.save(bank["key"], saved["question"], "新的完整答案。", request_id,
                           question_id=saved["id"], expected_revision=revision)
    prompt_count = len(model.prompts)
    repeated = manager.save(bank["key"], saved["question"], "新的完整答案。", request_id,
                            question_id=saved["id"], expected_revision=revision)
    assert repeated == updated and len(model.prompts) == prompt_count
    with pytest.raises(EditorError) as error:
        manager.save(bank["key"], saved["question"], "过期草稿。", token(),
                     question_id=saved["id"], expected_revision=revision)
    assert error.value.status == 409 and len(model.prompts) == prompt_count


def test_embedding_failure_preserves_published_file_and_vectors(editor):
    manager, bank, saved = create_question(editor)
    _, model = editor
    path = manager.root / f"{saved['id']}.md"
    file_before = path.read_bytes()
    child_ids = [r["id"] for r in manager.rag.database.representation_rows()]
    model.fail_embeddings = True
    with pytest.raises(RuntimeError):
        manager.save(bank["key"], saved["question"], "索引失败时不会覆盖。", token(),
                     question_id=saved["id"], expected_revision=saved["revision"])
    assert manager.get(saved["id"])["answer"] == saved["answer"]
    assert path.read_bytes() == file_before
    assert child_ids == [r["id"] for r in manager.rag.database.representation_rows()]
    assert not list(manager.root.glob("*.pending"))


def test_file_publish_failure_restores_old_index_without_another_embedding(editor, monkeypatch):
    manager, bank, saved = create_question(editor)
    _, model = editor
    original_replace = type(manager.root).replace

    def fail_publish(path, target):
        if path.suffix == ".pending":
            raise OSError("disk publication failed")
        return original_replace(path, target)

    monkeypatch.setattr(type(manager.root), "replace", fail_publish)
    model.embedding_inputs.clear()
    with pytest.raises(OSError):
        manager.save(bank["key"], saved["question"], "不能发布的新版本。", token(),
                     question_id=saved["id"], expected_revision=saved["revision"])
    assert manager.get(saved["id"]) == saved
    assert "不能发布" not in manager.rag.database.chunk_rows()[0][2]
    assert len(model.embedding_inputs) == 1


def test_invalid_model_metadata_and_duplicate_name_do_not_write_questions(editor):
    manager, model = editor
    request_id = token()
    bank = manager.create_bank("Java", request_id)
    assert manager.create_bank("Java", request_id)["key"] == bank["key"]
    with pytest.raises(EditorError, match="同名"):
        manager.create_bank("Java", token())
    model.info["difficulty"] = True
    with pytest.raises(EditorError) as error:
        manager.save(bank["key"], "题目", "答案", token())
    assert error.value.status == 502 and manager.rag.stats() == (0, 0)
    model.bank_info["domain"] = "../../unsafe"
    with pytest.raises(EditorError):
        manager.create_bank("另一题库", token())


def test_old_markdown_import_cannot_replace_online_revision(editor, tmp_path):
    manager, bank, saved = create_question(editor)
    from techrag.question_editor import _markdown

    source = tmp_path / "legacy"
    source.mkdir()
    old = manager._rows()[saved["id"]][0]
    (source / "old.md").write_text(_markdown(replace(old, reference_answer="过期答案")),
                                  encoding="utf-8")
    before = manager.rag.database.representation_rows()
    assert index_question_directory(manager.rag, source) == 0
    assert manager.get(saved["id"])["answer"] == saved["answer"]
    assert manager.rag.database.representation_rows() == before


def test_model_difficulty_labels_and_scoring_list_are_normalized(editor):
    manager, model = editor
    bank = manager.create_bank("Java", token())
    model.info.update({"difficulty": "基础", "scoring_points": ["说明原理。", "指出并发限制。"]})
    saved = manager.save(bank["key"], "HashMap 原理？", "哈希定位，不能随意并发修改。", token())
    assert saved["difficulty"] == 1
    assert saved["scoring_points"] == "1. 说明原理。\n2. 指出并发限制。"


def test_editor_api_create_edit_refresh_banks_and_export(editor, tmp_path):
    manager, _ = editor
    engine = InterviewEngine(manager.rag)
    with TestClient(create_app(engine=engine, question_dir=manager.root)) as client:
        bank = client.post("/api/library/banks", json={"name": "Java"}).json()
        assert bank["count"] == 0
        assert client.get("/api/banks").json()["banks"] == []
        response = client.post(f"/api/library/banks/{bank['key']}/questions", json={
            "question": "HashMap 原理？", "answer": "哈希定位、处理冲突，必要时扩容。",
        })
        assert response.status_code == 201
        saved = response.json()
        assert client.get("/api/banks").json()["banks"][0]["name"] == "Java"
        assert client.get("/api/knowledge").json()["domains"] == [{"id": "java", "name": "Java"}]
        response = client.put(f"/api/library/questions/{saved['id']}", json={
            "question": saved["question"], "answer": "HashMap 通过哈希值定位桶，再比较键。",
            "expected_revision": saved["revision"],
        })
        assert response.status_code == 200
        assert client.get(f"/api/library/questions/{saved['id']}").json() == response.json()
        exported = client.get(f"/api/library/banks/{bank['key']}/export")
        path = tmp_path / "export.md"
        path.write_text(exported.text, encoding="utf-8")
        assert load_question_file(path)[0].reference_answer == response.json()["answer"]
        assert exported.headers["content-disposition"].startswith("attachment")
        assert client.get("/api/library/questions/missing").status_code == 404
        assert client.post(f"/api/library/banks/{bank['key']}/questions", json={
            "question": "   ", "answer": "答案",
        }).status_code == 400
        assert client.post(f"/api/library/banks/{bank['key']}/questions", json={
            "question": "题目", "answer": "答案", "difficulty": 1,
        }).status_code == 422
