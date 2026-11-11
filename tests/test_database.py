import json
from types import SimpleNamespace

import pytest
from chromadb.errors import InternalError

from techrag.database import ChunkRecord, Database, RepresentationRecord
from techrag.interview_store import InterviewStore
from techrag.migration import import_legacy_snapshot
from techrag.search import vector_search


def _chunk(text="parent", vector=None):
    return ChunkRecord(text, [RepresentationRecord(text, vector or [1.0, 0.0], {text: 1})])


def test_temporary_hnsw_reader_failure_retries_the_same_filtered_read(monkeypatch):
    database = Database.__new__(Database)
    calls = []
    def query(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise InternalError("Error creating hnsw segment reader: Nothing found on disk")
        return {"metadatas": [[{"parent_id": 7}]], "distances": [[0.0]]}
    database.children = SimpleNamespace(query=query)
    database.get_record = lambda *args: None
    monkeypatch.setattr("techrag.database.time.sleep", lambda *_: None)
    result = database.query_children([[1., 0.]], parent_ids=[7], n_results=1)
    assert result["metadatas"][0][0]["parent_id"] == 7
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    assert calls[0]["where"] == {"parent_id": {"$in": [7]}}


@pytest.mark.parametrize("message,attempts", [
    ("Error creating hnsw segment reader: Nothing found on disk", 3),
    ("different persistent error", 1),
])
def test_persistent_or_unrelated_database_errors_are_not_hidden(monkeypatch, message, attempts):
    database = Database.__new__(Database)
    calls = []
    def query(**kwargs):
        calls.append(kwargs)
        raise InternalError(message)
    database.children = SimpleNamespace(query=query)
    database.get_record = lambda *args: None
    monkeypatch.setattr("techrag.database.time.sleep", lambda *_: None)
    with pytest.raises(InternalError, match=message):
        database.query_children([[1., 0.]], parent_ids=[7], n_results=1)
    assert len(calls) == attempts


def test_chroma_persists_parents_vectors_evaluations_and_reports(tmp_path):
    path = tmp_path / "chroma"
    with Database(path) as database:
        database.replace_document(source="test", checksum="v1", metadata={"tags": ["a", "b"]},
                                  chunks=[_chunk()])
        database.insert_evaluation("question", ["test"], "answer")
        database.put_record("session", "s1", {"id": "s1"}, session_id="s1")
        store = InterviewStore(database)
        for index in (2, 1):
            store.append_turn(session_id="s1", turn_index=index, phase="questioning", domain="rag",
                              question_id="R1", question="question", answer="answer",
                              evaluation={"overall": 80}, decision={"action": "next"},
                              sources=["test"])
        store.save_report(session_id="s1", markdown_content="report", pdf_path="report.pdf",
                          checksum="hash", status="ready")
    with Database(path) as database:
        assert database.stats() == (1, 1)
        assert database.document_rows()[0][1] == {"tags": ["a", "b"]}
        assert vector_search(database, [1.0, 0.0], top_k=1)[0].source == "test"
        assert database.evaluation_rows() == [("question", ["test"], "answer")]
        store = InterviewStore(database)
        assert [row["turn_index"] for row in store.turns("s1")] == [1, 2]
        assert store.report_data("s1")["markdown_content"] == "report"
        assert store.delete_session("s1")
        assert store.turns("s1") == []
        assert store.report_data("s1") is None
        assert database.stats() == (1, 1)


def test_failed_replacement_keeps_previous_generation(tmp_path, monkeypatch):
    with Database(tmp_path / "chroma") as database:
        original = database.replace_document(source="test", checksum="v1", metadata={},
                                             chunks=[_chunk("original")])
        write = database._write_batches

        def fail_children(collection, *args):
            write(collection, *args)
            if collection.name == "techrag_children":
                raise RuntimeError("simulated write failure")

        monkeypatch.setattr(database, "_write_batches", fail_children)
        with pytest.raises(RuntimeError, match="simulated"):
            database.replace_document(source="test", checksum="v2", metadata={},
                                      chunks=[_chunk("replacement")])
        assert database.existing_document("test") == (original, "v1")
        assert database.chunk_rows()[0][2] == "original"
        assert database.children.count() == 1
        assert database.parents.count() == 1


def test_chroma_query_expands_children_and_filters_before_ranking(tmp_path, monkeypatch):
    with Database(tmp_path / "chroma") as database:
        database.replace_document(source="many", checksum="1", metadata={"domain": "rag"}, chunks=[
            ChunkRecord("complete parent", [
                RepresentationRecord(f"child {index}", [1.0, 0.0], {"child": 1})
                for index in range(25)
            ])
        ])
        database.replace_document(source="second", checksum="1", metadata={"domain": "rag"},
                                  chunks=[_chunk("second", [0.9, 0.1])])
        database.replace_document(source="excluded", checksum="1", metadata={"domain": "agent"},
                                  chunks=[_chunk("excluded")])
        calls = []
        query = database.query_children

        def track(*args, **kwargs):
            calls.append(kwargs["n_results"])
            return query(*args, **kwargs)

        monkeypatch.setattr(database, "query_children", track)
        results = vector_search(database, [1.0, 0.0], top_k=2, metadata_filter={"domain": "rag"})
        assert [row.source for row in results] == ["many", "second"]
        assert len(calls) > 1
        assert results[0].content == "complete parent"
        assert results[0].vector_score == pytest.approx(1.0)
        with pytest.raises(ValueError, match="dimension"):
            vector_search(database, [1.0], top_k=2)


def test_migration_is_idempotent_and_preserves_newer_sessions(tmp_path):
    snapshot = {
        "documents": [{"id": 1, "source": "legacy", "checksum": "old", "metadata": "{}"}],
        "chunks": [{"id": 1, "document_id": 1, "position": 0, "content": "legacy database",
                    "embedding": [1.0, 0.0]}],
        "interview_sessions": [{"id": "s1", "state": '{"id":"s1","turn_count":1}'}],
        "interview_turns": [{"session_id": "s1", "turn_index": 1, "phase": "questioning",
                             "domain": "rag", "question_id": "R1", "question": "question",
                             "answer": "answer", "evaluation": '{"overall":80}',
                             "decision": '{"action":"next"}', "sources": '["legacy"]'}],
        "interview_reports": [{"session_id": "s1", "markdown_content": "report",
                               "pdf_path": None, "checksum": "hash", "status": "ready",
                               "error_message": None, "generated_at": "2026-10-05"}],
        "evaluations": [{"id": 1, "question": "question", "expected_sources": '["legacy"]',
                         "reference_answer": "answer"}],
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    with Database(tmp_path / "chroma") as database:
        counts = import_legacy_snapshot(database, path)
        assert counts == {"documents": 1, "sessions": 1, "turns": 1, "reports": 1, "evaluations": 1}
        assert vector_search(database, [1.0, 0.0], top_k=1)[0].source == "legacy"
        database.put_record("session", "s1", {"id": "s1", "turn_count": 2}, session_id="s1")
        assert all(count == 0 for count in import_legacy_snapshot(database, path).values())
        assert database.get_record("session", "s1")["turn_count"] == 2
        assert InterviewStore(database).turns("s1")[0]["evaluation"] == {"overall": 80}
        assert InterviewStore(database).report_data("s1")["generated_at"] == "2026-10-05"


def test_legacy_database_path_is_rejected_without_modification(tmp_path):
    path = tmp_path / "legacy.duckdb"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="Chroma directory"):
        Database(path)
    assert path.read_bytes() == b"original"
