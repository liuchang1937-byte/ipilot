from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .database import ChunkRecord, Database, RepresentationRecord
from .text import tokenize


def _object(value):
    return json.loads(value) if isinstance(value, str) else value


def import_legacy_snapshot(database: Database, path: str | Path) -> dict[str, int]:
    """Import a read-only JSON export, retaining embeddings without calling a model.

    Existing interview records win so repeating an import cannot roll back a
    resumed session. Existing document sources are also skipped.
    """
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks_by_document = defaultdict(list)
    representations_by_chunk = defaultdict(list)
    frequencies = defaultdict(dict)
    for row in snapshot.get("chunks", []):
        chunks_by_document[row["document_id"]].append(row)
    for row in snapshot.get("chunk_representations", []):
        representations_by_chunk[row["chunk_id"]].append(row)
    for row in snapshot.get("representation_terms", []):
        frequencies[(row["chunk_id"], row["representation_index"])][row["term"]] = row["frequency"]
    for row in snapshot.get("index_metadata", []):
        database.ensure_index_setting(row["key"], row["value"])
    counts = {"documents": 0, "sessions": 0, "turns": 0, "reports": 0, "evaluations": 0}
    for document in snapshot.get("documents", []):
        existing = database.existing_document(document["source"])
        if existing:
            continue
        chunks = []
        for chunk in sorted(chunks_by_document[document["id"]], key=lambda row: row["position"]):
            representations = [
                RepresentationRecord(
                    content=row["content"], embedding=row["embedding"],
                    frequencies=frequencies[(chunk["id"], row["representation_index"])]
                                or dict(Counter(tokenize(row["content"]))),
                    kind=row["kind"],
                )
                for row in sorted(representations_by_chunk[chunk["id"]],
                                  key=lambda row: row["representation_index"])
            ] or [RepresentationRecord(
                chunk["content"], chunk["embedding"], dict(Counter(tokenize(chunk["content"]))),
            )]
            chunks.append(ChunkRecord(chunk["content"], representations))
        database.replace_document(
            source=document["source"], checksum=document["checksum"],
            metadata=_object(document["metadata"]), chunks=chunks,
        )
        counts["documents"] += 1
    for row in snapshot.get("interview_sessions", []):
        if database.get_record("session", row["id"]) is None:
            database.put_record("session", row["id"], _object(row["state"]), session_id=row["id"])
            counts["sessions"] += 1
    for row in snapshot.get("interview_turns", []):
        key = f"{row['session_id']}:{row['turn_index']}"
        if database.get_record("turn", key) is None:
            payload = {name: row[name] for name in (
                "turn_index", "phase", "domain", "question_id", "question", "answer",
            )}
            payload.update({name: _object(row[name]) if row[name] else None
                            for name in ("evaluation", "decision", "sources")})
            database.put_record("turn", key, payload, session_id=row["session_id"],
                                turn_index=row["turn_index"])
            counts["turns"] += 1
    for row in snapshot.get("interview_reports", []):
        if database.get_record("report", row["session_id"]) is None:
            payload = {name: row[name] for name in (
                "markdown_content", "pdf_path", "checksum", "status", "error_message",
                "generated_at",
            )}
            database.put_record("report", row["session_id"], payload, session_id=row["session_id"])
            counts["reports"] += 1
    for row in snapshot.get("evaluations", []):
        key = f"legacy:{row['id']}"
        if database.get_record("evaluation", key) is None:
            database.put_record("evaluation", key, {
                "question": row["question"], "expected_sources": _object(row["expected_sources"]),
                "reference_answer": row["reference_answer"],
            })
            counts["evaluations"] += 1
    return counts
