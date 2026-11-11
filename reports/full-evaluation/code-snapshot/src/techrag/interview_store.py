from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from .database import Database


class InterviewStore:
    """Store interview state and history as JSON documents in Chroma."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def save_session(self, session: Any) -> None:
        self.database.put_record("session", session.id, asdict(session), session_id=session.id)

    def load_session_data(self, session_id: str) -> dict[str, object] | None:
        return self.database.get_record("session", session_id)

    def append_turn(
        self, *, session_id: str, turn_index: int, phase: str, domain: str | None,
        question_id: str | None, question: str, answer: str,
        evaluation: dict[str, object] | None, decision: dict[str, object] | None,
        sources: Sequence[str] = (),
    ) -> None:
        key = f"{session_id}:{turn_index}"
        if self.database.get_record("turn", key) is not None:
            raise ValueError(f"interview turn already exists: {key}")
        self.database.put_record("turn", key, {
            "turn_index": turn_index, "phase": phase, "domain": domain,
            "question_id": question_id, "question": question, "answer": answer,
            "evaluation": evaluation, "decision": decision, "sources": list(sources),
        }, session_id=session_id, turn_index=turn_index)

    def turns(self, session_id: str) -> list[dict[str, object]]:
        return sorted(self.database.list_records("turn", session_id=session_id),
                      key=lambda row: int(row["turn_index"]))

    def save_report(
        self, *, session_id: str, markdown_content: str | None, pdf_path: str | None,
        checksum: str | None, status: str, error_message: str | None = None,
    ) -> None:
        existing = self.report_data(session_id)
        self.database.put_record("report", session_id, {
            "markdown_content": markdown_content, "pdf_path": pdf_path,
            "checksum": checksum, "status": status, "error_message": error_message,
            "generated_at": (existing or {}).get("generated_at") or datetime.now(UTC).isoformat(),
        }, session_id=session_id)

    def report_data(self, session_id: str) -> dict[str, object] | None:
        return self.database.get_record("report", session_id)

    def delete_session(self, session_id: str) -> bool:
        if self.load_session_data(session_id) is None:
            return False
        for kind in ("turn", "report", "session"):
            self.database.delete_records(kind, session_id=session_id)
        return True
