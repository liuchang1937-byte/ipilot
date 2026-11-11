"""One-time, read-only export: uv run --no-project --with duckdb this_script.py SOURCE OUTPUT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

TABLES = (
    "documents", "chunks", "chunk_representations", "representation_terms",
    "evaluations", "index_metadata", "interview_sessions", "interview_turns", "interview_reports",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a legacy database without changing it")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with duckdb.connect(str(args.source), read_only=True) as db:
        names = {row[0] for row in db.execute("SHOW TABLES").fetchall()}
        snapshot = {}
        for name in TABLES:
            if name in names:
                cursor = db.execute("SELECT * FROM " + name)  # noqa: S608
                keys = [column[0] for column in cursor.description]
                snapshot[name] = [dict(zip(keys, row, strict=True)) for row in cursor.fetchall()]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(snapshot, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(json.dumps({name: len(rows) for name, rows in snapshot.items()}))


if __name__ == "__main__":
    main()
