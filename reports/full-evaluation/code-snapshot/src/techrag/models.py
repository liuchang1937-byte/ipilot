from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls, content: str, *, source: str = "memory", metadata: dict[str, Any] | None = None
    ) -> Document:
        return cls(content=content, source=source, metadata=metadata or {})

    @classmethod
    def from_path(cls, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Document:
        file_path = Path(path)
        from .documents import document_to_markdown

        return cls(
            content=document_to_markdown(file_path),
            source=str(file_path.resolve()),
            metadata=metadata or {},
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: int
    document_id: int
    content: str
    source: str
    metadata: dict[str, Any]
    score: float
    vector_score: float
    keyword_score: float
    position: int = 0
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    chunks: tuple[SearchResult, ...]

    @property
    def content(self) -> str:
        return "\n\n".join(chunk.content for chunk in self.chunks)

    @property
    def source(self) -> str:
        return self.chunks[0].source if self.chunks else ""

    def to_xml(self, index: int) -> str:
        return (
            f'<document index="{index}" source="{self.source}">\n'
            f"{self.content}\n</document>"
        )


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: tuple[SearchResult, ...]


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    question: str
    expected_sources: tuple[str, ...]
    reference_answer: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    cases: int
    recall_at_k: float
    mean_reciprocal_rank: float
    answer_coverage: float | None = None
