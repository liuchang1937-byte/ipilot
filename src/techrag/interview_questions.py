from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from .models import Document
from .rag import TechRAG

_QUESTION_HEADING = re.compile(r"^##\s+([A-Z]+\d+)｜(.+?)\s*$", re.MULTILINE)
_FIELD = re.compile(
    r"^-[ \t]*(难度|标签|知识点|前置知识|考察方向|参考答案|评分要点|参考资料)：[ \t]*",
    re.MULTILINE,
)
_DOCUMENT_FIELD = re.compile(
    r"^(?:-[ \t]*)?(题目编号|领域|难度|标签|考察方向|参考答案|评分要点)[：:][ \t]*",
    re.MULTILINE,
)
_DIFFICULTY = {"基础": 1, "中等": 2, "进阶": 3}


@dataclass(frozen=True, slots=True)
class InterviewQuestion:
    id: str
    text: str
    domain: str
    difficulty: int
    tags: tuple[str, ...]
    reference_answer: str
    scoring_points: str
    source_path: str
    interview_type: str = "technical"
    knowledge_points: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()

    @property
    def routing_points(self) -> tuple[str, ...]:
        return self.knowledge_points or self.tags or (self.text,)

    def to_document(self) -> Document:
        tags = "、".join(self.tags)
        content = (
            f"# {self.text}\n\n"
            f"题目编号：{self.id}\n\n"
            f"领域：{self.domain}\n\n"
            f"难度：{self.difficulty}\n\n"
            f"标签：{tags}\n\n"
            f"参考答案：{self.reference_answer}\n\n"
            f"评分要点：{self.scoring_points}"
        )
        return Document.from_text(
            content,
            source=f"interview-question://{self.id}",
            metadata={
                "collection": "interview-question",
                "interview_type": self.interview_type,
                "question_id": self.id,
                "domain": self.domain,
                "difficulty": self.difficulty,
                "tags": list(self.tags),
                "source_path": self.source_path,
                "status": "approved",
                "knowledge_points": list(self.knowledge_points),
                "prerequisites": list(self.prerequisites),
            },
        )


def load_question_file(path: str | Path) -> list[InterviewQuestion]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    domain = _frontmatter_value(text, "domain") or file_path.stem
    interview_type = _frontmatter_value(text, "interview_type") or "technical"
    if interview_type not in {"technical", "behavioral"}:
        raise ValueError(f"{file_path}: invalid interview_type {interview_type}")
    matches = list(_QUESTION_HEADING.finditer(text))
    questions: list[InterviewQuestion] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.end() : end]
        fields = _parse_fields(section, _FIELD)
        missing = [name for name in ("难度", "参考答案", "评分要点") if not fields.get(name)]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{file_path}: question {match.group(1)} missing {joined}")
        difficulty_text = fields["难度"]
        if difficulty_text not in _DIFFICULTY:
            raise ValueError(
                f"{file_path}: question {match.group(1)} has invalid difficulty {difficulty_text}"
            )
        tags = tuple(part.strip() for part in fields.get("标签", "").split("、") if part.strip())
        questions.append(
            InterviewQuestion(
                id=match.group(1),
                text=match.group(2).strip(),
                domain=domain,
                difficulty=_DIFFICULTY[difficulty_text],
                tags=tags,
                reference_answer=fields["参考答案"],
                scoring_points=fields["评分要点"],
                source_path=str(file_path.resolve()),
                interview_type=interview_type,
                knowledge_points=_split_points(fields.get("知识点", "")),
                prerequisites=_split_points(fields.get("前置知识", "")),
            )
        )
    return questions


def load_question_directory(path: str | Path) -> list[InterviewQuestion]:
    directory = Path(path)
    if not directory.is_dir():
        raise ValueError(f"question directory does not exist: {directory}")
    questions = [
        question
        for file_path in sorted(directory.glob("*.md"))
        for question in load_question_file(file_path)
    ]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for question in questions:
        if question.id in seen:
            duplicates.add(question.id)
        seen.add(question.id)
    if duplicates:
        raise ValueError(f"duplicate question ids: {', '.join(sorted(duplicates))}")
    return questions


def index_question_directory(rag: TechRAG, path: str | Path) -> int:
    questions = load_question_directory(path)
    # Original imports must not replace a newer version saved in the web editor.
    online_ids = {str(metadata.get("question_id"))
                  for _, metadata in rag.database.document_rows()
                  if metadata.get("editor_version") == "question-editor-v1"}
    documents = [question.to_document() for question in questions
                 if question.id not in online_ids]
    rag.add_many(documents)
    return len(documents)


def question_from_content(content: str, metadata: dict[str, object]) -> InterviewQuestion:
    if metadata.get("editor_version") == "question-editor-v1":
        return InterviewQuestion(
            id=str(metadata["question_id"]), text=str(metadata["editor_question"]),
            domain=str(metadata["domain"]), difficulty=int(metadata["difficulty"]),
            tags=tuple(str(item) for item in metadata["tags"]),
            reference_answer=str(metadata["editor_answer"]),
            scoring_points=str(metadata["editor_scoring_points"]),
            source_path=str(metadata["source_path"]),
            interview_type=str(metadata["interview_type"]),
            knowledge_points=tuple(str(item) for item in metadata["knowledge_points"]),
            prerequisites=tuple(str(item) for item in metadata["prerequisites"]),
        )
    parsed = _parse_fields(content, _DOCUMENT_FIELD)
    fields = {
        label: parsed.get(label, "")
        for label in ("题目编号", "领域", "难度", "标签", "参考答案", "评分要点")
    }
    heading = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if not heading or not fields["题目编号"]:
        raise ValueError("retrieved content is not an interview question")
    tags = tuple(part.strip() for part in fields["标签"].split("、") if part.strip())
    return InterviewQuestion(
        id=fields["题目编号"],
        text=heading.group(1).strip(),
        domain=str(metadata.get("domain", fields["领域"])),
        difficulty=int(metadata.get("difficulty", fields["难度"] or 1)),
        tags=tags,
        reference_answer=fields["参考答案"],
        scoring_points=fields["评分要点"],
        source_path=str(metadata.get("source_path", "")),
        interview_type=str(metadata.get("interview_type", "technical")),
        knowledge_points=tuple(str(item) for item in metadata.get("knowledge_points", [])),
        prerequisites=tuple(str(item) for item in metadata.get("prerequisites", [])),
    )


def _split_points(value: str) -> tuple[str, ...]:
    parts = (part.strip() for part in re.split(r"[、,，]", value))
    return tuple(dict.fromkeys(part for part in parts if part))


def _parse_fields(content: str, pattern: re.Pattern[str]) -> dict[str, str]:
    """Retain all paragraphs until the next field, excluding question separators."""

    matches = list(pattern.finditer(content))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        raw = content[match.end() : end]
        # Editor exports indent entire fields, protecting headings/field-like
        # lines inside answers while retaining the original code indentation.
        lines = [line for line in raw.splitlines() if line.strip()]
        if raw.startswith("\n") and lines and all(line.startswith("    ") for line in lines):
            raw = textwrap.dedent(raw)
        value = raw.strip()
        fields[match.group(1)] = re.sub(r"\n\s*---\s*$", "", value).strip()
    return fields


def _frontmatter_value(text: str, key: str) -> str | None:
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return None
    value = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", match.group(1), re.MULTILINE)
    return value.group(1).strip() if value else None
