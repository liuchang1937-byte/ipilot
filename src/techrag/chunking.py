from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .providers import Embedder
from .text import chunk_text, count_tokens, normalize_markdown, split_sentences


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    body: str
    contextualized: str


@dataclass(frozen=True, slots=True)
class _MarkdownSection:
    headings: tuple[str, ...]
    heading_lines: tuple[str, ...]
    content: str


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def markdown_chunk_drafts(
    text: str,
    embedder: Embedder,
    *,
    chunk_size: int,
    overlap: int,
    threshold: float,
    semantic: bool = True,
) -> list[ChunkDraft]:
    """Split Markdown by heading hierarchy, semantic units, and token budget."""

    sections = _markdown_sections(normalize_markdown(text))
    drafts: list[ChunkDraft] = []
    for section in sections:
        heading_prefix = "\n".join(section.heading_lines)
        content_budget = max(1, chunk_size - count_tokens(heading_prefix))
        units = _split_oversized_units(_markdown_units(section.content), content_budget)
        groups = _semantic_groups(
            units,
            embedder,
            chunk_size=content_budget,
            overlap=overlap,
            threshold=threshold,
            semantic=semantic,
        )
        heading_path = " > ".join(section.headings)
        for group in groups or ([""] if heading_prefix else []):
            body = "\n\n".join(part for part in (heading_prefix, group) if part).strip()
            contextualized = (
                f"文档章节 / Section: {heading_path}\n\n{body}" if heading_path else body
            )
            if body:
                drafts.append(ChunkDraft(body=body, contextualized=contextualized))
    return drafts


def _markdown_sections(text: str) -> list[_MarkdownSection]:
    sections: list[_MarkdownSection] = []
    stack: dict[int, tuple[str, str]] = {}
    content: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        value = "\n".join(content).strip()
        if value:
            ordered = [stack[level] for level in sorted(stack)]
            sections.append(
                _MarkdownSection(
                    headings=tuple(item[1] for item in ordered),
                    heading_lines=tuple(item[0] for item in ordered),
                    content=value,
                )
            )
        content.clear()

    for line in text.splitlines():
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            content.append(line)
            continue
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            stack = {key: value for key, value in stack.items() if key < level}
            stack[level] = (f"{'#' * level} {match.group(2).strip()}", match.group(2).strip())
        else:
            content.append(line)
    flush()
    if sections:
        return sections
    if stack:
        ordered = [stack[level] for level in sorted(stack)]
        return [
            _MarkdownSection(
                headings=tuple(item[1] for item in ordered),
                heading_lines=tuple(item[0] for item in ordered),
                content="",
            )
        ]
    return [_MarkdownSection((), (), text)]


def _markdown_units(text: str) -> list[str]:
    """Keep fenced code, tables, and list runs intact while splitting paragraphs."""

    units: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in text.splitlines():
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            current.append(line)
            if not in_fence:
                units.append("\n".join(current).strip())
                current = []
            continue
        if in_fence:
            current.append(line)
        elif not line.strip():
            if current:
                units.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        units.append("\n".join(current).strip())
    return [unit for unit in units if unit]


def _split_oversized_units(units: Sequence[str], chunk_size: int) -> list[str]:
    output: list[str] = []
    for unit in units:
        if count_tokens(unit) <= chunk_size:
            output.append(unit)
            continue
        sentences = split_sentences(unit)
        if len(sentences) <= 1:
            output.extend(_hard_token_split(unit, chunk_size))
            continue
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and count_tokens(candidate) > chunk_size:
                output.append(current)
                current = sentence
            else:
                current = candidate
            if count_tokens(current) > chunk_size:
                output.extend(_hard_token_split(current, chunk_size))
                current = ""
        if current:
            output.append(current)
    return output


def _hard_token_split(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if current and count_tokens(candidate) > limit:
            pieces.append(current.strip())
            current = character
        else:
            current = candidate
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _semantic_groups(
    units: Sequence[str],
    embedder: Embedder,
    *,
    chunk_size: int,
    overlap: int,
    threshold: float,
    semantic: bool,
) -> list[str]:
    if not units:
        return []
    embeddings = embedder.embed(units) if semantic and len(units) > 1 else []
    if embeddings and len(embeddings) != len(units):
        raise ValueError("embedder returned an unexpected number of Markdown unit vectors")
    groups: list[str] = []
    current: list[str] = []
    for index, unit in enumerate(units):
        candidate = "\n\n".join([*current, unit])
        size_break = bool(current) and count_tokens(candidate) > chunk_size
        semantic_break = False
        if current and embeddings:
            left = np.asarray(embeddings[index - 1], dtype=np.float32)
            right = np.asarray(embeddings[index], dtype=np.float32)
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            similarity = float(np.dot(left, right) / denominator) if denominator else 0.0
            semantic_break = similarity < threshold and count_tokens("\n\n".join(current)) >= (
                chunk_size // 3
            )
        if size_break or semantic_break:
            groups.append("\n\n".join(current))
            current = _overlap_units(current, overlap)
            while current and count_tokens("\n\n".join([*current, unit])) > chunk_size:
                current.pop(0)
        current.append(unit)
    if current:
        groups.append("\n\n".join(current))
    return groups


def _overlap_units(units: Sequence[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    kept: list[str] = []
    used = 0
    for unit in reversed(units):
        tokens = count_tokens(unit)
        if kept and used + tokens > overlap:
            break
        kept.append(unit)
        used += tokens
        if used >= overlap:
            break
    return list(reversed(kept))


def retrieval_child_chunks(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Create small retrieval units while preserving Markdown atomic blocks."""

    units = _split_oversized_units(_markdown_units(normalize_markdown(text)), chunk_size)
    children: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = "\n\n".join([*current, unit])
        if current and count_tokens(candidate) > chunk_size:
            children.append("\n\n".join(current))
            current = _overlap_units(current, overlap)
            while current and count_tokens("\n\n".join([*current, unit])) > chunk_size:
                current.pop(0)
        current.append(unit)
    if current:
        children.append("\n\n".join(current))
    return children


def semantic_chunk_text(
    text: str,
    embedder: Embedder,
    *,
    chunk_size: int,
    overlap: int,
    threshold: float,
) -> list[str]:
    """Group adjacent sentences using embedding similarity and a size ceiling."""

    sentences = split_sentences(text)
    if len(sentences) < 2:
        return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    embeddings = embedder.embed(sentences)
    if len(embeddings) != len(sentences):
        raise ValueError("embedder returned an unexpected number of sentence vectors")

    chunks: list[str] = []
    current = sentences[0]
    previous = np.asarray(embeddings[0], dtype=np.float32)
    for sentence, raw_embedding in zip(sentences[1:], embeddings[1:], strict=True):
        embedding = np.asarray(raw_embedding, dtype=np.float32)
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(embedding))
        similarity = float(np.dot(previous, embedding) / denominator) if denominator else 0.0
        candidate = f"{current} {sentence}".strip()
        semantic_break = similarity < threshold and len(current) >= chunk_size // 3
        if len(candidate) > chunk_size or semantic_break:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip()
        else:
            current = candidate
        previous = embedding
    if current:
        chunks.append(current)
    return chunks


def add_contextual_headings(chunks: Sequence[str]) -> list[ChunkDraft]:
    """Carry the latest Markdown heading into chunks for context-aware embeddings."""

    heading_stack: dict[int, str] = {}
    drafts: list[ChunkDraft] = []
    for chunk in chunks:
        for level_text, heading in re.findall(r"(?m)^(#{1,6})\s+(.+)$", chunk):
            level = len(level_text)
            heading_stack[level] = heading.strip()
            heading_stack = {key: value for key, value in heading_stack.items() if key <= level}
        heading_path = " > ".join(value for _, value in sorted(heading_stack.items()))
        contextualized = f"文档章节 / Section: {heading_path}\n\n{chunk}" if heading_path else chunk
        drafts.append(ChunkDraft(body=chunk, contextualized=contextualized))
    return drafts
