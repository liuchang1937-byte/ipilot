from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_LATIN_WORD = re.compile(r"[a-z0-9]+(?:[._+#-][a-z0-9]+)*", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])(?=\s|$)|\n{2,}")
_TOKEN_PIECE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z0-9]+(?:[._+#-][a-z0-9]+)*|[^\w\s]",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def normalize_markdown(text: str) -> str:
    """Normalize Markdown without destroying code, list, or table indentation."""

    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n{4,}", "\n\n\n", "\n".join(lines)).strip()


def count_tokens(text: str) -> int:
    """Return a deterministic mixed Chinese/English token estimate."""

    return len(_TOKEN_PIECE.findall(text))


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English without a language-specific dependency.

    English uses normalized words; contiguous CJK text emits unigrams and bigrams,
    which gives useful keyword recall without relying on whitespace segmentation.
    """

    normalized = normalize_text(text).lower()
    tokens = _LATIN_WORD.findall(normalized)
    for run in _CJK_RUN.findall(normalized):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(normalize_text(text)) if part.strip()]


def chunk_text(text: str, *, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """Split Chinese/English text on sentence boundaries with character overlap."""

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        # Hard-split pathological sentences while preserving all content.
        pieces = [sentence[i : i + chunk_size] for i in range(0, len(sentence), chunk_size)]
        for piece in pieces:
            is_heading = piece.startswith("#") or current.startswith("#")
            separator = "\n" if current and is_heading else " "
            candidate = f"{current}{separator if current else ''}{piece}".strip()
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                prefix = _overlap_tail(current, overlap)
                current = f"{prefix} {piece}".strip()
                if len(current) > chunk_size:
                    current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :] if first_space >= 0 else tail


def unique_terms(tokens: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(tokens))
