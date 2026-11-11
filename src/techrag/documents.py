from __future__ import annotations

import re
from pathlib import Path
from statistics import median
from typing import Any

from .text import normalize_markdown


class DocumentConversionError(RuntimeError):
    pass


def pdf_to_markdown(path: str | Path) -> str:
    """Convert a PDF to Markdown using layout-aware pdftext extraction."""

    pdf_path = Path(path)
    try:
        from pdftext.extraction import dictionary_output
    except ModuleNotFoundError as exc:
        raise DocumentConversionError(
            "PDF support requires the document dependencies: uv sync --extra documents"
        ) from exc

    pages: list[dict[str, Any]] = dictionary_output(pdf_path, sort=True, keep_chars=False)
    spans = [
        span
        for page in pages
        for block in page.get("blocks", [])
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("text", "").strip()
    ]
    body_sizes = [float(span.get("font", {}).get("size", 0) or 0) for span in spans]
    useful_sizes = [size for size in body_sizes if size > 1]
    body_size = median(useful_sizes) if useful_sizes else 10.0

    converted_pages: list[str] = []
    for page_number, page in enumerate(pages, start=1):
        output: list[str] = []
        for block in page.get("blocks", []):
            lines: list[str] = []
            for line in block.get("lines", []):
                line_spans = line.get("spans", [])
                text = "".join(str(span.get("text", "")) for span in line_spans).strip()
                if not text or re.fullmatch(rf"(?:page\s*)?{page_number}", text, re.IGNORECASE):
                    continue
                sizes = [
                    float(span.get("font", {}).get("size", 0) or 0)
                    for span in line_spans
                ]
                size = max(sizes, default=body_size)
                if size >= body_size * 1.6:
                    text = f"# {text}"
                elif size >= body_size * 1.3:
                    text = f"## {text}"
                elif size >= body_size * 1.12:
                    text = f"### {text}"
                lines.append(text)
            if lines:
                output.append("\n".join(lines))
        converted_pages.append("\n\n".join(output))
    return normalize_markdown("\n\n---\n\n".join(page for page in converted_pages if page))


def document_to_markdown(path: str | Path) -> str:
    """Convert supported files to Markdown; PDF uses a dedicated converter."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_markdown(file_path)
    if suffix in {".md", ".markdown", ".txt"}:
        return normalize_markdown(file_path.read_text(encoding="utf-8"))
    try:
        from markitdown import MarkItDown
    except ModuleNotFoundError as exc:
        raise DocumentConversionError(
            "Office, HTML and other formats require: uv sync --extra documents"
        ) from exc
    result = MarkItDown().convert(str(file_path))
    text = result.text_content
    if not text or not text.strip():
        raise DocumentConversionError(f"MarkItDown produced no text for {file_path}")
    return normalize_markdown(text)
