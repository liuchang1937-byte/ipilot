import hashlib
from collections.abc import Sequence

from techrag.chunking import markdown_chunk_drafts
from techrag.text import chunk_text, count_tokens, tokenize


class _Embedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [float(byte) for byte in hashlib.sha256(text.encode()).digest()[:8]]
            for text in texts
        ]


def test_mixed_language_tokenization() -> None:
    tokens = tokenize("DuckDB 支持向量搜索。Tech-RAG works!")
    assert "duckdb" in tokens
    assert "tech-rag" in tokens
    assert "支持" in tokens
    assert "向量" in tokens


def test_bilingual_chunking_keeps_content() -> None:
    text = "第一句介绍数据库。第二句介绍检索！DuckDB stores vectors. RAG retrieves context."
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert chunks
    assert "第一句" in chunks[0]
    assert "DuckDB" in " ".join(chunks)


def test_markdown_chunking_preserves_hierarchy_and_fenced_code() -> None:
    text = """# Guide

Intro paragraph.

## Install

Run this command:

```python
value = "# not a heading"
print(value)
```

Final explanation with several words for splitting.
"""
    drafts = markdown_chunk_drafts(
        text,
        _Embedder(),
        chunk_size=30,
        overlap=5,
        threshold=-1,
        semantic=False,
    )
    assert drafts
    assert all(count_tokens(draft.body) <= 40 for draft in drafts)
    install = [draft for draft in drafts if "## Install" in draft.body]
    assert install
    assert all("# Guide" in draft.body for draft in install)
    assert any('value = "# not a heading"' in draft.body for draft in drafts)
