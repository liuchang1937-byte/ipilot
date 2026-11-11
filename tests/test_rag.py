from __future__ import annotations

import hashlib
from collections.abc import Sequence

from techrag import Document, TechRAG, TechRAGConfig


class FakeModel:
    dimensions = 64

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for character in text.lower():
                digest = hashlib.sha256(character.encode()).digest()
                vector[int.from_bytes(digest[:2], "big") % self.dimensions] += 1.0
            vectors.append(vector)
        return vectors

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        assert "DuckDB" in messages[-1]["content"]
        return "DuckDB 是嵌入式分析数据库。[1]"


def test_index_search_and_answer(tmp_path) -> None:
    model = FakeModel()
    config = TechRAGConfig(db_path=tmp_path / "test-chroma", chunk_size=100, chunk_overlap=20)
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        document_id = rag.add(
            Document.from_text(
                "DuckDB 是一个嵌入式分析数据库。It runs inside the application process.",
                source="duckdb.md",
                metadata={"language": "zh-en"},
            )
        )
        assert rag.add(
            Document.from_text(
                "DuckDB 是一个嵌入式分析数据库。It runs inside the application process.",
                source="duckdb.md",
            )
        ) == document_id
        assert rag.stats() == (1, 1)

        chinese = rag.search("什么是嵌入式分析数据库？")
        english = rag.search("Which database runs in the application process?")
        assert chinese[0].source == "duckdb.md"
        assert chinese[0].keyword_score > 0
        assert english[0].source == "duckdb.md"
        assert rag.ask("DuckDB 是什么？").text.endswith("[1]")


def test_reindex_changed_source_replaces_old_chunks(tmp_path) -> None:
    model = FakeModel()
    config = TechRAGConfig(db_path=tmp_path / "test-chroma", chunk_size=100, chunk_overlap=20)
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add(Document.from_text("旧内容。", source="same.md"))
        rag.add(Document.from_text("新内容。", source="same.md"))
        assert rag.stats() == (1, 1)
        assert rag.search("新内容")[0].content == "新内容。"


def test_add_file_and_text_accept_metadata(tmp_path) -> None:
    model = FakeModel()
    config = TechRAGConfig(db_path=tmp_path / "test-chroma", chunk_size=100, chunk_overlap=20)
    source_file = tmp_path / "guide.md"
    source_file.write_text("DuckDB stores vectors.", encoding="utf-8")

    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add_file(source_file, metadata={"language": "en"})
        rag.add_text(
            "DuckDB 存储向量。",
            source="guide-zh.md",
            metadata={"language": "zh"},
        )

        english = rag.search("DuckDB vectors", metadata_filter={"language": "en"})
        chinese = rag.search("DuckDB 向量", metadata_filter={"language": "zh"})

        assert len(english) == 1
        assert english[0].source == str(source_file.resolve())
        assert english[0].metadata == {"language": "en"}
        assert len(chinese) == 1
        assert chinese[0].source == "guide-zh.md"
        assert chinese[0].metadata == {"language": "zh"}
