from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Sequence
from types import SimpleNamespace

from techrag import (
    Document,
    EvaluationCase,
    TechRAG,
    TechRAGConfig,
    add_context,
    evaluate,
)
from techrag.documents import document_to_markdown
from techrag.providers import DashScopeDeepSeekProvider


class FeatureModel:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 32
            for token in text.lower():
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:2], "big") % 32
                vector[index] += 1.0
            vectors.append(vector)
        return vectors

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        if "Allowed metadata" in prompt:
            return '{"topic":"database"}'
        return "answer [1]"

    def stream_complete(self, messages: Sequence[dict[str, str]]) -> Iterator[str]:
        yield "answer"
        yield " [1]"


def _rag(tmp_path, **config_values) -> TechRAG:
    model = FeatureModel()
    config = TechRAGConfig(
        db_path=tmp_path / "features-chroma",
        chunk_size=100,
        chunk_overlap=10,
        **config_values,
    )
    return TechRAG(config, _embedder=model, _chat_model=model)


def test_search_modes_metadata_filter_and_self_query(tmp_path) -> None:
    with _rag(tmp_path, self_query=True) as rag:
        rag.add(
            Document.from_text(
                "DuckDB columnar analytics database.",
                source="database.md",
                metadata={"topic": "database", "language": ["en", "zh"]},
            )
        )
        rag.add(
            Document.from_text(
                "Python supports generators and iterators.",
                source="python.md",
                metadata={"topic": "programming"},
            )
        )
        keyword = rag.search(
            "DuckDB", mode="keyword", metadata_filter={"topic": "database"}
        )
        vector = rag.search("columnar", mode="vector")
        hybrid = rag.search("analytics database", mode="hybrid")
        assert [result.source for result in keyword] == ["database.md"]
        assert vector
        assert hybrid[0].source == "database.md"
        # self-query inferred topic=database and filters out the programming document.
        assert all(result.source == "database.md" for result in rag.search("database details"))


def test_multivector_spans_stream_delete_and_evaluation(tmp_path) -> None:
    with _rag(tmp_path, semantic_chunking=False, self_query=False) as rag:
        rag.add(
            Document.from_text(
                "# Storage\n\nDuckDB stores data by column. "
                "It runs in process and supports analytical SQL. "
                "Vector embeddings can be persisted as float lists.",
                source="guide.md",
                metadata={"topic": "database"},
            )
        )
        representation_count = len(rag.database.representation_rows())
        assert representation_count >= rag.stats()[1]
        assert rag.retrieve_context("analytical SQL")

        retrieved = []
        assert "".join(rag.stream("What does DuckDB support?", on_retrieval=retrieved.extend))
        assert retrieved
        assert "".join(rag.stream("你好")) == "answer [1]"

        report = evaluate(
            rag,
            [EvaluationCase("What runs in process?", ("guide.md",), "DuckDB")],
        )
        assert report.recall_at_k == 1.0
        assert report.mean_reciprocal_rank == 1.0
        message = add_context("question", rag.retrieve_context("SQL"))
        assert "<context>" in message["content"]
        assert rag.delete_by_metadata({"topic": "database"}) == 1
        assert rag.stats() == (0, 0)


def test_plain_document_conversion_and_pdf_routing(tmp_path, monkeypatch) -> None:
    text_path = tmp_path / "note.txt"
    text_path.write_text("中英 text", encoding="utf-8")
    assert document_to_markdown(text_path) == "中英 text"

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"not a real PDF")
    monkeypatch.setattr("techrag.documents.pdf_to_markdown", lambda path: "# PDF")
    assert document_to_markdown(pdf_path) == "# PDF"


def test_fixed_model_provider_is_the_default(tmp_path) -> None:
    config = TechRAGConfig(db_path=tmp_path / "models-chroma")
    with TechRAG(config) as rag:
        assert isinstance(rag.embedder, DashScopeDeepSeekProvider)
        assert rag.embedder.embedding_model == "dashscope/text-embedding-v4"
        assert rag.chat_model.chat_model == "deepseek/deepseek-v4-flash"


def test_provider_loads_dotenv_as_project_configuration(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "DASHSCOPE_API_KEY=from-dotenv\nDEEPSEEK_API_KEY=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-environment")

    DashScopeDeepSeekProvider()

    assert os.environ["DASHSCOPE_API_KEY"] == "from-dotenv"
    assert os.environ["DEEPSEEK_API_KEY"] == "from-dotenv"


def test_dashscope_embedding_batches_requests(monkeypatch) -> None:
    calls = []

    class FakeLiteLLM:
        @staticmethod
        def embedding(*, model, input):
            calls.append((model, input))
            return SimpleNamespace(
                data=[
                    {"index": index, "embedding": [float(len(text))]}
                    for index, text in enumerate(input)
                ]
            )

    provider = DashScopeDeepSeekProvider(embedding_batch_size=2)
    monkeypatch.setattr(provider, "_litellm", lambda: FakeLiteLLM)
    assert provider.embed(["a", "bb", "ccc", "dddd", "eeeee"]) == [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ]
    assert [len(batch) for _, batch in calls] == [2, 2, 1]


def test_mqe_and_hyde_expand_merge_and_deduplicate(tmp_path) -> None:
    class ExpansionModel(FeatureModel):
        def complete(self, messages: Sequence[dict[str, str]]) -> str:
            prompt = messages[-1]["content"]
            if "JSON array" in prompt:
                return '["embedded database", "analytical SQL", "embedded database"]'
            if "hypothetical passage" in prompt:
                return "DuckDB is an embedded database for analytical SQL."
            return super().complete(messages)

    model = ExpansionModel()
    config = TechRAGConfig(
        db_path=tmp_path / "expanded-chroma",
        chunk_size=100,
        chunk_overlap=10,
        enable_mqe=True,
        enable_hyde=True,
        candidate_pool_multiplier=4,
    )
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add(Document.from_text("DuckDB embedded analytical SQL database.", source="db.md"))
        rag.add(Document.from_text("Python iterator protocol.", source="python.md"))
        results = rag.search("database analytics", top_k=2)
        assert results
        assert results[0].source == "db.md"
        assert len({result.chunk_id for result in results}) == len(results)


def test_per_request_reranker_reorders_search_and_ask_candidates(tmp_path) -> None:
    class ReverseReranker:
        def rank(self, query: str, documents: Sequence[str]) -> list[float]:
            assert query == "database"
            return [float(index) for index in range(len(documents))]

    model = FeatureModel()
    config = TechRAGConfig(
        db_path=tmp_path / "rerank-chroma",
        chunk_size=100,
        chunk_overlap=10,
        search_mode="keyword",
        rerank_oversample=2,
    )
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add_text("database database", source="strong.md")
        rag.add_text("database", source="weak.md")

        reranker = ReverseReranker()
        result = rag.search("database", top_k=1, reranker=reranker)
        answer = rag.ask("database", top_k=1, adaptive=False, reranker=reranker)

        assert result[0].source == "weak.md"
        assert result[0].rerank_score == 1.0
        assert answer.sources[0].source == "weak.md"


def test_search_without_reranker_keeps_first_stage_scores(tmp_path) -> None:
    with _rag(tmp_path) as rag:
        rag.add_text("DuckDB database", source="db.md")

        result = rag.search("DuckDB", top_k=1)

        assert result[0].rerank_score is None


def test_child_representations_retrieve_and_return_parent_context(tmp_path) -> None:
    model = FeatureModel()
    config = TechRAGConfig(
        db_path=tmp_path / "hierarchical-chroma",
        chunk_size=300,
        chunk_overlap=0,
        child_chunk_size=30,
        child_chunk_overlap=0,
        semantic_chunking=False,
        search_mode="keyword",
    )
    content = (
        "# Storage\n\n"
        "DuckDB runs inside the application process and supports analytical SQL.\n\n"
        "A UNIQUE_MARKER vector is persisted as a float list in a table.\n\n"
        "The complete section also explains updates, transactions, and local files."
    )
    with TechRAG(config, _embedder=model, _chat_model=model) as rag:
        rag.add_text(content, source="storage.md")
        representation_count = len(rag.database.representation_rows())

        results = rag.search("UNIQUE_MARKER", top_k=5)

        assert rag.stats() == (1, 1)
        assert representation_count >= 2
        assert len(results) == 1
        assert results[0].content == content
