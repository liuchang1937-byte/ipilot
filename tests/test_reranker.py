import sys
from types import SimpleNamespace

import pytest

from techrag.providers import BGEReranker, OnnxCrossEncoder, ProviderError


def test_bge_reranker_is_lazy_and_returns_float_scores(monkeypatch) -> None:
    reranker = BGEReranker()
    fake = SimpleNamespace(compute_score=lambda pairs, normalize: [0.2, 0.8])
    monkeypatch.setattr(reranker, "_load_model", lambda: fake)

    scores = reranker.rank("query", ["first", "second"])

    assert scores == [0.2, 0.8]


def test_bge_reranker_explains_missing_optional_dependency(monkeypatch) -> None:
    reranker = BGEReranker()
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)

    with pytest.raises(ProviderError, match="uv sync --extra rerank"):
        reranker._load_model()


def test_onnx_cross_encoder_requires_prepared_local_assets(tmp_path):
    reranker = OnnxCrossEncoder(tmp_path / "absent")
    assert not reranker.ready
    assert reranker.rank("q", []) == []
    with pytest.raises(ProviderError, match="prepare-reranker"):
        reranker.rank("q", ["document"])
