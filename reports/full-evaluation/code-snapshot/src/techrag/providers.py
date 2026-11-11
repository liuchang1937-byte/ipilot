from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from dotenv import find_dotenv, load_dotenv

EMBEDDING_MODEL = "dashscope/text-embedding-v4"
CHAT_MODEL = "deepseek/deepseek-v4-flash"


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    def complete(self, messages: Sequence[dict[str, str]]) -> str: ...


class Reranker(Protocol):
    def rank(self, query: str, documents: Sequence[str]) -> list[float]: ...


class ProviderError(RuntimeError):
    pass


class OnnxCrossEncoder:
    """Local multilingual query/document joint scoring with a pinned INT8 model."""

    repository = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    revision = "1427fd652930e4ba29e8149678df786c240d8825"
    files = {
        "model.onnx": ("onnx/model_quint8_avx2.onnx",
            "6c2513767fb63d008a4377bef7a7a3555433d9436342bb53e35a3a72ffc52d4b"),
        "tokenizer.json": ("tokenizer.json",
            "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626"),
    }

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._session: Any = None
        self._tokenizer: Any = None

    @property
    def ready(self) -> bool:
        return all((self.model_dir / name).is_file() for name in self.files)

    def prepare(self) -> None:
        """Download public model assets only; no queries or documents are sent."""
        import hashlib
        import urllib.request

        self.model_dir.mkdir(parents=True, exist_ok=True)
        for name, (remote, expected) in self.files.items():
            destination = self.model_dir / name
            if destination.is_file():
                with destination.open("rb") as existing:
                    if hashlib.file_digest(existing, "sha256").hexdigest() == expected:
                        continue
            temporary = destination.with_suffix(destination.suffix + ".part")
            url = f"https://huggingface.co/{self.repository}/resolve/{self.revision}/{remote}"
            try:
                with (
                    urllib.request.urlopen(url, timeout=60) as response,
                    temporary.open("wb") as out,
                ):
                    while block := response.read(1024 * 1024):
                        out.write(block)
                with temporary.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ProviderError("Cross Encoder model checksum mismatch")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)

    def _load(self) -> None:
        if self._session is not None:
            return
        if not self.ready:
            raise ProviderError("Run `techrag prepare-reranker` to install the local Cross Encoder")
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=512)
        tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"), sess_options=settings,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer, self._session = tokenizer, session

    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        import numpy as np

        self._load()
        scores = []
        for start in range(0, len(documents), 4):
            encodings = self._tokenizer.encode_batch([
                (query, document) for document in documents[start:start + 4]
            ])
            values = {
                "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
                "token_type_ids": np.asarray([e.type_ids for e in encodings], dtype=np.int64),
            }
            inputs = {i.name: values[i.name] for i in self._session.get_inputs()}
            logits = np.asarray(self._session.run(None, inputs)[0]).reshape(-1)
            scores.extend(float(v) for v in 1 / (1 + np.exp(-np.clip(logits, -60, 60))))
        return scores


class BGEReranker:
    """Optional multilingual Cross-Encoder reranker backed by FlagEmbedding."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        use_fp16: bool = False,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.normalize = normalize
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagReranker
        except ModuleNotFoundError as exc:
            raise ProviderError(
                "Install the rerank extra with `uv sync --extra rerank` to use BGEReranker"
            ) from exc
        self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
        return self._model

    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        pairs = [[query, document] for document in documents]
        raw_scores = self._load_model().compute_score(pairs, normalize=self.normalize)
        if isinstance(raw_scores, (int, float)):
            return [float(raw_scores)]
        return [float(score) for score in raw_scores]


class DashScopeDeepSeekProvider:
    """The project's fixed DashScope embedding and DeepSeek chat provider."""

    embedding_model = EMBEDDING_MODEL
    chat_model = CHAT_MODEL

    def __init__(self, *, embedding_batch_size: int = 10) -> None:
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=True)
        self.embedding_batch_size = embedding_batch_size

    @staticmethod
    def _litellm():
        try:
            import litellm
        except ModuleNotFoundError as exc:
            message = "Install the project dependencies to use the configured models"
            raise ProviderError(message) from exc
        return litellm

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        litellm = self._litellm()
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = list(texts[start : start + self.embedding_batch_size])
            response = litellm.embedding(model=EMBEDDING_MODEL, input=batch)
            data = sorted(response.data, key=lambda item: item["index"])
            embeddings.extend(list(map(float, item["embedding"])) for item in data)
        return embeddings

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        response = self._litellm().completion(model=CHAT_MODEL, messages=list(messages))
        return str(response.choices[0].message.content or "")

    def stream_complete(self, messages: Sequence[dict[str, str]]) -> Iterator[str]:
        stream = self._litellm().completion(
            model=CHAT_MODEL, messages=list(messages), stream=True
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield str(token)
