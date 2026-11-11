from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .chunking import (
    ChunkDraft,
    add_contextual_headings,
    markdown_chunk_drafts,
    retrieval_child_chunks,
    semantic_chunk_text,
)
from .config import TechRAGConfig
from .database import ChunkRecord, Database, RepresentationRecord
from .models import Answer, ChunkSpan, Document, SearchResult
from .providers import (
    EMBEDDING_MODEL,
    ChatModel,
    DashScopeDeepSeekProvider,
    Embedder,
    Reranker,
)
from .search import (
    MetadataFilter,
    fuse_hybrid_results,
    hybrid_search,
    keyword_search,
    rerank_results,
    retrieve_chunk_spans,
    retrieve_chunks,
    vector_search,
    vector_search_many,
)
from .text import chunk_text, normalize_markdown, normalize_text, tokenize
from .timing import timed_stage

_SYSTEM_PROMPT = """你是一个严谨的中英双语技术助手。请只根据给定资料回答问题。
如果资料不足，请明确说明不知道，不要编造。保持用户提问所使用的主要语言。
引用资料时使用 [1]、[2] 形式标注来源。"""

_NO_CONTEXT_SYSTEM_PROMPT = """你是一个中英双语技术助手。当前问题无需检索知识库。
保持用户提问所使用的主要语言，并简洁回答。"""

_CHUNKING_VERSION = "hierarchical-v2"


class TechRAG:
    def __init__(
        self,
        config: TechRAGConfig | None = None,
        *,
        _embedder: Embedder | None = None,
        _chat_model: ChatModel | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config or TechRAGConfig()
        provider = (
            DashScopeDeepSeekProvider()
            if _embedder is None or _chat_model is None
            else None
        )
        self.embedder = _embedder or provider
        self.chat_model = _chat_model or provider
        self.reranker = reranker
        self.database = Database(self.config.db_path)

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> TechRAG:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def add(self, document: Document) -> int:
        self.database.ensure_index_setting("embedding_model", EMBEDDING_MODEL)
        content = normalize_markdown(document.content)
        if not content:
            raise ValueError("document content cannot be empty")
        question_parent = document.metadata.get("collection") == "interview-question"
        if question_parent:
            question_id = document.metadata.get("question_id")
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError("interview questions require a nonempty question_id")
            if document.source != f"interview-question://{question_id}":
                raise ValueError("interview question source must match its question_id")
            signature = {
                "version": "interview-question-parent-v1",
                "child_chunk_size": self.config.child_chunk_size,
                "child_chunk_overlap": self.config.child_chunk_overlap,
            }
        else:
            signature = {
                "version": _CHUNKING_VERSION,
                "markdown": self.config.markdown_chunking,
                "semantic": self.config.semantic_chunking,
                "threshold": self.config.semantic_threshold,
                "size": self.config.chunk_size,
                "overlap": self.config.chunk_overlap,
                "hierarchical_retrieval": self.config.hierarchical_retrieval,
                "child_chunk_size": self.config.child_chunk_size,
                "child_chunk_overlap": self.config.child_chunk_overlap,
            }
        chunking_signature = json.dumps(signature, sort_keys=True)
        checksum = hashlib.sha256(f"{chunking_signature}\n{content}".encode()).hexdigest()
        existing = self.database.existing_document(document.source)
        if existing and existing[1] == checksum:
            if question_parent:
                updates = {
                    key: document.metadata.get(key, [])
                    for key in ("knowledge_points", "prerequisites")
                }
                if document.metadata.get("editor_version") == "question-editor-v1":
                    updates.update(document.metadata)
                self.database.update_question_routing(document.source, updates)
            return existing[0]

        if question_parent:
            # A question is the atomic scoring context, even when its answer
            # exceeds the generic document budget or has semantic boundaries.
            drafts = [ChunkDraft(body=content, contextualized=content)]
        elif self.config.markdown_chunking:
            drafts = markdown_chunk_drafts(
                content,
                self.embedder,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                threshold=self.config.semantic_threshold,
                semantic=self.config.semantic_chunking,
            )
        elif self.config.semantic_chunking:
            chunks = semantic_chunk_text(
                content,
                self.embedder,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                threshold=self.config.semantic_threshold,
            )
            drafts = add_contextual_headings(chunks)
        else:
            chunks = chunk_text(
                content,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            drafts = add_contextual_headings(chunks)
        representation_groups = (
            [_question_representation_texts(document, content, self.config)]
            if question_parent
            else [_representation_texts(draft, self.config) for draft in drafts]
        )
        embedding_inputs = [
            representation
            for representations in representation_groups
            for representation in representations
        ]
        input_ranges: list[tuple[int, int]] = []
        cursor = 0
        for representations in representation_groups:
            input_ranges.append((cursor, cursor + len(representations)))
            cursor += len(representations)
        all_embeddings = self.embedder.embed(embedding_inputs)
        if len(all_embeddings) != len(embedding_inputs):
            raise ValueError("embedder returned an unexpected number of vectors")
        records: list[ChunkRecord] = []
        for draft, representations, (start, end) in zip(
            drafts, representation_groups, input_ranges, strict=True
        ):
            embeddings = all_embeddings[start:end]
            records.append(
                ChunkRecord(
                    content=draft.body,
                    representations=[
                        RepresentationRecord(
                            content=text,
                            embedding=embedding,
                            frequencies=dict(Counter(tokenize(text))),
                        )
                        for text, embedding in zip(representations, embeddings, strict=True)
                    ],
                )
            )
        return self.database.replace_document(
            source=document.source,
            checksum=checksum,
            metadata=document.metadata,
            chunks=records,
        )

    def add_file(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Convert and index a file, resolving its source to an absolute path."""

        return self.add(Document.from_path(path, metadata=metadata))

    def add_text(
        self,
        text: str,
        *,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Index in-memory text under a caller-provided source identifier."""

        return self.add(Document.from_text(text, source=source, metadata=metadata))

    def add_many(self, documents: Sequence[Document]) -> list[int]:
        return [self.add(document) for document in documents]

    def delete(self, source: str) -> bool:
        if self.database.delete_document(source):
            return True
        path = Path(source)
        return path.exists() and self.database.delete_document(str(path.resolve()))

    def delete_by_metadata(self, metadata_filter: MetadataFilter) -> int:
        if not metadata_filter:
            raise ValueError("metadata_filter cannot be empty")
        sources = [
            source
            for source, metadata in self.database.document_rows()
            if _metadata_matches(metadata, metadata_filter)
        ]
        return sum(self.database.delete_document(source) for source in sources)

    @timed_stage("retrieval")
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: str | None = None,
        metadata_filter: MetadataFilter | None = None,
        enable_mqe: bool | None = None,
        enable_hyde: bool | None = None,
        reranker: Reranker | None = None,
        enable_rerank: bool = True,
        trace: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query = normalize_text(query)
        if not query:
            raise ValueError("query cannot be empty")
        self.database.ensure_index_setting("embedding_model", EMBEDDING_MODEL)
        mode = mode or self.config.search_mode
        if mode not in {"vector", "keyword", "hybrid"}:
            raise ValueError("mode must be vector, keyword, or hybrid")
        if self.config.self_query:
            inferred = self._infer_metadata_filter(query)
            metadata_filter = {**inferred, **(metadata_filter or {})}
        result_count = top_k or self.config.top_k
        use_mqe = self.config.enable_mqe if enable_mqe is None else enable_mqe
        use_hyde = self.config.enable_hyde if enable_hyde is None else enable_hyde
        use_hyde = use_hyde and mode != "keyword"
        active_reranker = (reranker or self.reranker) if enable_rerank else None
        if trace is not None:
            trace.update({"mode": mode, "queries": [{"kind": "original", "text": query}],
                          "warnings": [], "reranked": False})
        if use_mqe or use_hyde:
            return self._expanded_search(
                query,
                top_k=result_count,
                mode=mode,
                metadata_filter=metadata_filter,
                enable_mqe=use_mqe,
                enable_hyde=use_hyde,
                reranker=active_reranker,
                trace=trace,
            )
        candidate_count = (
            result_count * self.config.rerank_oversample
            if active_reranker is not None
            else result_count
        )
        if mode == "keyword":
            results = keyword_search(
                self.database,
                query,
                top_k=candidate_count,
                metadata_filter=metadata_filter,
            )
        else:
            embeddings = self.embedder.embed([query])
            if len(embeddings) != 1:
                raise ValueError("embedder must return one vector for one query")
            if mode == "vector":
                results = vector_search(
                    self.database,
                    embeddings[0],
                    top_k=candidate_count,
                    metadata_filter=metadata_filter,
                )
            else:
                results = hybrid_search(
                    self.database,
                    query,
                    embeddings[0],
                    top_k=candidate_count,
                    vector_weight=self.config.vector_weight,
                    metadata_filter=metadata_filter,
                )
        if active_reranker is not None:
            results = rerank_results(query, results, active_reranker)
        if trace is not None:
            trace.update({"candidate_count": len(results),
                          "reranked": active_reranker is not None})
        return results[:result_count]

    def _expanded_search(
        self,
        query: str,
        *,
        top_k: int,
        mode: str,
        metadata_filter: MetadataFilter | None,
        enable_mqe: bool,
        enable_hyde: bool,
        reranker: Reranker | None,
        trace: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Expand, retrieve, deduplicate, fuse, and rerank a query."""

        expansions: list[tuple[str, str]] = [("original", query)]
        warnings: list[str] = []
        if enable_mqe:
            try:
                variants = self._generate_mqe(query)
                expansions.extend(("mqe", item[:800]) for item in variants)
                if not variants:
                    warnings.append("MQE 未生成有效改写，保留原问题检索。")
            except Exception:
                warnings.append("MQE 暂不可用，保留原问题检索。")
        if enable_hyde:
            try:
                hypothetical = self._generate_hyde(query)
                if hypothetical:
                    expansions.append(("hyde", hypothetical[:1600]))
                else:
                    warnings.append("HyDE 未生成有效假设文本，保留原问题检索。")
            except Exception:
                warnings.append("HyDE 暂不可用，保留原问题检索。")

        unique: list[tuple[str, str]] = []
        seen: set[str] = set()
        for kind, value in expansions:
            normalized = normalize_text(value)
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                unique.append((kind, normalized))

        candidate_count = top_k * self.config.candidate_pool_multiplier
        query_embeddings: list[list[float]] = []
        if mode != "keyword":
            query_embeddings = self.embedder.embed([value for _, value in unique])
            if len(query_embeddings) != len(unique):
                raise ValueError("embedder returned an unexpected number of expanded-query vectors")
        vector_candidate_count = candidate_count * 3 if mode == "hybrid" else candidate_count
        vector_results = (
            vector_search_many(
                self.database,
                query_embeddings,
                top_k=vector_candidate_count,
                metadata_filter=metadata_filter,
            )
            if query_embeddings
            else []
        )

        result_lists: list[list[SearchResult]] = []
        for index, (kind, value) in enumerate(unique):
            if kind == "hyde" and mode == "keyword":
                continue
            if mode == "keyword":
                results = keyword_search(
                    self.database,
                    value,
                    top_k=candidate_count,
                    metadata_filter=metadata_filter,
                )
            elif mode == "vector" or kind == "hyde":
                results = vector_results[index][:candidate_count]
            else:
                keywords = keyword_search(
                    self.database,
                    value,
                    top_k=vector_candidate_count,
                    metadata_filter=metadata_filter,
                )
                results = fuse_hybrid_results(
                    vector_results[index],
                    keywords,
                    top_k=candidate_count,
                    vector_weight=self.config.vector_weight,
                )
            result_lists.append(results)

        # Rank fusion makes scores from vector, keyword, MQE, and HyDE comparable.
        scores: Counter[int] = Counter()
        by_id: dict[int, SearchResult] = {}
        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                scores[result.chunk_id] += 1.0 / (60 + rank)
                previous = by_id.get(result.chunk_id)
                by_id[result.chunk_id] = replace(
                    result,
                    vector_score=max(result.vector_score, previous.vector_score if previous else 0),
                    keyword_score=max(result.keyword_score,
                                      previous.keyword_score if previous else 0),
                )
        pool_size = max(candidate_count, top_k * self.config.rerank_oversample)
        ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[
            :pool_size
        ]
        merged = [
            replace(by_id[chunk_id], score=float(scores[chunk_id]))
            for chunk_id in ranked_ids
        ]
        if reranker is not None:
            merged = rerank_results(query, merged, reranker)
        if trace is not None:
            trace.update({"queries": [{"kind": kind, "text": value} for kind, value in unique],
                          "candidate_count": len(merged), "warnings": warnings,
                          "reranked": reranker is not None})
        return merged[:top_k]

    @timed_stage("mqe")
    def _generate_mqe(self, query: str) -> list[str]:
        prompt = (
            "Generate three diverse search queries that preserve the user's intent and improve "
            "retrieval recall. Return only a JSON array of strings. Keep the query language unless "
            f"a cross-language variant is useful.\nQuery: {query}"
        )
        response = self.chat_model.complete([{"role": "user", "content": prompt}])
        return _parse_json_string_list(response, limit=3)

    @timed_stage("hyde")
    def _generate_hyde(self, query: str) -> str:
        prompt = (
            "Write a concise hypothetical passage that would directly answer the query. It is used "
            "only for vector retrieval, so include likely technical terminology and do not add "
            f"citations or commentary.\nQuery: {query}"
        )
        return self.chat_model.complete([{"role": "user", "content": prompt}]).strip()

    def vector_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        reranker: Reranker | None = None,
    ) -> list[SearchResult]:
        return self.search(
            query,
            top_k=top_k,
            mode="vector",
            metadata_filter=metadata_filter,
            reranker=reranker,
        )

    def keyword_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        reranker: Reranker | None = None,
    ) -> list[SearchResult]:
        return self.search(
            query,
            top_k=top_k,
            mode="keyword",
            metadata_filter=metadata_filter,
            reranker=reranker,
        )

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        reranker: Reranker | None = None,
    ) -> list[SearchResult]:
        return self.search(
            query,
            top_k=top_k,
            mode="hybrid",
            metadata_filter=metadata_filter,
            reranker=reranker,
        )

    def retrieve_chunks(self, chunk_ids: Sequence[int]) -> list[SearchResult]:
        return retrieve_chunks(self.database, chunk_ids)

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        *,
        reranker: Reranker | None = None,
    ) -> list[SearchResult]:
        active_reranker = reranker or self.reranker
        if active_reranker is None:
            raise ValueError("rerank requires an injected Reranker")
        return rerank_results(query, results, active_reranker)

    def search_and_rerank_chunks(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        reranker: Reranker | None = None,
    ) -> list[SearchResult]:
        active_reranker = reranker or self.reranker
        if active_reranker is None:
            raise ValueError("search_and_rerank_chunks requires a Reranker")
        return self.search(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            reranker=active_reranker,
        )

    def search_and_rerank_chunk_spans(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        reranker: Reranker | None = None,
    ) -> list[ChunkSpan]:
        results = self.search_and_rerank_chunks(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            reranker=reranker,
        )
        return self.retrieve_chunk_spans(results)

    def retrieve_chunk_spans(
        self, results: Sequence[SearchResult], *, neighbors: int | None = None
    ) -> list[ChunkSpan]:
        return retrieve_chunk_spans(
            self.database,
            results,
            neighbors=self.config.chunk_neighbors if neighbors is None else neighbors,
        )

    def retrieve_context(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        reranker: Reranker | None = None,
    ) -> list[ChunkSpan]:
        results = self.search(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            reranker=reranker,
        )
        return retrieve_chunk_spans(
            self.database, results, neighbors=self.config.chunk_neighbors
        )

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        history: Sequence[dict[str, str]] | None = None,
        adaptive: bool | None = None,
        reranker: Reranker | None = None,
    ) -> Answer:
        messages, sources = self._prepare_messages(
            question,
            top_k=top_k,
            metadata_filter=metadata_filter,
            history=history,
            adaptive=adaptive,
            reranker=reranker,
        )
        text = self.chat_model.complete(messages)
        return Answer(text=text, sources=tuple(sources))

    def stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
        history: Sequence[dict[str, str]] | None = None,
        adaptive: bool | None = None,
        on_retrieval: Callable[[list[SearchResult]], None] | None = None,
        reranker: Reranker | None = None,
    ) -> Iterator[str]:
        messages, sources = self._prepare_messages(
            question,
            top_k=top_k,
            metadata_filter=metadata_filter,
            history=history,
            adaptive=adaptive,
            reranker=reranker,
        )
        if on_retrieval and sources:
            on_retrieval(sources)
        stream_method = getattr(self.chat_model, "stream_complete", None)
        if callable(stream_method):
            yield from stream_method(messages)
        else:
            yield self.chat_model.complete(messages)

    def rag(
        self,
        messages: list[dict[str, str]],
        *,
        metadata_filter: MetadataFilter | None = None,
        on_retrieval: Callable[[list[SearchResult]], None] | None = None,
        reranker: Reranker | None = None,
    ) -> Iterator[str]:
        user_messages = [message for message in messages if message.get("role") == "user"]
        if not user_messages:
            raise ValueError("messages must contain a user message")
        question = user_messages[-1]["content"]
        history = messages[:-1]
        parts: list[str] = []
        for part in self.stream(
            question,
            metadata_filter=metadata_filter,
            history=history,
            on_retrieval=on_retrieval,
            reranker=reranker,
        ):
            parts.append(part)
            yield part
        messages.append({"role": "assistant", "content": "".join(parts)})

    async def async_rag(
        self,
        messages: list[dict[str, str]],
        *,
        metadata_filter: MetadataFilter | None = None,
        on_retrieval: Callable[[list[SearchResult]], None] | None = None,
        reranker: Reranker | None = None,
    ) -> AsyncIterator[str]:
        """Async-compatible streaming API; model calls remain provider-dependent."""

        for token in self.rag(
            messages,
            metadata_filter=metadata_filter,
            on_retrieval=on_retrieval,
            reranker=reranker,
        ):
            yield token

    def expand_document_metadata(
        self, document: Document, fields: Mapping[str, str]
    ) -> Document:
        prompt = (
            "Extract metadata from the document. Return JSON only. "
            f"Fields and descriptions: {json.dumps(fields, ensure_ascii=False)}\n\n"
            f"Document:\n{document.content[:8000]}"
        )
        extracted = _parse_json_object(
            self.chat_model.complete([{"role": "user", "content": prompt}])
        )
        return Document(
            content=document.content,
            source=document.source,
            metadata={**document.metadata, **extracted},
        )

    def stats(self) -> tuple[int, int]:
        return self.database.stats()

    def _infer_metadata_filter(self, query: str) -> dict[str, object]:
        allowed = self.database.metadata_values()
        if not allowed:
            return {}
        prompt = (
            "Map the query to metadata filters. Use only provided keys and exact values. "
            "Return {} when no reliable filter exists. Return JSON only.\n"
            f"Allowed metadata: {json.dumps(allowed, ensure_ascii=False)}\nQuery: {query}"
        )
        candidate = _parse_json_object(
            self.chat_model.complete([{"role": "user", "content": prompt}])
        )
        return {
            key: value
            for key, value in candidate.items()
            if key in allowed and _allowed_metadata_value(value, allowed[key])
        }

    def _prepare_messages(
        self,
        question: str,
        *,
        top_k: int | None,
        metadata_filter: MetadataFilter | None,
        history: Sequence[dict[str, str]] | None,
        adaptive: bool | None,
        reranker: Reranker | None,
    ) -> tuple[list[dict[str, str]], list[SearchResult]]:
        should_adapt = self.config.adaptive_retrieval if adaptive is None else adaptive
        should_retrieve = not should_adapt or _needs_retrieval(question)
        sources: list[SearchResult] = []
        context = ""
        if should_retrieve:
            sources = self.search(
                question,
                top_k=top_k,
                metadata_filter=metadata_filter,
                reranker=reranker,
            )
            spans = retrieve_chunk_spans(
                self.database, sources, neighbors=self.config.chunk_neighbors
            )
            context = _format_context(spans, self.config.max_context_chars)
        system = _SYSTEM_PROMPT if should_retrieve else _NO_CONTEXT_SYSTEM_PROMPT
        messages = [{"role": "system", "content": system}, *(history or [])]
        if should_retrieve:
            content = (
                f"<context>\n{context or '(无可用资料)'}\n</context>"
                f"\n\n问题 / Question:\n{question}"
            )
        else:
            content = question
        messages.append({"role": "user", "content": content})
        return messages, sources


def _question_representation_texts(
    document: Document, content: str, config: TechRAGConfig
) -> list[str]:
    """Index in-question passages with a shared identity and title prefix."""

    heading = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if not heading:
        raise ValueError("interview questions require a question heading")
    prefix = (
        f"# {heading.group(1)}\n"
        f"题目编号: {document.metadata['question_id']}\n"
        f"领域: {document.metadata.get('domain', '')}"
    )
    # Identity fields belong in the shared prefix/metadata rather than becoming
    # standalone retrieval passages. The complete parent retains every field.
    body = content[heading.end():].strip()
    body = re.sub(r"^(题目编号|领域|难度|标签)[：:].*$", "", body, flags=re.MULTILINE)
    children = retrieval_child_chunks(
        body,
        chunk_size=config.child_chunk_size,
        overlap=config.child_chunk_overlap,
    )
    return [f"{prefix}\n\n{child}" for child in children] or [prefix]


def _representation_texts(draft: ChunkDraft, config: TechRAGConfig) -> list[str]:
    if not config.hierarchical_retrieval:
        return [draft.contextualized]
    child_size = min(config.child_chunk_size, config.chunk_size)
    child_overlap = min(config.child_chunk_overlap, child_size - 1)
    children = retrieval_child_chunks(
        draft.body,
        chunk_size=child_size,
        overlap=child_overlap,
    )
    if not children:
        return [draft.contextualized]
    prefix = ""
    if draft.contextualized.endswith(draft.body):
        prefix = draft.contextualized[: -len(draft.body)].strip()
    return ["\n\n".join(part for part in (prefix, child) if part) for child in children]


def _format_context(spans: Sequence[ChunkSpan], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for index, span in enumerate(spans, start=1):
        part = span.to_xml(index)
        if parts and used + len(part) > max_chars:
            break
        parts.append(part[: max_chars - used])
        used += len(parts[-1])
    return "\n\n".join(parts)


def _needs_retrieval(question: str) -> bool:
    normalized = re.sub(r"[\s!！。.?,，]+", "", question).lower()
    greetings = {
        "hi",
        "hello",
        "hey",
        "你好",
        "您好",
        "谢谢",
        "thankyou",
        "你是谁",
        "whoareyou",
    }
    return normalized not in greetings


def _parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _parse_json_string_list(text: str, *, limit: int) -> list[str]:
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        value = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()][:limit]


def _allowed_metadata_value(value: object, allowed: Sequence[object]) -> bool:
    if isinstance(value, list):
        return all(item in allowed for item in value)
    return value in allowed


def _metadata_matches(
    metadata: Mapping[str, object], metadata_filter: MetadataFilter
) -> bool:
    for key, expected in metadata_filter.items():
        if key not in metadata:
            return False
        actual_values = metadata[key] if isinstance(metadata[key], list) else [metadata[key]]
        expected_values = expected if isinstance(expected, (list, tuple, set)) else [expected]
        if not any(value in actual_values for value in expected_values):
            return False
    return True


def add_context(
    user_prompt: str,
    context: Sequence[ChunkSpan],
    *,
    max_context_chars: int = 12_000,
) -> dict[str, str]:
    """Build a cache-friendly user message containing XML-delimited context."""

    formatted = _format_context(context, max_context_chars)
    return {
        "role": "user",
        "content": (
            f"<context>\n{formatted or '(无可用资料)'}\n</context>"
            f"\n\n问题 / Question:\n{user_prompt}"
        ),
    }
