from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

from .database import Database
from .models import ChunkSpan, SearchResult
from .providers import Reranker
from .text import tokenize, unique_terms
from .timing import timed_stage

MetadataFilter = Mapping[str, object | Sequence[object]]


def _metadata_matches(
    metadata: Mapping[str, object], metadata_filter: MetadataFilter | None
) -> bool:
    if not metadata_filter:
        return True
    for key, expected in metadata_filter.items():
        if key not in metadata:
            return False
        actual_values = metadata[key] if isinstance(metadata[key], list) else [metadata[key]]
        expected_values = expected if isinstance(expected, (list, tuple, set)) else [expected]
        if not any(value in actual_values for value in expected_values):
            return False
    return True


def _rows(
    database: Database, metadata_filter: MetadataFilter | None
) -> dict[int, tuple[object, ...]]:
    rows: dict[int, tuple[object, ...]] = {}
    for row in database.chunk_rows():
        metadata = json.loads(str(row[6])) if row[6] else {}
        if _metadata_matches(metadata, metadata_filter):
            rows[int(row[0])] = row
    return rows


def _result(
    row: tuple[object, ...],
    *,
    score: float,
    vector_score: float = 0.0,
    keyword_score: float = 0.0,
) -> SearchResult:
    return SearchResult(
        chunk_id=int(row[0]),
        document_id=int(row[1]),
        content=str(row[2]),
        source=str(row[5]),
        metadata=json.loads(str(row[6])) if row[6] else {},
        score=score,
        vector_score=vector_score,
        keyword_score=keyword_score,
        position=int(row[7]),
    )


def vector_search(
    database: Database,
    query_embedding: list[float],
    *,
    top_k: int,
    metadata_filter: MetadataFilter | None = None,
) -> list[SearchResult]:
    return vector_search_many(
        database, [query_embedding], top_k=top_k, metadata_filter=metadata_filter
    )[0]


@timed_stage("vector_search")
def vector_search_many(
    database: Database,
    query_embeddings: Sequence[list[float]],
    *,
    top_k: int,
    metadata_filter: MetadataFilter | None = None,
) -> list[list[SearchResult]]:
    """Query Chroma's cosine index, then aggregate child hits by parent."""
    if not query_embeddings:
        return []
    rows = _rows(database, metadata_filter)
    if not rows:
        return [[] for _ in query_embeddings]
    total_children = database.children.count()
    if not total_children:
        return [[] for _ in query_embeddings]
    candidate_count = min(max(top_k * 4, 10), total_children)
    target_parents = min(top_k, len(rows))
    while True:
        result = database.query_children(
            list(query_embeddings), parent_ids=list(rows), n_results=candidate_count,
        )
        scores: list[dict[int, float]] = []
        exhausted = []
        for metadata, distances in zip(result["metadatas"], result["distances"], strict=True):
            query_scores: dict[int, float] = {}
            for child, distance in zip(metadata, distances, strict=True):
                parent_id = int(child["parent_id"])
                # Chroma cosine distance = 1 - cosine; retain the public 0..1 score scale.
                score = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
                query_scores[parent_id] = max(query_scores.get(parent_id, 0.0), score)
            scores.append(query_scores)
            exhausted.append(len(metadata) < candidate_count)
        if candidate_count >= total_children or all(
            len(query_scores) >= target_parents or done
            for query_scores, done in zip(scores, exhausted, strict=True)
        ):
            break
        # Avoid one question's many children crowding out other parent candidates.
        candidate_count = min(candidate_count * 2, total_children)
    output = []
    for query_scores in scores:
        ranked = sorted(query_scores, key=lambda key: (-query_scores[key], key))[:top_k]
        output.append([
            _result(rows[key], score=query_scores[key], vector_score=query_scores[key])
            for key in ranked
        ])
    return output


@timed_stage("keyword_search")
def keyword_search(
    database: Database,
    query: str,
    *,
    top_k: int,
    metadata_filter: MetadataFilter | None = None,
) -> list[SearchResult]:
    rows = _rows(database, metadata_filter)
    if not rows:
        return []
    query_terms = unique_terms(tokenize(query))
    lengths = {
        (chunk_id, representation_index): length
        for chunk_id, representation_index, length in database.representation_lengths()
        if chunk_id in rows
    }
    if not lengths:
        return []
    matches = [
        match
        for match in database.matching_representation_terms(query_terms)
        if (match[0], match[1]) in lengths
    ]
    representations_by_term: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for chunk_id, representation_index, term, _frequency in matches:
        representations_by_term[term].add((chunk_id, representation_index))
    representation_count = len(lengths)
    average_length = sum(lengths.values()) / representation_count
    representation_scores: dict[tuple[int, int], float] = defaultdict(float)
    k1, b = 1.5, 0.75
    for chunk_id, representation_index, term, frequency in matches:
        key = (chunk_id, representation_index)
        length = lengths[key]
        df = len(representations_by_term[term])
        inverse_frequency = math.log(
            1 + (representation_count - df + 0.5) / (df + 0.5)
        )
        denominator = frequency + k1 * (1 - b + b * length / max(average_length, 1))
        representation_scores[key] += (
            inverse_frequency * frequency * (k1 + 1) / denominator
        )
    scores: dict[int, float] = defaultdict(float)
    for (chunk_id, _representation_index), score in representation_scores.items():
        scores[chunk_id] = max(scores[chunk_id], score)
    maximum = max(scores.values(), default=0.0)
    if maximum:
        scores = defaultdict(float, {key: value / maximum for key, value in scores.items()})
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [
        _result(rows[chunk_id], score=scores[chunk_id], keyword_score=scores[chunk_id])
        for chunk_id in ranked
    ]


def reciprocal_rank_fusion(
    vector_results: Sequence[SearchResult],
    keyword_results: Sequence[SearchResult],
    *,
    vector_weight: float,
    rank_constant: int = 60,
) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for rank, result in enumerate(vector_results, start=1):
        scores[result.chunk_id] += vector_weight / (rank_constant + rank)
    for rank, result in enumerate(keyword_results, start=1):
        scores[result.chunk_id] += (1 - vector_weight) / (rank_constant + rank)
    return scores


def hybrid_search(
    database: Database,
    query: str,
    query_embedding: list[float],
    *,
    top_k: int,
    vector_weight: float,
    metadata_filter: MetadataFilter | None = None,
) -> list[SearchResult]:
    candidate_count = max(top_k * 3, top_k)
    vectors = vector_search(
        database, query_embedding, top_k=candidate_count, metadata_filter=metadata_filter
    )
    keywords = keyword_search(
        database, query, top_k=candidate_count, metadata_filter=metadata_filter
    )
    return fuse_hybrid_results(
        vectors, keywords, top_k=top_k, vector_weight=vector_weight
    )


def fuse_hybrid_results(
    vectors: Sequence[SearchResult],
    keywords: Sequence[SearchResult],
    *,
    top_k: int,
    vector_weight: float,
) -> list[SearchResult]:
    fused = reciprocal_rank_fusion(vectors, keywords, vector_weight=vector_weight)
    by_id = {result.chunk_id: result for result in [*vectors, *keywords]}
    vector_scores = {result.chunk_id: result.vector_score for result in vectors}
    keyword_scores = {result.chunk_id: result.keyword_score for result in keywords}
    ranked = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:top_k]
    return [
        SearchResult(
            chunk_id=by_id[chunk_id].chunk_id,
            document_id=by_id[chunk_id].document_id,
            content=by_id[chunk_id].content,
            source=by_id[chunk_id].source,
            metadata=by_id[chunk_id].metadata,
            score=fused[chunk_id],
            vector_score=vector_scores.get(chunk_id, 0.0),
            keyword_score=keyword_scores.get(chunk_id, 0.0),
            position=by_id[chunk_id].position,
        )
        for chunk_id in ranked
    ]


def rerank_results(
    query: str,
    results: Sequence[SearchResult],
    reranker: Reranker,
) -> list[SearchResult]:
    if not results:
        return []
    scores = reranker.rank(query, [result.content for result in results])
    if len(scores) != len(results):
        raise ValueError("reranker returned an unexpected number of scores")
    reranked = [
        SearchResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            content=result.content,
            source=result.source,
            metadata=result.metadata,
            score=result.score,
            vector_score=result.vector_score,
            keyword_score=result.keyword_score,
            position=result.position,
            rerank_score=float(score),
        )
        for result, score in zip(results, scores, strict=True)
    ]
    return sorted(reranked, key=lambda result: (-(result.rerank_score or 0.0), -result.score))


def retrieve_chunks(database: Database, chunk_ids: Sequence[int]) -> list[SearchResult]:
    return [_result(row, score=0.0) for row in database.chunks_by_id(list(chunk_ids))]


def retrieve_chunk_spans(
    database: Database,
    results: Sequence[SearchResult],
    *,
    neighbors: int = 1,
) -> list[ChunkSpan]:
    all_results = {
        (int(row[1]), int(row[7])): _result(row, score=0.0) for row in database.chunk_rows()
    }
    spans: list[ChunkSpan] = []
    seen: set[int] = set()
    for result in results:
        chunks = []
        for position in range(result.position - neighbors, result.position + neighbors + 1):
            neighbor = all_results.get((result.document_id, position))
            if neighbor and neighbor.chunk_id not in seen:
                chunks.append(neighbor)
                seen.add(neighbor.chunk_id)
        if chunks:
            spans.append(ChunkSpan(tuple(chunks)))
    return spans
