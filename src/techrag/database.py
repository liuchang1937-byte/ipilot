from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.errors import InternalError

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RepresentationRecord:
    content: str
    embedding: list[float]
    frequencies: dict[str, int]
    kind: str = "child"


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    content: str
    representations: list[RepresentationRecord]


def _new_id() -> int:
    return uuid.uuid4().int & ((1 << 63) - 1)


def _source_id(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


class Database:
    """Chroma storage; document manifests publish fully written generations."""

    def __init__(self, path: Path) -> None:
        path = Path(path)
        if path.is_file() or path.suffix.lower() == ".duckdb":
            raise ValueError("--db expects a Chroma directory; migrate legacy files first")
        self.client = chromadb.PersistentClient(
            path=str(path.resolve()), settings=Settings(anonymized_telemetry=False)
        )
        self.documents = self._collection("techrag_documents")
        self.parents = self._collection("techrag_parents")
        self.children = self.client.get_or_create_collection(
            "techrag_children", embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        self.records = self._collection("techrag_records")

    def _collection(self, name: str):
        return self.client.get_or_create_collection(name, embedding_function=None)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def existing_document(self, source: str) -> tuple[int, str] | None:
        result = self.documents.get(ids=[_source_id(source)])
        if not result["ids"]:
            return None
        metadata = result["metadatas"][0]
        return int(metadata["document_id"]), str(metadata["checksum"])

    def update_question_routing(self, source: str, routing: dict[str, object]) -> None:
        """Refresh non-embedded routing fields without regenerating vectors."""
        manifest = self.documents.get(ids=[_source_id(source)])
        if not manifest["ids"]:
            return
        meta = manifest["metadatas"][0]
        payload = json.loads(meta["payload"])
        if all(payload.get(key) == value for key, value in routing.items()):
            return
        payload.update(routing)
        encoded = json.dumps(payload, ensure_ascii=False)
        parents = self.parents.get(where={"document_id": int(meta["document_id"])})
        if parents["ids"]:
            self.parents.update(ids=parents["ids"], metadatas=[
                {**item, "payload": encoded} for item in parents["metadatas"]
            ])
        self.documents.update(ids=manifest["ids"], metadatas=[{**meta, "payload": encoded}])

    def document_rows(self) -> list[tuple[str, dict[str, object]]]:
        return [
            (str(metadata["source"]), json.loads(metadata["payload"]))
            for metadata in self.documents.get()["metadatas"]
        ]

    def snapshot_document(self, source: str) -> dict[str, Any] | None:
        """Capture an active generation for local rollback without re-embedding."""
        existing = self.existing_document(source)
        if existing is None:
            return None
        parents = self.parents.get(where={"document_id": existing[0]})
        children = self.children.get(
            where={"document_id": existing[0]}, include=["documents", "metadatas", "embeddings"],
        )
        representations: dict[str, list[tuple[int, RepresentationRecord]]] = {}
        for content, meta, vector in zip(children["documents"], children["metadatas"],
                                         children["embeddings"], strict=True):
            representations.setdefault(str(meta["parent_id"]), []).append((
                int(meta["representation_index"]), RepresentationRecord(
                    content=content, embedding=[float(v) for v in vector],
                    frequencies=json.loads(meta["frequencies"]), kind=meta["kind"],
                ),
            ))
        chunks = []
        for key, content, _meta in sorted(zip(parents["ids"], parents["documents"],
                                             parents["metadatas"], strict=True),
                                         key=lambda item: int(item[2]["position"])):
            chunks.append(ChunkRecord(content=content, representations=[
                rep for _, rep in sorted(representations[key], key=lambda item: item[0])
            ]))
        return {"source": source, "checksum": existing[1],
                "metadata": json.loads(parents["metadatas"][0]["payload"]), "chunks": chunks}

    def replace_document(
        self, *, source: str, checksum: str, metadata: dict[str, object],
        chunks: list[ChunkRecord],
    ) -> int:
        if not chunks or any(not chunk.representations for chunk in chunks):
            raise ValueError("each document requires parents with retrieval representations")
        dimensions = {len(rep.embedding) for chunk in chunks for rep in chunk.representations}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("all embeddings must have the same nonzero dimension")
        if any(not math.isfinite(value)
               for chunk in chunks for rep in chunk.representations for value in rep.embedding):
            raise ValueError("embeddings must contain only finite numbers")
        dimension = str(dimensions.pop())
        stored = self.get_record("index-setting", "embedding_dimensions")
        if stored and stored["value"] != dimension:
            raise ValueError("embedding dimension changed; use the same embedding model")
        existing = self.existing_document(source)
        document_id = _new_id()
        parent_ids, bodies, parent_metadata = [], [], []
        child_ids, passages, child_metadata, vectors = [], [], [], []
        payload = json.dumps(metadata, ensure_ascii=False)
        for position, chunk in enumerate(chunks):
            parent_id = _new_id()
            parent_ids.append(str(parent_id))
            bodies.append(chunk.content)
            parent_metadata.append({
                "document_id": document_id, "position": position, "source": source,
                "payload": payload,
                "token_count": sum(chunk.representations[0].frequencies.values()),
            })
            for index, rep in enumerate(chunk.representations):
                child_ids.append(f"{parent_id}:{index}")
                passages.append(rep.content)
                vectors.append(rep.embedding)
                child_metadata.append({
                    "document_id": document_id, "parent_id": parent_id,
                    "representation_index": index, "kind": rep.kind,
                    "frequencies": json.dumps(rep.frequencies, ensure_ascii=False),
                    "token_count": sum(rep.frequencies.values()),
                    "question_id": str(metadata.get("question_id", "")),
                    "domain": str(metadata.get("domain", "")),
                    "collection": str(metadata.get("collection", "")),
                })
        try:
            self._write_batches(self.parents, parent_ids, bodies, parent_metadata,
                                [[1.0]] * len(parent_ids))
            self._write_batches(self.children, child_ids, passages, child_metadata, vectors)
            self.ensure_index_setting("embedding_dimensions", dimension)
            # This manifest is the commit point; previous data stays readable until here.
            self.documents.upsert(
                ids=[_source_id(source)], documents=[source], embeddings=[[1.0]],
                metadatas=[{"document_id": document_id, "source": source,
                            "checksum": checksum, "payload": payload}],
            )
        except Exception:
            self._delete_generation(document_id)
            raise
        if existing:
            self._delete_generation(existing[0])
        return document_id

    def _write_batches(self, collection, ids, documents, metadatas, embeddings) -> None:
        size = min(self.client.get_max_batch_size(), 1000)
        for start in range(0, len(ids), size):
            collection.add(
                ids=ids[start:start + size], documents=documents[start:start + size],
                metadatas=metadatas[start:start + size], embeddings=embeddings[start:start + size],
            )

    def _delete_generation(self, document_id: int) -> None:
        self.children.delete(where={"document_id": document_id})
        self.parents.delete(where={"document_id": document_id})

    def _active_document_ids(self) -> set[int]:
        return {int(metadata["document_id"]) for metadata in
                self.documents.get(include=["metadatas"])["metadatas"]}

    def chunk_rows(self) -> list[tuple[object, ...]]:
        result = self.parents.get()
        active = self._active_document_ids()
        rows = [
            (int(parent_id), int(meta["document_id"]), content, [], int(meta["token_count"]),
             meta["source"], meta["payload"], int(meta["position"]))
            for parent_id, content, meta in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            ) if int(meta["document_id"]) in active
        ]
        return sorted(rows, key=lambda row: (int(row[1]), int(row[7])))

    def chunks_by_id(self, chunk_ids: list[int]) -> list[tuple[object, ...]]:
        by_id = {int(row[0]): row for row in self.chunk_rows()}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def representation_rows(self, *, source: str | None = None) -> list[dict[str, Any]]:
        active = self._active_document_ids()
        if source is not None:
            existing = self.existing_document(source)
            active = {existing[0]} if existing else set()
        if not active:
            return []
        result = self.children.get(
            where={"document_id": {"$in": list(active)}}, include=["documents", "metadatas"],
        )
        return [
            {"id": record_id, "content": content, **meta}
            for record_id, content, meta in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        ]

    def representation_lengths(self) -> list[tuple[int, int, int]]:
        return [(int(row["parent_id"]), int(row["representation_index"]), int(row["token_count"]))
                for row in self.representation_rows()]

    def matching_representation_terms(self, terms: list[str]) -> list[tuple[int, int, str, int]]:
        if not terms:
            return []
        matches = []
        for row in self.representation_rows():
            frequencies = json.loads(row["frequencies"])
            for term in terms:
                if term in frequencies:
                    matches.append((int(row["parent_id"]), int(row["representation_index"]),
                                    term, int(frequencies[term])))
        return matches

    def query_children(
        self, query_embeddings: list[list[float]], *, parent_ids: list[int], n_results: int,
    ) -> dict[str, Any]:
        dimension = self.get_record("index-setting", "embedding_dimensions")
        if dimension and any(len(vector) != int(dimension["value"]) for vector in query_embeddings):
            raise ValueError("embedding dimension changed; use the same embedding model")
        for attempt in range(3):
            try:
                return self.children.query(
                    query_embeddings=query_embeddings, n_results=n_results,
                    where={"parent_id": {"$in": parent_ids}},
                    include=["metadatas", "distances"],
                )
            except InternalError as error:
                if ("Error creating hnsw segment reader: Nothing found on disk" not in str(error)
                        or attempt == 2):
                    raise
                # Retry the same read only. Never rebuild, delete or relax the filter.
                _LOG.warning("Chroma HNSW reader unavailable; retrying read (%s/2)", attempt + 1)
                time.sleep(.05 * (attempt + 1))
        raise AssertionError("unreachable query retry state")

    def stats(self) -> tuple[int, int]:
        return self.documents.count(), len(self.chunk_rows())

    def delete_document(self, source: str) -> bool:
        existing = self.existing_document(source)
        if not existing:
            return False
        self.documents.delete(ids=[_source_id(source)])
        self._delete_generation(existing[0])
        return True

    def metadata_values(self) -> dict[str, list[object]]:
        values: dict[str, list[object]] = {}
        for _, metadata in self.document_rows():
            for key, value in metadata.items():
                bucket = values.setdefault(key, [])
                for candidate in value if isinstance(value, list) else [value]:
                    if candidate not in bucket:
                        bucket.append(candidate)
        return values

    def ensure_index_setting(self, key: str, value: str) -> None:
        record = self.get_record("index-setting", key)
        if record and record["value"] != value:
            raise ValueError(
                f"index was created with {key}={record['value']}; received incompatible {value!r}"
            )
        if not record:
            self.put_record("index-setting", key, {"value": value})

    def put_record(
        self, kind: str, key: str, payload: dict[str, Any], **filters: str | int,
    ) -> None:
        self.records.upsert(
            ids=[f"{kind}:{key}"], documents=[json.dumps(payload, ensure_ascii=False)],
            embeddings=[[1.0]], metadatas=[{"record_type": kind, **filters}],
        )

    def get_record(self, kind: str, key: str) -> dict[str, Any] | None:
        result = self.records.get(ids=[f"{kind}:{key}"])
        return json.loads(result["documents"][0]) if result["ids"] else None

    def list_records(self, kind: str, **filters: str | int) -> list[dict[str, Any]]:
        conditions = [{"record_type": kind}, *({key: value} for key, value in filters.items())]
        where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        return [json.loads(document) for document in self.records.get(where=where)["documents"]]

    def delete_records(self, kind: str, **filters: str | int) -> None:
        conditions = [{"record_type": kind}, *({key: value} for key, value in filters.items())]
        where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        self.records.delete(where=where)

    def insert_evaluation(
        self, question: str, expected_sources: list[str], reference_answer: str | None
    ) -> int:
        record_id = _new_id()
        self.put_record("evaluation", str(record_id), {
            "id": record_id, "question": question, "expected_sources": expected_sources,
            "reference_answer": reference_answer,
        })
        return record_id

    def evaluation_rows(self) -> list[tuple[str, list[str], str | None]]:
        return [(row["question"], row["expected_sources"], row["reference_answer"])
                for row in self.list_records("evaluation")]
