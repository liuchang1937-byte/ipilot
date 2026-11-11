from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class TechRAGConfig:
    """Runtime configuration for chunking, retrieval, generation, and Chroma."""

    db_path: Path = Path("data/chroma")
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 5
    vector_weight: float = 0.65
    search_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    semantic_chunking: bool = True
    semantic_threshold: float = 0.45
    markdown_chunking: bool = True
    hierarchical_retrieval: bool = True
    child_chunk_size: int = 180
    child_chunk_overlap: int = 30
    rerank_oversample: int = 4
    chunk_neighbors: int = 1
    adaptive_retrieval: bool = True
    self_query: bool = False
    enable_mqe: bool = False
    enable_hyde: bool = False
    candidate_pool_multiplier: int = 4
    max_context_chars: int = 12_000
    reranker_dir: Path = Path("models/mmarco-cross-encoder")
    knowledge_min_rerank_score: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "db_path", Path(self.db_path))
        object.__setattr__(self, "reranker_dir", Path(self.reranker_dir))
        if not 0 <= self.knowledge_min_rerank_score <= 1:
            raise ValueError("knowledge_min_rerank_score must be between 0 and 1")
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 <= self.vector_weight <= 1:
            raise ValueError("vector_weight must be between 0 and 1")
        if not -1 <= self.semantic_threshold <= 1:
            raise ValueError("semantic_threshold must be between -1 and 1")
        if self.child_chunk_size < 20:
            raise ValueError("child_chunk_size must be at least 20")
        if not 0 <= self.child_chunk_overlap < self.child_chunk_size:
            raise ValueError(
                "child_chunk_overlap must be >= 0 and smaller than child_chunk_size"
            )
        if self.rerank_oversample < 1:
            raise ValueError("rerank_oversample must be positive")
        if self.candidate_pool_multiplier < 1:
            raise ValueError("candidate_pool_multiplier must be positive")
        if self.chunk_neighbors < 0:
            raise ValueError("chunk_neighbors cannot be negative")
        if self.max_context_chars < 1000:
            raise ValueError("max_context_chars must be at least 1000")
