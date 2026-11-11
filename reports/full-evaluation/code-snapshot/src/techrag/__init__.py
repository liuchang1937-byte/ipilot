"""Public API for TechRAG."""

from .config import TechRAGConfig
from .documents import document_to_markdown, pdf_to_markdown
from .evaluation import answer_evals, evaluate, generate_evals, insert_evals, load_evals
from .interview import InterviewEngine, InterviewSession, InterviewStep
from .interview_questions import (
    InterviewQuestion,
    index_question_directory,
    load_question_directory,
    load_question_file,
)
from .models import (
    Answer,
    ChunkSpan,
    Document,
    EvaluationCase,
    EvaluationReport,
    SearchResult,
)
from .providers import BGEReranker, Reranker
from .rag import TechRAG, add_context
from .report_pdf import export_report_pdf

__all__ = [
    "Answer",
    "add_context",
    "answer_evals",
    "BGEReranker",
    "ChunkSpan",
    "Document",
    "document_to_markdown",
    "EvaluationCase",
    "EvaluationReport",
    "evaluate",
    "export_report_pdf",
    "generate_evals",
    "insert_evals",
    "InterviewEngine",
    "InterviewQuestion",
    "InterviewSession",
    "InterviewStep",
    "index_question_directory",
    "load_question_directory",
    "load_question_file",
    "load_evals",
    "pdf_to_markdown",
    "Reranker",
    "SearchResult",
    "TechRAG",
    "TechRAGConfig",
]
