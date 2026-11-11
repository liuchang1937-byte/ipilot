from pathlib import Path

import pytest

from techrag.cli import _parser, _run_interview
from techrag.interview import InterviewStep


def test_add_parses_relative_paths_and_metadata() -> None:
    args = _parser().parse_args(
        ["add", "docs/guide.md", "--metadata", '{"language":"zh","topic":"rag"}']
    )

    assert args.paths == [Path("docs/guide.md")]
    assert args.metadata == {"language": "zh", "topic": "rag"}


def test_metadata_arguments_require_json_objects() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["search", "query", "--filter", '["zh"]'])


def test_bge_reranker_is_a_global_option() -> None:
    args = _parser().parse_args(["--reranker", "bge", "search", "query"])

    assert args.reranker == "bge"


def test_terminal_labels_followup_and_preserves_response(monkeypatch, capsys) -> None:
    step = InterviewStep(
        session_id="demo", phase="questioning",
        interviewer_message="你提到了工具调用，真正执行工具的是谁？",
        decision="followup",
    )

    def stop(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", stop)
    _run_interview(None, step)
    output = capsys.readouterr().out
    assert "[本题追问]" in output
    assert "你提到了工具调用" in output
