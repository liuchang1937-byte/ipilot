"""Stable rubric IDs and verifiable excerpts from candidate answers."""

from __future__ import annotations

import re
from typing import Any

STATUS_LABELS = {
    "mastered": "掌握",
    "partial": "部分掌握",
    "incorrect": "存在错误",
    "not_addressed": "未涉及",
    "unverified": "待核验",
}


def rubric_points(text: str) -> list[dict[str, str]]:
    parts = [
        re.sub(r"^\s*(?:[-*]|\d+[.)、．])\s*", "", part).strip()
        for part in re.split(r"[；;\n]+", text)
    ]
    parts = list(dict.fromkeys(p for p in parts if p))
    if len(parts) > 16:
        parts = parts[:15] + ["；".join(parts[15:])]
    return [{"id": f"P{i}", "point": point} for i, point in enumerate(parts, 1)]


def point_reviews(
    raw: Any,
    rubric: list[dict[str, str]],
    answers: list[str],
    *,
    unavailable_reason: str = "模型未提供可核验的逐点评价。",
) -> list[dict[str, Any]]:
    """A substantive judgement needs an exact quote from a numbered answer."""
    rows: dict[str, list[dict]] = {}
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and isinstance(item.get("point_id"), str):
            rows.setdefault(item["point_id"], []).append(item)
    output = []
    for point in rubric:
        candidates = rows.get(point["id"], [])
        row = candidates[0] if len(candidates) == 1 else {}
        status = row.get("status", "unverified")
        reason = row.get("reason")
        reason = reason.strip()[:1000] if isinstance(reason, str) else ""
        evidence = []
        quotes = row.get("evidence", [])
        valid = isinstance(quotes, list)
        for entry in quotes if valid else []:
            if not isinstance(entry, dict):
                valid = False
                break
            index, quote = entry.get("answer_index"), entry.get("quote")
            if (
                type(index) is not int
                or not 1 <= index <= len(answers)
                or not isinstance(quote, str)
                or not quote.strip()
                or len(quote) > 1500
                or quote not in answers[index - 1]
            ):
                valid = False
                break
            evidence.append({"answer_index": index, "quote": quote})
        if (
            not isinstance(status, str)
            or status not in STATUS_LABELS
            or not reason
            or not valid
            or (status in {"mastered", "partial", "incorrect"} and not evidence)
            or (status == "not_addressed" and evidence)
        ):
            status, evidence, reason = "unverified", [], unavailable_reason
        output.append(
            {
                "point_id": point["id"],
                "point": point["point"],
                "status": status,
                "label": STATUS_LABELS[status],
                "evidence": evidence,
                "reason": reason,
            }
        )
    return output
