"""Opt-in, request-local measurements; nested stages are inclusive."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_TIMINGS: ContextVar[dict[str, float] | None] = ContextVar("techrag_timings", default=None)


@contextmanager
def collect_timings() -> Iterator[dict[str, float]]:
    timings: dict[str, float] = {}
    token = _TIMINGS.set(timings)
    try:
        yield timings
    finally:
        _TIMINGS.reset(token)


def timed_stage(name: str):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            timings = _TIMINGS.get()
            if timings is None:
                return function(*args, **kwargs)
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
        return measured
    return decorate
