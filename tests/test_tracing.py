"""Tests for sextant.tracing."""

from __future__ import annotations

import time

from sextant.tracing import (
    CallbackTracer,
    LoggingTracer,
    NoOpTracer,
    default_tracer,
    set_default_tracer,
    span,
)


def test_noop_tracer_returns_silent():
    t = NoOpTracer()
    s = t.start_span("foo")
    assert t.end_span(s, x=1) is None
    assert t.emit_event("e", k=2) is None


def test_logging_tracer_records_spans():
    t = LoggingTracer()
    s = t.start_span("op", arg=1)
    time.sleep(0.01)
    t.end_span(s, result=42)
    assert len(t.spans) == 1
    rec = t.spans[0]
    assert rec["name"] == "op"
    assert rec["input"] == {"arg": 1}
    assert rec["output"] == {"result": 42}
    assert rec["duration_ms"] >= 5.0


def test_logging_tracer_records_events():
    t = LoggingTracer()
    t.emit_event("cache_hit", key="abc")
    assert len(t.events) == 1
    assert t.events[0]["name"] == "cache_hit"
    assert t.events[0]["attrs"]["key"] == "abc"


def test_logging_tracer_sink_invoked_on_span_end():
    seen = []
    t = LoggingTracer(sink=seen.append)
    s = t.start_span("x")
    t.end_span(s, y=1)
    assert len(seen) == 1


def test_callback_tracer_on_span_end():
    seen = []
    t = CallbackTracer(on_span_end=lambda name, attrs: seen.append((name, attrs)))
    s = t.start_span("op", arg=1)
    t.end_span(s, result=99)
    assert seen[0][0] == "op"
    assert seen[0][1]["arg"] == 1
    assert seen[0][1]["result"] == 99
    assert "duration_ms" in seen[0][1]


def test_callback_tracer_on_event():
    seen = []
    t = CallbackTracer(on_event=lambda name, attrs: seen.append((name, attrs)))
    t.emit_event("started", id=42)
    assert seen == [("started", {"id": 42})]


def test_callback_tracer_no_handler_is_silent():
    """If no on_span_end provided, end_span should silently do nothing."""
    t = CallbackTracer()
    s = t.start_span("x")
    t.end_span(s)
    t.emit_event("e")
    # No exceptions, no state.


def test_span_helper_with_none_tracer_is_safe():
    """The internal `span()` context manager handles None gracefully."""
    with span(None, "op") as out:
        assert out is None


def test_span_helper_with_real_tracer_passes_output_back():
    t = LoggingTracer()
    with span(t, "op", x=1) as out:
        out["y"] = 2
    rec = t.spans[0]
    assert rec["input"]["x"] == 1
    assert rec["output"]["y"] == 2


def test_default_tracer_global():
    initial = default_tracer()
    custom = LoggingTracer()
    try:
        set_default_tracer(custom)
        assert default_tracer() is custom
    finally:
        set_default_tracer(initial)
