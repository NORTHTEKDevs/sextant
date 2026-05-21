"""Provider-agnostic tracing for sextant primitives.

Every primitive accepts an optional `tracer=` keyword that defaults to a
no-op. When provided, the primitive emits structured events you can pipe
into Langfuse, Phoenix, OpenTelemetry, Honeycomb, stdout -- anywhere.

Design notes
------------

- Sextant doesn't depend on any tracing library. The `Tracer` protocol is
  six methods: start_span, end_span, emit_event, plus three context-manager
  conveniences. You implement them however your stack wants.
- Spans are nested by call site, not by thread-local context. If you need
  thread-local context (for OTel), wrap your underlying tracer and use
  its own context-manager.
- Events carry a `primitive` name ("cove", "self_consistency", ...) plus
  a `step` name ("baseline", "plan", "sample") plus arbitrary kwargs.

Bundled tracers:
  - NoOpTracer    -- silent. The default.
  - LoggingTracer -- prints/yields events. Useful for debugging + as a
                    reference implementation.
  - CallbackTracer(on_span_end=fn) -- minimal adapter; fn(name, attrs) per span.

To wire to Langfuse:

    from langfuse.client import Langfuse
    lf = Langfuse(...)

    class LangfuseTracer:
        def start_span(self, name, **attrs):
            self._span = lf.span(name=name, input=attrs)
            return self._span
        def end_span(self, span, **attrs):
            span.end(output=attrs)
        def emit_event(self, name, **attrs):
            lf.event(name=name, input=attrs)

    cove(complete, query="...", tracer=LangfuseTracer())
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SpanInfo:
    """Returned from start_span. Just a handle; meaning depends on the tracer."""
    name: str
    started_at: float
    attrs: dict[str, Any] = field(default_factory=dict)


class Tracer(Protocol):
    """Minimal tracer interface. Implement this for your favorite backend."""

    def start_span(self, name: str, **attrs: Any) -> Any: ...
    def end_span(self, span: Any, **attrs: Any) -> None: ...
    def emit_event(self, name: str, **attrs: Any) -> None: ...


class NoOpTracer:
    """Default. Does nothing. Sextant primitives call into this in the hot path."""

    def start_span(self, name: str, **attrs: Any) -> SpanInfo:
        return SpanInfo(name=name, started_at=time.monotonic(), attrs={})

    def end_span(self, span: Any, **attrs: Any) -> None:
        return None

    def emit_event(self, name: str, **attrs: Any) -> None:
        return None


class LoggingTracer:
    """Records every span and event in memory. Useful for tests + debugging."""

    def __init__(self, sink: Callable[[dict], None] | None = None) -> None:
        self.spans: list[dict] = []
        self.events: list[dict] = []
        self.sink = sink

    def start_span(self, name: str, **attrs: Any) -> SpanInfo:
        return SpanInfo(name=name, started_at=time.monotonic(), attrs=dict(attrs))

    def end_span(self, span: SpanInfo, **attrs: Any) -> None:
        record = {
            "name": span.name,
            "started_at": span.started_at,
            "duration_ms": (time.monotonic() - span.started_at) * 1000,
            "input": span.attrs,
            "output": attrs,
        }
        self.spans.append(record)
        if self.sink is not None:
            self.sink(record)

    def emit_event(self, name: str, **attrs: Any) -> None:
        record = {"name": name, "attrs": attrs, "ts": time.monotonic()}
        self.events.append(record)
        if self.sink is not None:
            self.sink(record)


class CallbackTracer:
    """Thin adapter: pass a single `on_span_end` callback and we'll call it
    with (name, attrs_dict) for every completed span. Most lightweight way
    to plug an existing logger / Langfuse / Phoenix sink in."""

    def __init__(
        self,
        on_span_end: Callable[[str, dict], None] | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.on_span_end = on_span_end
        self.on_event = on_event

    def start_span(self, name: str, **attrs: Any) -> SpanInfo:
        return SpanInfo(name=name, started_at=time.monotonic(), attrs=dict(attrs))

    def end_span(self, span: SpanInfo, **attrs: Any) -> None:
        if self.on_span_end is None:
            return
        merged = {
            "started_at": span.started_at,
            "duration_ms": (time.monotonic() - span.started_at) * 1000,
            **span.attrs, **attrs,
        }
        self.on_span_end(span.name, merged)

    def emit_event(self, name: str, **attrs: Any) -> None:
        if self.on_event is not None:
            self.on_event(name, attrs)


# Helpers ------------------------------------------------------------------

@contextmanager
def span(tracer: Tracer | None, name: str, **attrs: Any):
    """Context manager wrapper used internally by primitives. Skips if tracer is None."""
    if tracer is None:
        yield None
        return
    s = tracer.start_span(name, **attrs)
    output: dict[str, Any] = {}
    try:
        yield output
    finally:
        tracer.end_span(s, **output)


def default_tracer() -> Tracer:
    """Module-level default; can be set with `set_default_tracer()`."""
    global _DEFAULT
    return _DEFAULT


def set_default_tracer(tracer: Tracer) -> None:
    """All primitives that don't get an explicit tracer use this one."""
    global _DEFAULT
    _DEFAULT = tracer


_DEFAULT: Tracer = NoOpTracer()
