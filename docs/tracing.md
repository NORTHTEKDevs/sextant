# Tracing

Every primitive accepts an optional `tracer=` kwarg. By default it's a
no-op; pass any object implementing the `Tracer` protocol and sextant
will emit structured spans + events you can pipe to your observability
backend.

## The protocol

```python
class Tracer(Protocol):
    def start_span(self, name: str, **attrs) -> Any: ...
    def end_span(self, span: Any, **attrs) -> None: ...
    def emit_event(self, name: str, **attrs) -> None: ...
```

Three implementations ship with sextant:

| Class | Use case |
|---|---|
| `NoOpTracer` | Default. Records nothing. Zero overhead. |
| `LoggingTracer` | Records spans + events in memory. For tests + debugging. |
| `CallbackTracer(on_span_end, on_event)` | Forward to your own logger / Langfuse / Datadog. |

## Wire to Langfuse

```python
from langfuse import Langfuse
from sextant import CallbackTracer, cove

lf = Langfuse(...)

tracer = CallbackTracer(
    on_span_end=lambda name, attrs: lf.span(
        name=name,
        input=attrs.get("query"),
        metadata=attrs,
    ).end(),
)

cove(complete, query="...", tracer=tracer)
```

## Wire to OpenTelemetry

```python
from opentelemetry import trace
from sextant import CallbackTracer

otel_tracer = trace.get_tracer("sextant")

def on_span_end(name, attrs):
    with otel_tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            s.set_attribute(k, str(v))

cove(complete, query="...", tracer=CallbackTracer(on_span_end=on_span_end))
```

## Default tracer

To avoid threading `tracer=` through every call, set a global default:

```python
from sextant import LoggingTracer, set_default_tracer

set_default_tracer(LoggingTracer(sink=print))

# Now every primitive call records to the default unless overridden.
```

## What spans are emitted

| Primitive | Spans |
|---|---|
| `cove` | `sextant.cove`, `sextant.cove.baseline`, `sextant.cove.plan`, `sextant.cove.answer` (one per question), `sextant.cove.final` |
| `debate` | `sextant.debate`, `sextant.debate.draft` (per agent), `sextant.debate.revise` (per agent per round), `sextant.debate.judge` |
| `reflexion` | `sextant.reflexion`, `sextant.reflexion.attempt` (per iteration), `sextant.reflexion.critic` (per iteration) |

Other primitives emit fewer spans (or none); the cost is in the wrapped
LLM calls, which your provider client surfaces in its own telemetry.
