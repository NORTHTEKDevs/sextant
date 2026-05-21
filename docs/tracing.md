# Tracing

Every primitive accepts an optional `tracer=` kwarg. By default it's a
no-op; pass any object implementing the `Tracer` protocol and lemmas
will emit structured spans + events you can pipe to your observability
backend.

## The protocol

```python
class Tracer(Protocol):
    def start_span(self, name: str, **attrs) -> Any: ...
    def end_span(self, span: Any, **attrs) -> None: ...
    def emit_event(self, name: str, **attrs) -> None: ...
```

Three implementations ship with lemmas:

| Class | Use case |
|---|---|
| `NoOpTracer` | Default. Records nothing. Zero overhead. |
| `LoggingTracer` | Records spans + events in memory. For tests + debugging. |
| `CallbackTracer(on_span_end, on_event)` | Forward to your own logger / Langfuse / Datadog. |

## Wire to Langfuse

```python
from langfuse import Langfuse
from lemmas import CallbackTracer, cove

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
from lemmas import CallbackTracer

otel_tracer = trace.get_tracer("lemmas")

def on_span_end(name, attrs):
    with otel_tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            s.set_attribute(k, str(v))

cove(complete, query="...", tracer=CallbackTracer(on_span_end=on_span_end))
```

## Default tracer

To avoid threading `tracer=` through every call, set a global default:

```python
from lemmas import LoggingTracer, set_default_tracer

set_default_tracer(LoggingTracer(sink=print))

# Now every primitive call records to the default unless overridden.
```

## What spans are emitted

| Primitive | Spans |
|---|---|
| `cove` | `lemmas.cove`, `lemmas.cove.baseline`, `lemmas.cove.plan`, `lemmas.cove.answer` (one per question), `lemmas.cove.final` |
| `debate` | `lemmas.debate`, `lemmas.debate.draft` (per agent), `lemmas.debate.revise` (per agent per round), `lemmas.debate.judge` |
| `reflexion` | `lemmas.reflexion`, `lemmas.reflexion.attempt` (per iteration), `lemmas.reflexion.critic` (per iteration) |

Other primitives emit fewer spans (or none); the cost is in the wrapped
LLM calls, which your provider client surfaces in its own telemetry.
