# sextant

**Reliability primitives for any LLM API.** No framework, no SDK lock-in --
just small modules that wrap whatever provider you use.

```bash
pip install sextant
```

```python
from sextant import cove
from sextant.adapters import openai_complete
from openai import OpenAI

complete = openai_complete(OpenAI(), model="gpt-4o-mini")
r = cove(complete, query="Who invented the laser?")
print(r.final)
```

## What's in the box

| Primitive | Paper | What it does |
|---|---|---|
| [`cove`](primitives/cove.md) | [Dhuliawala 2023](https://arxiv.org/abs/2309.11495) | Generate, plan verification questions, answer each independently, revise. |
| [`self_consistency`](primitives/self_consistency.md) | [Wang 2022](https://arxiv.org/abs/2203.11171) | Sample N, vote on plurality answer. |
| [`best_of_n`](primitives/best_of_n.md) | classic | Sample N, score each, pick best. |
| [`reflexion`](primitives/reflexion.md) | [Shinn 2023](https://arxiv.org/abs/2303.11366) | Try -> critique -> retry loop with feedback. |
| [`debate`](primitives/debate.md) | [Du, Li, Mordatch 2023](https://arxiv.org/abs/2305.14325) | Multi-agent debate; agents revise after seeing others. |
| [`DriftDetector`](primitives/drift.md) | rolling centroid | Detect prompt drift per bucket via z-score. |
| [`race`](primitives/race.md) | [Dean & Barroso 2013](https://research.google/pubs/the-tail-at-scale/) | Hedged execution: race N callables, first wins. |

Every primitive has an [async sibling](async.md) and accepts an optional
[tracer](tracing.md) for observability.

## Why sextant exists

Modern LLM platforms (LangChain, LlamaIndex, LiteLLM) give you routing and
abstractions. They don't give you the inference-time reliability methods
from the research literature -- you end up reimplementing CoVe and
self-consistency by hand in every project. Sextant is the lowest-friction
implementation of those methods, designed to drop into any stack.

## What sextant does NOT do

- No router (use LiteLLM, Portkey, or your own).
- No observability backend (use Langfuse, Phoenix, Datadog -- sextant just
  emits events via a tracer protocol).
- No retrieval / RAG (use LlamaIndex, LangChain, a real vector DB).
- No agent loop framework.

These are deliberate. Sextant is a small library, not a framework.
