# `race` -- Hedged Execution

> Dean & Barroso, ["The Tail at Scale" (2013)](https://research.google/pubs/the-tail-at-scale/)

Run N callables in parallel; return whichever succeeds first; cancel the
losers. Generic -- not LLM-specific.

```python
from lemmas import race

result = race([
    ("openai",    lambda: openai_complete(msgs)),
    ("anthropic", lambda: anthropic_complete(msgs)),
], timeout_secs=10.0)

print(result.winner, result.value, f"{result.latency_ms:.0f}ms")
```

## When to use it

Latency matters more than cost.

- voice loops where user-visible p99 is unacceptable
- first-token-fast chat racing Haiku against gpt-4o-mini
- racing a self-hosted model against a managed one, with auto-fallover
- racing two embedding providers, web fetches, database shards (any callables)

## Cost model

If all callables eventually succeed: N x compute.
In practice losers get cancelled mid-flight (HTTP connections closed,
provider calls aborted) so you pay closer to **1.2-1.5x** for meaningful
tail-latency wins.

## Failures don't kill the race

```python
def bad():
    raise RuntimeError("backend X is down")

def good():
    return "ok"

r = race([("bad", bad), ("good", good)])  # winner='good', losers=[...]
```

Set `fail_fast=True` if you want the first exception to propagate.

## Async parity

```python
from lemmas.asyncio import arace

r = await arace([
    ("a", async_call_a),
    ("b", async_call_b),
])
```
