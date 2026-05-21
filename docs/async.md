# Async API

Every primitive has an async sibling under `sextant.asyncio`. The
N-sample primitives (`aself_consistency`, `abest_of_n`) parallelize their
LLM calls via `asyncio.gather`, turning N x latency into ~1 x latency.

```python
from sextant.asyncio import acove, aself_consistency, abest_of_n, areflexion, arace
```

## Async adapters

Sextant doesn't (yet) ship async adapters for every provider. Roll your own:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def async_complete(messages: list[dict]) -> str:
    resp = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages,
        temperature=0.7, max_tokens=512,
    )
    return resp.choices[0].message.content
```

## `acove`

```python
r = await acove(async_complete, query="...", n_questions=5)
```

Steps 1 (baseline), 2 (plan), 4 (revise) are sequential because they
depend on each other; step 3 (N verification answers) runs concurrently.

## `aself_consistency` / `abest_of_n`

N samples drawn concurrently:

```python
r = await aself_consistency(async_complete, messages=[...], n=10)
r = await abest_of_n(async_complete, messages=[...], scorer=my_scorer, n=5)
```

`abest_of_n` accepts both sync and async scorers. If async, it's awaited.

## `arace`

```python
from sextant.asyncio import arace

r = await arace([
    ("primary",   lambda: primary_provider(messages)),
    ("backup",    lambda: backup_provider(messages)),
])
```

## `areflexion`

```python
r = await areflexion(async_complete, query="...", critic=my_critic, max_iterations=4)
```

Critic may be sync or async; sync critics are called synchronously inside
the loop, async ones are awaited.
