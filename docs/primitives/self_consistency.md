# `self_consistency`

> Wang et al. 2022, [arxiv:2203.11171](https://arxiv.org/abs/2203.11171) (Google Research)

Sample N completions at temperature > 0, return the plurality answer.
Beats greedy decoding on reasoning benchmarks by 10-20 points (GSM8K,
SVAMP, AQuA, ARC, StrategyQA).

```python
from sextant import self_consistency

r = self_consistency(
    complete,  # temperature > 0 baked into your CompleteFn
    messages=[{"role": "user", "content": "What is 13 * 17?"}],
    n=7,
    extractor="last_number",
)
print(r.answer, r.confidence, r.vote_counts)
```

## Four extractors

| Extractor | When to use |
|---|---|
| `last_line` | Default. "The answer is X." patterns. |
| `last_number` | Arithmetic and counting. |
| `regex` | Custom regex; group 1 is the answer. |
| `similarity` | Open-ended generation. Embeds all samples, returns the one nearest the semantic centroid. Requires `embed_fn=`. |

The `similarity` extractor is sextant-specific. It lets you do
self-consistency on tasks where there's no discrete answer to vote on
(summaries, code, creative writing).

## Cost

N model calls. Wang recommends N=20-40 for hard reasoning benchmarks;
N=5 is enough for most tasks.

## Async parity

```python
from sextant.asyncio import aself_consistency

# N samples run concurrently via asyncio.gather -- ~1x latency instead of Nx.
r = await aself_consistency(async_complete, messages=[...], n=10)
```
