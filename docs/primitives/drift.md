# `DriftDetector`

Per-bucket rolling embedding centroid + exponentially-weighted variance.
Flags traffic-shape changes (abuse, eval-set staleness, prompt drift) via
z-score, with an absolute-distance fallback for the degenerate case.

```python
from sextant import DriftDetector
from sextant.adapters import openai_embed
from openai import OpenAI

embed = openai_embed(OpenAI(), model="text-embedding-3-small")
d = DriftDetector(embed_fn=embed, z_threshold=3.0, warmup_n=20)

for prompt in incoming:
    s = d.observe(bucket="feature-search", text=prompt)
    if s.is_drift:
        slack.post(f"drift on search: z={s.z_score:.2f}")
```

## How it works

1. Unit-normalize the new embedding.
2. distance = 1 - cosine(centroid, new)
3. z = distance / sqrt(rolling_variance)
4. Update centroid (EMA) and variance (EMA).
5. Past warmup, flag drift if |z| > threshold OR
   (std ~ 0 AND distance > distance_threshold). The second clause catches
   the case where warmup observations are all identical (e.g. a load
   tester replaying the same prompt) -- variance stays at zero, so the
   z-score is blinded; absolute distance is the safety net.

## Persistence

State is in-memory by default. Pass `persist_fn` + `load_fn` to share
across processes:

```python
DriftDetector(
    embed_fn=embed,
    persist_fn=lambda bucket, state: redis.set(f"drift:{bucket}", json.dumps(state)),
    load_fn=lambda bucket: json.loads(redis.get(f"drift:{bucket}") or "null"),
)
```

## Use cases

- Detect a customer's traffic shape has shifted (abuse, new use case).
- Detect a deployed prompt template is being used "wrong" vs. what you tested on.
- Detect your eval set has gone stale relative to live traffic.
