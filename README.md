# sextant

[![CI](https://github.com/NORTHTEKDevs/sextant/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/NORTHTEKDevs/sextant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/NORTHTEKDevs/sextant)

**Five reliability primitives for any LLM API.** No new framework, no SDK
lock-in -- just five small modules that wrap whatever provider you already
use (OpenAI, Anthropic, Gemini, Groq, vLLM, local llama.cpp, anything).

```bash
pip install sextant            # once published to PyPI
# or until then:
pip install git+https://github.com/NORTHTEKDevs/sextant.git
```

```python
from openai import OpenAI
from sextant import cove
from sextant.adapters import openai_complete

complete = openai_complete(OpenAI(), model="gpt-4o-mini")
result = cove(complete, query="Who invented the laser?")
print(result.final)        # revised, fact-checked answer
print(result.baseline)     # original first draft
print(result.questions)    # what the model asked itself
```

That's the whole API surface for one primitive. There are four of them.

---

## What's in the box

| Module | Paper | What it does |
|---|---|---|
| `cove` | [Dhuliawala et al. 2023 (Meta)](https://arxiv.org/abs/2309.11495) | Chain-of-Verification. Generate, plan verification questions, answer each independently, revise. Reduces hallucination on long-form factual answers. |
| `self_consistency` | [Wang et al. 2022](https://arxiv.org/abs/2203.11171) | Sample N completions at temperature > 0, return the plurality answer. Beats greedy decoding on reasoning benchmarks by 10-20 points. |
| `best_of_n` | classic test-time compute pattern | Sample N, score each with a scorer fn (LLM-as-judge, length, keywords, or your own reward model), return the highest-scoring. The natural companion to self-consistency for open-ended tasks. |
| `DriftDetector` | rolling embedding centroid + Welford's variance | Per-bucket drift detection. Flags traffic-shape changes (abuse, eval-set staleness, prompt drift) via z-score with absolute-distance fallback. |
| `race` | [Dean & Barroso "The Tail at Scale" 2013](https://research.google/pubs/the-tail-at-scale/) | Hedged execution. Race N callables in parallel, return whichever finishes first. Generic -- not LLM-specific. |

**Async parity.** Every primitive has an async sibling under `sextant.asyncio`:
`acove`, `aself_consistency`, `abest_of_n`, `arace`. The N-sample primitives
(`aself_consistency`, `abest_of_n`) parallelize their LLM calls with
`asyncio.gather` -- so what would have been 5x sequential latency becomes
~1x concurrent.

All four are **backend-agnostic**: each takes a callable, not a client.
You can use them with any provider, in any combination, in any framework.

---

## Why this exists

Modern LLM platforms (LangChain, LlamaIndex, LiteLLM, etc.) give you
*routing* and *abstractions*. They don't give you the inference-time
reliability methods from the research literature. You end up reimplementing
CoVe and self-consistency by hand in every project.

These four primitives are the ones I've reached for repeatedly. Together
they cover:

- **factuality at inference time** (CoVe)
- **reasoning robustness** (self-consistency)
- **observability of behavior change** (drift)
- **tail-latency control** (hedged execution)

Each one is ~150 lines. The whole library is < 1 kLoC. No magic, no
dependencies beyond numpy.

---

## 1. CoVe -- Chain-of-Verification

```python
from sextant import cove
from sextant.adapters import anthropic_complete
from anthropic import Anthropic

complete = anthropic_complete(Anthropic(), model="claude-haiku-4-5-20251001")

r = cove(complete,
         query="List the five highest mountains in North America.",
         n_questions=5)

print("BASELINE (may contain hallucinations):")
print(r.baseline)
print("\nVERIFICATION:")
for q, a in zip(r.questions, r.answers):
    print(f"  Q: {q}\n  A: {a}")
print(f"\nREVISED ANSWER (had {r.revisions} revision pass):")
print(r.final)
```

**How it works:**

1. **Baseline.** Model produces a first answer.
2. **Plan.** Model generates N verification questions about the kinds of
   claims a good answer would contain. It does this *without seeing the
   baseline*, so questions stay unbiased.
3. **Execute.** Each question is answered independently. Bad claims can't
   verify each other.
4. **Revise.** Model rewrites the baseline using the Q/A pairs, correcting
   or removing anything contradicted by its own verifications.

**Cost:** N+2 model calls.
**Wins on:** TriviaQA, WikiData lists, biographies, multi-fact questions.
**Doesn't help on:** math, code, single-fact lookups.

---

## 2. Self-consistency sampling

```python
from sextant import self_consistency
from sextant.adapters import openai_complete
from openai import OpenAI

# IMPORTANT: bake temperature > 0 into your complete fn.
complete = openai_complete(OpenAI(), model="gpt-4o-mini", temperature=0.7)

prompt = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
    "every morning and bakes muffins with four. She sells the remainder "
    "at the farmers' market for $2 per fresh duck egg. How much in "
    "dollars does she make every day at the farmers' market?\n"
    "Think step by step, then end with 'Answer: <number>'."
)

r = self_consistency(complete,
                      messages=[{"role": "user", "content": prompt}],
                      n=7,
                      extractor="last_number")

print(f"plurality answer: {r.answer}  (confidence={r.confidence:.2f})")
print(f"vote counts: {r.vote_counts}")
```

**Four extractors:**

| Extractor | When to use |
|---|---|
| `last_line` | Default. Works for "The answer is X." patterns. |
| `last_number` | Math word problems. |
| `regex` | Custom -- you supply the pattern; group 1 is the answer. |
| `similarity` | Open-ended generation. Embeds all N samples, returns the one closest to the centroid. Requires an `embed_fn`. |

**Cost:** N model calls. The original paper recommends N=20-40 for hard
reasoning benchmarks; N=5 is enough for most tasks.

The `similarity` extractor is sextant-specific -- it lets you do
self-consistency on tasks where there's no discrete answer to vote on
(summaries, code generation, creative writing). It picks the sample
nearest the semantic centroid, which empirically picks the "median"
generation -- the one that best represents what the model would say on
average.

---

## 3. DriftDetector

```python
from sextant import DriftDetector
from sextant.adapters import openai_embed
from openai import OpenAI

embed = openai_embed(OpenAI(), model="text-embedding-3-small")
detector = DriftDetector(embed_fn=embed, z_threshold=3.0, warmup_n=20)

# On every incoming prompt to a deployed feature:
for prompt in incoming_prompts:
    s = detector.observe(bucket="feature-search-v1", text=prompt)
    if s.is_drift:
        slack.post(f"prompt drift on search-v1: z={s.z_score:.2f} "
                   f"after {s.n} observations")
```

**What it does:**

- Maintains a unit-norm centroid of recent prompt embeddings per bucket.
- On each new prompt, computes cosine distance to the centroid.
- Updates centroid and variance via exponential moving average.
- Returns a z-score; flags drift when |z| > threshold *and* the bucket has
  passed warmup.

**State is in-memory by default.** Pass `persist_fn` + `load_fn` to
serialize across processes (SQLite, Redis, whatever):

```python
detector = DriftDetector(
    embed_fn=embed,
    persist_fn=lambda bucket, state: redis.set(f"drift:{bucket}", json.dumps(state)),
    load_fn=lambda bucket: json.loads(redis.get(f"drift:{bucket}") or "null"),
)
```

**Use cases:**

- Detect that a customer's traffic has shifted (potential abuse or new use case).
- Detect that a deployed prompt template is being used differently than tested.
- Detect when your eval set has gone stale relative to live traffic.

---

## 4. `race` -- hedged execution

```python
from sextant import race
from sextant.adapters import openai_complete, anthropic_complete

primary = openai_complete(...)        # gpt-4o
backup  = anthropic_complete(...)     # claude-haiku

messages = [{"role": "user", "content": "Summarize this article: ..."}]

result = race([
    ("openai",    lambda: primary(messages)),
    ("anthropic", lambda: backup(messages)),
], timeout_secs=10.0)

print(f"winner: {result.winner} after {result.latency_ms:.0f}ms")
print(result.value)
```

**What it does:**

Runs all N callables in parallel in a `ThreadPoolExecutor`. Returns
whichever **succeeds** first; cancels the rest; records losers for
telemetry. Failures don't kill the race (set `fail_fast=True` to change
that).

**This is generic.** It's not LLM-specific -- any zero-arg callables work.
Use it to race two embedding providers, two web fetches, two database
shards, whatever.

**Cost model:** if all callables eventually succeed, you pay N × compute.
In practice the losers get cancelled mid-flight (HTTP connections closed,
provider calls aborted) so you pay closer to 1.2× -- 1.5× for major tail
latency wins.

---

## 5. `best_of_n` -- sample-and-score

The natural companion to self-consistency. Where self-consistency uses
*voting* to pick the answer, `best_of_n` uses a *scorer function*:

```python
from sextant import best_of_n, llm_judge_scorer
from sextant.adapters import openai_complete

complete = openai_complete(OpenAI(), model="gpt-4o-mini", temperature=0.7)
judge    = openai_complete(OpenAI(), model="gpt-4o-mini", temperature=0.0)

r = best_of_n(
    complete,
    messages=[{"role": "user", "content": "Write a haiku about Anchorage."}],
    scorer=llm_judge_scorer(judge,
        rubric="Rate this haiku 0-10 on imagery, meter, and surprise. "
                "Respond with only the number."),
    n=5,
)
print(r.answer)         # winning haiku
print(r.scores)         # all 5 scores
```

When to use which:

| | self_consistency | best_of_n |
|---|---|---|
| Task has a discrete answer | yes (vote on it) | overkill |
| Task is open-ended | no (no token to vote on) | yes (score each) |
| You have a reward model | not used | plug it in as `scorer` |
| You want LLM-as-judge | no | yes (use `llm_judge_scorer`) |

Three scorer factories are included: `llm_judge_scorer`, `length_scorer`,
`keyword_scorer`. You can also pass any `Callable[[str], float]`.

---

## Async API

Every primitive has an async sibling:

```python
from sextant.asyncio import acove, aself_consistency, abest_of_n, arace

# N samples run concurrently instead of sequentially:
result = await aself_consistency(async_complete, messages=[...], n=10)

# Race async coroutines:
result = await arace([
    ("openai",    lambda: openai_complete_async(msgs)),
    ("anthropic", lambda: anthropic_complete_async(msgs)),
])
```

For `aself_consistency` and `abest_of_n`, this turns N x latency into ~1 x
latency. For `acove`, the N verification answers fan out concurrently
(steps 1, 2, 4 are still sequential because they depend on each other).

---

## Adapters (optional)

`sextant.adapters` includes thin wrappers for the popular SDKs so you
don't have to write the `(messages) -> str` glue yourself:

```python
from sextant.adapters import (
    openai_complete,       # any openai.OpenAI() client
    openai_embed,
    anthropic_complete,    # any anthropic.Anthropic() client
    echo_complete,         # deterministic stub for tests
    varying_echo_complete, # cycling stub for self_consistency tests
)
```

If you use a provider that isn't here, write your own three-line adapter:

```python
def my_complete(messages: list[dict]) -> str:
    resp = my_client.do_chat(messages=messages, ...)
    return resp.text
```

That's the whole interface.

---

## Combining primitives

These compose. A high-stakes RAG endpoint might do:

```python
# Hedge between two LLM providers, then verify the winner's output.
def cove_with_hedge(query: str) -> str:
    def run_a(): return primary(messages_for(query))
    def run_b(): return backup(messages_for(query))
    raw = race([("openai", run_a), ("anthropic", run_b)]).value
    # Now use CoVe to clean up that fastest-arriving answer.
    return cove(primary, query=query).final
```

Or self-consistency + drift together:

```python
# Sample N answers AND record the prompt for drift monitoring.
detector.observe(bucket=f"key:{api_key}", text=prompt)
result = self_consistency(complete, messages=[...], n=5)
return result.answer
```

---

## What sextant does NOT do

- **No router.** Use LiteLLM, Portkey, or your own.
- **No observability backend.** Pipe the result objects to Langfuse, Helicone,
  Datadog, or print them. Sextant gives you the data; you decide where it goes.
- **No retrieval / RAG.** Use LlamaIndex, LangChain, or a real vector DB.
- **No agent loop.** Use the framework of your choice.
- **No streaming.** All four primitives operate on complete responses.
  (CoVe needs the full baseline; self-consistency needs full samples;
  hedged execution returns the first complete response; drift is per-prompt.)

These are deliberate. Sextant is a small library, not a framework.

---

## Performance notes

- **CoVe** adds N+2 model calls. For 5 questions on Claude Haiku, that's
  ~$0.002 of additional spend per query and ~6× the wall time of greedy.
  Reserve it for high-stakes factual answers.
- **self_consistency** is embarrassingly parallel; sextant doesn't parallelize
  internally (your `complete` is synchronous), but wrapping it in an async
  caller is straightforward.
- **DriftDetector** is O(d) per observation where d is the embedding
  dimension. The bottleneck is the embedding API call, not the math.
- **`race`** uses `concurrent.futures.ThreadPoolExecutor`; safe for the
  blocking HTTP clients that every provider SDK currently uses.

---

## CLI

For a quick spin without writing code, `sextant` ships with a tiny CLI:

```bash
# Offline (uses the deterministic echo stub):
python -m sextant cove "Who painted the Mona Lisa?"
python -m sextant self_consistency "What is 2+2?" --n 3

# With a real provider (OpenAI):
OPENAI_API_KEY=sk-... python -m sextant cove "Where was Marie Curie born?"
OPENAI_API_KEY=sk-... python -m sextant self_consistency \
    "Janet has 16 eggs..." --n 5 --extractor last_number
```

Pass `--help` for the full list of subcommands and flags.

---

## Development

```bash
git clone https://github.com/NORTHTEKDevs/sextant.git
cd sextant
pip install -e .[dev]
pytest tests/                          # ~6s; no network calls.
ruff check .
```

Tests use deterministic stub `CompleteFn` / `EmbedFn` callables. No API
keys required.

---

## Releasing to PyPI

Sextant uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC, no manual token). Tag-driven release flow:

```bash
git tag v0.2.1
git push origin v0.2.1
# .github/workflows/release.yml builds + publishes automatically
```

One-time setup on PyPI (maintainers only):

1. Sign in at https://pypi.org and go to **Account settings → Publishing**.
2. Click **Add a new pending publisher** and fill in:
   - PyPI Project Name: `sextant`
   - Owner: `NORTHTEKDevs`
   - Repository name: `sextant`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
3. The next tag push will publish automatically.

---

## Citing

If you use sextant in a paper or product writeup:

```
@software{sextant,
  author = {Baer, Kristian},
  title  = {Sextant: reliability primitives for LLM APIs},
  year   = {2026},
  url    = {https://github.com/NORTHTEKDevs/sextant}
}
```

For the underlying methods, cite the original papers:

- Dhuliawala et al. (2023). *Chain-of-Verification Reduces Hallucination
  in Large Language Models*. arxiv:2309.11495.
- Wang et al. (2022). *Self-Consistency Improves Chain of Thought
  Reasoning in Language Models*. arxiv:2203.11171.
- Dean & Barroso (2013). *The Tail at Scale*. Communications of the ACM.

---

## License

MIT. Use it however you like.
