"""Self-consistency sampling.

Wang et al. 2022 (arxiv 2203.11171), Google Research.

Idea: for chain-of-thought reasoning where greedy decoding fails, sample N
completions at temperature > 0 and pick the answer that the plurality of
samples agree on. Empirically beats greedy on GSM8K, SVAMP, AQuA, ARC,
StrategyQA -- often by 10-20 points of accuracy.

Sextant supports four answer extractors:

  last_line     Take the last non-empty line. Good for "The answer is X."
  last_number   Last number in the response. Good for arithmetic.
  regex         Custom regex; group 1 (if present) is the answer.
  similarity    No string extraction; pick the sample closest to the
                  semantic centroid of all samples. Requires an embed_fn.
                  Useful for open-ended generation where the "right" answer
                  isn't a discrete token but a position in semantic space.

Cost is N model calls (linear). For most tasks N=5 gives near-saturating
gains; the paper recommends N=20-40 for hard reasoning benchmarks.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from sextant.types import CompleteFn, EmbedFn, Message


@dataclass
class SelfConsistencyResult:
    answer: str
    confidence: float                  # fraction of samples agreeing (or
                                        # semantic-centroid headroom in [0,1])
    samples: list[str]                 # all N raw completions
    extracted: list[str]               # per-sample extracted answer
    vote_counts: dict[str, int] = field(default_factory=dict)
    n: int = 0


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def self_consistency(
    complete: CompleteFn,
    messages: list[Message],
    n: int = 5,
    extractor: str = "last_line",
    extractor_regex: str | None = None,
    embed_fn: EmbedFn | None = None,
) -> SelfConsistencyResult:
    """Sample N completions and return the plurality answer.

    Note: `complete` must produce different outputs across calls (i.e. you
    set temperature > 0 in your provider call). Sextant doesn't inject
    temperature -- it's wrapped into your CompleteFn.

    Args:
        complete:        the LLM callable.
        messages:        the prompt as a list of Message dicts.
        n:               number of samples.
        extractor:       "last_line" | "last_number" | "regex" | "similarity".
        extractor_regex: required if extractor == "regex".
        embed_fn:        required if extractor == "similarity".

    Returns:
        SelfConsistencyResult.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if extractor == "regex" and not extractor_regex:
        raise ValueError("extractor='regex' requires extractor_regex=...")
    if extractor == "similarity" and embed_fn is None:
        raise ValueError(
            "extractor='similarity' requires embed_fn=... (a callable that "
            "maps list[str] -> numpy.ndarray of shape (n, d))")

    samples = [(complete(messages) or "") for _ in range(n)]
    extracted = [_extract(s, extractor, extractor_regex) for s in samples]

    if extractor == "similarity":
        answer, confidence = _semantic_vote(samples, embed_fn)  # type: ignore[arg-type]
        return SelfConsistencyResult(
            answer=answer, confidence=confidence,
            samples=samples, extracted=extracted,
            vote_counts=dict(Counter(extracted)), n=n,
        )

    counts = Counter(e for e in extracted if e)
    if not counts:
        return SelfConsistencyResult(
            answer="", confidence=0.0,
            samples=samples, extracted=extracted, n=n,
        )
    answer, votes = counts.most_common(1)[0]
    return SelfConsistencyResult(
        answer=answer, confidence=votes / n,
        samples=samples, extracted=extracted,
        vote_counts=dict(counts), n=n,
    )


def _extract(text: str, extractor: str, regex: str | None) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if extractor == "last_line":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    if extractor == "last_number":
        matches = _NUM_RE.findall(text)
        return matches[-1] if matches else ""
    if extractor == "regex":
        if not regex:
            return text
        m = re.search(regex, text, re.DOTALL)
        if m:
            return (m.group(1) if m.groups() else m.group(0)).strip()
        return ""
    if extractor == "similarity":
        return text   # handled by _semantic_vote
    return text


def _semantic_vote(samples: list[str], embed_fn: EmbedFn) -> tuple[str, float]:
    """Embed all samples, return the one closest to the centroid.

    confidence = (max_sim - mean_sim) / max(eps, 1 - mean_sim)
    -- roughly the fraction of available headroom above the average. ~1.0
    when one sample stands out; ~0.0 when all samples are equally close
    to the centroid.
    """
    import numpy as np
    vecs = np.asarray(embed_fn(samples))
    if vecs.ndim != 2 or vecs.shape[0] != len(samples):
        raise ValueError(
            f"embed_fn must return shape (n, d); got {vecs.shape}")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vecs / norms
    centroid = unit.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    sims = unit @ centroid
    best = int(np.argmax(sims))
    max_sim = float(sims[best])
    mean_sim = float(sims.mean())
    headroom = 1.0 - mean_sim
    confidence = (max_sim - mean_sim) / max(1e-9, headroom)
    confidence = max(0.0, min(1.0, confidence))
    return samples[best], confidence
