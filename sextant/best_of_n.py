"""Best-of-N sampling with a scoring function.

The natural companion to self-consistency. Where self_consistency uses
*voting* to pick an answer, best_of_n uses a *scorer*:

  - Sample N completions at temperature > 0.
  - Run each through a scorer fn that returns a float (higher = better).
  - Return the highest-scoring sample.

When to use which:

  self_consistency  -- the task has a discrete answer (math, multi-choice,
                        classification). Voting works because the answer is
                        the same string across most samples.

  best_of_n         -- the task is open-ended (essays, code, summaries).
                        There's no canonical token to vote on; a learned or
                        rule-based scorer ranks them instead.

Scorer interface:

  ScoreFn = Callable[[str], float]

Some common scorer constructions:

  - Reward model (cheap classifier).
  - LLM-as-judge (another `complete` call asking the model to grade it).
  - Heuristic: response length, presence of citations, regex matches, etc.
  - Embedding-similarity to a reference answer.

best_of_n returns *all* scores so you can inspect why the winner won.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sextant.types import CompleteFn, Message


ScoreFn = Callable[[str], float]
"""Map a candidate string to a scalar score. Higher = better."""


@dataclass
class BestOfNResult:
    answer: str                         # winning sample
    score: float                        # winning score
    samples: list[str]                  # all N samples in generation order
    scores: list[float] = field(default_factory=list)
    n: int = 0
    winner_index: int = 0


def best_of_n(
    complete: CompleteFn,
    messages: list[Message],
    scorer: ScoreFn,
    n: int = 5,
) -> BestOfNResult:
    """Sample N completions, score each, return the best.

    Args:
        complete: the LLM callable. Should use temperature > 0 so samples differ.
        messages: prompt.
        scorer:   maps each completion string to a float; higher wins.
        n:        number of samples.

    Returns:
        BestOfNResult with winning answer, score, all samples, all scores.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    samples: list[str] = [(complete(messages) or "") for _ in range(n)]
    scores: list[float] = [float(scorer(s)) for s in samples]
    winner_index = max(range(n), key=lambda i: scores[i])
    return BestOfNResult(
        answer=samples[winner_index],
        score=scores[winner_index],
        samples=samples,
        scores=scores,
        n=n,
        winner_index=winner_index,
    )


# ---- Common scorer factories -----------------------------------------------

def llm_judge_scorer(
    complete: CompleteFn,
    rubric: str = "Rate the following response from 0 to 10 on quality, "
                  "factuality, and clarity. Respond with only the number.",
) -> ScoreFn:
    """Use an LLM as the scoring function.

    Returns a scorer that wraps each candidate in a judge prompt and parses
    a number from the judge's response. Falls back to 0.0 on parse failure.
    """
    import re
    num_re = re.compile(r"-?\d+(?:\.\d+)?")

    def _score(candidate: str) -> float:
        prompt = f"{rubric}\n\nRESPONSE:\n{candidate}"
        raw = complete([{"role": "user", "content": prompt}])
        matches = num_re.findall(raw or "")
        if not matches:
            return 0.0
        try:
            return float(matches[0])
        except ValueError:
            return 0.0

    return _score


def length_scorer(target_chars: int = 500) -> ScoreFn:
    """Prefer responses close to `target_chars`. Useful as a sanity-check scorer."""
    def _score(candidate: str) -> float:
        return -abs(len(candidate) - target_chars)
    return _score


def keyword_scorer(keywords: list[str],
                    case_sensitive: bool = False) -> ScoreFn:
    """One point per keyword present."""
    def _score(candidate: str) -> float:
        text = candidate if case_sensitive else candidate.lower()
        kws = keywords if case_sensitive else [k.lower() for k in keywords]
        return float(sum(1 for k in kws if k in text))
    return _score
