"""Async versions of the four primitives.

The big win here is that self_consistency and best_of_n are embarrassingly
parallel -- N independent LLM calls. Sync versions serialize them; async
versions launch them concurrently with asyncio.gather.

Public API:

  acove(...)              Chain-of-Verification, with N+2 awaits.
  aself_consistency(...)  Sample N concurrently, vote.
  abest_of_n(...)         Sample N concurrently, score (optionally async), pick.
  arace(...)              Race async coroutines, return first to complete.

All take AsyncCompleteFn (or AsyncCompleteFn + sync scorer for abest_of_n)
instead of sync CompleteFn. Adapters for OpenAI / Anthropic async clients
are in sextant.adapters_async.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from sextant.best_of_n import BestOfNResult
from sextant.cove import CoVeStep, CoVeResult, _split_questions
from sextant.cove import (
    _BASELINE_TEMPLATE, _PLAN_TEMPLATE, _ANSWER_TEMPLATE, _FINAL_TEMPLATE,
)
from sextant.hedged import HedgeResult
from sextant.self_consistency import (
    SelfConsistencyResult, _extract, _semantic_vote,
)
from sextant.types import AsyncCompleteFn, AsyncEmbedFn, EmbedFn, Message


# ---- Async CoVe ----------------------------------------------------------

async def acove(
    complete: AsyncCompleteFn,
    query: str,
    n_questions: int = 4,
    system: str | None = None,
) -> CoVeResult:
    """Async Chain-of-Verification. The N verification answers run concurrently."""
    steps: list[CoVeStep] = []

    async def _ask(user_content: str) -> str:
        msgs: list[Message] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_content})
        out = await complete(msgs)
        return (out or "").strip()

    # 1. Baseline.
    baseline_prompt = _BASELINE_TEMPLATE.format(query=query)
    baseline = await _ask(baseline_prompt)
    steps.append(CoVeStep(name="baseline", content=baseline,
                           raw_input=baseline_prompt))

    # 2. Plan.
    plan_prompt = _PLAN_TEMPLATE.format(query=query, n=n_questions)
    plan_raw = await _ask(plan_prompt)
    questions = _split_questions(plan_raw, n_questions)
    steps.append(CoVeStep(name="plan", content=plan_raw,
                           raw_input=plan_prompt))

    # 3. Execute -- run all questions concurrently.
    if questions:
        answers = await asyncio.gather(*[
            _ask(_ANSWER_TEMPLATE.format(question=q)) for q in questions
        ])
    else:
        answers = []
    for q, a in zip(questions, answers):
        steps.append(CoVeStep(name="answer", content=a, raw_input=q))

    # 4. Revise.
    qa_block = "\n".join(f"Q: {q}\nA: {a}"
                          for q, a in zip(questions, answers))
    final_prompt = _FINAL_TEMPLATE.format(query=query, baseline=baseline,
                                            qa_block=qa_block)
    final = await _ask(final_prompt)
    steps.append(CoVeStep(name="final", content=final,
                           raw_input=final_prompt))

    revisions = 0 if final.strip() == baseline.strip() else 1
    return CoVeResult(final=final, baseline=baseline,
                       questions=questions, answers=list(answers),
                       steps=steps, revisions=revisions)


# ---- Async self_consistency ---------------------------------------------

async def aself_consistency(
    complete: AsyncCompleteFn,
    messages: list[Message],
    n: int = 5,
    extractor: str = "last_line",
    extractor_regex: str | None = None,
    embed_fn: EmbedFn | AsyncEmbedFn | None = None,
) -> SelfConsistencyResult:
    """Async self-consistency. N samples drawn concurrently.

    `embed_fn` may be sync or async; if async, it is awaited.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if extractor == "regex" and not extractor_regex:
        raise ValueError("extractor='regex' requires extractor_regex=...")
    if extractor == "similarity" and embed_fn is None:
        raise ValueError("extractor='similarity' requires embed_fn=...")

    samples = await asyncio.gather(*[complete(messages) for _ in range(n)])
    samples = [(s or "") for s in samples]
    extracted = [_extract(s, extractor, extractor_regex) for s in samples]

    if extractor == "similarity":
        if inspect.iscoroutinefunction(embed_fn):
            vecs = await embed_fn(samples)  # type: ignore[misc]
            sync_embed = lambda _texts: vecs  # noqa: E731
        else:
            sync_embed = embed_fn  # type: ignore[assignment]
        answer, confidence = _semantic_vote(samples, sync_embed)  # type: ignore[arg-type]
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


# ---- Async best_of_n ----------------------------------------------------

AsyncScoreFn = Callable[[str], Awaitable[float]]


async def abest_of_n(
    complete: AsyncCompleteFn,
    messages: list[Message],
    scorer: Callable[[str], float] | AsyncScoreFn,
    n: int = 5,
) -> BestOfNResult:
    """Async best-of-N. Samples + (if async) scoring run concurrently."""
    if n < 1:
        raise ValueError("n must be >= 1")
    samples = await asyncio.gather(*[complete(messages) for _ in range(n)])
    samples = [(s or "") for s in samples]

    if inspect.iscoroutinefunction(scorer):
        scores = await asyncio.gather(*[scorer(s) for s in samples])  # type: ignore[misc]
    else:
        scores = [float(scorer(s)) for s in samples]  # type: ignore[operator]
    scores = [float(x) for x in scores]
    winner = max(range(n), key=lambda i: scores[i])
    return BestOfNResult(
        answer=samples[winner], score=scores[winner],
        samples=samples, scores=scores, n=n, winner_index=winner,
    )


# ---- Async race ---------------------------------------------------------

async def arace(
    targets: list[tuple[str, Callable[[], Awaitable]]],
    timeout_secs: float = 30.0,
    fail_fast: bool = False,
) -> HedgeResult:
    """Race async coroutines; return whichever succeeds first."""
    if not targets:
        raise ValueError("arace: at least one target required")
    started = time.monotonic()
    losers: list[dict] = []

    async def _wrap(label: str, fn):
        try:
            return label, await fn(), None
        except BaseException as e:
            return label, None, e

    tasks = [asyncio.create_task(_wrap(label, fn)) for label, fn in targets]
    pending = set(tasks)
    last_error: BaseException | None = None
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                timeout=timeout_secs if pending == set(tasks) else None,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done and pending:
                for t in pending:
                    t.cancel()
                raise TimeoutError(
                    f"arace timed out after {timeout_secs}s")
            for t in done:
                label, value, err = await t
                if err is not None:
                    losers.append({
                        "label": label, "error": str(err),
                        "latency_ms": (time.monotonic() - started) * 1000,
                    })
                    if fail_fast:
                        for p in pending:
                            p.cancel()
                        raise err
                    if isinstance(err, BaseException):
                        last_error = err
                    continue
                # Winner: cancel remaining and return.
                elapsed_ms = (time.monotonic() - started) * 1000
                for p in pending:
                    p.cancel()
                    losers.append({
                        "label": "?", "cancelled": True,
                        "latency_ms": elapsed_ms,
                    })
                return HedgeResult(winner=label, value=value,
                                    latency_ms=elapsed_ms, losers=losers)
        if last_error is not None:
            raise last_error
        raise RuntimeError("arace: no successful result")
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
