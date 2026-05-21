"""Tests for sextant.asyncio.

The sync primitives are already tested in their own modules. These tests
focus on the things async actually changes:

  1. N-sample primitives parallelize (faster than serial would be).
  2. Async scorer / async embed_fn are awaited correctly.
  3. arace cancels losers and surfaces winners correctly.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from sextant.asyncio import abest_of_n, acove, arace, aself_consistency

# ---- helpers --------------------------------------------------------------

def make_async_complete(suffixes: list[str], per_call_delay: float = 0.0):
    """Async stub: cycles through suffixes, sleeps per_call_delay seconds."""
    state = {"i": 0}

    async def _fn(messages):
        if per_call_delay > 0:
            await asyncio.sleep(per_call_delay)
        i = state["i"] % len(suffixes)
        state["i"] += 1
        last = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None)
        body = (last.get("content") if last else "")
        return f"{body}\n{suffixes[i]}"

    return _fn


# ---- aself_consistency ----------------------------------------------------

@pytest.mark.asyncio
async def test_aself_consistency_returns_plurality():
    complete = make_async_complete(["Answer: 42", "Answer: 42", "Answer: 7"])
    r = await aself_consistency(
        complete, messages=[{"role": "user", "content": "?"}],
        n=3, extractor="last_line")
    assert r.answer == "Answer: 42"
    assert r.confidence == pytest.approx(2 / 3)
    assert r.n == 3


@pytest.mark.asyncio
async def test_aself_consistency_parallelizes():
    """N samples should complete in ~1x per-call delay, not Nx."""
    per_call = 0.15
    complete = make_async_complete(["Answer: A"] * 5, per_call_delay=per_call)
    t0 = time.monotonic()
    await aself_consistency(complete,
                             messages=[{"role": "user", "content": "?"}],
                             n=5)
    elapsed = time.monotonic() - t0
    # Serial would be ~0.75s; concurrent should be well under 0.4s.
    assert elapsed < per_call * 3.5, (
        f"expected concurrent run; got {elapsed:.2f}s for 5x{per_call}s")


# ---- abest_of_n ----------------------------------------------------------

@pytest.mark.asyncio
async def test_abest_of_n_with_sync_scorer():
    complete = make_async_complete(["short", "much longer answer here"])
    r = await abest_of_n(
        complete, messages=[{"role": "user", "content": "?"}],
        scorer=lambda s: float(len(s)), n=2)
    assert r.n == 2
    assert r.score >= len("much longer answer here")


@pytest.mark.asyncio
async def test_abest_of_n_with_async_scorer():
    complete = make_async_complete(["a", "bb", "ccc"])

    async def async_score(s: str) -> float:
        await asyncio.sleep(0.0)
        return float(len(s))

    r = await abest_of_n(
        complete, messages=[{"role": "user", "content": "?"}],
        scorer=async_score, n=3)
    assert r.winner_index == 2  # "ccc" is longest


# ---- acove ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_acove_full_pipeline():
    complete = make_async_complete(["I don't know.", "Q1?\nQ2?\nQ3?",
                                       "unknown", "unknown", "unknown",
                                       "revised"])
    r = await acove(complete, query="something", n_questions=3)
    assert r.baseline
    assert r.final
    assert len(r.questions) <= 3
    assert len(r.answers) == len(r.questions)


# ---- arace ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_arace_returns_first_to_finish():
    async def slow():
        await asyncio.sleep(0.3)
        return "slow"

    async def fast():
        await asyncio.sleep(0.05)
        return "fast"

    r = await arace([("slow", slow), ("fast", fast)])
    assert r.winner == "fast"
    assert r.value == "fast"


@pytest.mark.asyncio
async def test_arace_survives_failing_callable():
    async def bad():
        raise RuntimeError("kaboom")

    async def good():
        await asyncio.sleep(0.05)
        return 42

    r = await arace([("bad", bad), ("good", good)])
    assert r.winner == "good"
    assert r.value == 42


@pytest.mark.asyncio
async def test_arace_empty_raises():
    with pytest.raises(ValueError):
        await arace([])
