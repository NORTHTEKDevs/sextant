"""Speculative / hedged execution.

When latency matters more than cost, fire the same request at N backends
in parallel and return whichever finishes first. The losers' results are
recorded for telemetry but discarded.

This cuts tail latency dramatically (especially p99) at the price of N x
cost. Common applications:

  - voice loops where p99 STT/TTS latency is user-visible
  - first-token-fast chat where you race Haiku against gpt-4o-mini
  - critical-path calls where you race a self-hosted model against a
    managed one and fall through if the self-hosted is degraded

Sextant's race is generic: any zero-arg callables work, not just LLM calls.

Cost model: if all callables eventually succeed, you pay for N x compute.
If most fail and one succeeds, you pay for ~1 (others get cancelled).
"""

from __future__ import annotations

import concurrent.futures as _cf
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class HedgeResult:
    winner: str                # label of the callable that finished first
    value: Any                 # whatever it returned
    latency_ms: float          # wall-clock from race start to winner finish
    losers: list[dict[str, Any]] = field(default_factory=list)


def race(
    targets: list[tuple[str, Callable[[], T]]],
    timeout_secs: float = 30.0,
    fail_fast: bool = False,
) -> HedgeResult:
    """Run all callables in parallel; return whichever succeeds first.

    Args:
        targets:     list of (label, zero-arg-callable). Labels appear in
                       HedgeResult.winner / losers for telemetry.
        timeout_secs: hard timeout on the whole race.
        fail_fast:   if True, raise on the first exception instead of
                       waiting for any success. Default False -- a single
                       failing backend doesn't kill the race.

    Returns:
        HedgeResult with the winner's value and metadata on losers.

    Raises:
        ValueError       if targets is empty.
        TimeoutError     if no callable completes within timeout_secs.
        Exception        of the last-known kind, if every callable fails.
    """
    if not targets:
        raise ValueError("race: at least one target required")

    started = time.monotonic()
    losers: list[dict[str, Any]] = []
    with _cf.ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures: dict[_cf.Future, str] = {
            pool.submit(fn): label for label, fn in targets
        }
        done_first: _cf.Future | None = None
        winner_label = ""
        last_error: BaseException | None = None

        while futures and done_first is None:
            done, _ = _cf.wait(futures.keys(),
                                timeout=timeout_secs,
                                return_when=_cf.FIRST_COMPLETED)
            if not done:
                for fut in futures:
                    fut.cancel()
                raise TimeoutError(
                    f"hedged race timed out after {timeout_secs}s")
            for fut in done:
                label = futures.pop(fut)
                err = fut.exception()
                if err is not None:
                    losers.append({
                        "label": label, "error": str(err),
                        "latency_ms": (time.monotonic() - started) * 1000,
                    })
                    if fail_fast:
                        for f in futures:
                            f.cancel()
                        raise err
                    if isinstance(err, BaseException):
                        last_error = err
                    continue
                done_first = fut
                winner_label = label
                break

        if done_first is None:
            # All futures completed but every one failed.
            if last_error is not None:
                raise last_error
            raise RuntimeError("hedged race: no successful result")

        elapsed_ms = (time.monotonic() - started) * 1000
        value = done_first.result()
        for fut, label in futures.items():
            fut.cancel()
            losers.append({
                "label": label, "cancelled": True,
                "latency_ms": elapsed_ms,
            })
        return HedgeResult(winner=winner_label, value=value,
                            latency_ms=elapsed_ms, losers=losers)
