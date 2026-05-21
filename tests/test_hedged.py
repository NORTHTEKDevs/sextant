"""Tests for lemmas.hedged."""

from __future__ import annotations

import time

import pytest

from lemmas.hedged import race


def test_fast_wins_against_slow():
    def slow():
        time.sleep(0.3)
        return "slow"

    def fast():
        time.sleep(0.05)
        return "fast"

    r = race([("slow", slow), ("fast", fast)])
    assert r.winner == "fast"
    assert r.value == "fast"
    # The slow one is recorded as a cancelled loser.
    assert any(loser.get("label") == "slow" for loser in r.losers)


def test_failing_callable_doesnt_kill_the_race():
    def bad():
        raise RuntimeError("kaboom")

    def good():
        time.sleep(0.05)
        return 42

    r = race([("bad", bad), ("good", good)])
    assert r.winner == "good"
    assert r.value == 42
    # The failure shows up in losers with an "error" key.
    assert any("error" in loser for loser in r.losers)


def test_all_failures_propagates_last_error():
    def fail_a():
        raise ValueError("a")

    def fail_b():
        raise ValueError("b")

    with pytest.raises(ValueError):
        race([("a", fail_a), ("b", fail_b)])


def test_fail_fast_raises_immediately():
    def bad():
        raise RuntimeError("boom")

    def slow():
        time.sleep(2.0)
        return "slow"

    with pytest.raises(RuntimeError):
        race([("bad", bad), ("slow", slow)], fail_fast=True)


def test_empty_targets_raises_value_error():
    with pytest.raises(ValueError):
        race([])


def test_timeout_raises_timeout_error():
    def slow():
        time.sleep(5.0)
        return "x"

    with pytest.raises(TimeoutError):
        race([("a", slow)], timeout_secs=0.1)


def test_latency_ms_is_recorded():
    def quick():
        return "ok"

    r = race([("q", quick)])
    assert r.latency_ms >= 0.0
    assert r.latency_ms < 1000.0
