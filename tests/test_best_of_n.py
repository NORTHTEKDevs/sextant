"""Tests for lemmas.best_of_n."""

from __future__ import annotations

import pytest

from lemmas.adapters import varying_echo_complete
from lemmas.best_of_n import (
    best_of_n,
    keyword_scorer,
    length_scorer,
    llm_judge_scorer,
)


def test_best_of_n_picks_highest_scoring():
    # Stub returns samples with predictable lengths.
    suffixes = ["short", "medium answer", "this is a much longer answer overall"]
    complete = varying_echo_complete(suffixes)
    # length_scorer with target=200 -> prefers the longest (closest to 200)
    r = best_of_n(complete,
                   messages=[{"role": "user", "content": "?"}],
                   scorer=length_scorer(target_chars=200),
                   n=3)
    assert r.n == 3
    assert len(r.samples) == 3
    assert len(r.scores) == 3
    # Winner index points to the longest sample.
    winning_text = r.samples[r.winner_index]
    assert "much longer answer" in winning_text


def test_keyword_scorer_picks_match():
    suffixes = ["no keywords here", "this has alpha", "this has alpha and beta"]
    complete = varying_echo_complete(suffixes)
    r = best_of_n(complete,
                   messages=[{"role": "user", "content": "?"}],
                   scorer=keyword_scorer(["alpha", "beta"]),
                   n=3)
    assert r.score == 2.0
    assert "alpha and beta" in r.samples[r.winner_index]


def test_keyword_scorer_case_sensitivity():
    s = keyword_scorer(["Hello"], case_sensitive=True)
    assert s("Hello world") == 1.0
    assert s("hello world") == 0.0
    s2 = keyword_scorer(["Hello"], case_sensitive=False)
    assert s2("hello world") == 1.0


def test_length_scorer_prefers_target():
    s = length_scorer(target_chars=100)
    # A 100-char string scores 0 (the maximum -- closer to target).
    assert s("x" * 100) == 0.0
    # Longer or shorter is penalized.
    assert s("x" * 200) == -100.0
    assert s("x" * 50) == -50.0


def test_llm_judge_scorer_parses_numbers():
    # A judge stub that responds with a number proportional to length.
    def judge(_messages):
        # The candidate is inside the user message; just parse a fixed number.
        return "Score: 7.5"
    s = llm_judge_scorer(judge)
    assert s("anything") == 7.5


def test_llm_judge_scorer_handles_unparseable_response():
    def judge(_messages):
        return "I have no idea what to say."
    s = llm_judge_scorer(judge)
    assert s("anything") == 0.0


def test_zero_n_rejected():
    with pytest.raises(ValueError):
        best_of_n(varying_echo_complete(["x"]),
                   messages=[{"role": "user", "content": "?"}],
                   scorer=length_scorer(), n=0)
