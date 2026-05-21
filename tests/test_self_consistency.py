"""Tests for lemmas.self_consistency."""

from __future__ import annotations

import pytest

from lemmas.adapters import varying_echo_complete
from lemmas.self_consistency import self_consistency


def test_last_line_plurality_wins():
    suffixes = ["Answer: 42", "Answer: 42", "Answer: 7", "Answer: 42"]
    complete = varying_echo_complete(suffixes)
    r = self_consistency(complete,
                          messages=[{"role": "user", "content": "Q?"}],
                          n=4, extractor="last_line")
    assert r.answer == "Answer: 42"
    assert r.confidence == 0.75
    assert r.vote_counts["Answer: 42"] == 3
    assert r.n == 4
    assert len(r.samples) == 4


def test_last_number_extractor():
    suffixes = ["The total is 17", "It came to 17 dollars", "I think 12"]
    complete = varying_echo_complete(suffixes)
    r = self_consistency(complete,
                          messages=[{"role": "user", "content": "How much?"}],
                          n=3, extractor="last_number")
    assert r.answer == "17"
    assert r.confidence == pytest.approx(2 / 3)


def test_regex_extractor_with_group():
    suffixes = ["[A] yes", "[B] no", "[A] yes"]
    complete = varying_echo_complete(suffixes)
    r = self_consistency(complete,
                          messages=[{"role": "user", "content": "?"}],
                          n=3, extractor="regex",
                          extractor_regex=r"\[(A|B)\]")
    assert r.answer == "A"
    assert r.confidence == pytest.approx(2 / 3)


def test_regex_without_pattern_raises():
    with pytest.raises(ValueError):
        self_consistency(varying_echo_complete(["x"]),
                          messages=[{"role": "user", "content": "?"}],
                          n=1, extractor="regex")


def test_similarity_without_embed_fn_raises():
    with pytest.raises(ValueError):
        self_consistency(varying_echo_complete(["x"]),
                          messages=[{"role": "user", "content": "?"}],
                          n=2, extractor="similarity")


def test_similarity_extractor_picks_centroid():
    """Embed deterministically so we can predict the winner."""
    import numpy as np

    # Three "samples": [aaa, aab, far]
    # We embed by sentence -- pick a stub that returns:
    #   "aaa" -> [1, 0]
    #   "aab" -> [1, 0]
    #   "far" -> [0, 1]
    # Centroid = [0.667, 0.333]; "aaa" and "aab" are equidistant.
    def fake_embed(texts):
        out = []
        for t in texts:
            out.append([1.0, 0.0] if "a" in t else [0.0, 1.0])
        return np.asarray(out, dtype=np.float32)

    def stub_complete(_):
        # Each call returns a different sample (cycle).
        stub_complete.i = getattr(stub_complete, "i", 0)
        outs = ["aaa", "aab", "far"]
        v = outs[stub_complete.i % 3]
        stub_complete.i += 1
        return v

    r = self_consistency(stub_complete,
                          messages=[{"role": "user", "content": "x"}],
                          n=3, extractor="similarity",
                          embed_fn=fake_embed)
    assert r.answer in ("aaa", "aab")
    assert 0.0 <= r.confidence <= 1.0


def test_zero_n_rejected():
    with pytest.raises(ValueError):
        self_consistency(varying_echo_complete(["x"]),
                          messages=[{"role": "user", "content": "?"}],
                          n=0)


def test_no_extractable_answer_returns_empty():
    # last_number on a sample with no digits.
    complete = varying_echo_complete(["just words", "more words"])
    r = self_consistency(complete,
                          messages=[{"role": "user", "content": "?"}],
                          n=2, extractor="last_number")
    assert r.answer == ""
    assert r.confidence == 0.0
