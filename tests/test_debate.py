"""Tests for sextant.debate."""

from __future__ import annotations

import pytest

from sextant.debate import (
    _convergence_winner,
    _default_personas,
    _parse_judge_verdict,
    debate,
)


def _scripted_complete(scripts: dict[str, list[str]]):
    """Return a CompleteFn that returns different scripted outputs depending
    on which agent persona prefix appears in the prompt.

    `scripts` maps a substring (e.g. "analytical reasoner") to a list of
    sequential outputs for that agent across rounds.
    """
    state = {persona: 0 for persona in scripts}

    def _fn(messages):
        content = messages[-1].get("content", "") if messages else ""
        for needle, outs in scripts.items():
            if needle in content:
                i = state[needle] % len(outs)
                state[needle] += 1
                return outs[i]
        return ""

    return _fn


def test_debate_single_complete_fn_spawns_n_agents():
    """When `agents` is a callable, we should get n_agents_if_single drafts."""
    complete = _scripted_complete({
        "analytical reasoner": ["draft A1", "draft A2"],
        "critical skeptic": ["draft B1", "draft B2"],
        "creative thinker": ["draft C1", "draft C2"],
    })
    r = debate(query="Q?", agents=complete, rounds=1,
                n_agents_if_single=3)
    assert len(r.rounds) == 2  # round 0 + 1 revision
    assert len(r.rounds[0].drafts) == 3
    assert r.winner_label in r.rounds[-1].drafts


def test_debate_explicit_agents_and_judge():
    """When a judge is provided, it picks the winner."""
    def complete_a(_msgs):
        return "AAAA reasoning"
    def complete_b(_msgs):
        return "BBBB reasoning"
    def judge(_msgs):
        return "BOB: most convincing argument"
    r = debate(
        query="?",
        agents=[("ALICE", complete_a), ("BOB", complete_b)],
        rounds=0,
        judge=judge,
    )
    assert r.winner_label == "BOB"
    assert r.final == "BBBB reasoning"
    assert "BOB" in r.judgment


def test_debate_no_judge_uses_convergence_winner():
    """Without a judge, picks the draft most similar to the others."""
    # Two agents agree, one disagrees -> one of the two agreeing wins.
    drafts_round = {
        "A": "the sky is blue and so is the sea today",
        "B": "the sky is blue and so is the sea today",
        "C": "the moon orbits the earth and influences tides",
    }
    winner = _convergence_winner(drafts_round)
    assert winner in ("A", "B")


def test_debate_rounds_zero_returns_only_initial_drafts():
    complete = _scripted_complete({
        "analytical reasoner": ["X"],
        "critical skeptic": ["Y"],
    })
    r = debate("?", agents=complete, rounds=0, n_agents_if_single=2)
    assert len(r.rounds) == 1
    assert len(r.rounds[0].drafts) == 2


def test_debate_zero_agents_rejected():
    with pytest.raises(ValueError):
        debate("?", agents=[])


def test_debate_personas_length_mismatch_rejected():
    with pytest.raises(ValueError):
        debate("?", agents=[("A", lambda _m: "x"), ("B", lambda _m: "y")],
                personas=["only one"])


def test_default_personas_cycles_pool():
    p = _default_personas(7)
    assert len(p) == 7
    # First and 6th should be the same (pool length is 5).
    assert p[0] == p[5]


def test_parse_judge_verdict_matches_prefix():
    assert _parse_judge_verdict("ALICE: best", ["ALICE", "BOB"]) == "ALICE"
    assert _parse_judge_verdict("Bob has best reasoning",
                                  ["ALICE", "BOB"]) == "BOB"
    # Fallback: no match -> first label.
    assert _parse_judge_verdict("totally unrelated",
                                  ["ALICE", "BOB"]) == "ALICE"


def test_debate_with_tracer_records_spans():
    """Plugging a LoggingTracer in should record one span per draft + the outer."""
    from sextant import LoggingTracer
    tracer = LoggingTracer()
    def complete(_msgs):
        return "x"
    debate(
        query="?",
        agents=[("A", complete), ("B", complete)],
        rounds=1, tracer=tracer,
    )
    span_names = [s["name"] for s in tracer.spans]
    # outer + 2 drafts + 2 revisions = 5 spans
    assert "sextant.debate" in span_names
    assert span_names.count("sextant.debate.draft") == 2
    assert span_names.count("sextant.debate.revise") == 2
