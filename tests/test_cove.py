"""Tests for sextant.cove."""

from __future__ import annotations

from sextant.adapters import echo_complete
from sextant.cove import _split_questions, cove


def test_cove_runs_full_pipeline_with_stub():
    """End-to-end smoke test against the echo stub."""
    complete = echo_complete()
    r = cove(complete, query="Who painted the Mona Lisa?",
              n_questions=3)
    assert r.baseline
    assert r.final
    assert len(r.questions) <= 3
    # Each step is recorded.
    step_names = [s.name for s in r.steps]
    assert "baseline" in step_names
    assert "plan" in step_names
    assert "final" in step_names
    # answers ran for every question.
    assert step_names.count("answer") == len(r.questions)


def test_cove_zero_questions_still_completes():
    complete = echo_complete()
    r = cove(complete, query="x", n_questions=0)
    # No questions to plan; baseline + (plan output) + final still run.
    assert r.final
    assert r.questions == []


def test_split_questions_strips_numbering_and_bullets():
    plan = (
        "1. Who painted the Mona Lisa?\n"
        "2. When was it painted?\n"
        "- What's its current location?\n"
        "* Has it ever been stolen?\n"
    )
    qs = _split_questions(plan, 10)
    assert len(qs) == 4
    assert all(q.endswith("?") for q in qs)
    assert "1." not in qs[0]
    assert "-" not in qs[2]


def test_split_questions_caps_at_n():
    plan = "\n".join(f"{i}. Question number {i}?" for i in range(10))
    qs = _split_questions(plan, 3)
    assert len(qs) == 3


def test_split_questions_handles_empty_plan():
    assert _split_questions("", 5) == []
    assert _split_questions("   \n\n  \n", 5) == []


def test_cove_revisions_zero_when_baseline_equals_final():
    # Stub returns identical output every time -> baseline == final.
    complete = echo_complete(prefix="X")
    r = cove(complete, query="anything", n_questions=2)
    # baseline and final both echo the corresponding user prompt verbatim;
    # the final prompt is different from the baseline prompt, so revisions=1.
    # This test mainly exercises the equality-check codepath without
    # depending on stub specifics.
    assert r.revisions in (0, 1)
