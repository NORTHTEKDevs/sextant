"""Tests for sextant.reflexion."""

from __future__ import annotations

import pytest

from sextant.adapters import varying_echo_complete
from sextant.reflexion import llm_critic, programmatic_critic, reflexion


def test_reflexion_stops_when_critic_passes():
    """First attempt passes -> single iteration."""
    complete = varying_echo_complete(["good answer", "not good"])

    def always_pass(_candidate: str) -> tuple[bool, str]:
        return True, "PASS"

    r = reflexion(complete, query="?", critic=always_pass, max_iterations=4)
    assert r.passed is True
    assert r.iterations == 1
    assert len(r.steps) == 1


def test_reflexion_retries_until_pass():
    """Critic rejects first two, accepts third -> three iterations."""
    complete = varying_echo_complete(["v1", "v2", "v3", "v4"])
    state = {"calls": 0}

    def passes_on_third(_candidate: str) -> tuple[bool, str]:
        state["calls"] += 1
        return (state["calls"] >= 3, "needs work" if state["calls"] < 3 else "PASS")

    r = reflexion(complete, query="?", critic=passes_on_third,
                   max_iterations=5)
    assert r.passed is True
    assert r.iterations == 3
    assert len(r.steps) == 3
    # All but the last step were rejected.
    assert all(not s.critic_passed for s in r.steps[:-1])
    assert r.steps[-1].critic_passed is True


def test_reflexion_returns_last_attempt_if_never_passes():
    complete = varying_echo_complete(["x", "y", "z"])

    def always_fail(_candidate: str) -> tuple[bool, str]:
        return False, "still bad"

    r = reflexion(complete, query="?", critic=always_fail,
                   max_iterations=3)
    assert r.passed is False
    assert r.iterations == 3
    assert len(r.steps) == 3
    assert r.final  # we still return something


def test_reflexion_feedback_passed_to_retry():
    """The retry prompt should include the prior feedback."""
    seen_prompts: list[str] = []

    def capturing_complete(messages):
        # Capture the user content of every call.
        last = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None)
        seen_prompts.append(last.get("content", "") if last else "")
        return f"attempt-{len(seen_prompts)}"

    def fail_first_pass_second(candidate: str) -> tuple[bool, str]:
        return (candidate == "attempt-2",
                "PASS" if candidate == "attempt-2" else "use better wording")

    r = reflexion(capturing_complete, query="write a poem",
                   critic=fail_first_pass_second, max_iterations=3)
    assert r.passed is True
    assert r.iterations == 2
    # Second prompt should contain the prior feedback string.
    assert "use better wording" in seen_prompts[1]
    # And the prior attempt.
    assert "attempt-1" in seen_prompts[1]


def test_zero_max_iterations_rejected():
    with pytest.raises(ValueError):
        reflexion(varying_echo_complete(["x"]), query="?",
                   critic=lambda _c: (True, ""), max_iterations=0)


def test_programmatic_critic_factory_passes_through():
    def passes_if_long(c: str) -> tuple[bool, str]:
        return (len(c) > 20, "too short")
    critic = programmatic_critic(passes_if_long)
    passed, feedback = critic("short")
    assert passed is False
    assert feedback == "too short"
    passed, feedback = critic("this is a much longer string")
    assert passed is True


def test_llm_critic_passes_on_PASS_substring():
    def judge_say_pass(_messages):
        return "Looks good. Verdict: PASS"

    critic = llm_critic(judge_say_pass)
    passed, _verdict = critic("anything")
    assert passed is True


def test_llm_critic_fails_when_no_PASS():
    def judge_critique(_messages):
        return "The answer is missing the conclusion paragraph."

    critic = llm_critic(judge_critique)
    passed, verdict = critic("anything")
    assert passed is False
    assert "conclusion" in verdict


def test_llm_critic_case_insensitive():
    def judge_lowercase_pass(_messages):
        return "pass"
    critic = llm_critic(judge_lowercase_pass)
    passed, _ = critic("anything")
    assert passed is True
