"""Reflexion -- iterative self-improvement with verbal reinforcement.

Shinn et al. 2023 (arxiv 2303.11366), Northeastern + MIT + Princeton.

Different from CoVe (which is one shot of plan + verify + revise), Reflexion
is a *loop*:

  1. Attempt the task.
  2. Have the model critique its own answer against the goal.
  3. If the critic decides the answer is good enough, stop. Otherwise the
      critique becomes a "lesson" added to the prompt; go to step 1.
  4. Repeat until a max-iterations cap is hit.

Critic interface (you supply one):

  CriticFn = Callable[[str], CriticVerdict]
  CriticVerdict = (passed: bool, feedback: str)

Two built-in critic factories:

  - llm_critic(complete, rubric)
      Uses another LLM call to grade and give feedback. Looks for the
      string "PASS" anywhere in the verdict.

  - test_critic(test_fn)
      Programmatic. Pass a function that returns (passed, feedback).
      Useful for code-generation: run unit tests on the generated code.

Reflexion shines on tasks where you have a verifiable signal (unit tests,
exact-match against a known answer, a calibrated reward model) but the
first-shot answer is unreliable. It's strictly stronger than best-of-N
because the critique is *fed back* into the next attempt, so the model
learns from its mistakes within the same conversation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from sextant.types import CompleteFn, Message

CriticVerdict = tuple[bool, str]
"""(passed, feedback). feedback is shown to the model on retry."""

CriticFn = Callable[[str], CriticVerdict]
"""Map a candidate answer to (passed, feedback_text)."""


@dataclass
class ReflexionStep:
    iteration: int
    attempt: str
    critic_passed: bool
    critic_feedback: str


@dataclass
class ReflexionResult:
    final: str                                # the answer we're returning
    passed: bool                              # whether the critic ever said pass
    iterations: int                           # how many tries it took
    steps: list[ReflexionStep] = field(default_factory=list)


_INITIAL_TEMPLATE = (
    "{query}"
)

_RETRY_TEMPLATE = (
    "{query}\n\n"
    "PRIOR ATTEMPT:\n{prior}\n\n"
    "FEEDBACK ON THAT ATTEMPT:\n{feedback}\n\n"
    "Using the feedback, produce a better answer."
)


def reflexion(
    complete: CompleteFn,
    query: str,
    critic: CriticFn,
    max_iterations: int = 4,
    system: str | None = None,
) -> ReflexionResult:
    """Iterative attempt + critique loop.

    Args:
        complete:       the LLM callable.
        query:          the user's task statement.
        critic:         (candidate) -> (passed, feedback_str).
        max_iterations: hard cap. Stops earlier if critic returns passed=True.
        system:         optional system prompt prepended to every call.

    Returns:
        ReflexionResult.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    def _ask(content: str) -> str:
        msgs: list[Message] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content})
        out = complete(msgs)
        return (out or "").strip()

    steps: list[ReflexionStep] = []
    prior_attempt = ""
    prior_feedback = ""

    for i in range(1, max_iterations + 1):
        if i == 1:
            prompt = _INITIAL_TEMPLATE.format(query=query)
        else:
            prompt = _RETRY_TEMPLATE.format(
                query=query, prior=prior_attempt, feedback=prior_feedback)
        attempt = _ask(prompt)
        passed, feedback = critic(attempt)
        steps.append(ReflexionStep(
            iteration=i, attempt=attempt,
            critic_passed=passed, critic_feedback=feedback,
        ))
        if passed:
            return ReflexionResult(
                final=attempt, passed=True, iterations=i, steps=steps,
            )
        prior_attempt = attempt
        prior_feedback = feedback

    # Loop exhausted without passing. Return the last attempt anyway.
    return ReflexionResult(
        final=steps[-1].attempt, passed=False,
        iterations=max_iterations, steps=steps,
    )


# ---- Built-in critic factories --------------------------------------------

def llm_critic(
    complete: CompleteFn,
    rubric: str = (
        "Evaluate the candidate answer for correctness, completeness, and "
        "clarity. If it is good enough, respond with exactly the word PASS. "
        "Otherwise respond with a short critique pointing out what to fix. "
        "Be specific and actionable."
    ),
) -> CriticFn:
    """An LLM-as-critic. Returns (True, "PASS") iff PASS appears in the verdict."""
    def _critic(candidate: str) -> CriticVerdict:
        prompt = f"{rubric}\n\nCANDIDATE ANSWER:\n{candidate}"
        verdict = complete([{"role": "user", "content": prompt}]).strip()
        passed = "PASS" in verdict.upper()
        return passed, verdict

    return _critic


def programmatic_critic(
    fn: Callable[[str], CriticVerdict],
) -> CriticFn:
    """Wrap a programmatic check (e.g., 'does this code pass unit tests?').

    The `fn` is responsible for running whatever check makes sense and
    returning (passed, feedback). Useful for code generation, JSON-schema
    validation, exact-match-against-known-answer, etc.

    (Renamed from `test_critic` to avoid pytest collection collisions.)
    """
    def _critic(candidate: str) -> CriticVerdict:
        return fn(candidate)
    return _critic


# Backwards-compatible alias (pytest avoids collection because of `__test__`).
test_critic = programmatic_critic
test_critic.__test__ = False  # type: ignore[attr-defined]
