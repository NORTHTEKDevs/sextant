"""Chain-of-Verification (CoVe).

Dhuliawala et al. 2023 (arxiv 2309.11495), Meta AI.

Four-step pipeline that reduces hallucination on long-form factual answers:

  1. BASELINE:   model produces an initial response to the user's query.
  2. PLAN:       model generates verification questions targeting the kind
                  of factual claims a good answer would contain.
                  The plan is generated *without* the baseline in context,
                  to keep the questions unbiased.
  3. EXECUTE:    each verification question is answered independently, so
                  one wrong claim can't taint another's verification.
  4. FINAL:      model revises its baseline using the question/answer pairs,
                  correcting or dropping any claim a verification contradicts.

Cost is roughly N+2 model calls (N verification questions + baseline + final).
On TriviaQA + WikiData + MultiSpanQA, the Meta paper reports 20-50% relative
hallucination reduction vs. greedy decoding.

Lemmas's implementation is backend-agnostic. Plug in any callable that takes
a list of messages and returns the assistant text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lemmas.types import CompleteFn, Message


@dataclass
class CoVeStep:
    """One stage of the four-step pipeline. Kept for introspection."""
    name: str            # "baseline" | "plan" | "answer" | "final"
    content: str
    raw_input: str = ""


@dataclass
class CoVeResult:
    final: str                            # the revised answer to return
    baseline: str                         # the unrevised first answer
    questions: list[str]                  # verification questions the model wrote
    answers: list[str]                    # independent answers to each
    steps: list[CoVeStep] = field(default_factory=list)
    revisions: int = 0                    # 0 if final == baseline, else 1


_BASELINE_TEMPLATE = (
    "Answer the user question carefully. Provide your best answer.\n\n"
    "USER: {query}"
)

_PLAN_TEMPLATE = (
    "We're about to fact-check a draft answer to a question. Without "
    "looking at the draft, generate {n} concise verification questions "
    "that would help check the kind of factual claims a good answer would "
    "contain. One per line, no numbering or preamble.\n\n"
    "USER QUESTION: {query}"
)

_ANSWER_TEMPLATE = (
    "Answer this verification question briefly and factually. If you don't "
    "know, say 'unknown'. Output only the answer.\n\n"
    "QUESTION: {question}"
)

_FINAL_TEMPLATE = (
    "Revise the DRAFT ANSWER below so it is consistent with the "
    "VERIFICATION Q&A. If any claim in the draft is contradicted or "
    "unsupported by the verifications, correct or remove it. Keep the "
    "tone and length similar. Output only the revised answer.\n\n"
    "USER QUESTION: {query}\n\n"
    "DRAFT ANSWER:\n{baseline}\n\n"
    "VERIFICATION Q&A:\n{qa_block}"
)


def cove(
    complete: CompleteFn,
    query: str,
    n_questions: int = 4,
    system: str | None = None,
) -> CoVeResult:
    """Run the four-step CoVe pipeline.

    Args:
        complete: callable mapping list[Message] -> str (the LLM).
        query:    the user's question.
        n_questions: how many verification questions to generate.
        system:   optional system prompt prepended to each call.

    Returns:
        CoVeResult with the revised final answer plus intermediate steps.
    """
    steps: list[CoVeStep] = []

    def _ask(user_content: str) -> str:
        msgs: list[Message] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_content})
        return (complete(msgs) or "").strip()

    # 1. Baseline.
    baseline_prompt = _BASELINE_TEMPLATE.format(query=query)
    baseline = _ask(baseline_prompt)
    steps.append(CoVeStep(name="baseline", content=baseline,
                           raw_input=baseline_prompt))

    # 2. Plan verification questions.
    plan_prompt = _PLAN_TEMPLATE.format(query=query, n=n_questions)
    plan_raw = _ask(plan_prompt)
    questions = _split_questions(plan_raw, n_questions)
    steps.append(CoVeStep(name="plan", content=plan_raw,
                           raw_input=plan_prompt))

    # 3. Execute each question independently.
    answers: list[str] = []
    for q in questions:
        a = _ask(_ANSWER_TEMPLATE.format(question=q))
        answers.append(a)
        steps.append(CoVeStep(name="answer", content=a, raw_input=q))

    # 4. Final revision.
    qa_block = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers, strict=False))
    final_prompt = _FINAL_TEMPLATE.format(query=query, baseline=baseline,
                                            qa_block=qa_block)
    final = _ask(final_prompt)
    steps.append(CoVeStep(name="final", content=final,
                           raw_input=final_prompt))

    revisions = 0 if final.strip() == baseline.strip() else 1
    return CoVeResult(final=final, baseline=baseline,
                       questions=questions, answers=answers,
                       steps=steps, revisions=revisions)


def _split_questions(plan_text: str, n_questions: int) -> list[str]:
    """Extract up to n questions from the planner's free-form output."""
    lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
    out: list[str] = []
    for ln in lines:
        # Strip leading numbering or bullets: "1.", "1)", "-", "*".
        ln = re.sub(r"^\s*(?:\d+[\.\)]|[\-\*])\s*", "", ln).strip()
        if not ln:
            continue
        # Keep only up to the first '?' to avoid concatenated sentences.
        if "?" in ln:
            ln = ln.split("?")[0].strip() + "?"
        out.append(ln)
        if len(out) >= n_questions:
            break
    return out[:n_questions]
