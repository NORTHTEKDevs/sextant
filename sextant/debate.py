"""Multi-agent debate.

Du, Li, Mordatch 2023, arxiv 2305.14325. "Improving Factuality and Reasoning
in Language Models through Multiagent Debate."

The pattern: instead of a single model answering, multiple agents draft
independent answers, then each one revises after seeing the others' drafts
(with the implicit "we disagree, let's reconcile" pressure). After R rounds,
a judge picks the best final answer -- or one is chosen by plurality.

Sextant supports two debate shapes:

  1. Same-model debate (one CompleteFn, different system prompts).
     Cheap and surprisingly effective; the disagreement comes from
     temperature noise + different role priors.

  2. Cross-model debate (list of (label, CompleteFn) agents).
     Stronger: different model families have different blind spots, so
     debate between e.g. Claude + GPT + Gemini covers more ground.

Cost is N_agents * (R + 1) model calls, plus 1 for the judge.

Why ship this as a primitive rather than as a recipe in the README?
Because the round-by-round prompt scaffolding is fiddly to get right and
people mostly reinvent it badly. Plus it composes with `tracer=` for
observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sextant.tracing import Tracer, span
from sextant.types import CompleteFn, Message


@dataclass
class DebateRound:
    round_index: int
    drafts: dict[str, str]                # label -> draft text


@dataclass
class DebateResult:
    final: str                            # the chosen answer
    winner_label: str                     # which agent's last draft won
    rounds: list[DebateRound] = field(default_factory=list)
    judgment: str = ""                    # judge's verbatim verdict (if any)


_FIRST_ROUND_TEMPLATE = (
    "{system_persona}\n\n"
    "Answer this question carefully. Show your reasoning.\n\n"
    "QUESTION: {query}"
)

_REVISE_TEMPLATE = (
    "{system_persona}\n\n"
    "QUESTION: {query}\n\n"
    "Below are answers from other agents in round {prev_round}. Read them, "
    "consider where you agree or disagree, and produce your revised answer. "
    "Be willing to change your mind if another agent's reasoning is "
    "stronger; but if you remain confident, restate your position with "
    "additional support.\n\n"
    "{other_drafts}"
)

_JUDGE_TEMPLATE = (
    "Below are final answers from {n} agents on the same question. Pick "
    "the best one and respond with EXACTLY the agent's label followed by "
    "a colon, then a single-sentence justification. For example:\n"
    "  ALPHA: most rigorous reasoning, cites correct dates\n\n"
    "QUESTION: {query}\n\n{final_block}"
)


def debate(
    query: str,
    agents: list[tuple[str, CompleteFn]] | CompleteFn,
    rounds: int = 2,
    n_agents_if_single: int = 3,
    personas: list[str] | None = None,
    judge: CompleteFn | None = None,
    tracer: Tracer | None = None,
) -> DebateResult:
    """Run a multi-agent debate.

    Args:
        query:    the question to debate.
        agents:   either a list of (label, CompleteFn) pairs, OR a single
                   CompleteFn (in which case we run n_agents_if_single
                   copies of it with different personas).
        rounds:   number of debate rounds beyond the initial draft.
                   rounds=0 -> just initial drafts; rounds=2 -> initial +
                   2 revision passes.
        n_agents_if_single: only used when `agents` is a single CompleteFn.
        personas: optional list of persona strings, one per agent. Defaults
                   to ["You are an analytical reasoner.", "You are a
                   creative thinker.", ...]. Must match the number of agents
                   if provided.
        judge:    optional CompleteFn used to pick a winner. If omitted,
                   the winner is the agent whose final draft is most
                   similar to the others (a "convergence" winner).
        tracer:   optional Tracer for instrumentation.

    Returns:
        DebateResult.
    """
    # Normalize agents into a list of (label, complete) pairs.
    if callable(agents):
        complete = agents
        agents = [
            (f"AGENT_{chr(ord('A') + i)}", complete)
            for i in range(n_agents_if_single)
        ]
    if not agents:
        raise ValueError("debate: at least one agent required")
    if personas is None:
        personas = _default_personas(len(agents))
    if len(personas) != len(agents):
        raise ValueError("personas length must match number of agents")

    with span(tracer, "sextant.debate",
               query=query, n_agents=len(agents), rounds=rounds):
        # Round 0: initial independent drafts.
        drafts: dict[str, str] = {}
        for (label, complete), persona in zip(agents, personas, strict=True):
            with span(tracer, "sextant.debate.draft",
                       agent=label, round=0) as out:
                prompt = _FIRST_ROUND_TEMPLATE.format(
                    system_persona=persona, query=query)
                msgs: list[Message] = [{"role": "user", "content": prompt}]
                drafts[label] = (complete(msgs) or "").strip()
                if out is not None:
                    out["draft_len"] = len(drafts[label])
        recorded_rounds = [DebateRound(round_index=0, drafts=dict(drafts))]

        # Subsequent rounds: each agent sees the others' last drafts.
        for r in range(1, rounds + 1):
            new_drafts: dict[str, str] = {}
            for (label, complete), persona in zip(agents, personas, strict=True):
                with span(tracer, "sextant.debate.revise",
                           agent=label, round=r) as out:
                    other_drafts = "\n\n".join(
                        f"[{ol}]\n{ot}"
                        for ol, ot in drafts.items() if ol != label
                    )
                    prompt = _REVISE_TEMPLATE.format(
                        system_persona=persona, query=query,
                        prev_round=r - 1,
                        other_drafts=other_drafts,
                    )
                    msgs = [{"role": "user", "content": prompt}]
                    new_drafts[label] = (complete(msgs) or "").strip()
                    if out is not None:
                        out["draft_len"] = len(new_drafts[label])
            drafts = new_drafts
            recorded_rounds.append(
                DebateRound(round_index=r, drafts=dict(drafts)))

        # Judgment.
        if judge is not None:
            with span(tracer, "sextant.debate.judge") as out:
                final_block = "\n\n".join(
                    f"[{label}]\n{text}" for label, text in drafts.items())
                judge_prompt = _JUDGE_TEMPLATE.format(
                    n=len(agents), query=query, final_block=final_block)
                verdict = (judge([
                    {"role": "user", "content": judge_prompt}]) or "").strip()
                winner_label = _parse_judge_verdict(verdict, list(drafts))
                if out is not None:
                    out["winner"] = winner_label
            return DebateResult(
                final=drafts[winner_label],
                winner_label=winner_label,
                rounds=recorded_rounds,
                judgment=verdict,
            )

        # No judge: pick the convergence winner.
        winner_label = _convergence_winner(drafts)
        return DebateResult(
            final=drafts[winner_label],
            winner_label=winner_label,
            rounds=recorded_rounds,
            judgment="(convergence-based, no judge)",
        )


def _default_personas(n: int) -> list[str]:
    pool = [
        "You are an analytical reasoner. Be rigorous, cite evidence, and "
        "prefer mathematical or logical arguments.",
        "You are a critical skeptic. Look for missing assumptions, "
        "alternative explanations, and edge cases.",
        "You are a creative thinker. Consider unconventional angles and "
        "lateral connections.",
        "You are a pragmatic engineer. Prefer concrete examples, "
        "operational implications, and trade-offs.",
        "You are a domain historian. Anchor answers in the relevant "
        "history and prior art.",
    ]
    return [pool[i % len(pool)] for i in range(n)]


def _parse_judge_verdict(verdict: str, labels: list[str]) -> str:
    """Look for any label as a prefix or substring. Fallback to first label."""
    upper = verdict.upper()
    for label in labels:
        if upper.startswith(label.upper() + ":"):
            return label
    for label in labels:
        if label.upper() in upper:
            return label
    return labels[0]


def _convergence_winner(drafts: dict[str, str]) -> str:
    """Pick the draft most similar to all the others, by simple token overlap.

    Cheap heuristic that doesn't require embeddings. If you want a real
    semantic-similarity convergence winner, post-process drafts with your
    own embed_fn and pick by centroid distance.
    """
    labels = list(drafts.keys())
    if len(labels) == 1:
        return labels[0]
    scores: dict[str, float] = {}
    tokens = {label: set(text.lower().split()) for label, text in drafts.items()}
    for label, toks in tokens.items():
        total = 0.0
        for other, other_toks in tokens.items():
            if other == label or not other_toks:
                continue
            jaccard = len(toks & other_toks) / max(1, len(toks | other_toks))
            total += jaccard
        scores[label] = total
    return max(scores, key=lambda lbl: scores[lbl])
