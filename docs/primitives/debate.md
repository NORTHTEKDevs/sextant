# `debate` -- Multi-Agent Debate

> Du, Li, Mordatch 2023, [arxiv:2305.14325](https://arxiv.org/abs/2305.14325)
>
> "Improving Factuality and Reasoning in Language Models through Multiagent Debate"

Multiple agents draft independent answers, then each one revises after
seeing the others' drafts. After R rounds, a judge picks the best -- or
the convergence winner is selected automatically.

## Same-model debate

Cheap: three runs of the same model with different personas, temperature
noise drives the disagreement.

```python
from lemmas import debate
from lemmas.adapters import openai_complete
from openai import OpenAI

complete = openai_complete(OpenAI(), model="gpt-4o-mini", temperature=0.7)

r = debate(
    query="What were the most consequential US Supreme Court decisions of the 20th century?",
    agents=complete,                # single callable -> spawns N agents
    n_agents_if_single=3,
    rounds=2,
)
print(r.final)
print(f"winner: {r.winner_label}")
```

## Cross-model debate (stronger)

Different model families have different blind spots. Race them against
each other:

```python
from lemmas.adapters import openai_complete, anthropic_complete

openai_agent    = openai_complete(OpenAI(), model="gpt-4o-mini")
claude_agent    = anthropic_complete(Anthropic(), model="claude-haiku-4-5")
judge           = openai_complete(OpenAI(), model="gpt-4o", temperature=0)

r = debate(
    query="...",
    agents=[("OPENAI", openai_agent), ("CLAUDE", claude_agent)],
    rounds=2,
    judge=judge,
)
```

## Custom personas

```python
r = debate(
    query="...",
    agents=[("MATHEMATICIAN", complete), ("ENGINEER", complete)],
    personas=[
        "You are a pure mathematician. Prefer formal proofs.",
        "You are a practicing engineer. Prefer concrete examples and trade-offs.",
    ],
    rounds=2,
)
```

## With or without a judge

- **With `judge=`:** The judge model picks the winning agent. Recommended
  when you have a strong "grader" model available.
- **Without `judge=`:** The convergence winner is picked automatically --
  whichever agent's final draft is most similar (by token overlap) to the
  others. Useful when agents tend to agree after debate.

## Cost model

`debate(rounds=R, n=k_agents)` costs **k_agents * (R + 1) + 1** LLM calls
(plus one for the judge). For 3 agents and 2 rounds, that's 10 calls --
expensive but worthwhile for high-stakes factual or analytical tasks.

## Result shape

```python
@dataclass
class DebateResult:
    final: str             # the chosen answer
    winner_label: str      # which agent's final draft won
    rounds: list[DebateRound]
    judgment: str          # judge's verbatim verdict (or convergence note)

@dataclass
class DebateRound:
    round_index: int
    drafts: dict[str, str] # agent_label -> their draft this round
```
