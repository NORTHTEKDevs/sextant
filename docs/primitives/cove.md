# `cove` -- Chain-of-Verification

> Dhuliawala et al. 2023, [arxiv:2309.11495](https://arxiv.org/abs/2309.11495) (Meta AI)

Four-step pipeline that reduces hallucination on long-form factual answers:

1. **Baseline.** Model produces an initial answer.
2. **Plan.** Model generates N verification questions about the kind of
   claims a good answer would contain. Questions are generated *without
   seeing the baseline* so they stay unbiased.
3. **Execute.** Each verification question is answered independently --
   bad claims can't verify each other.
4. **Revise.** Model rewrites the baseline using the Q/A pairs, correcting
   or removing anything contradicted by the verifications.

```python
from sextant import cove

r = cove(complete, query="Who won Best Picture at the 1995 Academy Awards?",
          n_questions=4)
print(r.final)
print(r.questions)
print(r.answers)
```

## Cost

N+2 model calls. For N=4 on Claude Haiku, ~6 calls per query.
Wins on: TriviaQA, WikiData, biographies, multi-fact questions.
Doesn't help: math, code, single-fact lookups.

## Result shape

```python
@dataclass
class CoVeResult:
    final: str            # revised answer
    baseline: str         # original first answer
    questions: list[str]
    answers: list[str]
    steps: list[CoVeStep] # per-step audit trail
    revisions: int        # 0 if final == baseline, else 1
```
