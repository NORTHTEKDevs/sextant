# `best_of_n`

The natural companion to `self_consistency`. Where self-consistency uses
*voting*, best-of-N uses a *scorer*:

```python
from sextant import best_of_n, llm_judge_scorer

scorer = llm_judge_scorer(
    judge_complete,
    rubric="Rate this poem 0-10 on imagery, meter, and surprise. Reply with only the number.",
)
r = best_of_n(complete,
               messages=[{"role": "user", "content": "Write a haiku about Anchorage."}],
               scorer=scorer, n=5)
print(r.answer, r.score)
```

## When to use which

|  | self_consistency | best_of_n |
|---|---|---|
| Task has a discrete answer | voting | overkill |
| Task is open-ended | no token to vote on | score each |
| You have a reward model | unused | plug in as `scorer` |

## Built-in scorer factories

| Factory | What it does |
|---|---|
| `llm_judge_scorer(complete, rubric)` | LLM-as-judge; parses the first number from the verdict. |
| `length_scorer(target_chars=500)` | Prefers responses near a target length. |
| `keyword_scorer(keywords, case_sensitive=False)` | +1 per keyword present. |

You can pass any `Callable[[str], float]`.

## Cost

N model calls (+ N more if your scorer is LLM-based). Async variant
`abest_of_n` runs them concurrently.
