# `reflexion` -- Iterative Self-Improvement

> Shinn et al. 2023, [arxiv:2303.11366](https://arxiv.org/abs/2303.11366) (Northeastern + MIT + Princeton)

A loop: attempt -> critique -> retry. Different from CoVe (one-shot plan +
verify + revise), Reflexion iterates until a critic accepts the answer or
a max-iterations cap is hit. The critique becomes part of the next prompt
so the model learns from its mistakes within the same conversation.

```python
from sextant import reflexion, programmatic_critic

def run_tests(candidate: str) -> tuple[bool, str]:
    # Save the candidate code, run pytest, return (passed, feedback).
    ...

r = reflexion(complete, query="Write fizzbuzz in Python.",
               critic=programmatic_critic(run_tests),
               max_iterations=4)
print(r.final)
print("iterations:", r.iterations, "passed:", r.passed)
```

## Critic factories

| Factory | Use case |
|---|---|
| `llm_critic(complete, rubric)` | LLM-as-judge. Passes when verdict contains "PASS". |
| `programmatic_critic(fn)` | Wrap your own `(candidate) -> (passed, feedback)`. |
| `json_schema_critic(schema)` | Validate the candidate is JSON matching a schema. |

## When Reflexion beats `best_of_n`

When you have a **verifiable signal**: unit tests, exact-match against a
known answer, a JSON schema, a deterministic checker. In those cases
Reflexion is strictly stronger because the critic's feedback actually
guides the retry -- best-of-N samples blindly.

## Cost

Up to `max_iterations` attempts + critic calls. Stops early on first pass.

## Schema-validated structured output

```python
from sextant import reflexion, json_schema_critic

schema = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
}
r = reflexion(complete,
               query="Generate JSON for a person named Alice, 32 years old.",
               critic=json_schema_critic(schema),
               max_iterations=3)
```
