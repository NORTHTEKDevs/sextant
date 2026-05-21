# Quickstart

```bash
pip install lemmas            # core
pip install lemmas[openai]    # + OpenAI adapter
pip install lemmas[anthropic] # + Anthropic adapter
pip install lemmas[gemini]    # + Gemini adapter
pip install lemmas[groq]      # + Groq adapter
pip install lemmas[all]       # everything
```

## The 30-second tour

```python
from lemmas import cove, self_consistency, reflexion, debate, race
from lemmas.adapters import openai_complete
from openai import OpenAI

complete = openai_complete(OpenAI(), model="gpt-4o-mini", temperature=0.7)

# Factual answers, hallucination-reduced:
print(cove(complete, query="List 5 mountains in North America.").final)

# Math/logic, plurality-voted:
result = self_consistency(complete,
                            messages=[{"role": "user", "content": "What is 2+2?"}],
                            n=5, extractor="last_number")
print(result.answer, result.confidence)

# Iterative self-improvement with a verifiable critic:
result = reflexion(complete, query="Write fizzbuzz in Python.",
                    critic=my_test_runner)
print(result.final, "passed:", result.passed)
```

## Bring your own LLM

If you don't use OpenAI/Anthropic/Gemini/Groq, just write a 3-line adapter:

```python
def my_complete(messages: list[dict]) -> str:
    response = my_inference_server.do_chat(messages=messages, ...)
    return response.text
```

Or use the **zero-SDK** path for any OpenAI-compatible server:

```python
from lemmas.adapters import openai_compatible_complete

complete = openai_compatible_complete(
    base_url="http://localhost:8000",   # vLLM, llama.cpp, LM Studio, Ollama /v1
    api_key="...",                       # may be empty
    model="my-model",
)
```

## Test offline

Every primitive ships with a deterministic stub for testing without API keys:

```python
from lemmas.adapters import echo_complete, varying_echo_complete

complete = echo_complete()
cove(complete, query="anything").final            # deterministic
self_consistency(varying_echo_complete(["A","A","B"]),
                  messages=[{"role":"user","content":"?"}], n=3)
```

## CLI

Quick experimentation without code:

```bash
python -m lemmas cove "Who invented the laser?"
python -m lemmas self_consistency "What's 2+2?" --n 5
python -m lemmas best_of_n "Write a haiku" --keywords "snow,arctic"
python -m lemmas drift "hello world"   # needs OPENAI_API_KEY
```
