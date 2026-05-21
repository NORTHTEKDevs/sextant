# Quickstart

```bash
pip install sextant            # core
pip install sextant[openai]    # + OpenAI adapter
pip install sextant[anthropic] # + Anthropic adapter
pip install sextant[gemini]    # + Gemini adapter
pip install sextant[groq]      # + Groq adapter
pip install sextant[all]       # everything
```

## The 30-second tour

```python
from sextant import cove, self_consistency, reflexion, debate, race
from sextant.adapters import openai_complete
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
from sextant.adapters import openai_compatible_complete

complete = openai_compatible_complete(
    base_url="http://localhost:8000",   # vLLM, llama.cpp, LM Studio, Ollama /v1
    api_key="...",                       # may be empty
    model="my-model",
)
```

## Test offline

Every primitive ships with a deterministic stub for testing without API keys:

```python
from sextant.adapters import echo_complete, varying_echo_complete

complete = echo_complete()
cove(complete, query="anything").final            # deterministic
self_consistency(varying_echo_complete(["A","A","B"]),
                  messages=[{"role":"user","content":"?"}], n=3)
```

## CLI

Quick experimentation without code:

```bash
python -m sextant cove "Who invented the laser?"
python -m sextant self_consistency "What's 2+2?" --n 5
python -m sextant best_of_n "Write a haiku" --keywords "snow,arctic"
python -m sextant drift "hello world"   # needs OPENAI_API_KEY
```
