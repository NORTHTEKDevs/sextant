# Adapters

Adapters turn provider SDK clients into lemmas `CompleteFn` / `EmbedFn`
callables. Lemmas's core has zero provider dependencies; each adapter is
opt-in via extras (e.g. `pip install lemmas[openai]`).

## OpenAI

```python
from openai import OpenAI
from lemmas.adapters import openai_complete, openai_embed

complete = openai_complete(OpenAI(), model="gpt-4o-mini",
                            temperature=0.7, max_tokens=1024)
embed    = openai_embed(OpenAI(), model="text-embedding-3-small")
```

## Anthropic

```python
from anthropic import Anthropic
from lemmas.adapters import anthropic_complete

complete = anthropic_complete(Anthropic(), model="claude-haiku-4-5-20251001",
                                temperature=0.3, max_tokens=512)
```

The adapter lifts your `role: "system"` messages into Anthropic's
top-level `system` parameter for you.

## Gemini

```python
import google.generativeai as genai
genai.configure(api_key=...)
from lemmas.adapters import gemini_complete

# Either pass the module (we'll construct the model)...
complete = gemini_complete(genai, model="gemini-1.5-flash")
# ...or pass a pre-built GenerativeModel.
complete = gemini_complete(genai.GenerativeModel("gemini-1.5-flash"))
```

## Groq

```python
from groq import Groq
from lemmas.adapters import groq_complete

complete = groq_complete(Groq(), model="llama-3.1-8b-instant")
```

## OpenAI-compatible URL (zero SDK)

For vLLM, llama.cpp's server, Together, Fireworks, DeepSeek, Perplexity,
LM Studio, Ollama (`/v1` mode), Anyscale, or any homegrown OpenAI-shaped
endpoint:

```python
from lemmas.adapters import openai_compatible_complete, openai_compatible_embed

complete = openai_compatible_complete(
    base_url="http://localhost:8000",   # vLLM default
    api_key="",                          # often unused for local
    model="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.7,
)
embed = openai_compatible_embed(
    base_url="http://localhost:11434",  # Ollama
    model="nomic-embed-text",
)
```

This adapter uses `urllib` directly -- no SDK dependency.

## Test stubs

For unit tests and offline demos:

```python
from lemmas.adapters import echo_complete, varying_echo_complete

stub  = echo_complete()                                      # deterministic
cycle = varying_echo_complete(["A", "B", "C"])              # round-robins
```

## Roll your own

Three lines:

```python
def my_complete(messages: list[dict]) -> str:
    response = my_client.chat(messages=messages)
    return response.text
```

That's the whole interface.
