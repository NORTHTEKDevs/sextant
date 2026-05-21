"""Optional adapters that turn provider SDK clients into Lemmas callables.

Lemmas's core deliberately doesn't import any provider SDK. If you've
already wrapped your LLM in a `(messages) -> str` callable, you don't need
this module at all.

These adapters are just thin sugar so the README examples work without
asking the reader to write 20 lines of boilerplate. Each adapter is
gated behind an import-time check; you only need the SDKs you use.
"""

from __future__ import annotations

from typing import Any

from lemmas.types import CompleteFn, EmbedFn, Message


def openai_complete(
    client: Any,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **extra: Any,
) -> CompleteFn:
    """Adapt an openai.OpenAI() client into a Lemmas CompleteFn.

    `extra` is passed straight through to chat.completions.create -- you
    can use it for top_p, seed, tool_choice, etc.
    """
    def _fn(messages: list[Message]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[dict(m) for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        return (resp.choices[0].message.content or "").strip()
    return _fn


def openai_embed(
    client: Any,
    model: str = "text-embedding-3-small",
) -> EmbedFn:
    """Adapt an openai.OpenAI() client into a Lemmas EmbedFn."""
    import numpy as np

    def _fn(texts: list[str]) -> Any:
        resp = client.embeddings.create(model=model, input=texts)
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)
    return _fn


def anthropic_complete(
    client: Any,
    model: str = "claude-haiku-4-5-20251001",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> CompleteFn:
    """Adapt an anthropic.Anthropic() client into a Lemmas CompleteFn.

    Hides Anthropic's separate `system` field: a message with role='system'
    in your input list is lifted out and passed as the top-level `system`.
    """
    def _fn(messages: list[Message]) -> str:
        system_text = ""
        user_msgs = []
        for m in messages:
            if m.get("role") == "system":
                system_text = system_text + "\n" + (m.get("content") or "")
            else:
                user_msgs.append({
                    "role": m.get("role", "user"),
                    "content": m.get("content", ""),
                })
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": user_msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()
        resp = client.messages.create(**kwargs)
        out = []
        for block in getattr(resp, "content", []):
            if getattr(block, "type", "") == "text":
                out.append(block.text)
        return "".join(out).strip()
    return _fn


def echo_complete(prefix: str = "") -> CompleteFn:
    """Deterministic stub for tests / docs examples. Echoes the last user message."""
    def _fn(messages: list[Message]) -> str:
        last = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None)
        if last is None:
            return prefix
        return prefix + (last.get("content") or "")
    return _fn


def varying_echo_complete(suffixes: list[str]) -> CompleteFn:
    """Stub that cycles through suffixes -- useful for testing self_consistency."""
    state = {"i": 0}

    def _fn(messages: list[Message]) -> str:
        i = state["i"] % len(suffixes)
        state["i"] += 1
        last = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None)
        body = (last.get("content") if last else "")
        return f"{body}\n{suffixes[i]}"
    return _fn


# ---- Gemini ----

def gemini_complete(
    client: Any,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
) -> CompleteFn:
    """Adapt a google.generativeai.GenerativeModel(...) call into a CompleteFn.

    Pass `client = google.generativeai.GenerativeModel("gemini-1.5-flash")`
    (already configured via `genai.configure(api_key=...)`), or pass the
    `google.generativeai` module itself and we'll construct one.

    Gemini doesn't use the OpenAI message shape directly; we collapse the
    history to a single text turn (history + role tags) since that's the
    closest robust mapping for stateless calls.
    """
    def _fn(messages: list[Message]) -> str:
        # Build a Gemini-compatible call.
        # If `client` is the module, build a model instance.
        if hasattr(client, "GenerativeModel"):
            gm = client.GenerativeModel(model)
        else:
            gm = client
        # Flatten history into a single prompt.
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "") or ""
            parts.append(f"{role.upper()}: {content}")
        prompt = "\n\n".join(parts)
        resp = gm.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
        return (getattr(resp, "text", "") or "").strip()
    return _fn


def gemini_embed(
    client: Any,
    model: str = "text-embedding-004",
) -> EmbedFn:
    """Adapt google.generativeai.embed_content into an EmbedFn.

    Pass the `google.generativeai` module as `client`.
    """
    import numpy as np

    def _fn(texts: list[str]) -> Any:
        out = []
        for t in texts:
            resp = client.embed_content(model=f"models/{model}", content=t)
            out.append(resp["embedding"])
        return np.asarray(out, dtype=np.float32)
    return _fn


# ---- Groq (OpenAI-compatible API) ----

def groq_complete(
    client: Any,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> CompleteFn:
    """Adapt a groq.Groq() client. Same API shape as OpenAI."""
    def _fn(messages: list[Message]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[dict(m) for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    return _fn


# ---- Generic OpenAI-compatible URL (vLLM, llama.cpp server, Together, etc.) ----

def openai_compatible_complete(
    base_url: str,
    api_key: str = "",
    model: str = "default",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **extra: Any,
) -> CompleteFn:
    """Adapt any OpenAI-compatible HTTP endpoint (no SDK needed).

    Works with vLLM, llama.cpp's server, Together AI, Fireworks, DeepSeek,
    Anyscale, Perplexity, LM Studio, Ollama (with /v1 enabled), etc. As long
    as the server speaks OpenAI's POST /v1/chat/completions JSON shape.

    Uses urllib (zero deps) instead of an SDK so this works in every
    environment Lemmas runs in.
    """
    import json
    import urllib.request

    if base_url.endswith("/"):
        base_url = base_url[:-1]
    url = base_url + "/v1/chat/completions" if not base_url.endswith(
        "/chat/completions") else base_url

    def _fn(messages: list[Message]) -> str:
        body = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
            resp = json.loads(r.read())
        try:
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError):
            return ""
    return _fn


def openai_compatible_embed(
    base_url: str,
    api_key: str = "",
    model: str = "default",
) -> EmbedFn:
    """OpenAI-compatible HTTP embedding endpoint. Zero deps."""
    import json
    import urllib.request

    import numpy as np

    if base_url.endswith("/"):
        base_url = base_url[:-1]
    url = (base_url + "/v1/embeddings"
            if not base_url.endswith("/embeddings") else base_url)

    def _fn(texts: list[str]) -> Any:
        body = {"model": model, "input": texts}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            resp = json.loads(r.read())
        return np.asarray([d["embedding"] for d in resp["data"]],
                           dtype=np.float32)
    return _fn
