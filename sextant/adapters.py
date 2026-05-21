"""Optional adapters that turn provider SDK clients into Sextant callables.

Sextant's core deliberately doesn't import any provider SDK. If you've
already wrapped your LLM in a `(messages) -> str` callable, you don't need
this module at all.

These adapters are just thin sugar so the README examples work without
asking the reader to write 20 lines of boilerplate. Each adapter is
gated behind an import-time check; you only need the SDKs you use.
"""

from __future__ import annotations

from typing import Any

from sextant.types import CompleteFn, EmbedFn, Message


def openai_complete(
    client: Any,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **extra: Any,
) -> CompleteFn:
    """Adapt an openai.OpenAI() client into a Sextant CompleteFn.

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
    """Adapt an openai.OpenAI() client into a Sextant EmbedFn."""
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
    """Adapt an anthropic.Anthropic() client into a Sextant CompleteFn.

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
