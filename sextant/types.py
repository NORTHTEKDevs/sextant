"""Shared type aliases.

Sextant intentionally avoids depending on any provider SDK. Instead it
defines two thin callable interfaces that users wrap their LLM client in:

  CompleteFn:  (messages: list[Message]) -> str
  EmbedFn:     (texts: list[str]) -> np.ndarray of shape (n, d)

The `sextant.adapters` submodule provides ready-made factories for
OpenAI / Anthropic / Gemini that turn a client + model id into one of
these callables.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict


class Message(TypedDict, total=False):
    """OpenAI-shaped chat message."""
    role: str       # "system" | "user" | "assistant"
    content: str
    name: str
    tool_call_id: str


CompleteFn = Callable[[list[Message]], str]
"""Take a list of messages, return the assistant's text reply."""

EmbedFn = Callable[[list[str]], Any]
"""Take a list of strings, return a numpy.ndarray of shape (n, d)."""

AsyncCompleteFn = Callable[[list[Message]], Awaitable[str]]
"""Async equivalent of CompleteFn."""

AsyncEmbedFn = Callable[[list[str]], Awaitable[Any]]
"""Async equivalent of EmbedFn."""
