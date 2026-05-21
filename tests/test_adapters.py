"""Tests for the provider adapters in lemmas.adapters.

The tests don't call real APIs. They construct fake SDK clients that
implement the same interface and verify the adapter wires the calls
through correctly.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---- openai_complete -----------------------------------------------------

def test_openai_complete_wires_chat_completions():
    from lemmas.adapters import openai_complete

    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi there"))],
    )
    fn = openai_complete(client, model="gpt-4o-mini",
                          temperature=0.3, max_tokens=128)
    out = fn([{"role": "user", "content": "hello"}])
    assert out == "hi there"
    args, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 128
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]


# ---- openai_embed --------------------------------------------------------

def test_openai_embed_returns_numpy_array():
    import numpy as np

    from lemmas.adapters import openai_embed
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0]),
              SimpleNamespace(embedding=[4.0, 5.0, 6.0])],
    )
    fn = openai_embed(client, model="text-embedding-3-small")
    out = fn(["a", "b"])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)


# ---- anthropic_complete --------------------------------------------------

def test_anthropic_complete_lifts_system_message():
    from lemmas.adapters import anthropic_complete

    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="answer")],
    )
    fn = anthropic_complete(client, model="claude-haiku-4-5",
                              temperature=0.2, max_tokens=200)
    out = fn([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ])
    assert out == "answer"
    _args, kwargs = client.messages.create.call_args
    # System lifted out of messages.
    assert "system" in kwargs
    assert "be brief" in kwargs["system"]
    # User message preserved.
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


# ---- gemini_complete -----------------------------------------------------

def test_gemini_complete_flattens_history():
    from lemmas.adapters import gemini_complete

    # SimpleNamespace lacks GenerativeModel attr, so adapter uses it as the model directly.
    fake_model = SimpleNamespace(
        generate_content=MagicMock(return_value=SimpleNamespace(text="ok")))
    fn = gemini_complete(fake_model, model="gemini-1.5-flash")
    out = fn([
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
    ])
    assert out == "ok"
    args, kwargs = fake_model.generate_content.call_args
    prompt = args[0]
    assert "SYSTEM:" in prompt
    assert "rules" in prompt
    assert "USER:" in prompt
    assert "hello" in prompt
    cfg = kwargs.get("generation_config") or {}
    assert "temperature" in cfg
    assert "max_output_tokens" in cfg


def test_gemini_complete_accepts_module_form():
    """If `client` is the genai module, the adapter constructs a model."""
    from lemmas.adapters import gemini_complete

    fake_model = MagicMock()
    fake_model.generate_content.return_value = SimpleNamespace(text="x")
    fake_module = MagicMock()
    fake_module.GenerativeModel.return_value = fake_model
    fn = gemini_complete(fake_module, model="gemini-1.5-flash")
    out = fn([{"role": "user", "content": "hi"}])
    fake_module.GenerativeModel.assert_called_once_with("gemini-1.5-flash")
    assert out == "x"


# ---- groq_complete -------------------------------------------------------

def test_groq_complete_uses_openai_shape():
    from lemmas.adapters import groq_complete

    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="from groq"))],
    )
    fn = groq_complete(client, model="llama-3.1-8b-instant")
    assert fn([{"role": "user", "content": "hi"}]) == "from groq"


# ---- openai_compatible_complete ------------------------------------------

def test_openai_compatible_complete_http_call():
    from lemmas.adapters import openai_compatible_complete

    response_body = json.dumps({
        "choices": [{"message": {"content": "compat response"}}]
    }).encode()

    def fake_urlopen(_req, timeout=None):
        # Return a context-manager-shaped object with .read()
        cm = MagicMock()
        cm.__enter__.return_value = SimpleNamespace(read=lambda: response_body)
        cm.__exit__ = MagicMock(return_value=None)
        return cm

    fn = openai_compatible_complete(
        base_url="http://localhost:8000",
        api_key="test-key",
        model="my-model",
    )
    with patch("urllib.request.urlopen", fake_urlopen):
        out = fn([{"role": "user", "content": "x"}])
    assert out == "compat response"


def test_openai_compatible_complete_handles_trailing_slash():
    """The adapter should normalize trailing slashes in base_url."""
    from lemmas.adapters import openai_compatible_complete

    response_body = json.dumps({
        "choices": [{"message": {"content": "y"}}]
    }).encode()

    captured_url = {}

    def fake_urlopen(req, timeout=None):
        captured_url["url"] = req.full_url
        cm = MagicMock()
        cm.__enter__.return_value = SimpleNamespace(read=lambda: response_body)
        cm.__exit__ = MagicMock(return_value=None)
        return cm

    fn = openai_compatible_complete(
        base_url="http://localhost:8000/", api_key="k", model="m")
    with patch("urllib.request.urlopen", fake_urlopen):
        fn([{"role": "user", "content": "z"}])
    # No double slash.
    assert "//v1" not in captured_url["url"]
    assert captured_url["url"].endswith("/v1/chat/completions")


# ---- openai_compatible_embed ---------------------------------------------

def test_openai_compatible_embed():
    import numpy as np

    from lemmas.adapters import openai_compatible_embed
    response_body = json.dumps({
        "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}],
    }).encode()

    def fake_urlopen(_req, timeout=None):
        cm = MagicMock()
        cm.__enter__.return_value = SimpleNamespace(read=lambda: response_body)
        cm.__exit__ = MagicMock(return_value=None)
        return cm

    fn = openai_compatible_embed(
        base_url="http://localhost:11434", model="nomic-embed")
    with patch("urllib.request.urlopen", fake_urlopen):
        out = fn(["a", "b"])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
