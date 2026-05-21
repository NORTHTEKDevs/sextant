"""Race Anthropic Haiku vs OpenAI gpt-4o-mini, return whichever is first.

  pip install sextant[openai,anthropic]
  OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... \
    python examples/hedged_anthropic_vs_openai.py
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY"):
        print("set OPENAI_API_KEY and ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(2)

    from anthropic import Anthropic
    from openai import OpenAI
    from sextant import race
    from sextant.adapters import anthropic_complete, openai_complete

    o_complete = openai_complete(OpenAI(), model="gpt-4o-mini",
                                   temperature=0.3, max_tokens=200)
    a_complete = anthropic_complete(
        Anthropic(), model="claude-haiku-4-5-20251001",
        temperature=0.3, max_tokens=200)

    messages = [{"role": "user", "content": "In one sentence, define entropy."}]
    r = race([
        ("openai-gpt-4o-mini",       lambda: o_complete(messages)),
        ("anthropic-claude-haiku",   lambda: a_complete(messages)),
    ], timeout_secs=15.0)

    print(f"winner: {r.winner}")
    print(f"latency: {r.latency_ms:.0f}ms")
    print(f"value:\n{r.value}")
    print(f"\nlosers:")
    for loser in r.losers:
        print(f"  {loser}")


if __name__ == "__main__":
    main()
