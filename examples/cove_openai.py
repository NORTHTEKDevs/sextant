"""Chain-of-Verification with OpenAI.

  pip install lemmas[openai]
  OPENAI_API_KEY=sk-... python examples/cove_openai.py
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)
    from openai import OpenAI

    from lemmas import cove
    from lemmas.adapters import openai_complete

    query = " ".join(sys.argv[1:]) or "List five Nobel laureates in physics from the 1970s."
    complete = openai_complete(OpenAI(), model="gpt-4o-mini",
                                 temperature=0.3, max_tokens=512)
    r = cove(complete, query=query, n_questions=4)

    print("BASELINE:")
    print(r.baseline)
    print("\nVERIFICATION:")
    for q, a in zip(r.questions, r.answers, strict=False):
        print(f"  Q: {q}\n  A: {a}")
    print(f"\nFINAL (revisions={r.revisions}):")
    print(r.final)


if __name__ == "__main__":
    main()
