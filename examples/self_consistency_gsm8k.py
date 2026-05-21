"""Self-consistency on a GSM8K-style math word problem.

  pip install sextant[openai]
  OPENAI_API_KEY=sk-... python examples/self_consistency_gsm8k.py
"""

from __future__ import annotations

import os
import sys

PROMPT = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
    "morning and bakes muffins with four. She sells the remainder at the "
    "farmers' market for $2 per fresh duck egg. How much in dollars does "
    "she make every day at the farmers' market?\n\n"
    "Think step by step, then end with a line of the form 'Answer: <number>'."
)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)
    from openai import OpenAI

    from sextant import self_consistency
    from sextant.adapters import openai_complete

    # Temperature > 0 is what makes self-consistency work.
    complete = openai_complete(OpenAI(), model="gpt-4o-mini",
                                 temperature=0.7, max_tokens=400)

    r = self_consistency(
        complete,
        messages=[{"role": "user", "content": PROMPT}],
        n=7,
        extractor="last_number",
    )
    print(f"plurality answer: {r.answer}  (confidence={r.confidence:.2%})")
    print("votes:", r.vote_counts)
    print(f"\n# samples ({r.n}):")
    for i, s in enumerate(r.samples):
        print(f"\n--- sample {i+1} (extracted: {r.extracted[i]}) ---")
        print(s)


if __name__ == "__main__":
    main()
