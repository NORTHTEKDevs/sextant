"""Tiny benchmark: greedy vs self-consistency on a handful of GSM8K-style
math word problems.

Not a publishable benchmark -- just enough to feel the win on your own model.

  pip install lemmas[openai]
  OPENAI_API_KEY=sk-... python examples/benchmark_math.py
"""

from __future__ import annotations

import os
import re
import sys
import time

PROBLEMS = [
    {
        "q": ("Janet's ducks lay 16 eggs per day. She eats three for "
               "breakfast every morning and bakes muffins with four. She "
               "sells the remainder at the farmers' market for $2 per "
               "fresh duck egg. How much in dollars does she make every "
               "day at the farmers' market?"),
        "a": 18,
    },
    {
        "q": ("A robe takes 2 bolts of blue fiber and half that much white "
               "fiber. How many bolts in total does it take?"),
        "a": 3,
    },
    {
        "q": ("Josh decides to try flipping a house. He buys a house for "
               "$80,000 and then puts in $50,000 in repairs. This "
               "increased the value of the house by 150%. How much profit "
               "did he make?"),
        "a": 70_000,
    },
    {
        "q": ("James decides to run 3 sprints 3 times a week. He runs 60 "
               "meters each sprint. How many total meters does he run a "
               "week?"),
        "a": 540,
    },
    {
        "q": ("Every day Wendi feeds each of her chickens three cups of "
               "mixed chicken feed, containing seeds, mealworms and "
               "vegetables to help keep them healthy. She gives the "
               "chickens their feed in three separate meals. In the "
               "morning, she gives her flock of chickens 15 cups of feed. "
               "In the afternoon, she gives her chickens another 25 cups "
               "of feed. How many cups of feed does she need to give her "
               "chickens in the final meal of the day if the size of "
               "Wendi's flock is 20 chickens?"),
        "a": 20,
    },
]


SYSTEM = (
    "You solve math word problems carefully. Think step by step. End your "
    "response with a final line of the form 'Answer: <number>'."
)


_NUM_RE = re.compile(r"-?\d{1,}(?:,\d{3})*(?:\.\d+)?")


def extract_last_number(s: str) -> int | None:
    matches = _NUM_RE.findall(s or "")
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return int(float(raw))
    except ValueError:
        return None


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)
    from openai import OpenAI

    from lemmas import self_consistency
    from lemmas.adapters import openai_complete

    client = OpenAI()
    greedy = openai_complete(client, model="gpt-4o-mini",
                              temperature=0.0, max_tokens=400)
    sampled = openai_complete(client, model="gpt-4o-mini",
                               temperature=0.7, max_tokens=400)

    greedy_correct = 0
    sc_correct = 0
    t_greedy = 0.0
    t_sc = 0.0

    print(f"{'#':>2}  {'greedy':>7}  {'sc(5)':>7}  {'expected':>8}  question")
    print("-" * 90)
    for i, p in enumerate(PROBLEMS, 1):
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": p["q"]},
        ]

        # Greedy
        t0 = time.monotonic()
        g_text = greedy(msgs)
        t_greedy += time.monotonic() - t0
        g_ans = extract_last_number(g_text)
        g_ok = (g_ans == p["a"])
        greedy_correct += int(g_ok)

        # Self-consistency, N=5
        t0 = time.monotonic()
        sc_result = self_consistency(sampled, messages=msgs, n=5,
                                       extractor="last_number")
        t_sc += time.monotonic() - t0
        try:
            sc_ans = int(float(sc_result.answer.replace(",", "")))
        except (ValueError, AttributeError):
            sc_ans = None
        sc_ok = (sc_ans == p["a"])
        sc_correct += int(sc_ok)

        short = p["q"][:60] + ("..." if len(p["q"]) > 60 else "")
        print(f"{i:>2}  {str(g_ans):>7}{'+' if g_ok else ' '}  "
              f"{str(sc_ans):>7}{'+' if sc_ok else ' '}  "
              f"{p['a']:>8}  {short}")

    n = len(PROBLEMS)
    print("-" * 90)
    print(f"greedy:            {greedy_correct}/{n}  ({greedy_correct/n:.0%})  "
          f"{t_greedy:.1f}s total")
    print(f"self-consistency:  {sc_correct}/{n}  ({sc_correct/n:.0%})  "
          f"{t_sc:.1f}s total")
    print(f"delta:             +{sc_correct - greedy_correct} correct, "
          f"+{t_sc - t_greedy:.1f}s extra time")


if __name__ == "__main__":
    main()
