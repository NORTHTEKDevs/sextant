"""Lemmas works without any provider SDK. This example uses only the
echo stubs from lemmas.adapters so it runs offline.

  python examples/zero_deps_demo.py
"""

from __future__ import annotations

import time

from lemmas import cove, race, self_consistency
from lemmas.adapters import echo_complete, varying_echo_complete


def demo_cove() -> None:
    print("=== CoVe with echo stub ===")
    r = cove(echo_complete(), query="What year did the Berlin Wall fall?",
              n_questions=3)
    print(f"  questions extracted: {len(r.questions)}")
    print(f"  baseline length:     {len(r.baseline)}")
    print(f"  final length:        {len(r.final)}")


def demo_self_consistency() -> None:
    print("\n=== Self-consistency with varying-echo stub ===")
    suffixes = ["Answer: 42", "Answer: 42", "Answer: 7"]
    r = self_consistency(varying_echo_complete(suffixes),
                          messages=[{"role": "user", "content": "Q?"}],
                          n=3, extractor="last_line")
    print(f"  plurality answer:    {r.answer!r}")
    print(f"  confidence:          {r.confidence:.2f}")
    print(f"  vote counts:         {r.vote_counts}")


def demo_race() -> None:
    print("\n=== Hedged race ===")

    def slow():
        time.sleep(0.3)
        return "slow result"

    def fast():
        time.sleep(0.05)
        return "fast result"

    r = race([("slow", slow), ("fast", fast)])
    print(f"  winner:              {r.winner}")
    print(f"  value:               {r.value}")
    print(f"  latency:             {r.latency_ms:.0f}ms")


if __name__ == "__main__":
    demo_cove()
    demo_self_consistency()
    demo_race()
