"""Watch a stream of prompts for drift, using OpenAI embeddings.

  pip install sextant[openai]
  OPENAI_API_KEY=sk-... python examples/drift_monitor.py
"""

from __future__ import annotations

import os
import sys


PROMPTS = [
    "What's the weather in Anchorage tomorrow?",
    "Will it snow this weekend in Fairbanks?",
    "Tell me the forecast for Juneau next week.",
    "How cold does it get in Nome in January?",
    "What's the average snowfall in Alaska in December?",
    # ... 15 more weather prompts to warm up ...
    "Anchorage temperature trend last week?",
    "Will I need a jacket in Sitka tomorrow?",
    "Forecast for Kodiak fishing season?",
    "Wind speed prediction for Bethel?",
    "Rain probability in Ketchikan?",
    "Snowpack levels Alyeska resort?",
    "Aurora forecast tonight in Fairbanks?",
    "Tide tables for Homer this weekend?",
    "Will Denali be visible tomorrow morning?",
    "Frost warning for Mat-Su valley?",
    "Air quality in Anchorage today?",
    "Mountain weather pass conditions?",
    "Snow load forecast for the next 7 days?",
    "Will my flight to Adak be cancelled?",
    "What's the chance of fog in Yakutat tomorrow?",
    # Now the topic flips dramatically:
    "Explain how quantum entanglement works",
    "What's the difference between fermions and bosons?",
]


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)
    from openai import OpenAI
    from sextant import DriftDetector
    from sextant.adapters import openai_embed

    embed = openai_embed(OpenAI(), model="text-embedding-3-small")
    detector = DriftDetector(embed_fn=embed, z_threshold=2.0, warmup_n=10)

    print(f"{'#':>3}  {'distance':>10}  {'z':>8}  {'drift?':>6}  prompt")
    print("-" * 80)
    for i, prompt in enumerate(PROMPTS, 1):
        s = detector.observe(bucket="weather-feature", text=prompt)
        flag = "YES" if s.is_drift else ""
        short = prompt[:50] + ("..." if len(prompt) > 50 else "")
        print(f"{i:>3}  {s.distance:>10.4f}  {s.z_score:>8.2f}  {flag:>6}  {short}")


if __name__ == "__main__":
    main()
