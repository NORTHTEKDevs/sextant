"""Reflexion on a code-generation task: write a function, run unit tests,
retry with the test failure as feedback.

  pip install lemmas[openai]
  OPENAI_API_KEY=sk-... python examples/reflexion_code.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)
    from openai import OpenAI

    from lemmas import programmatic_critic, reflexion
    from lemmas.adapters import openai_complete

    complete = openai_complete(OpenAI(), model="gpt-4o-mini",
                                 temperature=0.3, max_tokens=400)

    query = textwrap.dedent("""\
        Write a Python function `def fizzbuzz(n: int) -> list[str]` that
        returns a list of length n. For each i in 1..n, the element is:
          - "FizzBuzz" if i is divisible by both 3 and 5
          - "Fizz" if i is divisible by 3
          - "Buzz" if i is divisible by 5
          - str(i) otherwise
        Output only the function definition. No commentary, no markdown fences.
    """)

    def extract_code(text: str) -> str:
        # Strip ```python fences if present.
        m = re.search(r"```(?:python)?\s*(.+?)```", text, re.DOTALL)
        return (m.group(1) if m else text).strip()

    def run_tests(candidate: str) -> tuple[bool, str]:
        """Save the candidate, run a tiny test, return (passed, feedback)."""
        code = extract_code(candidate)
        with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False) as f:
            f.write(code)
            f.write("\n\n")
            f.write(textwrap.dedent("""\
                # ---- inline tests ----
                cases = {
                    1: ["1"],
                    3: ["1", "2", "Fizz"],
                    5: ["1", "2", "Fizz", "4", "Buzz"],
                    15: ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz",
                         "Buzz","11","Fizz","13","14","FizzBuzz"],
                }
                for n, expected in cases.items():
                    got = fizzbuzz(n)
                    assert got == expected, f"fizzbuzz({n}) -> {got!r}; expected {expected!r}"
                print("ok")
            """))
            path = f.name
        try:
            r = subprocess.run([sys.executable, path],
                                capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False, "test run timed out (10s)"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if r.returncode == 0 and "ok" in r.stdout:
            return True, "PASS"
        # Pick out the most useful failure signal.
        feedback = (r.stderr or r.stdout).strip()
        # Trim to keep the model's retry prompt focused.
        return False, feedback[-800:]

    result = reflexion(complete, query=query,
                        critic=programmatic_critic(run_tests),
                        max_iterations=4)

    print(f"passed: {result.passed}")
    print(f"iterations: {result.iterations}")
    for i, step in enumerate(result.steps, 1):
        print(f"\n--- attempt {i} (passed={step.critic_passed}) ---")
        print(step.attempt[:400])
        if not step.critic_passed:
            print(f"\n  CRITIC: {step.critic_feedback[:200]}")


if __name__ == "__main__":
    main()
