"""Tiny CLI: `python -m lemmas <subcommand> <query>`.

Subcommands:
  cove "question"                Run Chain-of-Verification.
  self_consistency "question"    Plurality voting over N samples.
  best_of_n "question"           Sample N + length-scorer pick.
  drift "text"                   Observe one prompt in a named bucket.

If OPENAI_API_KEY is set, calls flow through openai/gpt-4o-mini. Otherwise
the offline echo stub is used so you can sanity-check the install.

This CLI is intentionally minimal -- it's an experimentation harness, not a
production tool. Wrap the library directly in real code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _build_complete(model: str, temperature: float, max_tokens: int) -> Any:
    """Return (complete_fn, backend_name)."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            from lemmas.adapters import openai_complete
            return (openai_complete(
                OpenAI(), model=model, temperature=temperature,
                max_tokens=max_tokens), f"openai/{model}")
        except ImportError:
            print("OPENAI_API_KEY set but openai package not installed; "
                  "run `pip install lemmas[openai]`.", file=sys.stderr)
            sys.exit(2)
    from lemmas.adapters import varying_echo_complete
    return (varying_echo_complete(
        ["Answer: 42", "Answer: 42", "Answer: 7",
         "Answer: 42", "Answer: 13"]), "stub")


def _build_embed() -> Any:
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            from lemmas.adapters import openai_embed
            return openai_embed(OpenAI(),
                                 model="text-embedding-3-small")
        except ImportError:
            print("OPENAI_API_KEY set but openai not installed",
                  file=sys.stderr)
            sys.exit(2)
    # No offline embed stub by default; drift CLI needs OPENAI_API_KEY.
    print("drift requires OPENAI_API_KEY (for embeddings)", file=sys.stderr)
    sys.exit(2)


def cmd_cove(args: argparse.Namespace) -> int:
    from lemmas import cove
    complete, backend = _build_complete(args.model, args.temperature,
                                          args.max_tokens)
    if not args.json:
        print(f"backend: {backend}\nquery:   {args.query}\n")
    r = cove(complete, query=args.query, n_questions=args.n_questions)
    if args.json:
        print(json.dumps({
            "final": r.final, "baseline": r.baseline,
            "questions": r.questions, "answers": r.answers,
            "revisions": r.revisions,
        }, indent=2))
        return 0
    print("BASELINE:")
    print(r.baseline)
    print("\nVERIFICATION:")
    for q, a in zip(r.questions, r.answers, strict=False):
        print(f"  Q: {q}\n  A: {a}")
    print(f"\nFINAL (revisions={r.revisions}):")
    print(r.final)
    return 0


def cmd_self_consistency(args: argparse.Namespace) -> int:
    from lemmas import self_consistency
    complete, backend = _build_complete(args.model, args.temperature,
                                          args.max_tokens)
    if not args.json:
        print(f"backend: {backend}\nquery:   {args.query}\n")
    r = self_consistency(
        complete, messages=[{"role": "user", "content": args.query}],
        n=args.n, extractor=args.extractor,
        extractor_regex=args.regex,
    )
    if args.json:
        print(json.dumps({
            "answer": r.answer, "confidence": r.confidence,
            "vote_counts": r.vote_counts, "n": r.n,
            "extracted": r.extracted,
        }, indent=2))
        return 0
    print(f"answer:      {r.answer!r}")
    print(f"confidence:  {r.confidence:.2%}")
    print(f"vote counts: {r.vote_counts}")
    return 0


def cmd_best_of_n(args: argparse.Namespace) -> int:
    from lemmas import best_of_n, keyword_scorer, length_scorer
    complete, backend = _build_complete(args.model, args.temperature,
                                          args.max_tokens)
    if args.keywords:
        scorer = keyword_scorer([k.strip() for k in args.keywords.split(",")])
        scorer_name = f"keyword({args.keywords})"
    else:
        scorer = length_scorer(target_chars=args.target_chars)
        scorer_name = f"length(target={args.target_chars})"
    if not args.json:
        print(f"backend: {backend}\nquery:   {args.query}\n"
              f"scorer:  {scorer_name}\n")
    r = best_of_n(
        complete, messages=[{"role": "user", "content": args.query}],
        scorer=scorer, n=args.n,
    )
    if args.json:
        print(json.dumps({
            "answer": r.answer, "score": r.score,
            "scores": r.scores, "winner_index": r.winner_index,
        }, indent=2))
        return 0
    print(f"winner_idx:  {r.winner_index}")
    print(f"score:       {r.score:.3f}")
    print(f"scores:      {[round(x, 2) for x in r.scores]}")
    print(f"\nwinning sample:\n{r.answer}")
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    from lemmas import DriftDetector
    embed = _build_embed()
    d = DriftDetector(embed_fn=embed,
                       z_threshold=args.z_threshold,
                       warmup_n=args.warmup_n)
    s = d.observe(args.bucket, args.text)
    out = {
        "bucket": s.bucket, "n": s.n,
        "distance": s.distance, "z_score": s.z_score,
        "is_drift": s.is_drift,
    }
    print(json.dumps(out, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    # Shared flags every subcommand accepts.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="gpt-4o-mini",
                         help="model id when OPENAI_API_KEY is set")
    common.add_argument("--temperature", type=float, default=0.7)
    common.add_argument("--max-tokens", type=int, default=512)
    common.add_argument("--json", action="store_true",
                         help="output as JSON instead of human-readable")

    p = argparse.ArgumentParser(
        prog="lemmas",
        description="Reliability primitives for any LLM API.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("cove", parents=[common],
                         help="Chain-of-Verification")
    sp.add_argument("query")
    sp.add_argument("--n-questions", type=int, default=4)
    sp.set_defaults(func=cmd_cove)

    sp = sub.add_parser("self_consistency", parents=[common],
                         help="Plurality voting over N samples")
    sp.add_argument("query")
    sp.add_argument("--n", type=int, default=5)
    sp.add_argument("--extractor", default="last_line",
                     choices=["last_line", "last_number", "regex", "similarity"])
    sp.add_argument("--regex", default=None,
                     help="required if --extractor regex")
    sp.set_defaults(func=cmd_self_consistency)

    sp = sub.add_parser("best_of_n", parents=[common],
                         help="Sample N + scorer")
    sp.add_argument("query")
    sp.add_argument("--n", type=int, default=5)
    sp.add_argument("--target-chars", type=int, default=500,
                     help="for length_scorer (default scorer)")
    sp.add_argument("--keywords", default=None,
                     help="comma-separated keywords to score by presence")
    sp.set_defaults(func=cmd_best_of_n)

    sp = sub.add_parser("drift", parents=[common],
                         help="Observe one prompt in a bucket")
    sp.add_argument("text")
    sp.add_argument("--bucket", default="default")
    sp.add_argument("--z-threshold", type=float, default=3.0)
    sp.add_argument("--warmup-n", type=int, default=20)
    sp.set_defaults(func=cmd_drift)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
