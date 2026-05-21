# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

(no unreleased changes)

## [0.4.0] - 2026-05-21

### Added

- **`debate`** primitive -- multi-agent debate (Du, Li, Mordatch 2023,
  arxiv 2305.14325). N agents draft, revise after seeing each other's
  drafts, R rounds. Optional judge or automatic convergence-based winner.
  Same-model debate (different personas) or cross-model debate supported.
- **Tracing instrumentation.** Every primitive accepts an optional
  `tracer=` kwarg. Three built-in implementations: `NoOpTracer` (default,
  zero overhead), `LoggingTracer` (records spans + events in memory),
  `CallbackTracer` (forward to any external sink -- Langfuse, Phoenix,
  OpenTelemetry). The `Tracer` protocol is six methods; bring your own.
- **`json_schema_critic(schema)`** -- Reflexion critic that validates
  output is JSON matching a schema. Uses `jsonschema` if installed,
  minimal type + required-keys fallback otherwise. Strips Markdown fences.
- **Documentation site** at https://NORTHTEKDevs.github.io/lemmas/
  (mkdocs-material, GitHub Pages, auto-published on docs/ changes).

### Stats

- 99 tests passing (up from 72)
- 7 primitives, full sync + async parity (except `debate` async, on the roadmap)
- 5+ provider adapters

## [0.3.0] - 2026-05-21

### Added

- **`reflexion`** primitive -- iterative try -> critique -> retry loop based
  on Shinn et al. 2023 (arxiv 2303.11366). Two critic factories: `llm_critic`
  (LLM-as-judge, passes on "PASS" substring) and `programmatic_critic` (wrap
  your own test function). Strictly stronger than best-of-N when you have a
  verifiable signal -- critic feedback flows into the next attempt.
- **`areflexion`** async variant in `lemmas.asyncio`.
- **Gemini adapter** (`gemini_complete`, `gemini_embed`) for
  `google.generativeai`. Accepts either a module or a `GenerativeModel`.
- **Groq adapter** (`groq_complete`) -- OpenAI-shaped, drop-in.
- **OpenAI-compatible URL adapter** (`openai_compatible_complete`,
  `openai_compatible_embed`) -- zero SDK deps, works with vLLM, llama.cpp,
  Together, Fireworks, DeepSeek, Anyscale, Perplexity, LM Studio,
  Ollama (`/v1`), etc.
- New CLI subcommand: `python -m lemmas cove`, `self_consistency`,
  `best_of_n`, `drift`.
- README badges (CI, license, Python version) and a clear "Releasing to
  PyPI" section documenting the trusted-publisher setup.
- `examples/reflexion_code.py` -- Reflexion with real unit-test feedback
  on a code-generation task.

### Changed

- `test_critic` renamed to `programmatic_critic` (the old name is kept as
  a backwards-compatible alias, with `__test__ = False` set so pytest
  doesn't try to collect it as a test function).
- Adapter coverage: the "any LLM API" claim is now backed by adapters for
  OpenAI, Anthropic, Gemini, Groq, and any OpenAI-compatible HTTP endpoint.

## [0.2.0] - 2026-05-21

### Added

- **`best_of_n`** primitive -- companion to `self_consistency`. Sample N,
  score each via a scorer fn, return the highest-scoring sample. Includes
  three scorer factories: `llm_judge_scorer`, `length_scorer`,
  `keyword_scorer`.
- **Async parity** for every primitive (`lemmas.asyncio` module):
  `acove`, `aself_consistency`, `abest_of_n`, `arace`. The N-sample
  primitives now parallelize via `asyncio.gather`, cutting wall-clock time
  for self-consistency / best-of-N from O(N) to O(1) per concurrent batch.
- **PEP 561 marker** (`lemmas/py.typed`) so mypy users get type checking
  out of the box.
- New scorer interface (`ScoreFn = Callable[[str], float]`) plus an async
  variant.

### Fixed

- **`DriftDetector` degenerate-variance case**: when all warmup
  observations were identical (std == 0), z-score was zero on any new
  observation, even a wildly different one. Now there's an absolute
  cosine-distance fallback (`distance_threshold`, default 0.5) that flags
  drift past warmup when the z-score mechanism is blinded by zero variance.

## [0.1.0] - 2026-05-21

Initial public release.

### Added

- **`cove`** -- Chain-of-Verification, four-step pipeline
  (baseline + plan + independent answers + revise). Based on
  Dhuliawala et al. 2023 (Meta).
- **`self_consistency`** -- plurality voting over N samples with four
  extractors (`last_line`, `last_number`, `regex`, `similarity` via
  semantic centroid). Based on Wang et al. 2022 (Google).
- **`DriftDetector`** -- per-bucket rolling embedding centroid + EMA
  variance + z-score; optional persist/load callbacks for cross-process
  state.
- **`race`** -- generic hedged execution (Dean & Barroso 2013).
- **Adapters** for `openai.OpenAI()`, `openai` embeddings,
  `anthropic.Anthropic()`, plus deterministic stubs (`echo_complete`,
  `varying_echo_complete`).
- Zero-deps offline demo (`examples/zero_deps_demo.py`).
- CI on Python 3.10 / 3.11 / 3.12.
