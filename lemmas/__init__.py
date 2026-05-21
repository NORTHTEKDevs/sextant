"""Lemmas -- reliability primitives for any LLM API.

  cove(...)               Chain-of-Verification (Meta, Dhuliawala 2023)
  self_consistency(...)   Plurality voting over N samples (Wang 2022)
  best_of_n(...)          Sample N + scorer (test-time compute)
  reflexion(...)          Try -> critique -> retry (Shinn 2023)
  debate(...)             Multi-agent debate (Du, Li, Mordatch 2023)
  DriftDetector(...)      Rolling-centroid prompt-drift detection
  race(...)               Hedged execution: race N callables, first wins

All backend-agnostic: pass in a `complete_fn` (or `embed_fn` for drift) and
they work with OpenAI, Anthropic, Gemini, Groq, Mistral, vLLM, llama.cpp,
or any OpenAI-compatible URL.

Quick start:

  from lemmas import cove
  from lemmas.adapters import openai_complete
  from openai import OpenAI

  complete = openai_complete(OpenAI(), model="gpt-4o-mini")
  result = cove(complete, query="Who invented the laser?")
  print(result.final)
"""

from lemmas.best_of_n import (
    BestOfNResult,
    ScoreFn,
    best_of_n,
    keyword_scorer,
    length_scorer,
    llm_judge_scorer,
)
from lemmas.cove import CoVeResult, CoVeStep, cove
from lemmas.debate import DebateResult, DebateRound, debate
from lemmas.drift import DriftDetector, DriftSample
from lemmas.hedged import HedgeResult, race
from lemmas.reflexion import (
    CriticFn,
    CriticVerdict,
    ReflexionResult,
    ReflexionStep,
    json_schema_critic,
    llm_critic,
    programmatic_critic,
    reflexion,
    test_critic,
)
from lemmas.self_consistency import SelfConsistencyResult, self_consistency
from lemmas.tracing import (
    CallbackTracer,
    LoggingTracer,
    NoOpTracer,
    Tracer,
    set_default_tracer,
)
from lemmas.types import (
    AsyncCompleteFn,
    AsyncEmbedFn,
    CompleteFn,
    EmbedFn,
    Message,
)

__version__ = "0.4.0"

__all__ = [
    "__version__",
    # types
    "Message",
    "CompleteFn",
    "EmbedFn",
    "AsyncCompleteFn",
    "AsyncEmbedFn",
    # CoVe
    "cove",
    "CoVeResult",
    "CoVeStep",
    # Self-consistency
    "self_consistency",
    "SelfConsistencyResult",
    # Best-of-N
    "best_of_n",
    "BestOfNResult",
    "ScoreFn",
    "llm_judge_scorer",
    "length_scorer",
    "keyword_scorer",
    # Reflexion
    "reflexion",
    "ReflexionResult",
    "ReflexionStep",
    "CriticFn",
    "CriticVerdict",
    "llm_critic",
    "programmatic_critic",
    "json_schema_critic",
    "test_critic",
    # Debate
    "debate",
    "DebateResult",
    "DebateRound",
    # Drift
    "DriftDetector",
    "DriftSample",
    # Hedged execution
    "race",
    "HedgeResult",
    # Tracing
    "Tracer",
    "NoOpTracer",
    "LoggingTracer",
    "CallbackTracer",
    "set_default_tracer",
]
