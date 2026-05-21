"""Sextant -- four reliability primitives for any LLM API.

  cove(...)               Chain-of-Verification (Meta, Dhuliawala 2023)
  self_consistency(...)   Plurality voting over N samples (Wang 2022)
  DriftDetector(...)      Rolling-centroid prompt-drift detection
  race(...)               Hedged execution: race N callables, first wins

All four are backend-agnostic: pass in a `complete_fn` (or in the drift case,
an `embed_fn`) and they work with OpenAI, Anthropic, Gemini, Groq, Mistral,
local llama.cpp, vLLM -- anything that can be wrapped in a callable.

Quick start:

  from sextant import cove
  from sextant.adapters import openai_complete
  from openai import OpenAI

  complete = openai_complete(OpenAI(), model="gpt-4o-mini")
  result = cove(complete, query="Who invented the laser?")
  print(result.final)
"""

from sextant.cove import CoVeResult, CoVeStep, cove
from sextant.drift import DriftDetector, DriftSample
from sextant.hedged import HedgeResult, race
from sextant.self_consistency import SelfConsistencyResult, self_consistency
from sextant.types import CompleteFn, EmbedFn, Message

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Message",
    "CompleteFn",
    "EmbedFn",
    "cove",
    "CoVeResult",
    "CoVeStep",
    "self_consistency",
    "SelfConsistencyResult",
    "DriftDetector",
    "DriftSample",
    "race",
    "HedgeResult",
]
