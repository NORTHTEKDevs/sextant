"""Prompt-drift detection via rolling embedding centroid.

For each `bucket` (e.g. a deployed feature, an API key, an endpoint), keep
a running unit-norm centroid of recent prompt embeddings plus an
exponentially-weighted variance of the centroid-distance.

On each new prompt:

  1. Embed it (unit-normalize).
  2. distance = 1 - cosine(centroid, new)
  3. z = distance / sqrt(variance)
  4. Update centroid + variance via exponential moving average.
  5. Flag is_drift if |z| > z_threshold and bucket has > warmup observations.

Use cases:

  - A customer's traffic shape suddenly changes (abuse, scraping, new use
    case). Worth a Slack ping.
  - A deployed prompt template is being used "wrong" by callers (their
    inputs no longer match the eval set you tuned on).
  - Your eval set has gone stale relative to live traffic.

State lives in memory by default. Pass a `persist_fn` + `load_fn` to
serialize across processes (SQLite, Redis, whatever you want).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from sextant.types import EmbedFn


@dataclass
class DriftSample:
    bucket: str
    distance: float        # 1 - cos(centroid, new)
    z_score: float         # distance / stdev_of_recent_distances
    n: int                 # how many observations this bucket has seen
    is_drift: bool         # |z| > threshold AND past warmup
    updated_at: float = 0.0


@dataclass
class _BucketState:
    centroid: Any          # numpy.ndarray of shape (d,)
    n: int
    variance: float
    last_distance: float
    updated_at: float


class DriftDetector:
    """Maintains per-bucket rolling embedding statistics.

    Args:
        embed_fn:    callable mapping list[str] -> numpy.ndarray (n, d).
        alpha:       EMA factor for centroid + variance updates.
        z_threshold: |z| threshold to flag drift.
        warmup_n:    minimum observations before is_drift can fire.
        persist_fn:  optional. Called as persist_fn(bucket, state_dict) after
                      every observe. Use to write to disk/Redis.
        load_fn:     optional. Called as load_fn(bucket) -> state_dict | None.
                      Used on first observation of an unseen bucket.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        alpha: float = 0.05,
        z_threshold: float = 3.0,
        warmup_n: int = 20,
        distance_threshold: float = 0.5,
        persist_fn: Callable[[str, dict], None] | None = None,
        load_fn: Callable[[str], dict | None] | None = None,
    ) -> None:
        """
        Args:
            embed_fn:           list[str] -> numpy.ndarray of shape (n, d).
            alpha:              EMA factor for centroid + variance updates.
            z_threshold:        |z| threshold to flag drift after warmup.
            warmup_n:           minimum observations before z-flagging kicks in.
            distance_threshold: absolute cosine-distance fallback for the case
                                 where variance is degenerate (e.g., all warmup
                                 prompts were identical). Past warmup, a single
                                 observation with distance > this value flags
                                 drift even if std == 0.
        """
        self.embed_fn = embed_fn
        self.alpha = alpha
        self.z_threshold = z_threshold
        self.warmup_n = warmup_n
        self.distance_threshold = distance_threshold
        self.persist_fn = persist_fn
        self.load_fn = load_fn
        self._buckets: dict[str, _BucketState] = {}
        self._lock = threading.Lock()

    def observe(self, bucket: str, text: str) -> DriftSample:
        import numpy as np
        vec = np.asarray(self.embed_fn([text])[0], dtype=np.float32)
        v = vec / (np.linalg.norm(vec) + 1e-9)
        now = time.time()

        with self._lock:
            state = self._buckets.get(bucket)
            if state is None and self.load_fn is not None:
                loaded = self.load_fn(bucket)
                if loaded is not None:
                    state = _BucketState(
                        centroid=np.asarray(loaded["centroid"], dtype=np.float32),
                        n=int(loaded["n"]),
                        variance=float(loaded["variance"]),
                        last_distance=float(loaded.get("last_distance", 0.0)),
                        updated_at=float(loaded.get("updated_at", now)),
                    )
                    self._buckets[bucket] = state
            if state is None:
                # First observation: centroid = this vector, no distance.
                self._buckets[bucket] = _BucketState(
                    centroid=v, n=1, variance=0.0,
                    last_distance=0.0, updated_at=now,
                )
                self._maybe_persist(bucket)
                return DriftSample(
                    bucket=bucket, distance=0.0, z_score=0.0,
                    n=1, is_drift=False, updated_at=now,
                )

            # Distance to centroid.
            sim = float(np.dot(state.centroid, v))
            distance = max(0.0, 1.0 - sim)

            # z-score against running variance.
            std = math.sqrt(state.variance) if state.variance > 0 else 0.0
            z = (distance / std) if std > 0 else 0.0

            # EMA update.
            new_centroid = (1 - self.alpha) * state.centroid + self.alpha * v
            new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-9)
            new_variance = (1 - self.alpha) * state.variance + self.alpha * (distance ** 2)

            state.centroid = new_centroid
            state.n += 1
            state.variance = new_variance
            state.last_distance = distance
            state.updated_at = now
            self._maybe_persist(bucket)

            # Past warmup, flag drift if either:
            #   (a) z-score exceeds threshold, OR
            #   (b) std is degenerate (~0) AND absolute distance exceeds the
            #       distance fallback. This catches the perfectly-stable
            #       baseline case where any topic shift is meaningful.
            past_warmup = state.n > self.warmup_n
            z_flag = abs(z) > self.z_threshold
            absolute_flag = (std < 1e-6 and distance > self.distance_threshold)
            is_drift = bool(past_warmup and (z_flag or absolute_flag))
            return DriftSample(
                bucket=bucket, distance=distance, z_score=z,
                n=state.n, is_drift=is_drift, updated_at=now,
            )

    def stats(self, bucket: str) -> dict[str, Any] | None:
        with self._lock:
            s = self._buckets.get(bucket)
            if s is None:
                return None
            return {
                "bucket": bucket, "n": s.n,
                "variance": s.variance,
                "last_distance": s.last_distance,
                "updated_at": s.updated_at,
            }

    def reset(self, bucket: str | None = None) -> None:
        """Drop all bucket state (or one bucket)."""
        with self._lock:
            if bucket is None:
                self._buckets.clear()
            else:
                self._buckets.pop(bucket, None)

    def _maybe_persist(self, bucket: str) -> None:
        if self.persist_fn is None:
            return
        state = self._buckets.get(bucket)
        if state is None:
            return
        try:
            self.persist_fn(bucket, {
                "centroid": state.centroid.tolist(),
                "n": state.n,
                "variance": state.variance,
                "last_distance": state.last_distance,
                "updated_at": state.updated_at,
            })
        except Exception:
            # Best-effort. Never raise from telemetry path.
            pass
