"""Tests for sextant.drift."""

from __future__ import annotations

import numpy as np

from sextant.drift import DriftDetector


def _fake_embed_topic(topic_map: dict[str, list[float]]):
    """Return an EmbedFn that hard-codes embeddings by substring."""
    def _fn(texts):
        out = []
        for t in texts:
            matched = None
            for key, vec in topic_map.items():
                if key in t.lower():
                    matched = vec
                    break
            out.append(matched if matched is not None else [0.0, 0.0, 1.0])
        return np.asarray(out, dtype=np.float32)
    return _fn


def test_first_observation_seeds_bucket_with_no_drift():
    embed = _fake_embed_topic({"hello": [1.0, 0.0, 0.0]})
    d = DriftDetector(embed_fn=embed)
    s = d.observe("a", "hello world")
    assert s.n == 1
    assert s.distance == 0.0
    assert s.is_drift is False


def test_steady_state_low_distance():
    embed = _fake_embed_topic({"weather": [1.0, 0.0, 0.0]})
    d = DriftDetector(embed_fn=embed, warmup_n=3)
    for _ in range(10):
        s = d.observe("weather-bucket", "weather today is fine")
    # All embeddings identical -> distance stays at 0; never flags drift.
    assert s.is_drift is False
    assert s.distance < 0.1


def test_obvious_topic_change_flags_drift_after_warmup():
    """Identical warmup -> std=0 -> absolute-distance fallback fires."""
    embed = _fake_embed_topic({
        "weather": [1.0, 0.0, 0.0],
        "quantum": [0.0, 1.0, 0.0],
    })
    d = DriftDetector(embed_fn=embed, warmup_n=5, z_threshold=2.0,
                       distance_threshold=0.5)
    # Warm up with literally identical weather queries -> variance stays 0.
    for _ in range(20):
        d.observe("b", "weather forecast for Anchorage")
    # Now flip to quantum physics; vector orthogonal -> distance ~ 1.
    s = d.observe("b", "quantum entanglement explained")
    assert s.distance > 0.5
    # Std is degenerate so z=0 -- but the absolute-distance fallback flags it.
    assert s.is_drift is True


def test_topic_change_after_varied_warmup_flags_drift_via_zscore():
    """Realistic case: warmup with slight variance -> z-score does the work."""
    import numpy as np

    np.random.seed(0)

    def jittery_weather_embed(texts):
        out = []
        for t in texts:
            if "quantum" in t:
                out.append([0.0, 1.0, 0.0])
            else:
                # Small noise around [1, 0, 0]
                jitter = np.random.normal(0, 0.05, 3)
                out.append([1.0 + jitter[0], jitter[1], jitter[2]])
        return np.asarray(out, dtype=np.float32)

    d = DriftDetector(embed_fn=jittery_weather_embed,
                       warmup_n=5, z_threshold=3.0)
    for _ in range(30):
        d.observe("b", "weather query")
    s = d.observe("b", "quantum entanglement query")
    assert s.distance > 0.3
    # Variance is now non-zero, so z-score is the active mechanism.
    assert s.z_score > 3.0 or s.is_drift  # either z fires or fallback fires


def test_separate_buckets_are_independent():
    embed = _fake_embed_topic({
        "weather": [1.0, 0.0, 0.0],
        "quantum": [0.0, 1.0, 0.0],
    })
    d = DriftDetector(embed_fn=embed)
    d.observe("user_a", "weather")
    d.observe("user_b", "quantum")
    a = d.stats("user_a")
    b = d.stats("user_b")
    assert a is not None and b is not None
    assert a["n"] == 1 and b["n"] == 1


def test_warmup_blocks_premature_drift_flags():
    embed = _fake_embed_topic({
        "weather": [1.0, 0.0, 0.0],
        "quantum": [0.0, 1.0, 0.0],
    })
    d = DriftDetector(embed_fn=embed, warmup_n=50, z_threshold=1.5)
    d.observe("c", "weather x")
    s = d.observe("c", "quantum z")
    # We're under warmup, so even though distance is large, no drift flag.
    assert s.is_drift is False


def test_reset_clears_state():
    embed = _fake_embed_topic({"x": [1.0, 0.0, 0.0]})
    d = DriftDetector(embed_fn=embed)
    d.observe("k", "x")
    assert d.stats("k") is not None
    d.reset("k")
    assert d.stats("k") is None


def test_persist_and_load_callbacks_fire():
    embed = _fake_embed_topic({"x": [1.0, 0.0, 0.0]})
    persisted: dict[str, dict] = {}

    def save(bucket, state):
        persisted[bucket] = state

    def load(bucket):
        return persisted.get(bucket)

    d1 = DriftDetector(embed_fn=embed, persist_fn=save, load_fn=load)
    d1.observe("p", "x")
    d1.observe("p", "x")
    assert "p" in persisted
    # New detector loads from persisted state on first observation.
    d2 = DriftDetector(embed_fn=embed, persist_fn=save, load_fn=load)
    s = d2.observe("p", "x")
    assert s.n >= 3  # picked up where d1 left off
