"""Tests for inspect_tools.metrics."""
from __future__ import annotations

import logging

import pytest
from inspect_ai.scorer import SampleScore, Score

from inspect_tools.metrics import score_at_depth, score_drop_pp


def _ss(value: float | int | str, target_tokens: int | None, sample_id: str | int = "s") -> SampleScore:
    md = {
        "inspect_tools": {
            "context_exhaustion": {"target_tokens": target_tokens},
        }
    }
    return SampleScore(
        score=Score(value=value),
        sample_id=sample_id,
        sample_metadata=md,
    )


def _ss_no_manifest(value: float, sample_id: str = "s") -> SampleScore:
    return SampleScore(score=Score(value=value), sample_id=sample_id, sample_metadata={})


def _ss_no_metadata(value: float, sample_id: str = "s") -> SampleScore:
    return SampleScore(score=Score(value=value), sample_id=sample_id, sample_metadata=None)


# === score_at_depth ===


def test_score_at_depth_continuous_linear():
    scores = []
    means = {4000: 0.9, 16000: 0.7, 64000: 0.5, 256000: 0.3}
    n = 24
    for depth, mean in means.items():
        for i in range(n):
            v = mean + (0.1 if i % 2 else -0.1)
            scores.append(_ss(v, depth, sample_id=f"{depth}-{i}"))
    m = score_at_depth(num_bootstrap=200, seed=42)
    out = m(scores)
    for depth, want_mean in means.items():
        key = str(depth)
        assert abs(out[key] - want_mean) < 1e-9
        assert out[f"{key}_ci_low"] <= out[key] <= out[f"{key}_ci_high"]
        assert out[f"{key}_n"] == float(n)


def test_score_at_depth_binary_all_one():
    scores = [_ss(1, 4000, sample_id=f"s{i}") for i in range(20)]
    out = score_at_depth()(scores)
    assert out["4000"] == 1.0
    assert out["4000_ci_low"] < 1.0
    assert out["4000_ci_high"] == 1.0


def test_score_at_depth_binary_all_zero():
    scores = [_ss(0, 4000, sample_id=f"s{i}") for i in range(20)]
    out = score_at_depth()(scores)
    assert out["4000"] == 0.0
    assert out["4000_ci_low"] == 0.0
    assert out["4000_ci_high"] > 0.0


def test_score_at_depth_binary_mixed():
    scores = [_ss(1 if i < 12 else 0, 4000, sample_id=f"s{i}") for i in range(20)]
    out = score_at_depth()(scores)
    assert out["4000"] == 0.6
    assert 0.0 < out["4000_ci_low"] < 0.6 < out["4000_ci_high"] < 1.0


def test_score_at_depth_bootstrap_deterministic():
    scores = [_ss(i / 10.0, 4000, sample_id=f"s{i}") for i in range(10)]
    out1 = score_at_depth(num_bootstrap=500, seed=7)(scores)
    out2 = score_at_depth(num_bootstrap=500, seed=7)(scores)
    assert out1 == out2


def test_score_at_depth_bootstrap_different_seeds_differ():
    scores = [_ss(i / 10.0, 4000, sample_id=f"s{i}") for i in range(10)]
    out_a = score_at_depth(num_bootstrap=500, seed=1)(scores)
    out_b = score_at_depth(num_bootstrap=500, seed=2)(scores)
    assert out_a["4000_ci_low"] != out_b["4000_ci_low"]


def test_score_at_depth_missing_target_tokens_dropped(caplog):
    good = [_ss(1, 4000, sample_id=f"g{i}") for i in range(5)]
    bad = [_ss_no_manifest(1.0, sample_id="b")]
    with caplog.at_level(logging.WARNING, logger="inspect_tools.metrics"):
        out = score_at_depth()(good + bad)
    assert out["4000_n"] == 5.0
    assert any("dropped 1" in rec.message for rec in caplog.records)


def test_score_at_depth_none_target_tokens_dropped():
    good = [_ss(1, 4000, sample_id=f"g{i}") for i in range(5)]
    literal = [_ss(1, None, sample_id="lit")]
    out = score_at_depth()(good + literal)
    assert out["4000_n"] == 5.0


def test_score_at_depth_none_sample_metadata_dropped():
    good = [_ss(1, 4000, sample_id=f"g{i}") for i in range(5)]
    nometa = [_ss_no_metadata(1.0, sample_id="n")]
    out = score_at_depth()(good + nometa)
    assert out["4000_n"] == 5.0


def test_score_at_depth_singleton_binary():
    out = score_at_depth()([_ss(1, 4000)])
    assert out["4000"] == 1.0
    assert out["4000_n"] == 1.0
    assert out["4000_ci_low"] <= 1.0 <= out["4000_ci_high"]


def test_score_at_depth_singleton_continuous():
    out = score_at_depth()([_ss(0.42, 4000)])
    assert out["4000"] == out["4000_ci_low"] == out["4000_ci_high"] == 0.42


def test_score_at_depth_key_naming_no_all_no_prefix():
    scores = [_ss(0.5, 4000), _ss(0.5, 16000)]
    out = score_at_depth()(scores)
    assert set(out.keys()) == {
        "4000", "4000_ci_low", "4000_ci_high", "4000_n",
        "16000", "16000_ci_low", "16000_ci_high", "16000_n",
    }
    assert "all" not in out
    assert not any(k.startswith("depth_") for k in out)


def test_score_at_depth_empty():
    out = score_at_depth()([])
    assert out == {}


# === score_drop_pp ===


def test_score_drop_pp_shortest_baseline():
    scores = (
        [_ss(1, 4000, sample_id=f"a{i}") for i in range(20)]
        + [_ss(1 if i < 16 else 0, 16000, sample_id=f"b{i}") for i in range(20)]
        + [_ss(1 if i < 12 else 0, 64000, sample_id=f"c{i}") for i in range(20)]
    )
    out = score_drop_pp(num_bootstrap=200, seed=0)(scores)
    assert "4000" not in out
    assert abs(out["16000"] - 20.0) < 1e-6
    assert abs(out["64000"] - 40.0) < 1e-6
    for k in ("16000", "64000"):
        assert out[f"{k}_ci_low"] <= out[k] <= out[f"{k}_ci_high"]


def test_score_drop_pp_explicit_baseline():
    scores = (
        [_ss(1, 4000, sample_id=f"a{i}") for i in range(20)]
        + [_ss(1 if i < 16 else 0, 16000, sample_id=f"b{i}") for i in range(20)]
        + [_ss(1 if i < 12 else 0, 64000, sample_id=f"c{i}") for i in range(20)]
    )
    out = score_drop_pp(baseline=16000, num_bootstrap=200, seed=0)(scores)
    assert "16000" not in out
    assert abs(out["4000"] - (-20.0)) < 1e-6
    assert abs(out["64000"] - 20.0) < 1e-6


def test_score_drop_pp_baseline_absent(caplog):
    scores = [_ss(1, 4000, sample_id=f"a{i}") for i in range(5)]
    with caplog.at_level(logging.WARNING, logger="inspect_tools.metrics"):
        out = score_drop_pp(baseline=99999)(scores)
    assert out == {}
    assert any("baseline=99999" in rec.message for rec in caplog.records)


def test_score_drop_pp_sign_aware_negative():
    scores = (
        [_ss(0, 4000, sample_id=f"a{i}") for i in range(10)]
        + [_ss(1, 16000, sample_id=f"b{i}") for i in range(10)]
    )
    out = score_drop_pp(num_bootstrap=200, seed=0)(scores)
    assert out["16000"] == -100.0


def test_score_drop_pp_deterministic():
    scores = (
        [_ss(0.8, 4000, sample_id=f"a{i}") for i in range(10)]
        + [_ss(0.5 + i * 0.05, 16000, sample_id=f"b{i}") for i in range(10)]
    )
    out1 = score_drop_pp(num_bootstrap=500, seed=11)(scores)
    out2 = score_drop_pp(num_bootstrap=500, seed=11)(scores)
    assert out1 == out2


def test_score_drop_pp_empty():
    assert score_drop_pp()([]) == {}


def test_score_drop_pp_only_baseline():
    scores = [_ss(1, 4000, sample_id=f"a{i}") for i in range(5)]
    out = score_drop_pp()(scores)
    assert out == {}


# === binary detection ===


def test_binary_detection_continuous_path():
    scores = [_ss(v, 4000, sample_id=f"s{i}") for i, v in enumerate([0.0, 1.0, 0.5])]
    out = score_at_depth(num_bootstrap=200, seed=0)(scores)
    assert abs(out["4000"] - 0.5) < 1e-9
    assert out["4000_ci_low"] != out["4000_ci_high"]


def test_binary_detection_int_path():
    scores = [_ss(v, 4000, sample_id=f"s{i}") for i, v in enumerate([0, 1, 1, 0])]
    out = score_at_depth()(scores)
    assert abs(out["4000"] - 0.5) < 1e-9
    assert 0.0 < out["4000_ci_low"] < 0.5 < out["4000_ci_high"] < 1.0
