"""Depth-aware metrics for context_exhaustion() runs.

score_at_depth:  mean per depth + 95% CI (Wilson if {0,1}; bootstrap otherwise).
score_drop_pp:   sign-aware (baseline_mean - depth_mean) * 100 per non-baseline depth.

Both group by sample_metadata["inspect_tools"]["context_exhaustion"]["target_tokens"];
samples missing that path (or target_tokens=None for literal-mode trials) are dropped.

CIs reflect cross-sample variance only — Inspect's epoch reducer collapses within-sample
variance before metrics run.
"""
from __future__ import annotations

import logging
import math
from typing import Literal, cast

import numpy as np
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Value,
    ValueToFloat,
    metric,
    value_to_float,
)

logger = logging.getLogger(__name__)

Z_95 = 1.959963984540054
_MANIFEST_PATH = ("inspect_tools", "context_exhaustion", "target_tokens")


def _group_by_depth(
    scores: list[SampleScore], to_float: ValueToFloat
) -> dict[int, list[float]]:
    groups: dict[int, list[float]] = {}
    dropped = 0
    for s in scores:
        md = s.sample_metadata
        if md is None:
            dropped += 1
            continue
        node: object = md
        for key in _MANIFEST_PATH:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, int):
            groups.setdefault(node, []).append(to_float(s.score.value))
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "depth-aware metric: dropped %d sample(s) missing "
            "metadata['inspect_tools']['context_exhaustion']['target_tokens']",
            dropped,
        )
    return groups


def _is_binary(values: list[float]) -> bool:
    return bool(values) and all(v == 0.0 or v == 1.0 for v in values)


def _wilson_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    p = sum(values) / n
    z = Z_95
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def _bootstrap_means(values: list[float], rng: np.random.Generator, n_boot: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return arr
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    return arr[idx].mean(axis=1)


def _bootstrap_ci(
    values: list[float], rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    means = _bootstrap_means(values, rng, n_boot)
    if means.size == 1:
        return mean, mean, mean
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi)


def _bootstrap_diff_ci(
    baseline_means: np.ndarray,
    baseline_pt: float,
    depth: list[float],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    diff_pp = (baseline_pt - sum(depth) / len(depth)) * 100.0
    d_means = _bootstrap_means(depth, rng, n_boot)
    diffs = (baseline_means - d_means) * 100.0
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return diff_pp, float(lo), float(hi)


def _ci_for_group(
    values: list[float], rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    if _is_binary(values):
        return _wilson_ci(values)
    return _bootstrap_ci(values, rng, n_boot)


@metric
def score_at_depth(
    num_bootstrap: int = 2000,
    seed: int = 0,
    to_float: ValueToFloat = value_to_float(),
) -> Metric:
    """Mean score per depth cell with 95% CI.

    Groups SampleScores by sample_metadata["inspect_tools"]["context_exhaustion"]
    ["target_tokens"]. Per-group CI: Wilson if every coerced value is in {0, 1},
    otherwise percentile bootstrap with num_bootstrap resamples.

    Returns {"<depth>": mean, "<depth>_ci_low": lo, "<depth>_ci_high": hi,
    "<depth>_n": count, ...} sorted by depth ascending. Empty input → {}.
    """

    def m(scores: list[SampleScore]) -> Value:
        groups = _group_by_depth(scores, to_float)
        if not groups:
            return cast(Value, {})
        rng = np.random.default_rng(seed)
        out: dict[str, float] = {}
        for depth in sorted(groups):
            values = groups[depth]
            mean, lo, hi = _ci_for_group(values, rng, num_bootstrap)
            key = str(depth)
            out[key] = mean
            out[f"{key}_ci_low"] = lo
            out[f"{key}_ci_high"] = hi
            out[f"{key}_n"] = float(len(values))
        return cast(Value, out)

    return m


@metric
def score_drop_pp(
    baseline: int | Literal["shortest"] = "shortest",
    num_bootstrap: int = 2000,
    seed: int = 0,
    to_float: ValueToFloat = value_to_float(),
) -> Metric:
    """Sign-aware percentage-point drop from baseline depth, with bootstrap CI of the difference.

    Drop = (baseline_mean - depth_mean) * 100. Positive = degradation; negative = improvement.

    baseline="shortest" (default) picks the smallest target_tokens observed; an explicit
    int pins the reference depth. If the explicit baseline is absent from the data, returns
    {} with a warning. Baseline depth is omitted from the output.

    Returns {"<depth>": drop_pp, "<depth>_ci_low": lo, "<depth>_ci_high": hi, ...}.
    """

    def m(scores: list[SampleScore]) -> Value:
        groups = _group_by_depth(scores, to_float)
        if not groups:
            return cast(Value, {})
        if baseline == "shortest":
            ref = min(groups)
        else:
            if baseline not in groups:
                logger.warning(
                    "score_drop_pp: baseline=%s not present in data (depths=%s); returning {}",
                    baseline,
                    sorted(groups),
                )
                return cast(Value, {})
            ref = baseline
        ref_values = groups[ref]
        rng = np.random.default_rng(seed)
        ref_pt = sum(ref_values) / len(ref_values)
        ref_means = _bootstrap_means(ref_values, rng, num_bootstrap)
        out: dict[str, float] = {}
        for depth in sorted(groups):
            if depth == ref:
                continue
            drop, lo, hi = _bootstrap_diff_ci(
                ref_means, ref_pt, groups[depth], rng, num_bootstrap
            )
            key = str(depth)
            out[key] = drop
            out[f"{key}_ci_low"] = lo
            out[f"{key}_ci_high"] = hi
        return cast(Value, out)

    return m
