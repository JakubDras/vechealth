"""
interpretation/noise_filter.py

Noise filtering for `compare()` verdicts, by metric reliability category.
`Report.compare()` (rust/report/src/compare.rs) reports `delta_pct`
identically for all metrics — it cannot tell "hubness_max jumped 40%
because that is this metric's ordinary noise" from "hubness_max jumped 40%
because something really broke". This module adds that layer, without
changing the Rust `compare()` itself (which deliberately stays "raw") and
without depending on any sampling of the input: it operates only on
already-computed values from two `Report`s.

Threshold source: the subsampling study of the preprint (sixty subsamples of
the baseline corpus, Table 31) — the coefficient of variation CV =
std_across_draws / mean_sampled_value at each sampling fraction. See
`_CATEGORY_NOISE_FLOOR_PCT` below for the derivation of the numbers.

IMPORTANT METHODOLOGICAL CAVEAT: the study measured CV under subsampling
(repeated draws of subsets from the same, fixed dataset), not the variance
between two INDEPENDENT full data snapshots over time (the typical
`compare()` use case). Using CV@50% (the largest fraction tested, closest to
the full-dataset scale) as a proxy for the latter is a reasonable but NOT
identical approximation — both quantities measure "how much this metric
naturally fluctuates between two comparable samples of the same kind of
data", but the study did not literally test two independent full datasets.
Documented deliberately, not hidden.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .metric_profiles import ReliabilityCategory, get_profile

__all__ = [
    "SignificanceVerdict",
    "NoiseAssessment",
    "FLATTEN_KEY_TO_METRIC_NAME",
    "assess_delta_significance",
    "assess_comparison",
]


# `Report.compare()` (Rust) keys `Comparison.deltas` by the flattened
# `"{group}.{field}"` names from `AllMetricsResult.flatten()`
# (rust/report/src/metrics.rs) — NOT the naming convention that
# `metric_profiles.py`'s `METRIC_PROFILES` uses (the statistic names of the
# preprint). This table is the one place bridging the two: each key is the
# raw result field a preprint statistic is read from.
#
# Only these 11 have a `MetricProfile` at all (see metric_profiles.py) —
# every other `Comparison.deltas` key (e.g. `hubness.orphans_fraction`,
# `anisotropy.top1_variance_ratio`, `duplicates.mean_1nn_distance`,
# `intrinsic_dim.median_id`, `qmas.mean_knn_distance`) was not part of the
# subsampling study and has no entry here — `assess_delta_significance`
# treats an unmapped key as UNASSESSED, not as a silent guess.
FLATTEN_KEY_TO_METRIC_NAME: dict[str, str] = {
    "hubness.max_occurrences": "hubness_max",
    "hubness.hubness_skewness": "hubness_skewness",
    "intrinsic_dim.mean_id": "intrinsic_dim_mean",
    "duplicates.ndds_fraction": "ndds_fraction",
    "dispersion.mean_1nn_distance": "dispersion_1nn",
    "dispersion.mean_knn_distance": "dispersion_10nn",
    "anisotropy.mean_vector_norm": "anisotropy_mean_norm",
    "outliers.outlier_fraction": "outlier_fraction",
    "snc.mean_snc_score": "snc_score",
    "qmas.mean_1nn_distance": "qmas_mean_1nn",
    "qmas.orphans_fraction": "qmas_orphans_pct",
}


# Noise floor per reliability category (percentage points of |delta_pct|),
# derived directly from the subsampling study (preprint Table 31): CV at 50%
# sampling — the largest fraction tested there, hence closest to the
# full-dataset scale (CV@100% is by definition uncomputable, since 100% is
# the ground truth itself). For each category the HIGHEST (most
# conservative) CV@50% among its members is taken, so the floor produces no
# false alarms for any metric in the category. The CV values were computed
# by this library (after the float64 numerics fix); category membership is
# rule v1:
#   A (anisotropy_mean_norm):                          CV@50% = 0.05%  -> floor 0.5
#   B (ndds_fraction):                                 CV@50% = 2.8%
#   C (dispersion_1nn/10nn, intrinsic_dim_mean,
#      qmas_mean_1nn, snc_score, hubness_skewness):     max CV@50% = 6.8%  (hubness_skewness)
#   D (hubness_max, qmas_orphans_pct):                  max CV@50% = 30.1% (hubness_max)
# Consequence: under rule v1, hubness_skewness falls into C, so the C floor
# is 6.8 pp for ALL C metrics (conservative: the floor must not produce
# false alarms for any member). Under rule v2 (sensitivity) it would be 1.0.
#
# `_MIN_FLOOR_PCT` sets the lower bound at 0.5pp: the CV table of the study
# has limited precision, so a literal 0.0% (category A) taken as a
# zero-tolerance threshold would mean "any change, however tiny, is
# significant" — 0.5pp is one full order of magnitude above the source
# data's measurement resolution, a purely technical safety margin, not an
# opinion about any specific metric's behavior.
_MIN_FLOOR_PCT = 0.5

_CATEGORY_NOISE_FLOOR_PCT: dict[ReliabilityCategory, float] = {
    ReliabilityCategory.A_STABLE: max(0.0, _MIN_FLOOR_PCT),
    ReliabilityCategory.B_CORRECTABLE: max(2.8, _MIN_FLOOR_PCT),
    ReliabilityCategory.C_CONVERGING: max(6.8, _MIN_FLOOR_PCT),
    ReliabilityCategory.D_UNSTABLE: max(30.1, _MIN_FLOOR_PCT),
    # ReliabilityCategory.UNKNOWN deliberately omitted — no credible
    # measurement in the study (outlier_fraction reads exactly 0 on the
    # healthy baseline corpus, so its MAPE is undefined), hence no specific
    # number can be justified. No entry = always UNASSESSED in assess_*.
}


class SignificanceVerdict(str, Enum):
    LIKELY_NOISE = "likely_noise"    # |delta_pct| within the category's measured noise
    SIGNIFICANT = "significant"      # |delta_pct| exceeds the category's noise floor
    UNASSESSED = "unassessed"        # no credible threshold — see `note`


@dataclass(frozen=True)
class NoiseAssessment:
    flatten_key: str
    canonical_metric_name: Optional[str]
    reliability_category: Optional[ReliabilityCategory]
    verdict: SignificanceVerdict
    noise_floor_pct: Optional[float]
    note: str


def assess_delta_significance(
        flatten_key: str,
        delta_pct: Optional[float],
        multiplier: float = 1.0,
) -> NoiseAssessment:
    """Assess a SINGLE delta from `Comparison.deltas` (key in the
    `"{group}.{field}"` format, see `FLATTEN_KEY_TO_METRIC_NAME`).

    `multiplier` scales the category's noise floor (default 1.0 — the raw
    value measured in the subsampling study, with no extra subjective
    adjustment). Raise it (e.g. 2.0) to require a clearer change before
    something is marked SIGNIFICANT (fewer false alarms, risk of missing a
    weaker signal); lower it (e.g. 0.5) for the opposite trade-off. An explicit parameter
    instead of a built-in opinion — consistent with the rest of this layer
    (`outlier_distance_threshold`, `duplicate_epsilon` are also explicit,
    tunable thresholds, not hardcoded)."""
    canonical_name = FLATTEN_KEY_TO_METRIC_NAME.get(flatten_key)
    if canonical_name is None:
        return NoiseAssessment(
            flatten_key=flatten_key,
            canonical_metric_name=None,
            reliability_category=None,
            verdict=SignificanceVerdict.UNASSESSED,
            noise_floor_pct=None,
            note=(
                f"'{flatten_key}' has no counterpart in metric_profiles.py "
                "— it was not part of the subsampling study, so there is no "
                "basis for a noise threshold. Treat any change of this value "
                "as requiring manual judgement."
            ),
        )

    profile = get_profile(canonical_name)
    category = profile.reliability_category

    if delta_pct is None:
        return NoiseAssessment(
            flatten_key=flatten_key,
            canonical_metric_name=canonical_name,
            reliability_category=category,
            verdict=SignificanceVerdict.UNASSESSED,
            noise_floor_pct=None,
            note=(
                f"'{canonical_name}' had a baseline value of exactly 0.0 — "
                "a percentage change is undefined, so the noise floor "
                "(expressed in percentage points) does not apply here. "
                "Compare `delta` (the absolute value) manually."
            ),
        )

    floor_pct = _CATEGORY_NOISE_FLOOR_PCT.get(category)
    if floor_pct is None:
        return NoiseAssessment(
            flatten_key=flatten_key,
            canonical_metric_name=canonical_name,
            reliability_category=category,
            verdict=SignificanceVerdict.UNASSESSED,
            noise_floor_pct=None,
            note=(
                f"'{canonical_name}' has reliability category "
                f"{category.value!r}: no variance level could be measured "
                "for it under subsampling (outlier_fraction reads exactly 0 on "
                "the healthy corpus used) — no basis for a noise threshold."
            ),
        )

    threshold_pct = floor_pct * multiplier
    if abs(delta_pct) > threshold_pct:
        verdict = SignificanceVerdict.SIGNIFICANT
        note = (
            f"'{canonical_name}' changed by {delta_pct:+.1f}%, above the "
            f"{threshold_pct:.1f}pp noise floor for category "
            f"{category.value} (measured under subsampling) — "
            "likely a real change, not measurement noise."
        )
    else:
        verdict = SignificanceVerdict.LIKELY_NOISE
        note = (
            f"'{canonical_name}' changed by {delta_pct:+.1f}%, within the "
            f"{threshold_pct:.1f}pp noise floor for category "
            f"{category.value} — this may be natural fluctuation between "
            "two data snapshots, not necessarily a real regression."
        )

    return NoiseAssessment(
        flatten_key=flatten_key,
        canonical_metric_name=canonical_name,
        reliability_category=category,
        verdict=verdict,
        noise_floor_pct=threshold_pct,
        note=note,
    )


def assess_comparison(comparison, multiplier: float = 1.0) -> dict[str, NoiseAssessment]:
    """Convenience wrapper over `assess_delta_significance` — takes a
    `vechealth.Comparison` (the result of `Report.compare()`) directly, or
    any object/mapping with a `deltas` attribute/key in the same shape
    (`dict[str, MetricDelta-like object with .delta_pct]`) — the latter
    variant exists mainly for tests, so they don't need a real, compiled
    `Comparison` from Rust.

    Returns a `flatten_key -> NoiseAssessment` mapping, one entry per
    metric in `comparison.deltas` — so metrics skipped by `compare()`
    itself (present in only one of the two `Report`s, see
    `Comparison.warnings`) do not appear here, just as they don't appear
    in `comparison.deltas`."""
    deltas = comparison.deltas if hasattr(comparison, "deltas") else comparison["deltas"]
    return {
        flatten_key: assess_delta_significance(flatten_key, delta.delta_pct, multiplier=multiplier)
        for flatten_key, delta in deltas.items()
    }
