"""
tests/test_noise_filter.py

Noise filtering in compare() by metric reliability category. Tests
interpretation/noise_filter.py — thresholds derived from the subsampling
study (CV at 50% sampling).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vechealth.interpretation.metric_profiles import ReliabilityCategory
from vechealth.interpretation.noise_filter import (
    SignificanceVerdict,
    assess_delta_significance,
    assess_comparison,
    FLATTEN_KEY_TO_METRIC_NAME,
)


def test_every_mapped_flatten_key_has_a_metric_profile():
    """Regression for typos in FLATTEN_KEY_TO_METRIC_NAME — every value must
    actually exist in METRIC_PROFILES (get_profile would raise KeyError if
    it did not)."""
    from vechealth.interpretation.metric_profiles import get_profile

    for canonical_name in FLATTEN_KEY_TO_METRIC_NAME.values():
        get_profile(canonical_name)  # must not raise


def test_category_a_stable_flags_tiny_change_as_significant():
    """Category A (anisotropy_mean_norm) has the lowest measured noise
    (CV@50%=0.0%, floored to 0.5pp) — even a moderate change should clear
    the threshold."""
    result = assess_delta_significance("anisotropy.mean_vector_norm", delta_pct=2.0)
    assert result.reliability_category == ReliabilityCategory.A_STABLE
    assert result.verdict == SignificanceVerdict.SIGNIFICANT
    assert result.noise_floor_pct == 0.5


def test_category_a_stable_flags_sub_floor_change_as_noise():
    result = assess_delta_significance("anisotropy.mean_vector_norm", delta_pct=0.2)
    assert result.verdict == SignificanceVerdict.LIKELY_NOISE


def test_category_d_unstable_has_much_higher_floor_than_category_a():
    """The core of the whole task: category D (hubness) needs a much larger
    change than category A to signal an alarm. Constants from the
    subsampling study as recomputed by the library, by the rule in
    noise_filter.py (CV@50% hubness_max = 30.1% vs anisotropy = 0.05% ->
    floor 0.5; preprint Table 31)."""
    a = assess_delta_significance("anisotropy.mean_vector_norm", delta_pct=5.0)
    d = assess_delta_significance("hubness.max_occurrences", delta_pct=5.0)
    assert a.verdict == SignificanceVerdict.SIGNIFICANT
    assert d.verdict == SignificanceVerdict.LIKELY_NOISE
    assert d.noise_floor_pct > a.noise_floor_pct
    assert d.noise_floor_pct == 30.1


def test_category_d_unstable_needs_huge_swing_to_be_significant():
    result = assess_delta_significance("hubness.max_occurrences", delta_pct=40.0)
    assert result.verdict == SignificanceVerdict.SIGNIFICANT
    result_below = assess_delta_significance("hubness.max_occurrences", delta_pct=-30.0)
    assert result_below.verdict == SignificanceVerdict.LIKELY_NOISE


def test_category_b_correctable_uses_ndds_floor():
    # CV@50% of ndds_fraction in the subsampling study, via the library, is 2.8%.
    result = assess_delta_significance("duplicates.ndds_fraction", delta_pct=3.0)
    assert result.reliability_category == ReliabilityCategory.B_CORRECTABLE
    assert result.noise_floor_pct == 2.8
    assert result.verdict == SignificanceVerdict.SIGNIFICANT


def test_category_c_converging_uses_max_member_floor():
    """The highest CV@50% in category C sets the shared category threshold —
    checked on another member. Categories follow rule v1: the highest is
    hubness_skewness (6.8%), snc_score only 0.35% (under rule v2 it would be
    qmas_orphans_pct, 1.0%)."""
    result = assess_delta_significance("snc.mean_snc_score", delta_pct=7.0)
    assert result.reliability_category == ReliabilityCategory.C_CONVERGING
    assert result.noise_floor_pct == 6.8
    assert result.verdict == SignificanceVerdict.SIGNIFICANT


def test_unmapped_flatten_key_is_unassessed():
    """A key absent from FLATTEN_KEY_TO_METRIC_NAME (not part of the
    subsampling study) — we do not guess a threshold."""
    result = assess_delta_significance("anisotropy.top1_variance_ratio", delta_pct=50.0)
    assert result.verdict == SignificanceVerdict.UNASSESSED
    assert result.canonical_metric_name is None
    assert result.noise_floor_pct is None


def test_unknown_category_metric_is_unassessed():
    """outlier_fraction has the UNKNOWN category (it reads exactly 0 on the
    healthy baseline corpus, so its MAPE is undefined) — we assign it no
    number."""
    result = assess_delta_significance("outliers.outlier_fraction", delta_pct=100.0)
    assert result.reliability_category == ReliabilityCategory.UNKNOWN
    assert result.verdict == SignificanceVerdict.UNASSESSED
    assert result.noise_floor_pct is None


def test_zero_baseline_delta_pct_none_is_unassessed():
    """compare() in Rust returns delta_pct=None when the baseline was
    exactly 0.0 (percentage change undefined) — must be handled explicitly,
    not crash on comparing None > threshold."""
    result = assess_delta_significance("duplicates.ndds_fraction", delta_pct=None)
    assert result.verdict == SignificanceVerdict.UNASSESSED
    assert "baseline" in result.note.lower() or "0.0" in result.note


def test_multiplier_scales_threshold():
    result_1x = assess_delta_significance("duplicates.ndds_fraction", delta_pct=4.0, multiplier=1.0)
    result_2x = assess_delta_significance("duplicates.ndds_fraction", delta_pct=4.0, multiplier=2.0)
    assert result_1x.verdict == SignificanceVerdict.SIGNIFICANT
    assert result_2x.verdict == SignificanceVerdict.LIKELY_NOISE
    assert result_2x.noise_floor_pct == result_1x.noise_floor_pct * 2.0


class _FakeMetricDelta:
    def __init__(self, delta_pct):
        self.delta_pct = delta_pct


class _FakeComparison:
    def __init__(self, deltas):
        self.deltas = deltas


def test_assess_comparison_covers_every_delta_key():
    comparison = _FakeComparison({
        "anisotropy.mean_vector_norm": _FakeMetricDelta(3.0),
        "hubness.max_occurrences": _FakeMetricDelta(3.0),
        "anisotropy.top1_variance_ratio": _FakeMetricDelta(3.0),
    })
    results = assess_comparison(comparison)
    assert set(results.keys()) == set(comparison.deltas.keys())
    assert results["anisotropy.mean_vector_norm"].verdict == SignificanceVerdict.SIGNIFICANT
    assert results["hubness.max_occurrences"].verdict == SignificanceVerdict.LIKELY_NOISE
    assert results["anisotropy.top1_variance_ratio"].verdict == SignificanceVerdict.UNASSESSED


def test_assess_comparison_accepts_plain_dict_with_deltas_key():
    comparison = {"deltas": {"duplicates.ndds_fraction": _FakeMetricDelta(10.0)}}
    results = assess_comparison(comparison)
    assert results["duplicates.ndds_fraction"].verdict == SignificanceVerdict.SIGNIFICANT


def test_reliability_messages_make_no_false_claims():
    """Message B does not claim the library corrects the result (no code
    does that); message D does not claim the instability concerns mainly
    small samples (D = spread GROWS with N)."""
    from vechealth.interpretation.reliability_messages import explain_metric_reliability
    b = explain_metric_reliability("ndds_fraction")
    assert "automatically corrected" not in b and "does NOT correct" in b
    for m in ("hubness_max", "qmas_orphans_pct"):
        d = explain_metric_reliability(m)
        assert "especially with smaller" not in d and "GROWS" in d


def test_categories_follow_rule_v1():
    """Categories follow rule v1 on the subsampling study as recomputed by
    the library (preprint Table 31)."""
    from vechealth.interpretation.metric_profiles import get_profile
    expected = {"anisotropy_mean_norm": "A", "ndds_fraction": "B", "hubness_max": "D", "qmas_orphans_pct": "D",
                "hubness_skewness": "C", "dispersion_1nn": "C", "dispersion_10nn": "C", "intrinsic_dim_mean": "C",
                "qmas_mean_1nn": "C", "snc_score": "C", "outlier_fraction": "unknown"}
    for m, c in expected.items():
        assert get_profile(m).reliability_category.value == c, m
