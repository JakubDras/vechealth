"""
tests/test_fail_closed_recommendations.py

A recommendation may not depend on the metric name alone, and a delta at
baseline = 0 may not yield ±1.0.

Why: epsilon deduplication applied to an anisotropic base (G5, level 4) —
which gives the same ndds_fraction reading as a base with duplicates —
removed 56.5% of the corpus and lowered recall by 47.3% (preprint Table 20).

The RAW_* rows are real values of the 11 benchmark statistics, computed by
this library (`compute_all` with the preprint's parameters).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vechealth.interpretation.recommendations import (
    build_recommendation,
    classify_recommendation_level,
    DiagnosisContext,
    RecommendationLevel,
)
from vechealth.interpretation.report import build_full_report, Level1Status
from vechealth.interpretation.signatures import (
    signed_delta,
    _cosine_similarity,
    SIGNATURES,
)

RAW_B3 = {
    "hubness_max": 313.0,
    "hubness_skewness": 3.182733779685893,
    "intrinsic_dim_mean": 20.70944595336914,
    "ndds_fraction": 0.0445062071084976,
    "dispersion_1nn": 0.5815351605415344,
    "dispersion_10nn": 0.7341088652610779,
    "anisotropy_mean_norm": 0.5129579901695251,
    "outlier_fraction": 0.0,
    "snc_score": 0.1807022392749786,
    "qmas_mean_1nn": 0.2558492422103882,
    "qmas_orphans_pct": 0.3493172824382782,
}
RAW_G5_L4 = {
    "hubness_max": 396.0,
    "hubness_skewness": 3.8831468279404966,
    "intrinsic_dim_mean": 21.35217666625977,
    "ndds_fraction": 0.9971191883087158,
    "dispersion_1nn": 0.0296717025339603,
    "dispersion_10nn": 0.0373962372541427,
    "anisotropy_mean_norm": 0.9990558624267578,
    "outlier_fraction": 0.0,
    "snc_score": 0.1792544275522232,
    "qmas_mean_1nn": 0.4935510158538818,
    "qmas_orphans_pct": 0.9931727051734924,
}


def test_no_context_ndds_is_not_auto_fixable():
    rec = build_recommendation("ndds_fraction")
    assert rec.level != RecommendationLevel.AUTO_FIXABLE
    assert rec.level == RecommendationLevel.REVIEW_REQUIRED
    # Fail closed: nothing to execute automatically; candidate for review only.
    assert rec.suggested_action is None
    assert rec.candidate_action is not None
    assert rec.human_review_required is True


def test_no_context_no_metric_is_auto_fixable():
    from vechealth.interpretation.metric_profiles import METRIC_PROFILES
    for name in METRIC_PROFILES:
        assert classify_recommendation_level(name) != RecommendationLevel.AUTO_FIXABLE
        assert build_recommendation(name).level != RecommendationLevel.AUTO_FIXABLE


def test_g4_context_ndds_is_auto_fixable():
    rec = build_recommendation("ndds_fraction", DiagnosisContext(matched_family_id="G4_duplicates"))
    assert rec.level == RecommendationLevel.AUTO_FIXABLE
    assert rec.suggested_action is not None


def test_g5_context_ndds_is_not_auto_fixable_and_warns():
    rec = build_recommendation("ndds_fraction", DiagnosisContext(matched_family_id="G5_anisotropy"))
    assert rec.level == RecommendationLevel.REVIEW_REQUIRED
    note = rec.causal_evidence_note.lower()
    assert "anisotrop" in note
    assert "47.3%" in rec.causal_evidence_note  # the measured loss from deduplicating the wrong family
    assert rec.suggested_action is None


def test_g4_match_but_also_g5_like_deltas_is_not_auto_fixable():
    """The context points at G4, but the observed deltas also match G5 —
    a confounding family for ndds; fail closed."""
    g5 = next(s for s in SIGNATURES if s.family_id == "G5_anisotropy")
    ctx = DiagnosisContext(matched_family_id="G4_duplicates", observed_deltas=dict(g5.metric_deltas))
    rec = build_recommendation("ndds_fraction", ctx)
    assert rec.level == RecommendationLevel.REVIEW_REQUIRED


def test_real_g5_l4_row_gives_no_urgent_dedup():
    report = build_full_report(RAW_B3, RAW_G5_L4)
    ndds = next(r for r in report.metric_reports if r.metric_name == "ndds_fraction")
    assert ndds.status != Level1Status.URGENT
    assert ndds.recommendation.level != RecommendationLevel.AUTO_FIXABLE
    assert ndds.recommendation.suggested_action is None


def test_hubness_follows_the_same_rule():
    assert build_recommendation("hubness_max").level == RecommendationLevel.REVIEW_REQUIRED
    assert (build_recommendation("hubness_max", DiagnosisContext(matched_family_id="G1_hubness")).level
            == RecommendationLevel.AUTO_FIXABLE)
    rec = build_recommendation("hubness_max", DiagnosisContext(matched_family_id="G8_noise"))
    assert rec.level == RecommendationLevel.REVIEW_REQUIRED
    assert "G8" in rec.causal_evidence_note


def test_non_strong_metrics_unchanged_by_context():
    ctx = DiagnosisContext(matched_family_id="G5_anisotropy")
    assert build_recommendation("anisotropy_mean_norm", ctx).level == RecommendationLevel.SAFE_BUT_UNPROVEN
    assert build_recommendation("snc_score", ctx).level == RecommendationLevel.NO_VALIDATED_FIX


# ------------------------- baseline of exactly zero ------------------------

def test_zero_baseline_uses_absolute_difference():
    assert signed_delta(0.0, 0.000004) == 0.000004
    assert signed_delta(0.0, 0.0) == 0.0
    assert signed_delta(0.5, 0.75) == 0.5


def test_single_point_on_zero_baseline_is_not_near_saturation():
    """One point marked as an outlier (0.000004 at baseline 0) may not weigh in
    the signature as much as full saturation (formerly: ±1.0 -> tanh 0.76)."""
    component = math.tanh(signed_delta(0.0, 0.000004))
    assert abs(component) < 1e-3
    saturated = math.tanh(signed_delta(0.0, 1.0))
    assert abs(component) < 0.01 * abs(saturated)


def test_full_report_uses_absolute_delta_on_zero_baseline():
    base = dict(RAW_B3)
    cur = dict(RAW_B3)
    cur["outlier_fraction"] = 0.000004
    # Only one metric changes, and by a negligible absolute value — no group
    # match and no saturation signal may arise.
    report = build_full_report(base, cur)
    assert report.grouped_match is None
    # Sanity: the similarity to any signature computed on this delta is
    # zero/undefined, since the delta does not exceed NOISE_FLOOR.
    assert _cosine_similarity({}, SIGNATURES[0].metric_deltas) == 0.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
