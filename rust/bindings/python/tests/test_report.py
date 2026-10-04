"""
tests/test_report.py

The composition layer: per-metric reports and the full report.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vechealth.interpretation.report import (
    build_metric_report,
    build_metric_report_from_drift,
    build_full_report,
    zero_labels_statement,
    Level1Status,
)
from vechealth.interpretation.limitations import LIMITATIONS
from vechealth.interpretation.deployment_context import DeploymentContext, IndexType


def test_build_metric_report_has_no_drift_without_baseline_current():
    """The compositional layer knows no baseline/current — drift must be None."""
    r = build_metric_report("hubness_max", diagnostic_strength=0.85)
    assert r.drift is None


def test_build_metric_report_from_drift_populates_drift():
    """The convenience layer MUST populate the drift field, unlike the
    compositional layer called directly."""
    r = build_metric_report_from_drift("hubness_max", 317, 890)
    assert r.drift is not None
    # No statistic has a validated direction of change, so none is labelled
    # "improved" or "regressed".
    assert r.drift.direction_label == "ambiguous"
    assert "regressed" not in r.level1_summary and "improved" not in r.level1_summary


def test_from_drift_uses_richer_level1_summary_than_direct():
    """The convenience layer must give a RICHER message than calling the
    compositional layer directly (it has access to baseline/current)."""
    generic = build_metric_report("hubness_max", diagnostic_strength=0.85)
    rich = build_metric_report_from_drift("hubness_max", 317, 890)
    assert generic.level1_summary != rich.level1_summary
    assert "baseline" in rich.level1_summary.lower()


def test_status_classification():
    """NORMAL below the noise floor; URGENT only when the benchmark measured a
    severe index-dependent loss for the matched pathology under the declared
    index; OBSERVE for every other change. A repair being available plays no
    part."""
    hnsw = DeploymentContext(index_type=IndexType.HNSW)

    # Matched G1 (hubness) on an HNSW index: the documented collapse applies.
    urgent = build_metric_report("hubness_max", diagnostic_strength=0.9,
                                 matched_family_id="G1_hubness", deployment_context=hnsw)
    assert urgent.status == Level1Status.URGENT

    # The same pattern without a declared index, or on exact search: no URGENT.
    no_index = build_metric_report("hubness_max", diagnostic_strength=0.9,
                                   matched_family_id="G1_hubness")
    assert no_index.status == Level1Status.OBSERVE
    exact = build_metric_report("hubness_max", diagnostic_strength=0.9,
                                matched_family_id="G1_hubness",
                                deployment_context=DeploymentContext(index_type=IndexType.EXACT))
    assert exact.status == Level1Status.OBSERVE

    # A family whose benchmark retention was not severe under HNSW: OBSERVE,
    # even though the repair for it (deduplication) is available.
    benign = build_metric_report("ndds_fraction", diagnostic_strength=0.9,
                                 matched_family_id="G4_duplicates", deployment_context=hnsw)
    assert benign.recommendation.level.value == "auto_fixable"
    assert benign.status == Level1Status.OBSERVE

    # An ambiguous match never escalates.
    ambiguous = build_metric_report("hubness_max", diagnostic_strength=0.9,
                                    deployment_context=hnsw, signature_ambiguous=True)
    assert ambiguous.status == Level1Status.OBSERVE

    # Below the noise floor: NORMAL, whatever the context.
    normal = build_metric_report("hubness_max", diagnostic_strength=0.05,
                                 matched_family_id="G1_hubness", deployment_context=hnsw)
    assert normal.status == Level1Status.NORMAL


def test_example_note_differs_point_level_vs_aggregate():
    """Metrics eligible for the future Hub Inspector must have a DIFFERENT
    message than purely aggregate metrics."""
    point_level = build_metric_report("hubness_max", diagnostic_strength=0.5)
    aggregate = build_metric_report("intrinsic_dim_mean", diagnostic_strength=0.5)
    assert "Hub Inspector" in point_level.example_note
    assert "Hub Inspector" not in aggregate.example_note
    assert point_level.example_note != aggregate.example_note


def test_example_provider_override():
    """If an example_provider returning a real example is given, it must be
    used instead of the default absence message."""
    def fake_provider(metric_name):
        return f"Example point for {metric_name}: ID=12345"

    r = build_metric_report("hubness_max", diagnostic_strength=0.5, example_provider=fake_provider)
    assert "ID=12345" in r.example_note


def test_group_context_caveat_only_when_grouped():
    r_grouped = build_metric_report_from_drift("hubness_max", 317, 890, is_part_of_group=True)
    r_not_grouped = build_metric_report_from_drift("hubness_max", 317, 890, is_part_of_group=False)
    assert r_grouped.group_context_caveat is not None
    assert "symptom, not a verdict" in r_grouped.group_context_caveat
    assert r_not_grouped.group_context_caveat is None


def test_full_report_sorted_by_priority_descending():
    baseline = {"hubness_max": 317, "anisotropy_mean_norm": 0.513, "snc_score": 0.18}
    current = {"hubness_max": 890, "anisotropy_mean_norm": 0.52, "snc_score": 0.19}
    report = build_full_report(baseline, current)
    scores = [r.priority.priority_score for r in report.metric_reports]
    assert scores == sorted(scores, reverse=True)


def test_full_report_includes_zero_labels_statement():
    report = build_full_report({"hubness_max": 317}, {"hubness_max": 890})
    assert report.zero_labels_statement == zero_labels_statement()
    assert "LLM" in report.zero_labels_statement


def test_full_report_propagates_ann_context_when_grouped():
    """Integration test: a matched signature must correctly feed
    ann_context_warning per metric."""
    baseline = {
        "hubness_max": 300, "hubness_skewness": 2.5, "intrinsic_dim_mean": 22.0,
        "ndds_fraction": 0.02, "dispersion_1nn": 0.5, "dispersion_10nn": 0.6,
        "anisotropy_mean_norm": 0.5, "outlier_fraction": 0.0, "snc_score": 0.2,
        "qmas_mean_1nn": 0.25, "qmas_orphans_pct": 0.35,
    }
    current = {
        "hubness_max": 33, "hubness_skewness": 0.43, "intrinsic_dim_mean": 268,
        "ndds_fraction": 0.0, "dispersion_1nn": 1.12, "dispersion_10nn": 1.08,
        "anisotropy_mean_norm": 0.05, "outlier_fraction": 0.5, "snc_score": 0.0002,
        "qmas_mean_1nn": 0.81, "qmas_orphans_pct": 1.0,
    }
    ctx = DeploymentContext(index_type=IndexType.HNSW)
    report = build_full_report(baseline, current, deployment_context=ctx)
    assert report.grouped_match is not None
    assert report.grouped_match.signature.family_id == "G8_noise"
    top = report.metric_reports[0]
    assert top.ann_warning is not None
    assert "9.8%" in top.ann_warning          # G8 retention under HNSW (preprint Table 22)
    assert top.status == Level1Status.URGENT  # the documented HNSW collapse applies


def test_full_report_carries_the_standing_limitations():
    report = build_full_report({"hubness_max": 317}, {"hubness_max": 890})
    assert report.limitations == LIMITATIONS
    text = " ".join(report.limitations)
    assert "did not predict retrieval quality" in text
    assert "symptom, not a verdict" in text
    assert "recalibrating" in text


def test_a_catastrophic_noise_pattern_is_never_described_as_improvement():
    """Under injected noise Recall@10 falls from 0.648 to 0.008 while hubness_max
    falls and intrinsic_dim_mean rises by an order of magnitude; neither
    reading may be described as an improvement."""
    baseline = {
        "hubness_max": 313, "hubness_skewness": 3.18, "intrinsic_dim_mean": 20.7,
        "ndds_fraction": 0.0445, "dispersion_1nn": 0.58, "dispersion_10nn": 0.73,
        "anisotropy_mean_norm": 0.513, "outlier_fraction": 0.0, "snc_score": 0.18,
        "qmas_mean_1nn": 0.256, "qmas_orphans_pct": 0.349,
    }
    current = {
        "hubness_max": 34, "hubness_skewness": 0.54, "intrinsic_dim_mean": 251.6,
        "ndds_fraction": 0.0, "dispersion_1nn": 1.31, "dispersion_10nn": 1.31,
        "anisotropy_mean_norm": 0.05, "outlier_fraction": 1.0, "snc_score": 0.0,
        "qmas_mean_1nn": 0.83, "qmas_orphans_pct": 1.0,
    }
    report = build_full_report(baseline, current)
    for mr in report.metric_reports:
        assert mr.drift.direction_label == "ambiguous"
        assert "improved" not in mr.level1_summary and "regressed" not in mr.level1_summary


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))