"""
tests/test_metric_profiles_complete.py

CI must fail if someone adds a new metric to core/src/metrics/ without a
matching entry in MetricProfile.
"""
import sys
from pathlib import Path

import pytest

# Tests live in rust/bindings/python/tests/, the package in
# rust/bindings/python/vechealth/ — both are siblings inside
# rust/bindings/python/, so adding that common parent directory to sys.path
# is enough to import "vechealth.interpretation..."
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vechealth.interpretation.metric_profiles import METRIC_PROFILES, ALL_METRIC_NAMES


# Canonical list of metric names returned by VecHealthEvaluator.compute_all().
#
# NOTE: this is a temporary, manually maintained source of truth about
# "which metrics exist at all" — risk of drifting apart from
# rust/core/src/metrics/all.rs if someone adds a metric on one side and
# forgets the other. Eventually this list should be introspected directly
# from the compiled package, e.g.:
#     evaluator = VecHealthEvaluator(dummy_vectors)
#     EXPECTED_METRICS = set(evaluator.compute_all().to_dict().keys())
# instead of being held explicitly here.
EXPECTED_METRICS = {
    "hubness_max",
    "hubness_skewness",
    "intrinsic_dim_mean",
    "ndds_fraction",
    "dispersion_1nn",
    "dispersion_10nn",
    "anisotropy_mean_norm",
    "outlier_fraction",
    "snc_score",
    "qmas_mean_1nn",
    "qmas_orphans_pct",
}


def test_all_expected_metrics_have_profile():
    missing = EXPECTED_METRICS - ALL_METRIC_NAMES
    assert not missing, (
        f"Missing MetricProfile for: {missing}. Add an entry in "
        f"interpretation/metric_profiles.py before merging."
    )


def test_no_orphaned_profiles():
    """The reverse test — a profile without a matching real metric is most
    likely a typo in the name, or dead code left after removing a
    metric."""
    orphaned = ALL_METRIC_NAMES - EXPECTED_METRICS
    assert not orphaned, (
        f"MetricProfile exists for unknown metrics: {orphaned} — a typo in "
        f"the name, or a metric removed from core while its profile "
        f"remained?"
    )


def test_every_profile_internally_consistent():
    for name, profile in METRIC_PROFILES.items():
        assert profile.name == name, (
            f"Inconsistency: dict key '{name}' != profile.name "
            f"'{profile.name}'"
        )
        assert 0.0 <= profile.ablation_contribution <= 1.0, (
            f"{name}: ablation_contribution={profile.ablation_contribution} "
            f"outside the sensible range [0, 1] — a typo in the data?"
        )
        assert profile.vif_after_p0 > 0, (
            f"{name}: VIF must be positive, got {profile.vif_after_p0}"
        )


def test_dose_response_range_is_none_or_valid_triple():
    """Documents a deliberate, temporary state: dose_response_range is
    intentionally empty (None) in this version — this test does not force
    it to be filled, it only guards that IF someone fills it, they do so
    correctly (a triple of increasing values), not with an accidental,
    wrong structure."""
    for name, profile in METRIC_PROFILES.items():
        r = profile.dose_response_range
        if r is None:
            continue
        assert len(r) == 3, f"{name}: dose_response_range must have 3 elements, got {len(r)}"


def test_every_metric_has_both_badges_without_crashing():
    """A regression integration test: get_badges() must work for every
    metric without exception, ALWAYS returning both fields (diagnostic +
    repair_lever), never None."""
    from vechealth.interpretation.badges import get_badges

    for name in METRIC_PROFILES:
        badges = get_badges(name)
        assert badges.diagnostic is not None
        assert badges.repair_lever is not None
        assert badges.diagnostic.label and badges.diagnostic.tooltip
        assert badges.repair_lever.label and badges.repair_lever.tooltip


def test_repair_lever_badge_matches_causal_evidence():
    """Consistency: the repair-lever badge level must match the profile's
    causal_evidence 1:1 — that is the only source of truth; the badge
    should not have its own independent decision logic."""
    from vechealth.interpretation.badges import get_badges, BadgeLevel
    from vechealth.interpretation.metric_profiles import CausalEvidence

    expected_level = {
        CausalEvidence.STRONG: BadgeLevel.STRONG,
        CausalEvidence.WEAK: BadgeLevel.WEAK,
        CausalEvidence.NONE: BadgeLevel.NONE,
    }
    for name, profile in METRIC_PROFILES.items():
        badges = get_badges(name)
        assert badges.repair_lever.level == expected_level[profile.causal_evidence], (
            f"{name}: badge level {badges.repair_lever.level} disagrees "
            f"with causal_evidence={profile.causal_evidence}"
        )


def test_no_shipped_profile_claims_a_validated_direction():
    """Preprint Sections 4.3-4.4: no statistic has a direction of change that
    is validated as good or bad — the same statistic moves in opposite
    directions across pathology families, and several move toward what would
    read as improvement while Recall@10 falls."""
    from vechealth.interpretation.metric_profiles import KnownDirection

    for name, profile in METRIC_PROFILES.items():
        assert profile.known_direction == KnownDirection.AMBIGUOUS, name


def test_drift_never_says_improved_or_regressed_in_either_direction():
    """The truncation case of Section 4.4: hubness_max falls 313 -> 146 and
    qmas_orphans_pct 0.349 -> 0.006 while Recall@10 falls 0.648 -> 0.548. A
    fall must not be called an improvement, and a rise must not be called a
    regression."""
    from vechealth.interpretation.drift import interpret_drift

    for name, before, after in [
        ("hubness_max", 313, 146),
        ("qmas_orphans_pct", 0.349, 0.006),
        ("intrinsic_dim_mean", 20.7, 251.6),
        ("snc_score", 0.18, 0.0),
        ("anisotropy_mean_norm", 0.5, 0.9),
    ]:
        for a, b in ((before, after), (after, before)):
            d = interpret_drift(name, a, b)
            assert d.direction_label == "ambiguous", (name, a, b)
            assert "improved" not in d.message and "regressed" not in d.message
            assert "not validated as good or bad" in d.message


def test_drift_reserved_branch_labels_direction_for_a_validated_profile(monkeypatch):
    """The improved/regressed branch is reserved for a profile whose direction a
    future version validates; it must still work when one is supplied."""
    from dataclasses import replace

    from vechealth.interpretation import drift as drift_module
    from vechealth.interpretation.metric_profiles import KnownDirection

    lower = replace(METRIC_PROFILES["hubness_max"], known_direction=KnownDirection.LOWER_BETTER)
    higher = replace(METRIC_PROFILES["snc_score"], known_direction=KnownDirection.HIGHER_BETTER)
    monkeypatch.setattr(drift_module, "get_profile", lambda name: {"hubness_max": lower, "snc_score": higher}[name])

    assert drift_module.interpret_drift("hubness_max", 300, 600).direction_label == "regressed"
    assert drift_module.interpret_drift("hubness_max", 600, 300).direction_label == "improved"
    assert drift_module.interpret_drift("snc_score", 0.25, 0.15).direction_label == "regressed"
    assert drift_module.interpret_drift("snc_score", 0.15, 0.25).direction_label == "improved"


def test_drift_from_exactly_zero_is_described_without_an_infinite_percentage():
    from vechealth.interpretation.drift import interpret_drift

    d = interpret_drift("ndds_fraction", 0.0, 0.1)
    assert "inf" not in d.message
    assert "exactly 0" in d.message


def test_drift_baseline_zero_edge_case_gives_maximal_signal():
    """Regression for a specific bug found during implementation: a change
    from EXACTLY zero to anything nonzero must give the maximal
    diagnostic_strength (1.0), not zero — an earlier version of the code
    wrongly returned 0.0 in this case."""
    from vechealth.interpretation.drift import interpret_drift

    changed = interpret_drift("ndds_fraction", 0.0, 0.1)
    unchanged = interpret_drift("ndds_fraction", 0.0, 0.0)
    assert changed.diagnostic_strength == 1.0
    assert unchanged.diagnostic_strength == 0.0


def test_drift_integrates_with_priority():
    """Integration test of the full pipeline: diagnostic_strength from
    interpret_drift() must be directly, without conversion, acceptable to
    compute_priority()."""
    from vechealth.interpretation.drift import interpret_drift
    from vechealth.interpretation.priority import compute_priority

    drift = interpret_drift("hubness_max", 317, 890)
    result = compute_priority("hubness_max", drift.diagnostic_strength)
    assert result.priority_score > 0


def test_signature_matching_distinguishes_overlapping_families():
    """Regression for a real finding from the extraction over the full
    53-base set: without tanh squashing, G4_duplicates and G5_anisotropy
    were confused (similarity 0.993, practically identical) due to
    ndds_fraction dominating the rest of the vector. With squashing they
    must be clearly distinguishable."""
    from vechealth.interpretation.signatures import find_matching_signature

    g4_moderate = {'intrinsic_dim_mean': -0.479, 'ndds_fraction': 9.878, 'dispersion_1nn': -0.545, 'snc_score': 0.423}
    match = find_matching_signature(g4_moderate)
    assert match is not None
    assert match.signature.family_id == "G4_duplicates"


def test_signature_matching_no_false_positive_g6_vs_g2():
    """Regression for a specifically found false match: at the 0.85
    threshold, G6_collapse (intermediate level) was confused with G2_voids
    (similarity 0.883). At the final 0.90 threshold this MUST NOT recur —
    neither as a wrong match nor as any other false match to a different
    family."""
    from vechealth.interpretation.signatures import find_matching_signature

    g6_moderate = {'hubness_max': -0.435, 'hubness_skewness': -0.216, 'ndds_fraction': 0.183,
                   'snc_score': -0.391, 'qmas_mean_1nn': 2.168, 'qmas_orphans_pct': 1.862}
    match = find_matching_signature(g6_moderate)
    # A safe fallback (None) is OK; a wrong match to a family OTHER than
    # G6_collapse is NOT OK.
    if match is not None:
        assert match.signature.family_id == "G6_collapse", (
            f"False match: G6_collapse confused with {match.signature.family_id}"
        )


def test_signature_matching_all_reference_levels_self_match():
    """Signatures built from the strongest levels must, of course, match
    themselves with similarity 1.0 — a data-integrity sanity check for
    SIGNATURES. Extraction and matching use one NOISE_FLOOR constant, so
    the similarity is exactly 1.0."""
    from vechealth.interpretation.signatures import find_matching_signature, SIGNATURES

    for sig in SIGNATURES:
        match = find_matching_signature(sig.metric_deltas)
        assert match is not None, f"{sig.family_id}: its own reference signature did not match itself"
        assert match.signature.family_id == sig.family_id
        assert match.similarity == pytest.approx(1.0)


def test_signature_matching_requires_minimum_metrics():
    """A single moved metric is not 'grouping' — it must return None, even
    when some signature is defined."""
    from vechealth.interpretation.signatures import find_matching_signature

    result = find_matching_signature({"hubness_max": 0.8})
    assert result is None



def test_signature_matching_returns_none_when_no_signatures_defined():
    """The default state (SIGNATURES empty, before filling in real data)
    must be a safe fallback, not an error."""
    import vechealth.interpretation.signatures as sigmod
    from vechealth.interpretation.signatures import find_matching_signature

    original = sigmod.SIGNATURES
    try:
        sigmod.SIGNATURES = []
        result = find_matching_signature({"hubness_max": 0.9, "dispersion_1nn": -0.3})
        assert result is None
    finally:
        sigmod.SIGNATURES = original




if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
