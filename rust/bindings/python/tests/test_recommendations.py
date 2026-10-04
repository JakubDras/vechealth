"""
tests/test_recommendations.py

The recommendation engine: levels, parameter sources, machine-readable warnings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vechealth.interpretation.recommendations import (
    build_recommendation,
    classify_recommendation_level,
    evidence_ceiling_level,
    DiagnosisContext,
    RecommendationLevel,
    ParameterSource,
    RESEARCH_STATUS,
)

# Fail-closed: AUTO_FIXABLE requires a context pointing at the family the
# repair was validated on. Calling build_recommendation(...) without a
# context and expecting AUTO_FIXABLE would encode behavior judged unsafe (the
# same deduplication on an anisotropic base removed 56.5% of the corpus), so
# the tests pass the context explicitly.
G4_CTX = DiagnosisContext(matched_family_id="G4_duplicates")
G1_CTX = DiagnosisContext(matched_family_id="G1_hubness")
from vechealth.interpretation.metric_profiles import METRIC_PROFILES


def test_every_metric_gets_exactly_one_of_three_levels():
    for name in METRIC_PROFILES:
        level = classify_recommendation_level(name)
        assert level in RecommendationLevel


def test_auto_fixable_only_hubness_and_ndds():
    """The only 3 metrics with causal_evidence=STRONG in the project — if this
    ever changes (a new causal experiment), this test must be deliberately
    updated, not silently stop applying. This is the evidence ceiling
    (evidence_ceiling_level), not the recommendation level — the latter is
    never AUTO_FIXABLE without a context."""
    auto_fixable = {
        name for name in METRIC_PROFILES
        if evidence_ceiling_level(name) == RecommendationLevel.AUTO_FIXABLE
    }
    assert auto_fixable == {"hubness_max", "hubness_skewness", "ndds_fraction"}


def test_parameter_source_distinguishes_ndds_from_hubness():
    """ndds and hubness may NOT share the same parameter_source, even though
    both are AUTO_FIXABLE."""
    ndds = build_recommendation("ndds_fraction", G4_CTX)
    hub = build_recommendation("hubness_max", G1_CTX)
    assert ndds.parameter_source == ParameterSource.DERIVED_FROM_DIAGNOSIS
    assert hub.parameter_source == ParameterSource.FIXED_VALIDATED_DEFAULT
    assert ndds.parameter_source != hub.parameter_source


def test_hubness_note_mentions_severity_limitation():
    """Regression for a specific caveat: the NICDM validation only at the
    strongest severity must be explicitly mentioned in the text."""
    rec = build_recommendation("hubness_max", G1_CTX)
    assert "strongest" in rec.parameter_source_note


def test_every_recommendation_has_machine_readable_warning():
    """CRITICAL test — every recommendation, regardless of level, MUST carry
    both machine-readable warning signals. This is a safeguard for agent
    systems consuming these reports, not only humans."""
    for name in METRIC_PROFILES:
        rec = build_recommendation(name)
        assert rec.human_review_required is True
        assert rec.research_status == RESEARCH_STATUS
        assert rec.research_status == "experimental_in_progress"


def test_every_recommendation_has_human_readable_disclaimer():
    for name in METRIC_PROFILES:
        rec = build_recommendation(name)
        assert rec.disclaimer
        assert "100%" in rec.disclaimer


def test_safe_but_unproven_only_anisotropy():
    swu = {
        name for name in METRIC_PROFILES
        if classify_recommendation_level(name) == RecommendationLevel.SAFE_BUT_UNPROVEN
    }
    assert swu == {"anisotropy_mean_norm"}


def test_no_validated_fix_has_no_suggested_action():
    """For a missing validated repair, suggested_action must be None, not an
    empty string or a placeholder — an explicit absence, not a hidden one."""
    rec = build_recommendation("snc_score")
    assert rec.level == RecommendationLevel.NO_VALIDATED_FIX
    assert rec.suggested_action is None
    assert rec.parameter_source is None


def test_safe_but_unproven_disclaimer_explicitly_warns_against_causal_interpretation():
    rec = build_recommendation("anisotropy_mean_norm")
    assert "do NOT treat" in rec.causal_evidence_note or "not treat" in rec.causal_evidence_note.lower()


def test_no_recommendation_names_a_function_that_does_not_exist():
    """Suggested actions describe the repairs of the preprint. None of them is
    implemented in this package, so none may be written as a call an agent
    could try to execute, and every recommendation says so in a flag."""
    import re

    call_like = re.compile(r"\b[a-z]+_[a-z_]+\(")
    for name in METRIC_PROFILES:
        for ctx in (None, G4_CTX, G1_CTX):
            rec = build_recommendation(name, ctx)
            assert rec.implemented_in_package is False
            for text in (rec.suggested_action, rec.candidate_action):
                assert text is None or not call_like.search(text), (name, text)


def test_repair_evidence_carries_the_preprints_figures_and_caveats():
    dedup = build_recommendation("ndds_fraction", G4_CTX).causal_evidence_note
    assert "15.0% to 15.2%" in dedup and "0.5495 to 0.6321" in dedup
    assert "clones carry identifiers absent from the relevance judgments" in dedup
    assert "162 genuine duplicates, 95 of them relevance-judged" in dedup
    nicdm = build_recommendation("hubness_max", G1_CTX).causal_evidence_note
    assert "7.1% to 7.8%" in nicdm and "about a tenth of the gap" in nicdm
    assert "changes the distance function" in nicdm


def test_anisotropy_recommendation_prefers_the_measured_safe_repairs():
    rec = build_recommendation("anisotropy_mean_norm")
    action = rec.suggested_action
    assert "centering" in action and "shrinkage whitening" in action
    assert "Do NOT apply full PCA whitening" in action
    assert "3.0e-6" in rec.causal_evidence_note or "3.0e-6" in action


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))