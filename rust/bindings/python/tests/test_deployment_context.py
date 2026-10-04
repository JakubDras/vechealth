"""
tests/test_deployment_context.py

The deployment context: index-conditioned figures (retention of exact-search
Recall@10, preprint Tables 22 and 24) and the hubness domain caveat.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vechealth.interpretation.deployment_context import (
    ANN_RETENTION,
    BASELINE_RETENTION,
    SEVERE_RETENTION_FRACTION,
    DeploymentContext,
    IndexType,
    ann_context_warning,
    domain_interpretation_caveat,
    index_loss_is_severe,
)

ALL_FAMILIES = {
    "G1_hubness", "G2_voids", "G3_imbalance", "G4_duplicates", "G5_anisotropy",
    "G6_collapse", "G7_fragmentation", "G8_noise", "G9_outliers",
}


def test_table_covers_every_benchmark_family():
    """Every family the benchmark measured under approximate search has an entry;
    a family without one would get silence, which reads as reassurance."""
    assert set(ANN_RETENTION) == ALL_FAMILIES


@pytest.mark.parametrize("family", sorted(ALL_FAMILIES))
@pytest.mark.parametrize("index_type", [IndexType.HNSW, IndexType.IVFPQ])
def test_every_family_gets_a_measurement_with_its_scope(family, index_type):
    msg = ann_context_warning(family, index_type)
    assert msg is not None
    entry = ANN_RETENTION[family]
    retention = entry.hnsw if index_type == IndexType.HNSW else entry.ivfpq
    assert f"{retention:.1f}%" in msg                      # the measurement itself
    assert "Qwen3-Embedding-0.6B" in msg and "BGE-large" in msg   # its scope
    assert "your own queries" in msg                       # what to do instead of trusting it
    assert "safe" not in msg.lower()                       # never a bare reassurance


def test_ann_warning_none_for_exact_search_unknown_index_and_no_family():
    assert ann_context_warning("G1_hubness", IndexType.EXACT) is None
    assert ann_context_warning("G1_hubness", IndexType.UNKNOWN) is None
    assert ann_context_warning(None, IndexType.HNSW) is None
    assert ann_context_warning("G99_not_a_family", IndexType.HNSW) is None


def test_hnsw_collapse_is_flagged_only_for_hubness_and_noise():
    """Under HNSW the benchmark's bases fell into two groups (retention 5.2%
    and 9.8% versus 98.4%-100.0%); no base lay between."""
    severe = {f for f in ALL_FAMILIES if index_loss_is_severe(f, IndexType.HNSW)}
    assert severe == {"G1_hubness", "G8_noise"}
    for family in ALL_FAMILIES - severe:
        assert ANN_RETENTION[family].hnsw >= 98.4


def test_ivfpq_flags_the_same_two_families_and_not_the_middle_group():
    severe = {f for f in ALL_FAMILIES if index_loss_is_severe(f, IndexType.IVFPQ)}
    assert severe == {"G1_hubness", "G8_noise"}
    # G4, G7, G9 sit below the baseline's 41.4% but are not flagged.
    for family in ("G4_duplicates", "G7_fragmentation", "G9_outliers"):
        assert ANN_RETENTION[family].ivfpq < BASELINE_RETENTION.ivfpq
        assert not index_loss_is_severe(family, IndexType.IVFPQ)


def test_severity_rule_gives_the_same_answer_on_the_second_encoder():
    """The grouping held on BGE-large (preprint Section 4.8): the same rule,
    applied to the BGE-large figures, singles out the same two families."""
    for hnsw in (True, False):
        base = BASELINE_RETENTION.hnsw_bge if hnsw else BASELINE_RETENTION.ivfpq_bge
        severe = set()
        for family, entry in ANN_RETENTION.items():
            value = entry.hnsw_bge if hnsw else entry.ivfpq_bge
            if value < SEVERE_RETENTION_FRACTION * base:
                severe.add(family)
        assert severe == {"G1_hubness", "G8_noise"}


def test_no_loss_flag_for_exact_search_or_unmatched_family():
    assert not index_loss_is_severe("G1_hubness", IndexType.EXACT)
    assert not index_loss_is_severe("G1_hubness", IndexType.UNKNOWN)
    assert not index_loss_is_severe(None, IndexType.HNSW)


def test_severe_and_mild_messages_differ_in_what_they_claim():
    severe = ann_context_warning("G1_hubness", IndexType.HNSW)
    mild = ann_context_warning("G5_anisotropy", IndexType.HNSW)
    assert "largest index-dependent loss" in severe
    assert "intermediate levels were not measured" in severe
    assert "No HNSW-specific loss was measured" in mild
    assert "does not establish that your index is unaffected" in mild


def test_ivfpq_message_states_that_the_baseline_depends_on_the_quantizer():
    msg = ann_context_warning("G6_collapse", IndexType.IVFPQ)
    assert "above the baseline" in msg
    assert "62.3%" in msg and "16 sub-quantizers" in msg
    assert "below the baseline" in ann_context_warning("G4_duplicates", IndexType.IVFPQ)


def test_no_loss_multiplier_is_quoted():
    """The multiplier moves by about +-25% between rebuilds of the same index
    (preprint Table 23), so only retention is reported."""
    for family in ALL_FAMILIES:
        for index_type in (IndexType.HNSW, IndexType.IVFPQ):
            msg = ann_context_warning(family, index_type)
            assert "larger than" not in msg and "x larger" not in msg


def test_domain_caveat_only_for_hubness_metrics():
    """The domain caveat concerns hubness only (the ClinicalTrials.gov case
    study) — other metrics do not carry the same documented risk of
    misinterpretation."""
    assert domain_interpretation_caveat("hubness_max", "medicine") is not None
    assert domain_interpretation_caveat("hubness_skewness", "medicine") is not None
    assert domain_interpretation_caveat("snc_score", "medicine") is None
    assert domain_interpretation_caveat("anisotropy_mean_norm", "medicine") is None


def test_domain_caveat_none_without_domain_hint():
    """Without a domain_hint (the user gave nothing) — no warning, no guessing."""
    assert domain_interpretation_caveat("hubness_max", None) is None


def test_deployment_context_is_immutable():
    ctx = DeploymentContext(index_type=IndexType.HNSW, domain_hint="test")
    with pytest.raises(Exception):
        ctx.index_type = IndexType.EXACT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
