"""
The signature-match threshold, the ambiguity margin, and messages depending
on the match state.

Data: tests/fixtures/round5_library_benchmark.csv — the 11 statistics for
the 53 benchmark bases, computed by this library at full precision.
"""
import csv
import re
from pathlib import Path

import pytest

import vechealth.interpretation.signatures as sigmod
from vechealth.interpretation.deployment_context import (
    ANN_RETENTION, DeploymentContext, IndexType, ann_context_warning,
)
from vechealth.interpretation.recommendations import (
    DiagnosisContext, RecommendationLevel, build_recommendation,
)
from vechealth.interpretation.report import build_full_report
from vechealth.interpretation.signatures import (
    SIGNATURES, MatchStatus, NOISE_FLOOR, _cosine_similarity, assess_signature_match,
    find_matching_signature, signed_delta,
)

FIXTURE = Path(__file__).parent / "fixtures" / "round5_library_benchmark.csv"
BASELINE = "B3_beir_multidomain"
METRICS = ["hubness_max", "hubness_skewness", "intrinsic_dim_mean", "ndds_fraction",
           "dispersion_1nn", "dispersion_10nn", "anisotropy_mean_norm",
           "outlier_fraction", "snc_score", "qmas_mean_1nn", "qmas_orphans_pct"]
PATTERN = re.compile(r"^(G\d+)_([a-z_]+?)_.*_level(\d+)$")


def _rows():
    with open(FIXTURE, newline="") as fh:
        return {r["base_id"]: {m: float(r[m]) for m in METRICS} for r in csv.DictReader(fh)}


def _families(rows):
    fams = {}
    for b in rows:
        mm = PATTERN.match(b)
        if mm:
            fams.setdefault(f"{mm.group(1)}_{mm.group(2)}", {})[int(mm.group(3))] = b
    return fams


def _deltas(rows, base_id):
    b = rows[BASELINE]
    return {m: signed_delta(b[m], rows[base_id][m]) for m in METRICS}


def test_signatures_equal_extraction_with_single_noise_constant():
    """SIGNATURES = the highest level of each family, by the extraction rule
    (round(delta, 3), |delta| >= NOISE_FLOOR), on library values — the same
    constant as at matching time."""
    rows = _rows()
    fams = _families(rows)
    by_id = {s.family_id: s for s in SIGNATURES}
    assert set(by_id) == set(fams)
    for f, levels in fams.items():
        d = _deltas(rows, levels[max(levels)])
        expected = {m: round(v, 3) for m, v in d.items() if abs(v) >= NOISE_FLOOR}
        assert by_id[f].metric_deltas == expected, f


def test_signature_matching_all_reference_levels_self_match_exactly():
    """With one noise constant, every signature matches itself with 1.0."""
    for sig in SIGNATURES:
        a = assess_signature_match(sig.metric_deltas)
        assert a.status == MatchStatus.MATCH, sig.family_id
        assert a.match.signature.family_id == sig.family_id
        assert a.match.similarity == pytest.approx(1.0)


def test_intended_true_matches_pass():
    """The two highest levels of each family give a MATCH to their own
    family — except those explicitly listed in LOST_INTENDED (the rule
    “safety wins”, if the calibration scan triggered it)."""
    rows = _rows()
    for f, levels in _families(rows).items():
        for lv in sorted(levels)[-2:]:
            b = levels[lv]
            a = assess_signature_match(_deltas(rows, b))
            if b in LOST_INTENDED:
                assert a.status != MatchStatus.MATCH or a.match.signature.family_id == f
                continue
            assert a.status == MatchStatus.MATCH, (b, a.status)
            assert a.match.signature.family_id == f, (b, a.match.signature.family_id)


def test_no_false_family_on_any_benchmark_level():
    """A full scan of G1-G9 (48 levels): no MATCH to a foreign family."""
    rows = _rows()
    for f, levels in _families(rows).items():
        for b in levels.values():
            m = find_matching_signature(_deltas(rows, b))
            assert m is None or m.signature.family_id == f, (b, m.signature.family_id)


@pytest.mark.parametrize("index_type", [IndexType.HNSW, IndexType.IVFPQ])
def test_g8_level0_gets_neither_g9_nor_reassuring_warning(index_type):
    """With the earlier threshold of 0.90, G8 L0 (library values) matched G9
    (0.906). A false match to G9 would have produced a message about the
    wrong family (benign under HNSW) for a pattern that is really noise
    (retention 9.8%, the collapse the benchmark measured). Now: no G9 family
    and no reassuring message."""
    rows = _rows()
    base = "G8_noise_B3_beir_multidomain_level0"
    a = assess_signature_match(_deltas(rows, base))
    assert not (a.status == MatchStatus.MATCH and a.match.signature.family_id == "G9_outliers")
    report = build_full_report(rows[BASELINE], rows[base], DeploymentContext(index_type=index_type))
    assert report.grouped_match is None or report.grouped_match.signature.family_id == "G8_noise"
    for mr in report.metric_reports:
        w = mr.ann_warning or ""
        assert "G9" not in w and "outlier" not in w.lower()
        assert "safe" not in w.lower() and "ok" != w.strip().lower()


def test_ann_warning_for_ambiguous_match_is_cautious_and_names_no_family():
    for it in (IndexType.HNSW, IndexType.IVFPQ):
        w = ann_context_warning("G1_hubness", it, ambiguous=True)
        assert w is not None and "cannot be established" in w
        assert "%" not in w                              # no specific-family figures
        for fam in ANN_RETENTION:
            assert fam not in w
    assert ann_context_warning(None, IndexType.EXACT, ambiguous=True) is None


def test_ambiguous_match_gives_no_family_anywhere(monkeypatch):
    """The ambiguous case (the margin raised so every pair is too close): no
    family in the report, a group message about ambiguity, a cautious ANN
    warning, a fail-closed recommendation."""
    monkeypatch.setattr(sigmod, "AMBIGUITY_MARGIN", 2.0)
    rows = _rows()
    base = "G4_duplicates_B3_beir_multidomain_level4"          # without the margin: MATCH to G4
    a = assess_signature_match(_deltas(rows, base))
    assert a.status == MatchStatus.AMBIGUOUS and a.match is None and a.best is not None
    assert find_matching_signature(_deltas(rows, base)) is None
    report = build_full_report(rows[BASELINE], rows[base], DeploymentContext(index_type=IndexType.HNSW))
    assert report.signature_status == MatchStatus.AMBIGUOUS
    assert report.grouped_match is None
    assert "no family is indicated" in report.grouped_message
    assert any(mr.ann_warning and "cannot be established" in mr.ann_warning for mr in report.metric_reports)
    ctx = DiagnosisContext.from_snapshots(rows[BASELINE], rows[base])
    assert ctx.matched_family_id is None and ctx.signature_ambiguous
    rec = build_recommendation("ndds_fraction", ctx)
    assert rec.level == RecommendationLevel.REVIEW_REQUIRED


def test_similarity_does_not_depend_on_key_order():
    a = {"hubness_max": 0.31, "ndds_fraction": 2.4, "snc_score": -0.2, "dispersion_1nn": -0.7}
    b = SIGNATURES[3].metric_deltas
    ra = dict(reversed(list(a.items())))
    rb = dict(reversed(list(b.items())))
    assert _cosine_similarity(a, b) == _cosine_similarity(ra, rb)


# Filled from the result of the calibration scan (the intended matches whose
# similarity falls below the threshold); empty = no lost intended matches.
LOST_INTENDED: set[str] = {"G6_collapse_B3_beir_multidomain_level3"}   # s1 0.8756 < T 0.907
