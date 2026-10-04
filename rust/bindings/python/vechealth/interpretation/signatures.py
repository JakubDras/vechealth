"""
interpretation/signatures.py

Grouping correlated changes into one message, instead of N separate,
unrelated alerts.

SIGNATURES are filled with real data from an extraction over the full
53-base benchmark, strongest level per family.

SOURCE: metrics computed by THIS library after the numerics fix (distances
and reductions in float64) —
`compute_all(k=10, k_intrinsic_dim=20, duplicate_epsilon=0.05,
outlier_distance_threshold=1.2)` on the 53 benchmark bases; deltas via
`signed_delta`; ONE constant noise-floor filter (0.15) — the same one used
at matching time (earlier extractions used 0.1). History: library values
from before the numerics fix and a 0.1 filter, and before that the values of
the older Python evaluator. The numbers in points 1-4 below come from those
earlier versions; the current calibration is point 5.

=== KEY FINDINGS FROM THE EMPIRICAL VALIDATION (before finalization) ===

1. Squashing (tanh) is NECESSARY, not cosmetic. Without it,
   `ndds_fraction` (which can reach relative values of 17-25 while most
   metrics stay within [-1, 2]) completely dominates the cosine
   similarity. Measured: G4_duplicates vs G5_anisotropy without squashing =
   0.993 (practically indistinguishable, despite entirely different
   mechanisms) — with squashing = 0.503 (correctly, moderately distinct).

2. SIMILARITY_THRESHOLD=0.90 (not 0.75 from the first version, not 0.85
   from the first fix) — empirically derived by a FULL scan of the matrix:
   each of the 8 INTERMEDIATE levels (not only the reference points)
   compared against EVERY one of the 9 reference signatures of other
   families (56 foreign pairs total). Exactly ONE dangerous collision was
   found: G6_collapse (intermediate level) vs G2_voids (reference
   signature) = 0.883 — the threshold must be clearly above this value, not
   just above the collision between the reference points themselves (0.823,
   which gave a false sense of safety in the first fix to 0.85).

3. Three families (G1_hubness, G6_collapse, G7_fragmentation) have an
   UNSTABLE geometric pattern — their own moderate severity level matches
   THEIR OWN reference signature more weakly (0.52, 0.81, 0.85
   respectively) than the final 0.90 threshold. This means grouping
   triggers rarely for these three families — mainly at the strongest
   severity, close to the reference signature itself. This is a
   DELIBERATE, safe trade-off (the design principle: "better not to group
   than to group wrongly"), not a bug to fix. Averaging several levels into a signature
   instead of a single point was also tried — it did not improve the result
   (inter-family collisions stayed at a similar level), so the simpler
   approach (single, strongest level) was kept.

4. RE-VALIDATION ON LIBRARY-DERIVED SIGNATURES (library values from before
   the numerics fix). The procedure from point 2 (each family's middle level vs the reference
   signatures of other families; 64 foreign pairs — G2 has fewer than
   MIN_METRICS_FOR_GROUPING metrics above NOISE_FLOOR at its middle level):
   max foreign pair 0.872 (G6_collapse -> G2_voids), 0 pairs >= 0.90 — the
   0.90 threshold unchanged. Middle-level self-similarity: G1 0.539, G6
   0.806, G9 0.835 (below the threshold), the other families >= 0.947 (G7
   0.984 — the 0.85 in the old version came from the ±1 rule for
   outlier_fraction). LIMITATION: scanning ALL intermediate levels (264
   foreign pairs) yields one false match above the threshold — G8_noise
   level 0 -> G9_outliers, 0.906 — so at the mildest noise level, grouping
   may point at the wrong family. The threshold was left unchanged
   (author's decision).

5. CURRENT CALIBRATION — THRESHOLD AND MARGIN FROM A RULE FIXED BEFORE THE
   SCAN (a full scan of the 48 levels of G1-G9).
   AMBIGUITY_MARGIN = the smallest gap (best minus runner-up) among the
   intended true matches (the two highest levels of each family) = 0.01723
   (G6 L3). SIMILARITY_THRESHOLD = the smallest value on a 0.001 grid above
   every false match at that margin = 0.907 (G8 L0 -> G9, 0.9062).
   CONFLICT: G6 L3 (intended, 0.8756) falls below the threshold — the
   emergency rule "safety wins" applies: the threshold stays, G6 L3 yields
   no match (no family message). Calibration result: 25 true matches, 0
   false, 0 ambiguous, 23 no-match.
   LIMITATION: the threshold and margin are calibrated on this benchmark
   only (one corpus, one encoder) — outside it they are not validated.
   Ambiguity NEVER produces a reassuring or a family message
   (assess_signature_match -> MatchStatus.AMBIGUOUS; deployment_context,
   recommendations, report).
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class PathologySignature:
    family_id: str
    family_label: str
    metric_deltas: dict[str, float]


SIGNATURES: list[PathologySignature] = [
    PathologySignature(
        family_id="G1_hubness",
        family_label="Hubness (dominant points in k-NN)",
        metric_deltas={'hubness_max': 4.556, 'hubness_skewness': 8.015, 'intrinsic_dim_mean': 0.454, 'ndds_fraction': -0.974, 'dispersion_1nn': -0.277, 'dispersion_10nn': -0.377, 'anisotropy_mean_norm': 0.249, 'snc_score': -0.226, 'qmas_mean_1nn': 0.442, 'qmas_orphans_pct': 1.595},
    ),
    PathologySignature(
        family_id="G2_voids",
        family_label="Void regions (empty semantic zones)",
        metric_deltas={'hubness_max': -0.329, 'hubness_skewness': -0.164, 'qmas_mean_1nn': 0.264, 'qmas_orphans_pct': 0.85},
    ),
    PathologySignature(
        family_id="G3_imbalance",
        family_label="Cluster imbalance (dominant category)",
        metric_deltas={'hubness_max': -0.613, 'hubness_skewness': -0.29, 'intrinsic_dim_mean': 0.28, 'ndds_fraction': -0.97, 'dispersion_1nn': 0.326, 'dispersion_10nn': 0.165, 'snc_score': -0.252, 'qmas_mean_1nn': 0.358, 'qmas_orphans_pct': 1.021},
    ),
    PathologySignature(
        family_id="G4_duplicates",
        family_label="Near-duplicates (approximate duplicates)",
        metric_deltas={'hubness_max': -0.198, 'intrinsic_dim_mean': -0.868, 'ndds_fraction': 21.477, 'dispersion_1nn': -0.995, 'dispersion_10nn': -0.151, 'snc_score': 0.675},
    ),
    PathologySignature(
        family_id="G5_anisotropy",
        family_label="Anisotropy (cone effect)",
        metric_deltas={'hubness_max': 0.272, 'hubness_skewness': 0.222, 'ndds_fraction': 21.412, 'dispersion_1nn': -0.949, 'dispersion_10nn': -0.949, 'anisotropy_mean_norm': 0.948, 'qmas_mean_1nn': 0.929, 'qmas_orphans_pct': 1.843},
    ),
    PathologySignature(
        family_id="G6_collapse",
        family_label="Embedding collapse (dimensionality collapse)",
        metric_deltas={'hubness_max': -0.831, 'hubness_skewness': -0.899, 'intrinsic_dim_mean': -0.66, 'ndds_fraction': 0.233, 'dispersion_1nn': -0.713, 'dispersion_10nn': -0.684, 'snc_score': -0.325, 'qmas_mean_1nn': 2.62, 'qmas_orphans_pct': 1.863},
    ),
    PathologySignature(
        family_id="G7_fragmentation",
        family_label="Fragmentation (disconnected semantic islands)",
        metric_deltas={'hubness_max': -0.214, 'ndds_fraction': -0.572, 'dispersion_1nn': 0.269, 'dispersion_10nn': 0.281, 'anisotropy_mean_norm': -0.615, 'qmas_mean_1nn': 0.491, 'qmas_orphans_pct': 1.411},
    ),
    PathologySignature(
        family_id="G8_noise",
        family_label="Systematic injected noise",
        metric_deltas={'hubness_max': -0.888, 'hubness_skewness': -0.826, 'intrinsic_dim_mean': 11.146, 'ndds_fraction': -1.0, 'dispersion_1nn': 1.249, 'dispersion_10nn': 0.795, 'anisotropy_mean_norm': -0.896, 'outlier_fraction': 1.0, 'snc_score': -0.999, 'qmas_mean_1nn': 2.235, 'qmas_orphans_pct': 1.863},
    ),
    PathologySignature(
        family_id="G9_outliers",
        family_label="Outlier contamination",
        metric_deltas={'hubness_max': 0.236, 'hubness_skewness': 0.522, 'intrinsic_dim_mean': 6.826, 'ndds_fraction': -0.839, 'dispersion_1nn': 0.812, 'dispersion_10nn': 0.504, 'anisotropy_mean_norm': -0.601, 'outlier_fraction': 0.6, 'snc_score': -0.638, 'qmas_mean_1nn': 0.16, 'qmas_orphans_pct': 0.569},
    ),
]


def signed_delta(baseline_value: float, current_value: float) -> float:
    """Signature component for one metric: the change relative to the
    baseline.

    Relative ((c - b) / |b|) when the baseline is nonzero. When the
    baseline is EXACTLY 0, the relative change is undefined — the ABSOLUTE
    difference (c - b) is returned then, not ±1.0. The previous rule (±1.0
    for any nonzero value) made one point on G7 (outlier_fraction =
    0.000004) weigh as much in the signature as full saturation on G8
    (outlier_fraction = 1.0). Metrics whose baseline is often zero
    (outlier_fraction, ndds_fraction) are fractions in [0, 1], so the
    absolute difference is on the same scale as the relative changes of the
    other metrics.

    One source of this rule for report.build_full_report and for the
    extraction of SIGNATURES from the benchmark — they must compute deltas
    identically, otherwise the observation and the signature live on
    different scales."""
    b = float(baseline_value)
    c = float(current_value)
    if b == 0.0:
        return c - b
    return (c - b) / abs(b)


def _squash(x: float) -> float:
    """Nonlinear squashing (tanh) — NECESSARY, not cosmetic (see finding 1
    in the module docstring). Prevents metrics with a naturally large
    relative-change scale (ndds_fraction) from dominating the rest of the
    vector in the cosine comparison."""
    return math.tanh(x)


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    # Keys sorted — set-iteration order depends on PYTHONHASHSEED,
    # and summation order affects the last bit of the result; with
    # AMBIGUITY_MARGIN equal to the borderline gap, the result must be
    # deterministic.
    keys = sorted(set(a) | set(b))
    va = [_squash(a.get(k, 0.0)) for k in keys]
    vb = [_squash(b.get(k, 0.0)) for k in keys]

    dot = sum(x * y for x, y in zip(va, vb))
    norm_a = math.sqrt(sum(x * x for x in va))
    norm_b = math.sqrt(sum(y * y for y in vb))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class SignatureMatch:
    signature: PathologySignature
    similarity: float
    matched_metrics: list[str]


# Derived by the rule of point 5 in the module docstring — the smallest value
# on a 0.001 grid above the highest false match in the full scan
# (G8 L0 -> G9, 0.9062). DO NOT change without re-running the scan.
SIMILARITY_THRESHOLD = 0.907

# The ambiguity margin. If the similarity to the best family exceeds the
# similarity to the runner-up by less than AMBIGUITY_MARGIN, the result is
# AMBIGUOUS — no family is pointed at (find_matching_signature returns None,
# DiagnosisContext gets no family, the ANN warning stays cautious — never
# family-specific or reassuring). Value derived by the same rule (point 5 in
# the module docstring).
AMBIGUITY_MARGIN = 0.0172267037992766   # min gap among intended matches (G6 L3), full precision

MIN_METRICS_FOR_GROUPING = 2
# ONE constant noise-floor threshold — used at matching time (here) and when
# the signatures were extracted from the benchmark.
NOISE_FLOOR = 0.15


class MatchStatus(str, Enum):
    MATCH = "match"          # one family above the threshold, with a margin over the runner-up
    AMBIGUOUS = "ambiguous"  # above the threshold, but the runner-up is closer than the margin
    NONE = "none"            # below the threshold, or too few metrics above the noise floor


@dataclass(frozen=True)
class SignatureAssessment:
    """The full result of the comparison against the signatures. `match` is
    set ONLY for MATCH; `best` and `runner_up` are diagnostic information
    (the two closest families), not a family indication."""
    status: MatchStatus
    match: Optional[SignatureMatch]
    best: Optional[SignatureMatch]
    runner_up: Optional[SignatureMatch]


def assess_signature_match(observed_deltas: dict[str, float]) -> SignatureAssessment:
    significant = {k: v for k, v in observed_deltas.items() if abs(v) >= NOISE_FLOOR}
    if len(significant) < MIN_METRICS_FOR_GROUPING or not SIGNATURES:
        return SignatureAssessment(MatchStatus.NONE, None, None, None)

    scored = []
    for sig in SIGNATURES:
        similarity = _cosine_similarity(significant, sig.metric_deltas)
        matched = [k for k in significant if k in sig.metric_deltas]
        scored.append(SignatureMatch(signature=sig, similarity=similarity, matched_metrics=matched))
    # stable: on equal similarity, the SIGNATURES order is kept
    scored.sort(key=lambda m: m.similarity, reverse=True)
    best = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None

    if best.similarity < SIMILARITY_THRESHOLD:
        return SignatureAssessment(MatchStatus.NONE, None, best, runner_up)
    if runner_up is not None and best.similarity - runner_up.similarity < AMBIGUITY_MARGIN:
        return SignatureAssessment(MatchStatus.AMBIGUOUS, None, best, runner_up)
    return SignatureAssessment(MatchStatus.MATCH, best, best, runner_up)


def find_matching_signature(
        observed_deltas: dict[str, float],
) -> Optional[SignatureMatch]:
    """A match to ONE family, or None — None also for an AMBIGUOUS result
    (safe by default for every caller); assess_signature_match returns the
    full state."""
    return assess_signature_match(observed_deltas).match


def similarity_to_family(observed_deltas: dict[str, float], family_id: str) -> Optional[float]:
    """The similarity of the observed deltas to the signature of ONE named
    family — the same filtering (NOISE_FLOOR) and the same measure as in
    find_matching_signature. Returns None when the family is not in
    SIGNATURES or when too few metrics exceed NOISE_FLOOR for the
    comparison to make sense.

    Used by recommendations.py to check whether the observation does NOT
    point at a family that produces the same reading as the repaired
    pathology (e.g. anisotropy vs duplication for ndds_fraction)."""
    significant = {k: v for k, v in observed_deltas.items() if abs(v) >= NOISE_FLOOR}
    if len(significant) < MIN_METRICS_FOR_GROUPING:
        return None
    for sig in SIGNATURES:
        if sig.family_id == family_id:
            return _cosine_similarity(significant, sig.metric_deltas)
    return None


def render_ambiguous_message(assessment: SignatureAssessment) -> str:
    """The message for an AMBIGUOUS result — it does not point at a family."""
    names = [m.signature.family_label for m in (assessment.best, assessment.runner_up) if m is not None]
    return (
        "The simultaneous metric changes match more than one pathology "
        f"pattern ({' / '.join(names)}) and it cannot be established which "
        "one — no family is indicated. Manual diagnosis recommended."
    )


def render_grouped_message(match: SignatureMatch) -> str:
    metrics_str = ", ".join(match.matched_metrics)
    return (
        f"The simultaneous metric changes ({metrics_str}) look like a "
        f"pattern close to: {match.signature.family_label} "
        f"(similarity {match.similarity:.0%})."
    )
