"""
interpretation/metric_profiles.py

Single, structured source of truth about the reliability and character of
each of VecHealth's 11 metrics. Every numeric value and category comes from
the benchmark's result files, as reported in the preprint — none is an
estimate or a value "from memory".

Sources (all computed by THIS library, after the float64 numerics fix, on the
53 benchmark bases):
  vif_after_p0, ablation_contribution — the variance inflation factor of the
    metric and the drop of the in-sample R^2 for Recall@10 when it is removed
    (preprint Table 13). NOTE: ablation is an IN-SAMPLE quantity from a model
    that does not generalize across pathology families (the leave-one-family-
    out R^2 is negative; the family-block permutation p is about 0.3-0.4) —
    the ablation ranking changes completely between metric-computation
    conventions. The field is purely descriptive; no library logic reads it
    (priority.py only uses causal_evidence and reliability_category).
  reliability_category — the subsampling study (sixty subsamples of the
    baseline corpus; preprint Table 31), RULE v1: A iff MAPE < 2% at every
    fraction; B iff mean/full in [0.9p, 1.1p] at p >= 5%; D iff CV@50% >
    CV@0.5% (the spread GROWS with N); C otherwise; unknown iff the full value
    = 0. Rule v2 (added after inspecting the data: D iff CV@50% >= 5%) is a
    sensitivity check only — it differs for hubness_skewness (v1 C, v2 D) and
    qmas_orphans_pct (v1 D, v2 C).

This file is the single place the interpretation layer reads these facts from.
Changes here propagate everywhere — this is intentional.

NOTE on the `dose_response_range` field: deliberately empty (None) for all
metrics in this version. Filling it requires a deliberate, human decision about
which benchmark bases represent the "mild"/"moderate"/"strong" level for each
metric — this must not be guessed or automated without review.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class ReliabilityCategory(str, Enum):
    """Reliability categories under subsampling (the subsampling study)."""
    A_STABLE = "A"           # MAPE < 2% at every sample fraction
    B_CORRECTABLE = "B"      # on a sample ~ p x full value (bias known; the library does NOT correct it)
    C_CONVERGING = "C"       # CV and MAPE decrease with sample size; bias disappears only on the full dataset
    D_UNSTABLE = "D"         # spread across draws GROWS with sample size (CV@50% > CV@0.5%)
    UNKNOWN = "unknown"      # not part of the subsampling study


class CausalEvidence(str, Enum):
    """Strength of repair-and-measure evidence (preprint Section 4.7): whether
    a targeted repair of the pathology this statistic reads was measured to
    change recall in the benchmark. It is evidence about the repair on that
    benchmark — not a demonstration that the statistic causes recall loss in
    general."""
    STRONG = "strong"  # a direct repair intervention really improved recall
    WEAK = "weak"       # strong statistical signal, but no credible causality
    NONE = "none"       # never tested causally (no repair algorithm)


class KnownDirection(str, Enum):
    """Whether a direction of a metric's drift is validated as 'good'.

    No statistic has one in the benchmark (preprint Sections 4.3-4.4,
    Table 15): the same statistic moves in opposite directions across
    pathology families, and several move toward what would read as
    improvement while Recall@10 falls. Every shipped profile is AMBIGUOUS;
    the other two values are reserved for a direction a future version
    validates."""
    LOWER_BETTER = "lower_better"
    HIGHER_BETTER = "higher_better"
    AMBIGUOUS = "ambiguous"  # no credible, unambiguous direction


@dataclass(frozen=True)
class MetricProfile:
    name: str
    reliability_category: ReliabilityCategory
    causal_evidence: CausalEvidence
    known_direction: KnownDirection
    vif_after_p0: float
    ablation_contribution: float
    cross_architecture_universal: bool   # directionally universal across encoders? (preprint Table 15)
    cross_corpus_replicated: bool        # does the direction replicate on new corpora? (preprint Section 4.9)
    dose_response_range: Optional[Tuple[float, float, float]] = None
    notes: str = ""


METRIC_PROFILES: dict[str, MetricProfile] = {

    "hubness_max": MetricProfile(
        name="hubness_max",
        reliability_category=ReliabilityCategory.D_UNSTABLE,
        causal_evidence=CausalEvidence.STRONG,
        known_direction=KnownDirection.AMBIGUOUS,
        vif_after_p0=18.2,
        ablation_contribution=0.0283,
        cross_architecture_universal=False,
        cross_corpus_replicated=True,
        notes=(
            "Together with hubness_skewness, the only statistics whose "
            "correlation sign with recall disagrees across three encoders "
            "(+0.18/-0.63/+0.42; preprint Table 15). The spread across "
            "subsamples GROWS with N (CV 7.5% at 0.5% sampling -> 30.1% at "
            "50%; category D, Table 31), so a single reading — even on a "
            "large sample — should be treated with caution. Under dimension "
            "truncation it falls together with recall, i.e. it moves toward "
            "'improvement' during degradation (Table 8) — outside the core "
            "set. The strongest repair evidence of any statistic: NICDM "
            "rescaling raised Recall@10 by 7.1% to 7.8% in three independent "
            "realizations (Section 4.7)."
        ),
    ),

    "hubness_skewness": MetricProfile(
        name="hubness_skewness",
        reliability_category=ReliabilityCategory.C_CONVERGING,
        causal_evidence=CausalEvidence.STRONG,
        known_direction=KnownDirection.AMBIGUOUS,
        vif_after_p0=17.8,
        ablation_contribution=0.0162,
        cross_architecture_universal=False,
        cross_corpus_replicated=True,
        notes=(
            "Same family as hubness_max and the same caveat about the "
            "correlation sign across encoders (-0.10/-0.52/+0.47; Table 15). "
            "Under subsampling it behaves differently: CV DECREASES (10.0% at "
            "0.5% sampling -> 6.8% at 50%) — category C under rule v1 "
            "(Table 31); the post-hoc rule v2 (CV@50% >= 5%) would give D."
        ),
    ),

    "intrinsic_dim_mean": MetricProfile(
        name="intrinsic_dim_mean",
        reliability_category=ReliabilityCategory.C_CONVERGING,
        causal_evidence=CausalEvidence.NONE,
        known_direction=KnownDirection.AMBIGUOUS,
        vif_after_p0=854.6,
        ablation_contribution=0.0012,
        cross_architecture_universal=True,
        cross_corpus_replicated=True,
        notes=(
            "Extremely high VIF (855; Table 13) — strongly redundant with the "
            "other statistics. Not specific: G8 (noise) raises it 16.9x more "
            "strongly than G6 (collapse) lowers it (Table 7). No repair was "
            "tested for it."
        ),
    ),

    "ndds_fraction": MetricProfile(
        name="ndds_fraction",
        reliability_category=ReliabilityCategory.B_CORRECTABLE,
        causal_evidence=CausalEvidence.STRONG,
        known_direction=KnownDirection.AMBIGUOUS,
        vif_after_p0=5.6,
        ablation_contribution=0.0172,
        cross_architecture_universal=True,
        cross_corpus_replicated=True,
        notes=(
            "The only AUTO_FIXABLE case with a repair parameter derived "
            "directly from the diagnosis — the epsilon used for detection is "
            "the epsilon used for deduplication — but ONLY when the diagnosis "
            "context indicates G4_duplicates and does not indicate "
            "G5_anisotropy (recommendations.py, fail-closed: the same "
            "deduplication on G5 L4 removed 56.5% of the corpus and lowered "
            "Recall@10 by 47.3%; Table 20). On G4 the repair raised Recall@10 "
            "by 15.0% to 15.2% in three realizations, partly because of the "
            "evaluation protocol (Section 4.7). On a subsample of fraction p "
            "the value is about p times the full-corpus value (category B); "
            "the library does not correct it."
        ),
    ),

    "dispersion_1nn": MetricProfile(
        name="dispersion_1nn",
        reliability_category=ReliabilityCategory.C_CONVERGING,
        causal_evidence=CausalEvidence.NONE,
        known_direction=KnownDirection.AMBIGUOUS,
        vif_after_p0=121.9,
        ablation_contribution=0.0044,
        cross_architecture_universal=True,
        cross_corpus_replicated=True,
        notes=(
            "Different pathology families push this metric in different "
            "directions (e.g. G4 duplicates down, G8 noise up) — there is no "
            "single universal 'good' drift direction; interpreting the "
            "direction requires context, not the value alone. In the core set "
            "(Table 15: specific, category C, consistent sign across three "
            "encoders); redundant with dispersion_10nn (rho = 0.95). Its mean "
            "is accumulated in float64."
        ),
    ),

    "dispersion_10nn": MetricProfile(
        name="dispersion_10nn",
        reliability_category=ReliabilityCategory.C_CONVERGING,
        causal_evidence=CausalEvidence.NONE,
        known_direction=KnownDirection.AMBIGUOUS,
        vif_after_p0=54.6,
        ablation_contribution=0.0082,
        cross_architecture_universal=True,
        cross_corpus_replicated=True,
        notes=(
            "Same ambiguous-direction caveat as dispersion_1nn. Not specific "
            "(G5 moves it 1.19x more strongly than G7/G8, Table 7) and "
            "redundant with dispersion_1nn — outside the core set (Table 15)."
        ),
    ),

    "anisotropy_mean_norm": MetricProfile(
        name="anisotropy_mean_norm",
        reliability_category=ReliabilityCategory.A_STABLE,
        causal_evidence=CausalEvidence.WEAK,
        known_direction=KnownDirection.AMBIGUOUS,
        vif_after_p0=12.1,
        ablation_contribution=0.0194,
        cross_architecture_universal=True,
        cross_corpus_replicated=True,
        notes=(
            "ANISOTROPY PARADOX (from the data only — ablation is not proof): "
            "directionally specific to G5 (+95% vs +25% for the next family "
            "in the same direction), the only metric fully stable under "
            "subsampling (category A), positive correlation sign with recall "
            "for three encoders — but on G5 recall GROWS with anisotropy: at "
            "the strongest injected level Recall@10 rose relative to the "
            "unperturbed baseline on all three corpora (B3 +2.7%, NQ +7.2%, "
            "MS MARCO +8.1%), and removing the anisotropy did not raise it. "
            "This describes anisotropy induced by interpolation toward the "
            "centroid, not that of trained encoders. TREAT AS A CREDIBLE "
            "GEOMETRY MEASUREMENT (stable, specific), NOT AS A SIGNAL OF "
            "RECALL HARM NOR A REPAIR LEVER. If repaired nonetheless: "
            "centering or the shrinkage variant only — naive whitening "
            "catastrophically destroys recall (Recall@10 fell from 0.665 to "
            "3.02e-6; Table 18)."
        ),
    ),

    "outlier_fraction": MetricProfile(
        name="outlier_fraction",
        reliability_category=ReliabilityCategory.UNKNOWN,
        causal_evidence=CausalEvidence.NONE,
        known_direction=KnownDirection.AMBIGUOUS,
        vif_after_p0=738.5,
        ablation_contribution=0.00041,
        cross_architecture_universal=False,
        cross_corpus_replicated=False,
        notes=(
            "Threshold: a chord distance on the unit sphere; the default is 1.2, "
            "the value used for the preprint's results (it is absolute — "
            "recalibrate it for other encoders or dimensionalities; the "
            "data-derived alternative, 3x the mean 1-NN distance, lands close "
            "to the largest possible chord distance, 2.0, on typical text "
            "embeddings and essentially flags nothing). With 1.2 the statistic "
            "reads exactly the injected fraction on G9 (0.05-0.6), saturates "
            "on G8, flags a single point on G7 at the two strongest levels, "
            "and is 0 on the unperturbed baseline and on the corpus with "
            "nothing injected (Tables 5 and 30). It is not specific: G8 "
            "moves it more than G9 does (Table 7). The identical rank correlation across the three "
            "encoders is mechanical — the statistic is non-zero on one of the "
            "nine bases — so it is not evidence of cross-architecture "
            "agreement. Extremely high VIF, low unique contribution."
        ),
    ),

    "snc_score": MetricProfile(
        name="snc_score",
        reliability_category=ReliabilityCategory.C_CONVERGING,
        causal_evidence=CausalEvidence.NONE,
        known_direction=KnownDirection.AMBIGUOUS,
        vif_after_p0=21.6,
        ablation_contribution=0.0285,
        cross_architecture_universal=True,
        cross_corpus_replicated=True,
        notes=(
            "No repair was tested for it. The library computes it over ALL "
            "points (no sample); the original evaluator averaged a 10,000-point "
            "sample, which on G9 consisted of the injected outliers (a seed "
            "collision; Section 4.10). In the core set (Table 15)."
        ),
    ),

    "qmas_mean_1nn": MetricProfile(
        name="qmas_mean_1nn",
        reliability_category=ReliabilityCategory.C_CONVERGING,
        causal_evidence=CausalEvidence.NONE,
        known_direction=KnownDirection.AMBIGUOUS,
        vif_after_p0=11.7,
        ablation_contribution=0.0021,
        cross_architecture_universal=True,
        cross_corpus_replicated=True,
        notes=(
            "No repair was tested for it. Under dimension truncation it falls "
            "together with recall, i.e. it moves toward 'improvement' during "
            "degradation (Table 8) — outside the core set (Table 15)."
        ),
    ),

    "qmas_orphans_pct": MetricProfile(
        name="qmas_orphans_pct",
        reliability_category=ReliabilityCategory.D_UNSTABLE,
        causal_evidence=CausalEvidence.NONE,
        known_direction=KnownDirection.AMBIGUOUS,
        vif_after_p0=6.1,
        ablation_contribution=0.0010,
        cross_architecture_universal=True,
        cross_corpus_replicated=True,
        notes=(
            "Redundant with qmas_mean_1nn (rho = 0.99). Unique contribution "
            "(in-sample ablation) 0.0010 on the library's values, and the "
            "ablation ranking is not stable. Under subsampling category D "
            "under rule v1 (CV grows 0.29% -> 1.02% — an absolutely small "
            "spread; C under rule v2; Table 31). It moves strongly with the "
            "vector dimension (Table 8), so its level is not comparable across "
            "dimensionalities. A candidate for moving from the main metrics to "
            "optional/advanced in the user interface."
        ),
    ),

}


def get_profile(metric_name: str) -> MetricProfile:
    """Return the metric's profile. Raises KeyError with a readable message
    if the metric has no entry yet — deliberately, so CI catches a newly
    added metric without a profile instead of silently returning None
    somewhere deeper in the interpretation layer. See
    tests/test_metric_profiles_complete.py."""
    try:
        return METRIC_PROFILES[metric_name]
    except KeyError:
        raise KeyError(
            f"No MetricProfile for '{metric_name}'. Every metric returned "
            f"by VecHealthEvaluator.compute_all() MUST have a matching "
            f"entry in METRIC_PROFILES (interpretation/metric_profiles.py) "
            f"before it reaches the interpretation layer."
        ) from None


ALL_METRIC_NAMES: frozenset[str] = frozenset(METRIC_PROFILES.keys())
