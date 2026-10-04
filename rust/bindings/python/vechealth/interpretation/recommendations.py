"""
interpretation/recommendations.py

The recommendation engine, an architecture of honesty — AUTO_FIXABLE /
REVIEW_REQUIRED / SAFE_BUT_UNPROVEN / NO_VALIDATED_FIX.

=== WHAT THE LEVELS MEAN ===

No repair is implemented in this package, and it never modifies the user's
vectors. The levels classify how far the benchmark's repair evidence reaches
for THIS diagnosis: AUTO_FIXABLE means the matched pathology is the one a
repair was measured on (the term follows the preprint, Section 4.10), not
that anything is applied automatically — `human_review_required` is always
True. The suggested actions are descriptions of the repairs in the preprint
(Section 3.7), flagged `implemented_in_package=False`.

=== FAIL-CLOSED ===

A recommendation does not depend on the metric name alone. The same
deduplication on an anisotropic base (the same ndds reading) removed 56.5% of
the corpus and lowered Recall@10 by 47.3% (preprint Table 20). AUTO_FIXABLE
therefore requires a diagnosis context (DiagnosisContext) pointing at the
family the repair was measured on, and not pointing at a family producing the
same reading for a different reason. Without context: REVIEW_REQUIRED.

=== AUTO_FIXABLE IS NOT UNIFORM ===

The two repairs behind it differ in where their parameter comes from:

  - `ndds_fraction` (dedup): the repair parameter (epsilon) follows
    STRUCTURALLY from the same threshold used for diagnosis — not an
    empirical tuning, but a property of the algorithm itself.
  - `hubness_max`/`hubness_skewness` (NICDM): k=10 was always used as a
    CONSTANT default (Recall@10 +7.1% to +7.8% in three realizations) — we
    never tested whether k should depend on the diagnosed hubness
    severity. Additionally: both repairs (NICDM and dedup) were measured
    ONLY at the strongest severity level of each family — it is not
    formally known whether the same repairs work equally well under a
    milder pathology.

Hence the `ParameterSource` field, distinguishing these two cases
explicitly instead of quietly equating them.

=== RESEARCH WARNING — DESIGNED FOR A NON-HUMAN RECIPIENT ===

These reports will likely be consumed by AI agents, not only humans. The
warning prose alone may be missed or misinterpreted by an agent.
Therefore EVERY recommendation carries THREE independent, redundant
warning signals, not one:
  1. `human_review_required: bool` — always True at this project stage, a
     boolean field for programmatic checks by an agent system
  2. `research_status: str` — a constant string `"experimental_in_progress"`,
     easy to match with a rule/condition, no NLP parsing required
  3. `disclaimer: str` — text readable by a human OR by a language model
     reading the report as text, not only as structured data
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .metric_profiles import get_profile, CausalEvidence
from .signatures import (
    SIMILARITY_THRESHOLD,
    MatchStatus,
    assess_signature_match,
    signed_delta,
    similarity_to_family,
)


class RecommendationLevel(str, Enum):
    AUTO_FIXABLE = "auto_fixable"
    # A validated repair EXISTS, but it has not been established that this
    # diagnosis is a case on which it was validated. This level exists so
    # that the engine fails closed. No other level describes this
    # state truthfully: SAFE_BUT_UNPROVEN claims the change is safe
    # (deduplication on an anisotropic database removed 56.5% of the corpus
    # and lowered recall by 47.3%), and NO_VALIDATED_FIX claims the repair
    # does not exist.
    REVIEW_REQUIRED = "review_required"
    SAFE_BUT_UNPROVEN = "safe_but_unproven"
    NO_VALIDATED_FIX = "no_validated_fix"


@dataclass(frozen=True)
class DiagnosisContext:
    """The diagnosis context without which no repair is AUTO_FIXABLE.

    `matched_family_id` — the family from signature matching
    (signatures.assess_signature_match), if one exceeded the threshold with
    a margin.
    `observed_deltas` — observed metric changes relative to the baseline
    (signatures.signed_delta), allowing a check whether the observation
    ALSO points at a family producing the same reading for a different
    reason."""
    matched_family_id: Optional[str] = None
    observed_deltas: Optional[dict[str, float]] = None
    # An AMBIGUOUS match — the family is not established
    # (matched_family_id = None), so the recommendation is fail-closed.
    signature_ambiguous: bool = False

    @classmethod
    def from_snapshots(cls, baseline: dict[str, float], current: dict[str, float]) -> "DiagnosisContext":
        deltas = {m: signed_delta(baseline[m], current[m]) for m in current if m in baseline}
        assessment = assess_signature_match(deltas)
        match = assessment.match                  # None when AMBIGUOUS
        return cls(matched_family_id=match.signature.family_id if match else None,
                   observed_deltas=deltas,
                   signature_ambiguous=assessment.status == MatchStatus.AMBIGUOUS)


class ParameterSource(str, Enum):
    """A distinction inside AUTO_FIXABLE — see the rationale in the module
    docstring. Do NOT conflate these two cases — they have different
    evidence strength."""
    DERIVED_FROM_DIAGNOSIS = "derived_from_diagnosis"
    FIXED_VALIDATED_DEFAULT = "fixed_validated_default"


# A constant, always-identical string — deliberately, so agent systems can
# match it with a simple comparison instead of NLP parsing.
RESEARCH_STATUS = "experimental_in_progress"


_LEVEL_BY_CAUSAL_EVIDENCE: dict[CausalEvidence, RecommendationLevel] = {
    CausalEvidence.STRONG: RecommendationLevel.AUTO_FIXABLE,
    CausalEvidence.WEAK: RecommendationLevel.SAFE_BUT_UNPROVEN,
    CausalEvidence.NONE: RecommendationLevel.NO_VALIDATED_FIX,
}

# The only two AUTO_FIXABLE cases in the project — an explicit, manually
# maintained map, not guessed from the metric name.
_PARAMETER_SOURCE: dict[str, ParameterSource] = {
    "ndds_fraction": ParameterSource.DERIVED_FROM_DIAGNOSIS,
    "hubness_max": ParameterSource.FIXED_VALIDATED_DEFAULT,
    "hubness_skewness": ParameterSource.FIXED_VALIDATED_DEFAULT,
}

# The family the repair was measured on (preprint Section 4.7), and the
# families that produce the SAME metric reading for a different reason.
# The confounding-family list comes from the benchmark: bases outside the
# target family whose change relative to the baseline falls within the
# target family's change range.
_REPAIR_TARGET_FAMILY: dict[str, str] = {
    "ndds_fraction": "G4_duplicates",
    "hubness_max": "G1_hubness",
    "hubness_skewness": "G1_hubness",
}

_CONFOUNDING_FAMILIES: dict[str, tuple[str, ...]] = {
    "ndds_fraction": ("G5_anisotropy",),
    "hubness_max": ("G3_imbalance", "G5_anisotropy", "G8_noise", "G9_outliers"),
    "hubness_skewness": ("G3_imbalance", "G5_anisotropy", "G8_noise", "G9_outliers"),
}

_REVIEW_WARNING: dict[str, str] = {
    "ndds_fraction": (
        "A high ndds_fraction does not distinguish duplication from "
        "anisotropy (a cone): on a database compressed toward the centroid, "
        "nearly every point has a neighbor within epsilon, even though "
        "there are no duplicates there. In the benchmark (anisotropy family, "
        "strongest level) the same deduplication removed 56.5% of the "
        "corpus, including 58.3% of all relevance-judged documents, and lowered "
        "Recall@10 by 47.3%, while ndds_fraction barely changed (0.9971 to "
        "0.9831). Deduplicate only after confirming the close pairs are "
        "actual duplicates (pair inspection, matching the duplication "
        "signature)."
    ),
    "hubness_max": (
        "Elevated hubness also appears with noise (G8), anisotropy (G5), "
        "outliers (G9) and imbalance (G3) — the same readings, a different "
        "cause. NICDM was measured only on hubness injected via "
        "attraction to hubs (G1: Recall@10 +7.1% to +7.8% in three "
        "independent realizations); its effect under the other causes was "
        "not tested."
    ),
}
_REVIEW_WARNING["hubness_skewness"] = _REVIEW_WARNING["hubness_max"]

_SUGGESTED_ACTION: dict[str, str] = {
    "ndds_fraction": (
        "epsilon-deduplication: for every document take its nearest other document and, "
        "when their chord distance is below epsilon (the same threshold that detected the "
        "problem), remove the one with the higher index"
    ),
    "hubness_max": (
        "NICDM rescaling of the retrieval distance (k = 10): divide each cosine distance by "
        "the geometric mean of the two points' local scales, a local scale being the mean "
        "cosine distance to a point's 10 nearest documents. This changes the distance "
        "function used at retrieval time"
    ),
    "anisotropy_mean_norm": (
        "centering (subtract the mean vector, then re-normalize; the smallest measured change "
        "in Recall@10, +0.2%), or shrinkage whitening with alpha = 0.1 (-3.0%). Do NOT apply "
        "full PCA whitening (Recall@10 fell from 0.6650 to 3.0e-6) or top-10 whitening "
        "(-89.4%)"
    ),
}
_SUGGESTED_ACTION["hubness_skewness"] = _SUGGESTED_ACTION["hubness_max"]

# What the benchmark measured for the repair each AUTO_FIXABLE metric points at
# (preprint Section 4.7, Tables 16-17), with the caveats that go with it.
_REPAIR_EVIDENCE: dict[str, str] = {
    "ndds_fraction": (
        "In the benchmark, epsilon-deduplication of a corpus in which every document had "
        "been cloned raised Recall@10 by 15.0% to 15.2% in three independent realizations "
        "(0.5495 to 0.6321). Part of that gain comes from the evaluation protocol: the "
        "clones carry identifiers absent from the relevance judgments, so every retrieved "
        "clone counts as a miss. The pass also removed 162 genuine duplicates, 95 of them "
        "relevance-judged, so Recall@10 stayed below the unperturbed baseline's 0.6479."
    ),
    "hubness_max": (
        "In the benchmark, NICDM rescaling on a store with injected hubs raised Recall@10 "
        "by 7.1% to 7.8% in three independent realizations (hubness_max 1739 to 514 in the "
        "first). That recovers about a tenth of the gap to the unperturbed baseline "
        "(+0.027 of 0.265), it changes the distance function used at retrieval time, and "
        "it was measured at one severity level only."
    ),
}
_REPAIR_EVIDENCE["hubness_skewness"] = _REPAIR_EVIDENCE["hubness_max"]


@dataclass(frozen=True)
class Recommendation:
    metric_name: str
    level: RecommendationLevel

    # --- Machine-readable signals (for agent systems) ---
    human_review_required: bool
    research_status: str

    # --- Content readable for a human / an LLM reading it as text ---
    headline: str
    disclaimer: str
    causal_evidence_note: str

    # --- Technical details, present only when applicable ---
    suggested_action: Optional[str] = None
    parameter_source: Optional[ParameterSource] = None
    parameter_source_note: Optional[str] = None

    # --- REVIEW_REQUIRED only ---
    # `suggested_action` is deliberately None there (an agent should not
    # execute it automatically); the candidate repair is in
    # `candidate_action`.
    review_reason: Optional[str] = None
    candidate_action: Optional[str] = None

    # --- Machine-readable: the described repair is NOT shipped in this
    # package (always False in this version). An agent must not look for a
    # function to call. ---
    implemented_in_package: bool = False


_GENERAL_DISCLAIMER = (
    "THIS IS THE OUTPUT OF AN ACTIVELY DEVELOPED RESEARCH PROJECT, NOT A "
    "100% CONFIRMED SOLUTION FOR EVERY CASE. We point at a problem and a "
    "potential repair approach — not a guaranteed, universal fix. It "
    "requires human review and a decision before execution, especially in "
    "a production environment."
)


def evidence_ceiling_level(metric_name: str) -> RecommendationLevel:
    """The highest level a metric can reach from causal-evidence strength
    alone (the old classification, independent of the diagnosis). This is
    NOT the recommendation level — that requires context, see
    classify_recommendation_level."""
    profile = get_profile(metric_name)
    return _LEVEL_BY_CAUSAL_EVIDENCE[profile.causal_evidence]


def _review_reason(metric_name: str, context: Optional[DiagnosisContext]) -> Optional[str]:
    """None if the context justifies AUTO_FIXABLE; otherwise the reason."""
    target = _REPAIR_TARGET_FAMILY.get(metric_name)
    if target is None:
        return "No family is defined on which the repair was validated."
    if context is None or (context.matched_family_id is None and not context.observed_deltas):
        return (
            "No diagnosis context (signature match or observed metric "
            "changes) — it cannot be established whether this is a case on "
            "which the repair was validated."
        )
    if context.matched_family_id != target:
        if context.matched_family_id:
            found = context.matched_family_id
        elif context.signature_ambiguous:
            found = "ambiguous (more than one family, family not established)"
        else:
            found = "no family above the threshold"
        return (
            f"The change pattern does not match the family the repair was "
            f"validated on ({target}); match: {found}."
        )
    if context.observed_deltas:
        for fam in _CONFOUNDING_FAMILIES.get(metric_name, ()):
            sim = similarity_to_family(context.observed_deltas, fam)
            if sim is not None and sim >= SIMILARITY_THRESHOLD:
                return (
                    f"The change pattern also matches family {fam} "
                    f"(similarity {sim:.2f} >= {SIMILARITY_THRESHOLD:.2f}), "
                    f"which produces the same reading for a different "
                    f"reason."
                )
    return None


def classify_recommendation_level(
        metric_name: str,
        context: Optional[DiagnosisContext] = None,
) -> RecommendationLevel:
    """The recommendation level for a SPECIFIC diagnosis. FAIL CLOSED:
    without context, or when the context does not point at the family the
    repair was validated on (or also points at a family producing the same
    reading), a metric with strong causal evidence gets REVIEW_REQUIRED,
    never AUTO_FIXABLE."""
    ceiling = evidence_ceiling_level(metric_name)
    if ceiling != RecommendationLevel.AUTO_FIXABLE:
        return ceiling
    if _review_reason(metric_name, context) is not None:
        return RecommendationLevel.REVIEW_REQUIRED
    return RecommendationLevel.AUTO_FIXABLE


def build_recommendation(
        metric_name: str,
        context: Optional[DiagnosisContext] = None,
) -> Recommendation:
    """Main entry point — returns a full recommendation, ALWAYS with the
    research warning, regardless of the level. `context` is optional
    (backward compatibility), but without it no recommendation is
    AUTO_FIXABLE."""
    profile = get_profile(metric_name)
    level = classify_recommendation_level(metric_name, context)

    if level == RecommendationLevel.REVIEW_REQUIRED:
        reason = _review_reason(metric_name, context)
        warning = _REVIEW_WARNING.get(metric_name, "")
        return Recommendation(
            metric_name=metric_name,
            level=level,
            human_review_required=True,
            research_status=RESEARCH_STATUS,
            headline=(
                f"A repair for '{metric_name}' was tested in the benchmark, but this "
                f"diagnosis was not confirmed as a case on which it was measured — "
                f"requires human review."
            ),
            disclaimer=_GENERAL_DISCLAIMER,
            causal_evidence_note=f"{reason} {warning}".strip(),
            suggested_action=None,
            parameter_source=None,
            parameter_source_note=None,
            review_reason=reason,
            candidate_action=_SUGGESTED_ACTION.get(metric_name),
        )

    if level == RecommendationLevel.AUTO_FIXABLE:
        param_source = _PARAMETER_SOURCE.get(metric_name)
        action = _SUGGESTED_ACTION.get(metric_name)

        if param_source == ParameterSource.DERIVED_FROM_DIAGNOSIS:
            param_note = (
                "The repair parameter follows structurally from the same "
                "threshold that detected the problem — this is not an "
                "empirical tuning."
            )
        else:  # FIXED_VALIDATED_DEFAULT
            param_note = (
                "The parameter is a CONSTANT, validated default value, NOT "
                "derived adaptively from your specific diagnosis. Validated "
                "only at the strongest tested pathology severity — under a "
                "milder one it may be less effective; we did not test that "
                "formally."
            )

        return Recommendation(
            metric_name=metric_name,
            level=level,
            human_review_required=True,
            research_status=RESEARCH_STATUS,
            headline=(
                f"The matched pattern is the one a repair for '{metric_name}' was "
                f"measured on in the benchmark (the repair is not implemented in this "
                f"package)."
            ),
            disclaimer=_GENERAL_DISCLAIMER,
            causal_evidence_note=_REPAIR_EVIDENCE[metric_name],
            suggested_action=action,
            parameter_source=param_source,
            parameter_source_note=param_note,
        )

    elif level == RecommendationLevel.SAFE_BUT_UNPROVEN:
        return Recommendation(
            metric_name=metric_name,
            level=level,
            human_review_required=True,
            research_status=RESEARCH_STATUS,
            headline=f"A geometry change was tested for '{metric_name}' — WITHOUT a recall benefit.",
            disclaimer=_GENERAL_DISCLAIMER,
            causal_evidence_note=(
                "On the anisotropy family, higher anisotropy went with slightly higher "
                "Recall@10 (0.6479 to 0.6654), and removing it did not raise recall: "
                "centering changed it by +0.2%, shrinkage whitening (alpha = 0.1) by -3.0%, "
                "full PCA whitening lowered it from 0.6650 to 3.0e-6 and top-10 whitening "
                "by 89.4% (preprint Section 4.7). At the strongest injected level Recall@10 "
                "also rose on two further corpora. This describes anisotropy induced by "
                "interpolation toward the centroid, not the anisotropy of trained "
                "encoders: do NOT treat it as a lever that improves retrieval quality."
            ),
            suggested_action=_SUGGESTED_ACTION.get(metric_name),
            parameter_source=None,
            parameter_source_note=None,
        )

    else:  # NO_VALIDATED_FIX
        return Recommendation(
            metric_name=metric_name,
            level=level,
            human_review_required=True,
            research_status=RESEARCH_STATUS,
            headline=f"No repair was tested for '{metric_name}'.",
            disclaimer=_GENERAL_DISCLAIMER,
            causal_evidence_note=(
                "No repair for the pathology this statistic reads was tested in the "
                "benchmark. Treat a change as a reading to investigate, not as a "
                "diagnosis of retrieval harm: the statistics did not predict retrieval "
                "quality across held-out pathology families. This is a deliberate gap, "
                "not an oversight."
            ),
            suggested_action=None,
            parameter_source=None,
            parameter_source_note=None,
        )
