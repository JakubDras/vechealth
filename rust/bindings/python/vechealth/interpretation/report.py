"""
interpretation/report.py

The composition layer tying the lower-level modules into a ready report.
Architectural pattern: matplotlib pyplot->Axes / scikit-learn
estimator->Pipeline — the low-level layer (interpret_drift,
compute_priority, get_badges, find_matching_signature,
build_recommendation) stays unchanged and fully
useful on its own ("VecHealth as an engine, like numpy"). This module adds
THREE new, progressively more convenient layers ON TOP, not instead:

    build_metric_report()            <- compositional (takes a ready
                                          diagnostic_strength, flexible,
                                          works outside drift mode too)
    build_metric_report_from_drift() <- a thin convenience wrapper that
                                          literally calls interpret_drift()
                                          + build_metric_report()
    build_full_report()              <- plug-and-play, a full metric set
                                          at once + signature grouping
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .metric_profiles import get_profile, MetricProfile
from .priority import compute_priority, PriorityResult, rank_metrics_by_priority
from .badges import get_badges, MetricBadges
from .drift import interpret_drift, DriftInterpretation
from .signatures import (
    assess_signature_match,
    MatchStatus,
    render_ambiguous_message,
    render_grouped_message,
    signed_delta,
    SignatureMatch,
    NOISE_FLOOR,
)
from .deployment_context import (
    DeploymentContext,
    IndexType,
    ann_context_warning,
    domain_interpretation_caveat,
    index_loss_is_severe,
)
from .limitations import LIMITATIONS
from .recommendations import (
    build_recommendation,
    DiagnosisContext,
    Recommendation,
)


# ============================================================
# Tone dictionary — one source of truth for the messages of this layer.
# HONEST CAVEAT: content written earlier (badges.py, drift.py,
# recommendations.py) predates this dictionary and may not match it 100%
# word-for-word — a deliberately documented, minor inconsistency.
# ============================================================

class Level1Status(str, Enum):
    URGENT = "urgent"
    OBSERVE = "observe"
    NORMAL = "normal"


_STATUS_LABEL: dict[Level1Status, str] = {
    Level1Status.URGENT: "URGENT",
    Level1Status.OBSERVE: "OBSERVE",
    Level1Status.NORMAL: "NORMAL",
}


def _classify_status(diagnostic_strength: float, index_loss_severe: bool) -> Level1Status:
    """NORMAL: the change is below the single noise-floor constant.

    URGENT: the change exceeds it AND the benchmark measured a severe
    index-dependent loss for the matched pathology under the index the user
    declared (`deployment_context.index_loss_is_severe`). That is a documented
    consequence of a specific pattern on a specific index type — not a
    prediction of the user's recall, which these statistics did not support.

    OBSERVE: any other change beyond the noise floor.

    Whether a repair exists plays no part: a status must not depend on it."""
    if diagnostic_strength < NOISE_FLOOR:
        return Level1Status.NORMAL
    if index_loss_severe:
        return Level1Status.URGENT
    return Level1Status.OBSERVE


# ============================================================
# The "mandatory example" rule — `example_provider` is an
# optional extension point: the Hub Inspector plugs in here
# (`vechealth.hub_inspector.hub_inspector_as_example_provider`). Without a
# provider — an explicit, honest absence message, not silence and not a
# fabricated example.
# ============================================================

# Metrics for which a SINGLE-POINT example makes conceptual sense
# (future Hub Inspector candidates) — as opposed to purely aggregate
# metrics, where pointing at "one point" is meaningless by definition.
_POINT_LEVEL_EXAMPLE_ELIGIBLE = frozenset({
    "hubness_max", "hubness_skewness", "ndds_fraction", "qmas_orphans_pct",
})

ExampleProvider = Callable[[str], Optional[str]]


def _resolve_example_note(metric_name: str, example_provider: Optional[ExampleProvider]) -> str:
    if example_provider is not None:
        result = example_provider(metric_name)
        if result:
            return result
    if metric_name in _POINT_LEVEL_EXAMPLE_ELIGIBLE:
        return (
            "No concrete-point example attached. Use the Hub Inspector "
            "(`vechealth.hub_inspector.inspect_hubs`) to list the points behind this "
            "reading, and pass `hub_inspector_as_example_provider(evaluator)` as "
            "`example_provider` to include one here."
        )
    return (
        "This metric is aggregate by nature — pointing at a single point "
        "as an example does not apply here."
    )


# ============================================================
# "Zero required labels" — generated ONCE per report
# (build_full_report), not per metric.
# ============================================================

def zero_labels_statement() -> str:
    return (
        "This diagnosis required no labeled queries and no LLM calls — "
        "only the geometry of your vectors."
    )


# ============================================================
# COMPOSITIONAL LAYER — build_metric_report()
# ============================================================

@dataclass(frozen=True)
class MetricReport:
    metric_name: str

    # --- Level 1 (always visible) ---
    status: Level1Status
    status_label: str
    level1_summary: str

    # --- Level 2 (expandable) ---
    level2_reasoning: str
    ann_warning: Optional[str]
    domain_caveat: Optional[str]
    recommendation: Recommendation
    example_note: str
    group_context_caveat: Optional[str]

    # --- Level 3 (expandable/verbose) ---
    drift: Optional[DriftInterpretation]
    priority: PriorityResult
    badges: MetricBadges
    profile: MetricProfile


_GROUP_CONTEXT_CAVEAT_TEMPLATE = (
    "This metric is part of a broader, grouped pattern (see the group message "
    "above). On its own a single statistic is a symptom, not a verdict: of the eight "
    "statistics that have a designated pathology family only five were specific to it "
    "in the benchmark, and a change can be a side effect of ANOTHER pathology. Trust "
    "the group message first."
)


def build_metric_report(
        metric_name: str,
        diagnostic_strength: float,
        deployment_context: Optional[DeploymentContext] = None,
        matched_family_id: Optional[str] = None,
        level1_summary_override: Optional[str] = None,
        example_provider: Optional[ExampleProvider] = None,
        is_part_of_group: bool = False,
        diagnosis_context: Optional[DiagnosisContext] = None,
        signature_ambiguous: bool = False,
) -> MetricReport:
    """The compositional layer — takes an ALREADY READY
    `diagnostic_strength`, does not compute it on its own. Works
    independently of drift mode (e.g. if a no-baseline mode is ever built)
    — this is the flexibility it was about: VecHealth as an engine, not
    only a finished product.

    The diagnosis context is passed to build_recommendation: an
    explicit `diagnosis_context`, or, if not given, just
    `matched_family_id`. Without either one, a recommendation cannot be
    AUTO_FIXABLE."""
    profile = get_profile(metric_name)
    priority = compute_priority(metric_name, diagnostic_strength)
    badges = get_badges(metric_name)
    if diagnosis_context is None and matched_family_id is not None:
        diagnosis_context = DiagnosisContext(matched_family_id=matched_family_id)
    recommendation = build_recommendation(metric_name, diagnosis_context)
    index_loss_severe = (
        deployment_context is not None
        and not signature_ambiguous
        and index_loss_is_severe(matched_family_id, deployment_context.index_type)
    )
    status = _classify_status(diagnostic_strength, index_loss_severe)

    level1_summary = level1_summary_override or (
        f"'{metric_name}': signal strength {diagnostic_strength:.2f} "
        f"(priority {priority.priority_score:.2f})."
    )

    ann_warning = None
    if deployment_context is not None:
        # For an ambiguous match, a cautious message with no family
        # (matched_family_id is None then).
        ann_warning = ann_context_warning(matched_family_id, deployment_context.index_type,
                                          ambiguous=signature_ambiguous)

    domain_caveat = None
    if deployment_context is not None:
        domain_caveat = domain_interpretation_caveat(metric_name, deployment_context.domain_hint)

    example_note = _resolve_example_note(metric_name, example_provider)

    # A caveat that a metric inside a recognized group is a symptom, not a
    # verdict, and may be a side effect of another pathology.
    group_context_caveat = _GROUP_CONTEXT_CAVEAT_TEMPLATE if is_part_of_group else None

    level2_reasoning = (
        f"{recommendation.causal_evidence_note} "
        f"[{recommendation.research_status}] {recommendation.disclaimer}"
    )

    return MetricReport(
        metric_name=metric_name,
        status=status,
        status_label=_STATUS_LABEL[status],
        level1_summary=level1_summary,
        level2_reasoning=level2_reasoning,
        ann_warning=ann_warning,
        domain_caveat=domain_caveat,
        recommendation=recommendation,
        example_note=example_note,
        group_context_caveat=group_context_caveat,
        drift=None,
        priority=priority,
        badges=badges,
        profile=profile,
    )


# ============================================================
# CONVENIENCE LAYER — build_metric_report_from_drift()
# Deliberately a THIN wrapper — all the logic lives in
# build_metric_report() above; this function only supplies
# diagnostic_strength from real data and swaps level1_summary for the
# richer version from interpret_drift().
# ============================================================

def build_metric_report_from_drift(
        metric_name: str,
        baseline_value: float,
        current_value: float,
        deployment_context: Optional[DeploymentContext] = None,
        matched_family_id: Optional[str] = None,
        example_provider: Optional[ExampleProvider] = None,
        is_part_of_group: bool = False,
        diagnosis_context: Optional[DiagnosisContext] = None,
        signature_ambiguous: bool = False,
) -> MetricReport:
    drift = interpret_drift(metric_name, baseline_value, current_value)
    report = build_metric_report(
        metric_name=metric_name,
        diagnostic_strength=drift.diagnostic_strength,
        deployment_context=deployment_context,
        matched_family_id=matched_family_id,
        level1_summary_override=drift.message,
        example_provider=example_provider,
        is_part_of_group=is_part_of_group,
        diagnosis_context=diagnosis_context,
        signature_ambiguous=signature_ambiguous,
    )
    # Swap the drift field (frozen dataclass -> a new instance, not mutation)
    from dataclasses import replace
    return replace(report, drift=drift)


# ============================================================
# PLUG-AND-PLAY LAYER — build_full_report()
# ============================================================

@dataclass(frozen=True)
class FullReport:
    zero_labels_statement: str
    metric_reports: list[MetricReport]  # sorted by priority_score, descending
    grouped_match: Optional[SignatureMatch]
    grouped_message: Optional[str]
    # MATCH / AMBIGUOUS / NONE (for AMBIGUOUS, grouped_match = None and
    # grouped_message = the ambiguity message).
    signature_status: MatchStatus = MatchStatus.NONE
    # What this report does NOT say (see `limitations.py`) — the same standing
    # statements for every report.
    limitations: tuple[str, ...] = LIMITATIONS


def build_full_report(
        baseline_snapshot: dict[str, float],
        current_snapshot: dict[str, float],
        deployment_context: Optional[DeploymentContext] = None,
        example_provider: Optional[ExampleProvider] = None,
) -> FullReport:
    """The plug-and-play layer — for someone who wants zero configuration.
    Two arguments (baseline, current), a ready report."""
    # Step 1: check whether several metrics at once form a recognizable pattern
    signed_deltas = {}
    for metric in current_snapshot:
        if metric not in baseline_snapshot:
            continue
        # The same rule as when building SIGNATURES — relative change, and
        # the absolute difference when the baseline is 0.
        signed_deltas[metric] = signed_delta(baseline_snapshot[metric], current_snapshot[metric])

    assessment = assess_signature_match(signed_deltas)
    match = assessment.match                      # None for AMBIGUOUS and NONE
    ambiguous = assessment.status == MatchStatus.AMBIGUOUS
    if match:
        grouped_message = render_grouped_message(match)
    elif ambiguous:
        grouped_message = render_ambiguous_message(assessment)
    else:
        grouped_message = None
    matched_family_id = match.signature.family_id if match else None
    matched_metrics_set = set(match.matched_metrics) if match else set()
    diagnosis_context = DiagnosisContext(
        matched_family_id=matched_family_id,
        observed_deltas=signed_deltas,
    )

    # Step 2: a full per-metric report, with the matched family's context (if any)
    reports = [
        build_metric_report_from_drift(
            metric_name=metric,
            baseline_value=baseline_snapshot[metric],
            current_value=current_snapshot[metric],
            deployment_context=deployment_context,
            matched_family_id=matched_family_id,
            example_provider=example_provider,
            is_part_of_group=(metric in matched_metrics_set),
            diagnosis_context=diagnosis_context,
            signature_ambiguous=ambiguous,
        )
        for metric in current_snapshot
        if metric in baseline_snapshot
    ]
    reports.sort(key=lambda r: r.priority.priority_score, reverse=True)

    return FullReport(
        zero_labels_statement=zero_labels_statement(),
        metric_reports=reports,
        grouped_match=match,
        grouped_message=grouped_message,
        signature_status=assessment.status,
        limitations=LIMITATIONS,
    )
