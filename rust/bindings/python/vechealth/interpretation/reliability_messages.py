"""
interpretation/reliability_messages.py

Generates sentences about (a) the confidence/reliability of a metric
reading (categories A-D, from the subsampling study) and (b) the strength
of repair evidence — two INDEPENDENT prioritization axes, never merged into
a single statement.

The "explicit, not implicit" rule — the confidence message is always
explicit in the text, never a silent default.
"""

from .metric_profiles import get_profile, ReliabilityCategory, CausalEvidence


_RELIABILITY_MESSAGES: dict[ReliabilityCategory, str] = {
    ReliabilityCategory.A_STABLE: (
        "This metric is stable regardless of sample size — you can trust a "
        "single reading."
    ),
    # No library code corrects a value computed on a sample, so the message
    # must not suggest it does. The proportionality was checked in the
    # subsampling study: the ratio of the sampled to the full value, divided
    # by p, is 0.97-1.05.
    ReliabilityCategory.B_CORRECTABLE: (
        "This metric has a known, predictable error when computed on a "
        "sample instead of the full dataset: on a sample that is a fraction "
        "p of the dataset, its value is approximately p times the full-"
        "dataset value. The library does NOT correct this — compute it on "
        "the full dataset or divide the sample result by p."
    ),
    ReliabilityCategory.C_CONVERGING: (
        "This metric converges to a stable value at larger sample sizes — "
        "treat a reading on a small sample as an approximation, not a "
        "final value."
    ),
    # Category D means exactly that the spread GROWS with sample size
    # (hubness_max: CV 7.5% -> 30.1%, qmas_orphans_pct: 0.29% -> 1.02%;
    # preprint Table 31), so the message must not say that a larger sample
    # helps.
    ReliabilityCategory.D_UNSTABLE: (
        "The spread of this metric across random samples GROWS with sample "
        "size, so a large sample does not make a single reading more "
        "certain — treat it cautiously and watch the trend over time, not a "
        "single number."
    ),
    ReliabilityCategory.UNKNOWN: (
        "We have not yet studied this metric's behavior under sampling — "
        "no additional caveats, but also no confirmed stability."
    ),
}

_CAUSAL_EVIDENCE_MESSAGES: dict[CausalEvidence, str] = {
    CausalEvidence.STRONG: (
        "In the benchmark, a targeted repair for this pathology raised Recall@10 in "
        "repeated realizations. The gain was partial and part of it depends on the "
        "evaluation protocol; it does not show that the statistic causes recall loss "
        "in general."
    ),
    CausalEvidence.WEAK: (
        "Despite a strong statistical signal, removing what this statistic reads did "
        "not raise recall in the benchmark — treat it as a diagnostic indicator, not a "
        "repair lever."
    ),
    CausalEvidence.NONE: (
        "No repair for the pathology this statistic reads was tested in the benchmark, "
        "so there is no repair evidence either way."
    ),
}


def explain_metric_reliability(metric_name: str) -> str:
    """One sentence on the metric's confidence/reliability (the subsampling
    axis) — ready to insert into level 1 or level 2 of a message
    (progressive disclosure)."""
    profile = get_profile(metric_name)
    return _RELIABILITY_MESSAGES[profile.reliability_category]


def explain_causal_evidence(metric_name: str) -> str:
    """One sentence on the strength of causal evidence (the repair axis) —
    the second, INDEPENDENT prioritization axis. Never combine this
    function's output with explain_metric_reliability() into one sentence —
    that is the core rule of this layer: 'diagnostic indicator' and 'repair
    lever' are two separate labels, not one shared scale."""
    profile = get_profile(metric_name)
    return _CAUSAL_EVIDENCE_MESSAGES[profile.causal_evidence]


def full_confidence_summary(metric_name: str) -> dict[str, str]:
    """Convenient aggregate of both axes at once, as separate fields —
    deliberately not one merged sentence, so the presentation layer can
    show them as two separate labels in the UI."""
    return {
        "reliability": explain_metric_reliability(metric_name),
        "causal_evidence": explain_causal_evidence(metric_name),
    }
