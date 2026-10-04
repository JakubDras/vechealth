"""
interpretation/priority.py

A two-axis prioritization system, not a single-axis threshold. An alert's
priority is a function of (diagnostic signal strength, causal evidence
strength, reliability category) — not of the metric value alone.

Deliberately three formula variants, not one — the choice has to come from
an empirical comparison on historical data (the 53 benchmark bases), not a
"gut-feel" choice. The default variant is meant to be selected AFTER that
comparison, not before it (see DEFAULT_VARIANT).

`diagnostic_strength` (input, 0.0-1.0) is deliberately a parameter of this
function, not something computed here — how it is determined (e.g. from
drift against a baseline, or in a baseline-free mode) is a separate
responsibility. This function deals ONLY with combining an already-computed
signal with the knowledge about the metric.
"""

from dataclasses import dataclass
from typing import Callable

from .metric_profiles import (
    MetricProfile,
    CausalEvidence,
    ReliabilityCategory,
    get_profile,
)


# Weights shared across variants, so changing one number does not
# require editing three separate places.

CAUSAL_WEIGHT: dict[CausalEvidence, float] = {
    CausalEvidence.STRONG: 1.0,
    CausalEvidence.WEAK: 0.3,
    CausalEvidence.NONE: 0.1,
}

RELIABILITY_WEIGHT: dict[ReliabilityCategory, float] = {
    ReliabilityCategory.A_STABLE: 1.0,
    ReliabilityCategory.B_CORRECTABLE: 0.9,
    ReliabilityCategory.C_CONVERGING: 0.8,
    ReliabilityCategory.D_UNSTABLE: 0.6,
    ReliabilityCategory.UNKNOWN: 0.7,
}

_CAUSAL_TIER: dict[CausalEvidence, int] = {
    CausalEvidence.STRONG: 2,
    CausalEvidence.WEAK: 1,
    CausalEvidence.NONE: 0,
}


# The three formula variants.

def priority_variant_multiplicative(diagnostic_strength: float, profile: MetricProfile) -> float:
    """Variant A — product of three factors. Causality and reliability act
    as DAMPING multipliers, not hard gates: even a metric with weak causal
    evidence can get a nonzero priority under an extreme diagnostic
    signal, but always lower than the identical signal under strong
    causality. Output range: [0, 1]."""
    return (
            diagnostic_strength
            * CAUSAL_WEIGHT[profile.causal_evidence]
            * RELIABILITY_WEIGHT[profile.reliability_category]
    )


def priority_variant_lexicographic(diagnostic_strength: float, profile: MetricProfile) -> float:
    """Variant B — causality as a HARD precedence gate, encoded as a large
    layer offset: NO diagnostic_strength value in a lower causality layer
    can outscore a higher causality layer. Reliability differentiates ONLY
    within the same layer. Output range: [0, 30) — not directly comparable
    with the other variants; only the RELATIVE ordering within this
    variant matters."""
    tier = _CAUSAL_TIER[profile.causal_evidence]
    within_tier_score = diagnostic_strength * RELIABILITY_WEIGHT[profile.reliability_category]
    return tier * 10.0 + within_tier_score  # *10 guarantees layer separation (within_tier_score <= 1.0)


def priority_variant_weighted_sum(diagnostic_strength: float, profile: MetricProfile) -> float:
    """Variant C — weighted sum, not a product: milder priority degradation
    for weak causality than in the multiplicative variant, since missing
    causal evidence does not zero out the diagnostic signal entirely.
    Output range: [0, 1]."""
    causal_score = {
        CausalEvidence.STRONG: 1.0,
        CausalEvidence.WEAK: 0.4,
        CausalEvidence.NONE: 0.0,
    }[profile.causal_evidence]
    reliability_score = RELIABILITY_WEIGHT[profile.reliability_category]
    return 0.5 * diagnostic_strength + 0.35 * causal_score + 0.15 * reliability_score


PRIORITY_VARIANTS: dict[str, Callable[[float, MetricProfile], float]] = {
    "multiplicative": priority_variant_multiplicative,
    "lexicographic": priority_variant_lexicographic,
    "weighted_sum": priority_variant_weighted_sum,
}

# NOTE: the default variant is to be CONFIRMED/CHANGED only after comparing
# the three variants on the full 53-base benchmark — "multiplicative" is
# here as a reasonable starting point (see the rationale in this file's
# inline comments), not a final decision.
DEFAULT_VARIANT = "multiplicative"


@dataclass(frozen=True)
class PriorityResult:
    metric_name: str
    priority_score: float
    diagnostic_strength: float
    causal_evidence: CausalEvidence
    reliability_category: ReliabilityCategory
    formula_variant: str


def compute_priority(
        metric_name: str,
        diagnostic_strength: float,
        variant: str = DEFAULT_VARIANT,
) -> PriorityResult:
    """Main entry point. `diagnostic_strength` must already be computed by
    the caller (e.g. the drift layer) and normalized to [0, 1] — this
    function does not do that."""
    if not 0.0 <= diagnostic_strength <= 1.0:
        raise ValueError(
            f"diagnostic_strength must be in [0, 1], got "
            f"{diagnostic_strength} for metric '{metric_name}'"
        )
    if variant not in PRIORITY_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {list(PRIORITY_VARIANTS)}"
        )

    profile = get_profile(metric_name)
    formula = PRIORITY_VARIANTS[variant]
    score = formula(diagnostic_strength, profile)

    return PriorityResult(
        metric_name=metric_name,
        priority_score=score,
        diagnostic_strength=diagnostic_strength,
        causal_evidence=profile.causal_evidence,
        reliability_category=profile.reliability_category,
        formula_variant=variant,
    )


def rank_metrics_by_priority(
        diagnostic_strengths: dict[str, float],
        variant: str = DEFAULT_VARIANT,
) -> list[PriorityResult]:
    """Convenience aggregate — takes a {metric_name: diagnostic_strength}
    dict (e.g. the result of one diagnostic run) and returns a list of
    PriorityResult sorted descending by priority_score. This is what the
    user actually sees as the 'alert order' in a report."""
    results = [
        compute_priority(name, strength, variant=variant)
        for name, strength in diagnostic_strengths.items()
    ]
    return sorted(results, key=lambda r: r.priority_score, reverse=True)
