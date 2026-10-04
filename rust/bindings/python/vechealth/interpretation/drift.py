"""
interpretation/drift.py

The primary, recommended working mode — interpreting a metric CHANGE against
the user's OWN baseline, not against a universal, arbitrary threshold.
Instead of guessing "what is normal in general", we check what changed
relative to a state the user themselves marked as a reference point.

This module is also where `diagnostic_strength` (in the prioritization layer
only a manually supplied parameter) actually comes from real data — as a
normalized, relative deviation from the baseline.

=== NO "IMPROVED" / "REGRESSED" LABELS ===

In the benchmark no statistic has a direction of change that is validated as
good or bad (preprint Sections 4.3-4.4 and Table 15): the same statistic moves
in opposite directions across pathology families, and several statistics move
toward what would read as improvement while Recall@10 falls (dimension
truncation). Every shipped `MetricProfile` therefore has
`known_direction = AMBIGUOUS`, and a change is described neutrally. The
"improved"/"regressed" branch below is reserved for a profile whose direction
a future version validates.
"""

from dataclasses import dataclass

from .metric_profiles import get_profile, KnownDirection


def _diagnostic_strength_from_drift(baseline_value: float, current_value: float) -> float:
    """Normalized [0,1] signal strength based on the relative change vs.
    the baseline.

    Edge case baseline=0: if current_value is ALSO 0, that is an actual
    absence of change (strength=0). If current_value != 0, a change from a
    'exactly zero' state to something nonzero is by definition maximally
    significant (the relative change would be infinite) — a realistic case
    for metrics like ndds_fraction, which are often exactly zero on healthy
    data."""
    if baseline_value == 0:
        return 0.0 if current_value == 0 else 1.0
    return min(1.0, abs(current_value - baseline_value) / abs(baseline_value))


@dataclass(frozen=True)
class DriftInterpretation:
    metric_name: str
    baseline_value: float
    current_value: float
    percent_change: float
    diagnostic_strength: float
    direction_label: str  # "ambiguous" | "no_change" ("improved" | "regressed" are reserved)
    message: str


def _describe_change(metric_name: str, baseline_value: float, current_value: float,
                     percent_change: float) -> str:
    if baseline_value == 0:
        return (
            f"'{metric_name}' changed from exactly 0 (your baseline) to {current_value:.4g}"
        )
    return (
        f"'{metric_name}' changed by {percent_change:+.1f}% relative to your baseline "
        f"(from {baseline_value:.4g} to {current_value:.4g})"
    )


def interpret_drift(
        metric_name: str,
        baseline_value: float,
        current_value: float,
) -> DriftInterpretation:
    """Returns a ready level-1 message (progressive disclosure) — one
    sentence, immediately readable, without needing to know MetricProfile's
    internal categories."""
    profile = get_profile(metric_name)

    if baseline_value == 0:
        percent_change = 0.0 if current_value == 0 else float("inf")
    else:
        percent_change = 100.0 * (current_value - baseline_value) / baseline_value

    diagnostic_strength = _diagnostic_strength_from_drift(baseline_value, current_value)

    if current_value == baseline_value:
        return DriftInterpretation(
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            percent_change=percent_change,
            diagnostic_strength=diagnostic_strength,
            direction_label="no_change",
            message=(
                f"'{metric_name}' is unchanged relative to your baseline "
                f"({baseline_value:.4g})."
            ),
        )

    change = _describe_change(metric_name, baseline_value, current_value, percent_change)

    # --- Direction not validated (every shipped profile) ---
    if profile.known_direction == KnownDirection.AMBIGUOUS:
        return DriftInterpretation(
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            percent_change=percent_change,
            diagnostic_strength=diagnostic_strength,
            direction_label="ambiguous",
            message=(
                f"{change}. The direction of this change is not validated as good or bad "
                f"— treat it as a symptom to investigate, not a verdict."
            ),
        )

    # --- Reserved: a profile whose direction has been validated ---
    is_lower_better = profile.known_direction == KnownDirection.LOWER_BETTER
    went_down = current_value < baseline_value
    improved = (is_lower_better and went_down) or (not is_lower_better and not went_down)
    direction_label = "improved" if improved else "regressed"
    return DriftInterpretation(
        metric_name=metric_name,
        baseline_value=baseline_value,
        current_value=current_value,
        percent_change=percent_change,
        diagnostic_strength=diagnostic_strength,
        direction_label=direction_label,
        message=f"{change}: {direction_label} relative to your baseline.",
    )
