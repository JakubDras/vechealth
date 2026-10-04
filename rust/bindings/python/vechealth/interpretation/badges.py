"""
interpretation/badges.py
"""

from dataclasses import dataclass
from enum import Enum

from .metric_profiles import get_profile, CausalEvidence


class BadgeLevel(str, Enum):
    NONE = "none"
    WEAK = "weak"
    STRONG = "strong"


@dataclass(frozen=True)
class Badge:
    label: str
    level: BadgeLevel
    tooltip: str


def diagnostic_indicator_badge(metric_name: str) -> Badge:
    return Badge(
        label="Diagnostic indicator",
        level=BadgeLevel.STRONG,  # always present, so always full strength
        tooltip=(
            f"'{metric_name}' is a measured property of the geometry of "
            f"your embedding space. On its own it only says 'this is "
            f"unusual', not 'this definitely harms retrieval' — see the "
            f"separate 'Repair lever' label."
        ),
    )


def repair_lever_badge(metric_name: str) -> Badge:
    profile = get_profile(metric_name)

    if profile.causal_evidence == CausalEvidence.STRONG:
        return Badge(
            label="Repair lever",
            level=BadgeLevel.STRONG,
            tooltip=(
                f"In the benchmark, a targeted repair of the pathology '{metric_name}' "
                f"reads raised Recall@10 in repeated realizations. The gain was partial "
                f"and depends on the evaluation protocol (preprint Section 4.7); it does "
                f"not show that the statistic causes recall loss in general."
            ),
        )
    elif profile.causal_evidence == CausalEvidence.WEAK:
        return Badge(
            label="Indicator only — repair may NOT help",
            level=BadgeLevel.WEAK,
            tooltip=(
                f"Despite a strong statistical signal, removing what '{metric_name}' "
                f"reads did not raise recall in the benchmark. Treat it as an indicator "
                f"to observe, not something to repair immediately."
            ),
        )
    else:  # CausalEvidence.NONE
        return Badge(
            label="Indicator only — no causality test",
            level=BadgeLevel.NONE,
            tooltip=(
                f"No repair for what '{metric_name}' reads was tested in the benchmark, "
                f"so there is no repair evidence either way. It is a reading to "
                f"investigate, not a diagnosis of retrieval harm."
            ),
        )


@dataclass(frozen=True)
class MetricBadges:
    """Both labels at once — deliberately two separate fields, not one
    merged string. A presentation layer renders them as two separate badges
    in the UI, never as one combined score."""
    diagnostic: Badge
    repair_lever: Badge


def get_badges(metric_name: str) -> MetricBadges:
    return MetricBadges(
        diagnostic=diagnostic_indicator_badge(metric_name),
        repair_lever=repair_lever_badge(metric_name),
    )


def render_badges_cli(metric_name: str) -> str:
    """Reference text renderer (CLI/log) — the simplest, always available
    rendering, useful for checking the module by hand."""
    badges = get_badges(metric_name)
    return f"{badges.diagnostic.label}   {badges.repair_lever.label}"
