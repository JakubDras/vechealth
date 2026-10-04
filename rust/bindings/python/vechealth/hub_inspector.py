"""
vechealth/hub_inspector.py

Hub Inspector — an interpretation layer on top of the real Rust engine
(core/src/hub_inspector.rs, exposed in the bindings as
VecHealthEvaluator.identify_hubs() and .get_knn()). It identifies the
concrete points responsible for high hubness, not just an aggregate
statistic — exactly the analysis that had to be done by hand on the
ClinicalTrials.gov corpus: hubness_max=155 alone said nothing; only manually
inspecting the concrete points revealed a meaningful (though unexpected)
semantic structure, not noise.

=== IMPORTANT: THIS IS THE SECOND VERSION OF THIS FILE ===

The first version (an earlier project stage) computed k-NN neighborhoods
in Python itself (brute-force numpy) — a temporary workaround from before
the Rust core (`identify_hubs`, core/src/hub_inspector.rs) existed at all.
This version computes nothing heavy in Python — both hub identification and
the k-NN neighborhoods needed for the category-dispersion analysis come
from the already-verified `VecHealthEvaluator` methods (`identify_hubs`,
`get_knn`).

=== DECISION: A FUNCTION, NOT A METHOD ON THE EVALUATOR ===

A method `evaluator.inspect_hubs(...)` would be the obvious design. It is a
plain function `inspect_hubs(evaluator, ...)` taking the evaluator as its first
argument instead — attaching new methods to a compiled PyO3 class at runtime
is fragile and needlessly complicates something that is pure Python
interpretation logic, not part of the engine. This keeps the same architectural boundary as the rest of
`interpretation/`.

=== NOTE ON `likely_pathology` — an UNCALIBRATED threshold, explicitly marked ===

As in `recommendations.py` — this heuristic threshold is not
empirically calibrated on a large real-world dataset, unlike, say,
SIMILARITY_THRESHOLD in signatures.py. Hence the same machine-readable
warning pattern (`research_status`, `human_review_required`) as in the
recommendation engine.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

HubId = Union[int, str]


@dataclass(frozen=True)
class HubEntry:
    index: int
    id: HubId
    occurrence_count: int


@dataclass(frozen=True)
class HubInspectionReport:
    hubs: list[HubEntry]
    category_dispersion: Optional[float]
    likely_pathology: Optional[bool]
    interpretation_note: str
    research_status: str
    human_review_required: bool


_LIKELY_PATHOLOGY_DISPERSION_THRESHOLD = 0.5  # uncalibrated, see the module docstring
_RESEARCH_STATUS = "experimental_in_progress"


def inspect_hubs(
        evaluator,
        k: int = 10,
        top_n: int = 20,
        id_array: Optional[Sequence[HubId]] = None,
        category_labels: Optional[Sequence] = None,
        batch_size: Optional[int] = None,
) -> HubInspectionReport:
    """The main Hub Inspector function. `evaluator` is a real, initialized
    `vechealth.VecHealthEvaluator`.

    Hub identification: `evaluator.identify_hubs()` (Rust, verified).
    Category-dispersion analysis (optional, only when `category_labels` is
    given): `evaluator.get_knn()` (Rust, verified) — Python computes only
    the entropy over the already-ready neighborhood matrix, nothing else."""
    raw_hubs = evaluator.identify_hubs(k=k, top_n=top_n, batch_size=batch_size)

    hubs = [
        HubEntry(
            index=h.index,
            id=(id_array[h.index] if id_array is not None else h.index),
            occurrence_count=h.occurrence_count,
        )
        for h in raw_hubs
    ]

    category_dispersion: Optional[float] = None
    likely_pathology: Optional[bool] = None

    if category_labels is not None:
        hub_indices = [h.index for h in hubs]
        category_dispersion = _compute_category_dispersion(
            evaluator, hub_indices, k, batch_size, category_labels
        )
        likely_pathology = category_dispersion > _LIKELY_PATHOLOGY_DISPERSION_THRESHOLD

    interpretation_note = _build_interpretation_note(hubs, category_dispersion, likely_pathology)

    return HubInspectionReport(
        hubs=hubs,
        category_dispersion=category_dispersion,
        likely_pathology=likely_pathology,
        interpretation_note=interpretation_note,
        research_status=_RESEARCH_STATUS,
        human_review_required=True,
    )


def _compute_category_dispersion(
        evaluator,
        hub_indices: list[int],
        k: int,
        batch_size: Optional[int],
        category_labels: Sequence,
) -> float:
    """Category dispersion among the points CITING a given hub — NOT the
    hub's own category. The same hand-verified entropy logic as in the
    first version of this file; only the neighborhood matrix now comes from
    `evaluator.get_knn()` (Rust), not manual Python computation."""
    _, indices = evaluator.get_knn(k=k, batch_size=batch_size)
    # indices: numpy array (n, k) — row i = indices of point i's k nearest neighbors

    all_categories = sorted(set(category_labels))
    num_all_categories = len(all_categories)

    dispersions, weights = [], []
    for hub_idx in hub_indices:
        citer_mask = (indices == hub_idx).any(axis=1)
        citer_categories = [category_labels[i] for i in citer_mask.nonzero()[0]]
        if not citer_categories:
            continue
        counts = Counter(citer_categories)
        total = sum(counts.values())
        if num_all_categories <= 1:
            entropy = 0.0
        else:
            probs = [c / total for c in counts.values()]
            raw_entropy = -sum(p * math.log(p) for p in probs)
            entropy = raw_entropy / math.log(num_all_categories)
        dispersions.append(entropy)
        weights.append(total)

    if not dispersions:
        return 0.0

    total_weight = sum(weights)
    return sum(d * w for d, w in zip(dispersions, weights)) / total_weight


def _build_interpretation_note(
        hubs: list[HubEntry],
        category_dispersion: Optional[float],
        likely_pathology: Optional[bool],
) -> str:
    if not hubs:
        return "No hubs identified in the provided set."

    top = hubs[0]
    base = (
        f"Strongest identified hub: id={top.id}, "
        f"occurrences in k-NN: {top.occurrence_count}."
    )

    if category_dispersion is None:
        return (
            f"{base} No category_labels given — no category-dispersion "
            f"analysis. Manual inspection of this point's content "
            f"recommended."
        )

    if likely_pathology:
        return (
            f"{base} Category dispersion: {category_dispersion:.2f} (high) "
            f"— hubs cited by many different categories. NOTE: this may "
            f"indicate a geometric pathology, BUT (the ClinicalTrials.gov case "
            f"study) it can also be a real, valuable semantic structure (e.g. shared "
            f"terminology across different topics) — this heuristic's "
            f"threshold is NOT empirically calibrated; treat it as a "
            f"starting point for manual inspection, not a verdict."
        )
    else:
        return (
            f"{base} Category dispersion: {category_dispersion:.2f} (low) "
            f"— hubs concentrated mainly in one category, consistent with "
            f"healthy thematic clustering. Still worth verifying manually; "
            f"this signal is a starting point, not a verdict."
        )


# ============================================================
# Integration with the interpretation layer (report.py)
# ============================================================

ExampleProvider = Callable[[str], Optional[str]]


def hub_inspector_as_example_provider(
        evaluator,
        id_array: Optional[Sequence[HubId]] = None,
        category_labels: Optional[Sequence] = None,
        k: int = 10,
        batch_size: Optional[int] = None,
) -> ExampleProvider:
    """Returns a function compatible with ExampleProvider from
    interpretation/report.py — hook-up:
    build_metric_report(..., example_provider=
    hub_inspector_as_example_provider(evaluator, ...))."""
    _cache: dict[int, HubInspectionReport] = {}

    def provider(metric_name: str) -> Optional[str]:
        if metric_name not in ("hubness_max", "hubness_skewness"):
            return None
        if k not in _cache:
            _cache[k] = inspect_hubs(
                evaluator, k=k, top_n=3,
                id_array=id_array, category_labels=category_labels,
                batch_size=batch_size,
            )
        report = _cache[k]
        if not report.hubs:
            return None
        top = report.hubs[0]
        return f"Example hub: id={top.id}, occurrences in k-NN: {top.occurrence_count}."

    return provider


# ============================================================
# "New hubs since the last healthy snapshot"
# ============================================================

@dataclass(frozen=True)
class NewHubsReport:
    new_hubs: list[HubEntry]
    persisted_hubs: list[HubEntry]
    resolved_hubs: list[HubEntry]
    interpretation_note: str


def compare_hub_snapshots(
        previous: HubInspectionReport,
        current: HubInspectionReport,
) -> NewHubsReport:
    """Comparison of two inspect_hubs() runs over time.

    IMPORTANT ASSUMPTION: comparing by `id` makes sense ONLY if both runs
    used the same, stable `id_array`. If both runs used raw positional
    indices and the vector set changed size/order between runs — the
    comparison will be MISLEADING. This function does not verify that
    automatically."""
    prev_ids = {h.id for h in previous.hubs}
    curr_ids = {h.id for h in current.hubs}

    new_hubs = [h for h in current.hubs if h.id not in prev_ids]
    persisted_hubs = [h for h in current.hubs if h.id in prev_ids]
    resolved_hubs = [h for h in previous.hubs if h.id not in curr_ids]

    parts = []
    if new_hubs:
        parts.append(
            f"{len(new_hubs)} NEW hub(s) since the last snapshot — worth "
            f"checking manually; these are points that were not problematic "
            f"before."
        )
    if resolved_hubs:
        parts.append(
            f"{len(resolved_hubs)} hub(s) from the previous snapshot are no "
            f"longer in the top — a good sign, but check whether it is a "
            f"real improvement or just a ranking shift."
        )
    if persisted_hubs and not new_hubs and not resolved_hubs:
        parts.append("No change — the same hubs as before.")
    if not parts:
        parts.append("Not enough data to compare.")

    return NewHubsReport(
        new_hubs=new_hubs,
        persisted_hubs=persisted_hubs,
        resolved_hubs=resolved_hubs,
        interpretation_note=" ".join(parts),
    )
