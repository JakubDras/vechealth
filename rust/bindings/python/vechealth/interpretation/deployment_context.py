"""
interpretation/deployment_context.py

The user's deployment context — the ANN index type and, optionally, the
data-domain character — decides which consequences of a diagnosis are worth
mentioning, without changing the metric value itself.

=== WHAT THE INDEX FIGURES ARE ===

For every pathology family, the benchmark behind this package measured how
much of its exact-search Recall@10 a store keeps when it is served by an
approximate index ("retention", in percent). They come from the preprint's
approximate-search experiments (Tables 22 and 24):

  - HNSW: M = 16, efConstruction 200, efSearch 128;
  - IVF-PQ: 8 sub-quantizers of 8 bits, nprobe 128, nlist = floor(4 sqrt(N));
  - one severity level per family (stated in `AnnRetention.level`);
  - Qwen3-Embedding-0.6B (primary) and BGE-large (second encoder), same
    246,460 documents.

Retention is reported instead of a "loss multiplier" because the multiplier
divides by the baseline's tiny HNSW loss and therefore moves by about +-25%
between rebuilds of the same index on identical input (preprint Table 23),
whereas the standard deviation of retention stays below 0.6 percentage
points.

Two caveats travel with these numbers and appear in the messages: the IVF-PQ
baseline (41.4%) is itself a product of the aggressive 8-byte quantization
(62.3% with 16 sub-quantizers), and the intermediate severity levels of the
two families that collapse under HNSW (hubness, noise) were not measured.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class IndexType(str, Enum):
    EXACT = "exact"
    HNSW = "hnsw"
    IVFPQ = "ivfpq"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DeploymentContext:
    index_type: IndexType = IndexType.UNKNOWN
    domain_hint: Optional[str] = None


@dataclass(frozen=True)
class AnnRetention:
    """Recall@10 retention, in percent of the exact-search Recall@10, for one
    benchmark base. `hnsw`/`ivfpq` are for Qwen3-Embedding-0.6B, `*_bge` for
    BGE-large."""

    label: str
    level: str
    hnsw: float
    ivfpq: float
    hnsw_bge: float
    ivfpq_bge: float


# The unperturbed baseline (preprint Tables 22 and 24).
BASELINE_RETENTION = AnnRetention(
    label="unperturbed baseline",
    level="no pathology injected",
    hnsw=99.4,
    ivfpq=41.4,
    hnsw_bge=99.84,
    ivfpq_bge=36.42,
)

# IVF-PQ retention of the unperturbed baseline with 16 sub-quantizers instead
# of 8 (preprint Table 25): the baseline figure is a property of the
# quantizer settings, not of the data alone.
BASELINE_IVFPQ_RETENTION_16_SUBQUANTIZERS = 62.3

# Every family of the benchmark, at the one severity level measured under
# approximate search (preprint Section 3.6; Tables 22 and 24).
ANN_RETENTION: dict[str, AnnRetention] = {
    "G1_hubness": AnnRetention(
        "hubness", "hub attraction alpha = 0.6, the strongest level",
        5.2, 4.3, 7.32, 4.92,
    ),
    "G2_voids": AnnRetention(
        "void regions", "deleted clusters f = 0.5",
        99.0, 45.8, 99.96, 46.79,
    ),
    "G3_imbalance": AnnRetention(
        "cluster imbalance", "dominant-domain share f = 0.5",
        99.6, 48.1, 99.73, 40.16,
    ),
    "G4_duplicates": AnnRetention(
        "near-duplicates", "every document cloned, the strongest level",
        98.7, 37.2, 99.37, 32.14,
    ),
    "G5_anisotropy": AnnRetention(
        "anisotropy", "interpolation toward the centroid alpha = 0.95, the strongest level",
        99.5, 58.6, 99.51, 52.25,
    ),
    "G6_collapse": AnnRetention(
        "embedding collapse", "64 kept coordinates",
        100.0, 70.6, 99.61, 57.17,
    ),
    "G7_fragmentation": AnnRetention(
        "fragmentation", "push away from the cluster centroid beta = 2.5, the strongest level",
        98.4, 31.7, 99.53, 22.17,
    ),
    "G8_noise": AnnRetention(
        "injected noise", "Gaussian noise sigma = 0.1",
        9.8, 1.8, 17.16, 1.26,
    ),
    "G9_outliers": AnnRetention(
        "outlier contamination", "replaced vectors f = 0.2",
        99.4, 31.2, 99.70, 27.65,
    ),
}

# A family is flagged as a severe index-dependent loss when it kept less than
# this fraction of what the unperturbed baseline kept under the same index.
# On the benchmark that singles out exactly the two families that collapse
# under HNSW (hubness, noise; no measured base lay between 9.8% and 98.4%),
# and the same two families under IVF-PQ, on both encoders. The 0.5 is a
# convention, not a calibrated threshold.
SEVERE_RETENTION_FRACTION = 0.5

_INDEX_LABEL = {IndexType.HNSW: "HNSW", IndexType.IVFPQ: "IVF-PQ"}


def _retention(entry: AnnRetention, index_type: IndexType, bge: bool = False) -> float:
    if index_type == IndexType.HNSW:
        return entry.hnsw_bge if bge else entry.hnsw
    return entry.ivfpq_bge if bge else entry.ivfpq


def index_loss_is_severe(family_id: Optional[str], index_type: IndexType) -> bool:
    """True when the benchmark measured a severe index-dependent loss for this
    family under this index type (see `SEVERE_RETENTION_FRACTION`). Always
    False for exact search, an unknown index, or an unmatched family — the
    absence of a measurement is not a finding."""
    if family_id is None or index_type not in _INDEX_LABEL:
        return False
    entry = ANN_RETENTION.get(family_id)
    if entry is None:
        return False
    baseline = _retention(BASELINE_RETENTION, index_type)
    return _retention(entry, index_type) < SEVERE_RETENTION_FRACTION * baseline


_TAIL_SEVERE = (
    "This is the largest index-dependent loss the benchmark measured. Only this one "
    "severity level was tested under {index} for this family, and intermediate levels "
    "were not measured. Measure exact-versus-approximate recall on a sample of your "
    "own queries before drawing conclusions."
)
_TAIL_MILD = (
    "No {index}-specific loss was measured for this family at that level. That does not "
    "establish that your index is unaffected: measure exact-versus-approximate recall on "
    "a sample of your own queries."
)


def ann_context_warning(
        family_id: Optional[str],
        index_type: IndexType,
        ambiguous: bool = False,
) -> Optional[str]:
    """What the benchmark measured for the matched pathology family under the
    declared approximate index, as retention of exact-search Recall@10. None
    when it does not apply (exact search, an unknown index, no matched
    family). It never returns a bare "all OK": every message quotes the
    measurement and its scope, and tells the reader to measure on their own
    queries.

    `ambiguous=True` (the signature match is AMBIGUOUS — two families too
    close together) yields a CAUTIOUS message that names no family and quotes
    no family-specific figure: the family cannot be established, and some
    families lose far more recall under an approximate index than under exact
    search. `family_id` is ignored then."""
    if index_type not in _INDEX_LABEL:
        return None
    index = _INDEX_LABEL[index_type]

    if ambiguous:
        return (
            f"You declared an {index} index. The observed metric changes match more than "
            "one pathology family and it cannot be established which one. Some families "
            "(in the benchmark, hubness and noise) kept only a small fraction of their "
            "exact-search recall under this index type, while others lost none — the "
            "impact may be severe, so measure exact-versus-approximate recall on a "
            "sample of your own queries before drawing any conclusion."
        )

    entry = ANN_RETENTION.get(family_id) if family_id is not None else None
    if entry is None:
        return None

    base = BASELINE_RETENTION
    got = _retention(entry, index_type)
    got_bge = _retention(entry, index_type, bge=True)
    base_val = _retention(base, index_type)
    base_bge = _retention(base, index_type, bge=True)

    if index_type == IndexType.HNSW:
        settings = "HNSW M = 16, efConstruction 200, efSearch 128"
        compare = ""
    else:
        settings = "IVF-PQ with 8 sub-quantizers of 8 bits, nprobe 128"
        compare = (
            f" That is {'below' if got < base_val else 'above'} the baseline, and the "
            f"baseline figure itself depends on the quantizer settings "
            f"({BASELINE_IVFPQ_RETENTION_16_SUBQUANTIZERS:.1f}% with 16 sub-quantizers)."
        )

    severe = index_loss_is_severe(family_id, index_type)
    tail = (_TAIL_SEVERE if severe else _TAIL_MILD).format(index=index)
    return (
        f"You declared an {index} index. In the benchmark behind this package (one corpus, "
        f"Qwen3-Embedding-0.6B, {settings}), a store with the {entry.label} pathology "
        f"({entry.level}) kept {got:.1f}% of its exact-search Recall@10 under {index} "
        f"(BGE-large: {got_bge:.2f}%), against {base_val:.1f}% for the unperturbed "
        f"baseline (BGE-large: {base_bge:.2f}%).{compare} {tail}"
    )


# ============================================================
# An interpretation caveat for domain_hint — DELIBERATELY not automatic
# classification (rationale in the docstring)
# ============================================================

_DOMAIN_SENSITIVE_METRICS = frozenset({"hubness_max", "hubness_skewness"})


def domain_interpretation_caveat(metric_name: str, domain_hint: Optional[str]) -> Optional[str]:
    """An interpretation caveat (NOT an automatic "this is a pathology" /
    "this is healthy" classification) for hubness in a domain context.

    The ClinicalTrials.gov case study (preprint Section 4.9) showed that
    telling "healthy thematic clustering" from "a real geometric pathology"
    without domain knowledge is unreliable: hubs scattered across categories
    partly reflected real semantic structure — shared immune-checkpoint-
    inhibitor terminology across different cancer types — not pure noise.
    Therefore this function does NOT try to classify "good/bad" — it only
    raises awareness that the number alone may not suffice."""
    if domain_hint is None:
        return None
    if metric_name not in _DOMAIN_SENSITIVE_METRICS:
        return None

    return (
        f"High hubness scattered across categories can be a real, valuable "
        f"semantic structure (e.g. shared terminology across different "
        f"topics in your domain '{domain_hint}') — we recommend manual "
        f"inspection (Hub Inspector), not acting automatically on the "
        f"number alone."
    )
