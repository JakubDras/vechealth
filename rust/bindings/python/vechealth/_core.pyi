"""Type stubs for the compiled `vechealth._core` extension module.

Hand-written and kept in sync manually with `rust/bindings/src/lib.rs` — PyO3
does not generate these automatically. Import from `vechealth`, not from this
module, unless you specifically need to bypass the pure-Python re-export layer.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VecHealthError(Exception):
    """Base class for all errors raised by vechealth."""

class DimensionMismatchError(VecHealthError):
    """Query/vector dimensionality does not match the indexed vectors."""

class KTooLargeError(VecHealthError):
    """Requested k is >= the number of available vectors."""

class KTooSmallError(VecHealthError):
    """Requested k is smaller than this metric requires."""

class EmptyInputError(VecHealthError):
    """The input matrix of vectors is empty."""

class AllVectorsDegenerateError(VecHealthError):
    """Every input vector has zero norm, so none can be normalized."""

class ConnectorError(VecHealthError):
    """Fetching vectors from an external source (a local file, Qdrant,
    Postgres) failed — I/O, parsing, a schema mismatch, or a network/auth
    problem. The message says which."""

class ReportError(VecHealthError):
    """Saving, loading, or parsing a `Report` failed — I/O or JSON
    (de)serialization. The message says which."""

# ---------------------------------------------------------------------------
# Typed results — all immutable, all fields read-only. Every one of these
# also has `.to_dict()` / `.to_json()`, using the same JSON schema as the
# matching field inside a `Report`.
# ---------------------------------------------------------------------------

class HubnessResult:
    hubness_skewness: float
    orphans_fraction: float
    max_occurrences: int
    def to_dict(self) -> dict[str, float | int]: ...
    def to_json(self) -> str: ...

class DispersionResult:
    mean_1nn_distance: float
    mean_knn_distance: float
    def to_dict(self) -> dict[str, float]: ...
    def to_json(self) -> str: ...

class AnisotropyResult:
    mean_vector_norm: float
    top1_variance_ratio: float
    top10_variance_ratio: float
    def to_dict(self) -> dict[str, float]: ...
    def to_json(self) -> str: ...

class OutliersResult:
    outlier_fraction: float
    max_1nn_distance: float
    std_1nn_distance: float
    def to_dict(self) -> dict[str, float]: ...
    def to_json(self) -> str: ...

class DuplicatesResult:
    ndds_fraction: float
    mean_1nn_distance: float
    min_distance_global: float
    def to_dict(self) -> dict[str, float]: ...
    def to_json(self) -> str: ...

class IntrinsicDimResult:
    mean_id: float
    median_id: float
    def to_dict(self) -> dict[str, float]: ...
    def to_json(self) -> str: ...

class QmasResult:
    mean_1nn_distance: float
    mean_knn_distance: float
    orphans_fraction: float
    def to_dict(self) -> dict[str, float]: ...
    def to_json(self) -> str: ...

class SncResult:
    mean_snc_score: float
    def to_dict(self) -> dict[str, float]: ...
    def to_json(self) -> str: ...

class AllMetricsResult:
    hubness: HubnessResult
    dispersion: DispersionResult
    anisotropy: AnisotropyResult
    outliers: OutliersResult
    duplicates: DuplicatesResult
    intrinsic_dim: IntrinsicDimResult
    snc: SncResult
    qmas: QmasResult | None
    def to_dict(self) -> dict[str, object]: ...
    def to_json(self) -> str: ...

# ---------------------------------------------------------------------------
# Report / Comparison — persistence and baseline comparison over
# `AllMetricsResult`. See TODO_List.md's "śledzenie w czasie / porównanie z
# baseline" item for the rationale.
# ---------------------------------------------------------------------------

class Report:
    """A self-contained, versioned snapshot: metrics plus the metadata
    needed to interpret them later (config used, dataset fingerprint,
    timestamp, arbitrary tags). Produced by
    :meth:`VecHealthEvaluator.compute_report`.
    """

    schema_version: int
    generated_at: str  # RFC 3339
    vechealth_version: str
    n_vectors: int
    dim: int
    content_hash: str
    tags: dict[str, str]
    metrics: AllMetricsResult

    def to_dict(self) -> dict[str, object]: ...
    def to_json(self) -> str: ...
    def flatten(self) -> dict[str, float]:
        """Flat ``"{group}.{field}" -> value`` view of `metrics`, ready for
        a metric store / experiment tracker or a future Prometheus
        exporter."""
        ...
    def save(self, path: str) -> None:
        """Writes this report as pretty-printed JSON to `path`. Raises
        :class:`ReportError` on I/O failure."""
        ...
    @staticmethod
    def load(path: str) -> Report:
        """Loads a report previously written by `save`. Raises
        :class:`ReportError` on I/O or parse failure."""
        ...
    def compare(self, baseline: Report) -> Comparison:
        """Compares this report (treated as "current") against `baseline`,
        metric by metric. Applies no thresholds — see
        `Comparison.warnings` for anything that would make the comparison
        unreliable."""
        ...

class MetricDelta:
    baseline: float
    current: float
    delta: float
    delta_pct: float | None

class Comparison:
    baseline_generated_at: str  # RFC 3339
    current_generated_at: str  # RFC 3339
    deltas: dict[str, MetricDelta]
    warnings: list[str]
    def to_dict(self) -> dict[str, object]: ...
    def to_json(self) -> str: ...

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class VecHealthEvaluator:
    """Stateful evaluator over a fixed set of vectors.

    KNN and normalization results are cached internally, so calling several
    ``compute_*`` methods on the same instance re-uses the same k-NN search
    instead of recomputing it. Every ``compute_*`` method releases the GIL
    for the duration of the Rust-side computation.
    """

    def __init__(self, vectors: npt.ArrayLike) -> None:
        """`vectors` accepts any NumPy array-like of numbers — most
        commonly a `float32` or `float64` array, but also plain nested
        Python sequences. Non-`float32` input is cast via NumPy's own
        `asarray`; an already-`float32`, C-contiguous array is used as-is
        with no copy."""
        ...
    @staticmethod
    def from_local(
        path: str,
        has_header: bool = True,
        columns: list[str] | None = None,
    ) -> VecHealthEvaluator:
        """Loads vectors from a local ``.npy``, ``.csv``, or ``.parquet``
        file, dispatching on extension. ``has_header`` only applies to
        ``.csv`` (first line skipped when true); ``columns`` only applies to
        ``.parquet`` (selects/reorders a column subset; ``None`` uses every
        column in schema order). Raises :class:`ConnectorError` on I/O,
        parse, or schema problems.
        """
        ...
    @staticmethod
    def from_qdrant(
        url: str,
        collection: str,
        api_key: str | None = None,
        page_size: int = 1000,
        timeout_secs: int = 30,
    ) -> VecHealthEvaluator:
        """Fetches every point's vector from a Qdrant collection via the
        ``scroll`` API (never ``search``/ANN). Pulls the complete
        collection — ``page_size`` only controls the network page size, not
        how many points are fetched. Warns via ``warnings.warn`` if the
        collection's distance metric isn't cosine. Raises
        :class:`ConnectorError` on network/auth/schema problems.
        """
        ...
    @staticmethod
    def from_pgvector(
        connection_string: str,
        table: str,
        vector_column: str,
        id_column: str,
        page_size: int = 5000,
    ) -> VecHealthEvaluator:
        """Fetches every row from a pgvector-backed Postgres table via
        keyset pagination on ``id_column`` (never ``OFFSET``). Pulls the
        complete table. ``id_column`` must be an integer primary/unique key.
        Connection is unencrypted (``NoTls``); use an SSH tunnel or a
        trusted network if TLS is required. Raises :class:`ConnectorError`
        on network/auth/schema problems.
        """
        ...
    @property
    def n_vectors(self) -> int: ...
    @property
    def dim(self) -> int: ...
    def get_knn(
        self, k: int, batch_size: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]: ...
    def compute_hubness(self, k: int = 10, batch_size: int = 2048) -> HubnessResult: ...
    def compute_dispersion(self, k: int = 10, batch_size: int = 2048) -> DispersionResult: ...
    def compute_anisotropy(self) -> AnisotropyResult: ...
    def compute_outliers(
        self, distance_threshold: float, batch_size: int = 2048
    ) -> OutliersResult: ...
    def compute_duplicates(
        self, epsilon: float = 0.05, batch_size: int = 2048
    ) -> DuplicatesResult: ...
    def compute_intrinsic_dim(
        self, k: int = 20, batch_size: int = 2048
    ) -> IntrinsicDimResult: ...
    def compute_qmas(
        self,
        queries: npt.ArrayLike,
        k: int = 10,
        batch_size: int = 2048,
    ) -> QmasResult: ...
    def compute_snc(self, k: int = 10, batch_size: int = 2048) -> SncResult: ...
    def compute_all(
        self,
        queries: npt.ArrayLike | None = None,
        k: int = 10,
        k_intrinsic_dim: int = 20,
        batch_size: int = 2048,
        duplicate_epsilon: float = 0.05,
        outlier_distance_threshold: float | None = None,
    ) -> AllMetricsResult: ...
    def compute_report(
        self,
        queries: npt.ArrayLike | None = None,
        k: int = 10,
        k_intrinsic_dim: int = 20,
        batch_size: int = 2048,
        duplicate_epsilon: float = 0.05,
        outlier_distance_threshold: float | None = None,
        tags: dict[str, str] | None = None,
    ) -> Report:
        """Same as `compute_all`, but wraps the result in a `Report` —
        savable, reloadable, and comparable against another `Report` later.
        """
        ...
