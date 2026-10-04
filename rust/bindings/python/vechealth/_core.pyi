"""Type stubs for the compiled `vechealth._core` extension module.

Hand-written and kept in sync manually with `rust/bindings/src/lib.rs` — PyO3
does not generate these automatically. Import from `vechealth`, not from this
module, unless you specifically need to bypass the pure-Python re-export layer.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

def compiled_features() -> list[str]:
    """Names of the optional native connectors compiled into this build
    (a subset of ``"qdrant"``, ``"pgvector"``, ``"lancedb"``)."""
    ...

def estimate_peak_memory_bytes(
    n_vectors: int,
    dim: int,
    batch_size: int | None = None,
    threads: int | None = None,
) -> int:
    """Estimated peak memory, in bytes, of the exact k-NN search on
    ``n_vectors`` vectors of ``dim`` dimensions: a few copies of the vectors
    plus one block of ``batch_size`` similarity rows per worker thread.
    ``batch_size=None`` uses the automatic choice and ``threads=None`` the
    number of worker threads of this process. Good to about 10% on the sizes
    it was measured on."""
    ...

def default_batch_size(n_vectors: int, threads: int | None = None) -> int:
    """The batch size the k-NN search uses when none is given: as large as
    keeps its similarity blocks within about 2 GB on ``threads`` worker
    threads (default: this process's), between 16 and 2048 rows."""
    ...

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

class MemoryUsageWarning(UserWarning):
    """Raised as a warning, before the work starts, when the estimated peak
    memory of an analysis exceeds 8 GB (or the value of the
    ``VECHEALTH_MEMORY_WARNING_GB`` environment variable)."""

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

class HubEntry:
    """One point of the k-NN graph and how many neighbourhoods it appears in."""
    index: int
    occurrence_count: int
    def to_dict(self) -> dict[str, int]: ...
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
# `AllMetricsResult`.
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
    @staticmethod
    def from_lancedb(
        uri: str,
        table: str,
        vector_column: str,
        batch_size: int = 1024,
    ) -> VecHealthEvaluator:
        """Fetches every row's vector from a LanceDB table via a plain
        columnar scan (never ``vector_search()``/ANN). Pulls the complete
        table — ``batch_size`` only controls the maximum rows per
        `RecordBatch` read at a time, not how many rows are fetched. ``uri``
        is most commonly a local directory path (LanceDB is embedded-first).
        Warns via ``warnings.warn`` if the table's vector index reports a
        distance metric other than cosine. Raises :class:`ConnectorError` on
        I/O, network, or schema problems. Native only in builds compiled with
        the ``lancedb`` Cargo feature; every other build (including the PyPI
        wheels) serves it through the official ``lancedb`` SDK
        (``pip install vechealth[lancedb]``) with the same contract.
        """
        ...
    @staticmethod
    def from_weaviate(
        url: str,
        collection: str,
        api_key: str | None = None,
        grpc_port: int = 50051,
        vector_name: str | None = None,
        page_size: int = 1000,
    ) -> VecHealthEvaluator:
        """Fetches every object's vector from a Weaviate collection via
        ``Collection.iterator()`` (never ``near_vector``/ANN search). Pulls
        the complete collection. Requires the optional ``weaviate-client``
        SDK (``pip install vechealth[weaviate]``) — this connector is a
        pure-Python wrapper (``vechealth.connectors.weaviate``), not part of
        the compiled ``_core`` extension; it is attached to this class at
        import time so its contract matches every other ``from_*``
        constructor. ``vector_name`` selects which named vector to read on a
        collection with more than one (required only in that case). Warns
        via ``warnings.warn`` if the collection's distance metric isn't
        cosine. Raises :class:`ConnectorError` on connection, auth, or
        schema problems.
        """
        ...
    @staticmethod
    def from_chroma(
        collection: str,
        path: str | None = None,
        host: str | None = None,
        port: int = 8000,
        ssl: bool = False,
        api_key: str | None = None,
        tenant: str = "default_tenant",
        database: str = "default_database",
        page_size: int = 1000,
    ) -> VecHealthEvaluator:
        """Fetches every embedding from a Chroma collection via
        ``Collection.get()`` (never ``.query()``/ANN search). Pulls the
        complete collection. Requires the optional ``chromadb`` SDK
        (``pip install vechealth[chroma]``) — this connector is a
        pure-Python wrapper (``vechealth.connectors.chroma``), not part of
        the compiled ``_core`` extension; it is attached to this class at
        import time so its contract matches every other ``from_*``
        constructor. Exactly one of ``path`` (embedded, on-disk) or ``host``
        (server/Chroma Cloud) must be given. Warns via ``warnings.warn`` if
        the collection's distance metric isn't cosine. Raises
        :class:`ConnectorError` on connection, auth, or schema problems.
        """
        ...
    @staticmethod
    def from_milvus(
        uri: str,
        collection: str,
        vector_field: str | None = None,
        user: str = "",
        password: str = "",
        token: str = "",
        db_name: str = "",
        batch_size: int = 1000,
    ) -> VecHealthEvaluator:
        """Fetches every vector from a Milvus collection via
        ``MilvusClient.query_iterator()`` (never ``.search()``/ANN). Pulls
        the complete collection. Requires the optional ``pymilvus`` SDK
        (``pip install vechealth[milvus]``) — this connector is a
        pure-Python wrapper (``vechealth.connectors.milvus``), not part of
        the compiled ``_core`` extension; it is attached to this class at
        import time so its contract matches every other ``from_*``
        constructor. ``uri`` follows ``pymilvus`` conventions: a
        ``http(s)://host:port`` server address, or a local ``.db`` file path
        for embedded Milvus Lite. ``vector_field`` selects which field to
        read on a collection with more than one dense vector field (required
        only in that case). Warns via ``warnings.warn`` if the collection's
        distance metric isn't cosine. Raises :class:`ConnectorError` on
        connection, auth, or schema problems.
        """
        ...
    @staticmethod
    def from_pinecone(
        api_key: str,
        index_name: str,
        host: str | None = None,
        namespace: str = "",
        batch_size: int = 100,
    ) -> VecHealthEvaluator:
        """Fetches every vector from a Pinecone index/namespace via
        ``Index.list()`` + ``Index.fetch()`` (never ``.query()``/ANN
        search). Pulls the complete index/namespace. Requires the optional
        ``pinecone`` SDK (``pip install vechealth[pinecone]``) — this
        connector is a pure-Python wrapper (``vechealth.connectors.
        pinecone``), not part of the compiled ``_core`` extension; it is
        attached to this class at import time so its contract matches every
        other ``from_*`` constructor. Pinecone has no cheap bulk-export
        path (unlike every other connector here): ``Index.list()`` is
        capped at 100 IDs per page by the API itself, so this is
        meaningfully chattier/slower than the others on a large index — a
        backup/snapshot is a better source than a live production index.
        ``batch_size`` is clamped to that 100-ID cap. Warns via
        ``warnings.warn`` if the index's distance metric isn't cosine.
        Raises :class:`ConnectorError` on connection, auth, or schema
        problems.
        """
        ...
    @property
    def n_vectors(self) -> int: ...
    @property
    def dim(self) -> int: ...
    def get_knn(
        self, k: int, batch_size: int | None = None
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]: ...
    def identify_hubs(
        self, k: int = 10, top_n: int = 20, batch_size: int | None = None
    ) -> list[HubEntry]:
        """The ``top_n`` points that appear most often among the ``k``
        nearest neighbours of other points, most frequent first."""
        ...
    def compute_hubness(self, k: int = 10, batch_size: int | None = None) -> HubnessResult: ...
    def compute_dispersion(self, k: int = 10, batch_size: int | None = None) -> DispersionResult: ...
    def compute_anisotropy(self) -> AnisotropyResult: ...
    def compute_outliers(
        self, distance_threshold: float, batch_size: int | None = None
    ) -> OutliersResult: ...
    def compute_duplicates(
        self, epsilon: float = 0.05, batch_size: int | None = None
    ) -> DuplicatesResult: ...
    def compute_intrinsic_dim(
        self, k: int = 20, batch_size: int | None = None
    ) -> IntrinsicDimResult: ...
    def compute_qmas(
        self,
        queries: npt.ArrayLike,
        k: int = 10,
        batch_size: int | None = None,
    ) -> QmasResult: ...
    def compute_snc(self, k: int = 10, batch_size: int | None = None) -> SncResult: ...
    def compute_all(
        self,
        queries: npt.ArrayLike | None = None,
        k: int = 10,
        k_intrinsic_dim: int = 20,
        batch_size: int | None = None,
        duplicate_epsilon: float = 0.05,
        outlier_distance_threshold: float | None = 1.2,
    ) -> AllMetricsResult:
        """Runs every metric in one pass. ``outlier_distance_threshold`` is a
        chord distance on the unit sphere (never above 2) and defaults to 1.2,
        the value used for the results in the accompanying preprint; it is
        absolute, so recalibrate it for other encoders or dimensionalities.
        ``None`` derives it from the data (3x the mean nearest-neighbour
        distance) — for typical high-dimensional text embeddings that lands
        close to or above 2.0, where essentially no point can be flagged.
        """
        ...
    def compute_report(
        self,
        queries: npt.ArrayLike | None = None,
        k: int = 10,
        k_intrinsic_dim: int = 20,
        batch_size: int | None = None,
        duplicate_epsilon: float = 0.05,
        outlier_distance_threshold: float | None = 1.2,
        tags: dict[str, str] | None = None,
    ) -> Report:
        """Same as `compute_all`, but wraps the result in a `Report` —
        savable, reloadable, and comparable against another `Report` later.
        """
        ...
