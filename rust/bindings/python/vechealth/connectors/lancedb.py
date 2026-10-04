"""LanceDB connector — a thin wrapper over the official `lancedb` Python SDK.

LanceDB also has a native Rust connector in `vechealth-connectors`, but it
sits behind the `lancedb` Cargo feature because the LanceDB/DataFusion
dependency tree is by far the heaviest in the project. The wheels published
to PyPI are built without it, so `VecHealthEvaluator.from_lancedb` is served
by this module instead (see `vechealth/__init__.py`); a build compiled with
`--features lancedb` keeps the native implementation. Both follow the same
contract and are covered by the same smoke test.

This module is the only place in the Python package that imports `lancedb`;
it is a soft/optional dependency, only needed by callers of `from_lancedb`
(`pip install vechealth[lancedb]`).

Uses a plain table scan (`Table.search()` with no query vector, restricted to
the vector column) — never a vector search, so the table's ANN index is not
touched. No sampling and no cap on rows: the complete table is pulled;
`batch_size` only controls how many rows are read per Arrow batch.
"""

from __future__ import annotations

import warnings

import numpy as np

from vechealth._core import ConnectorError, VecHealthEvaluator

__all__ = ["fetch_all", "from_lancedb"]

# LanceDB's index `distance_type` values, normalized to the naming the native
# connectors use (`cosine`/`dot`/`euclidean`) so the warning in `from_lancedb`
# reads identically across every connector.
_DISTANCE_METRIC_NAMES = {
    "cosine": "cosine",
    "dot": "dot",
    "l2": "euclidean",
}


def _require_lancedb():
    try:
        import lancedb
    except ImportError as exc:
        raise ConnectorError(
            "the LanceDB connector needs the official `lancedb` SDK — install it "
            "with `pip install vechealth[lancedb]`"
        ) from exc
    return lancedb


def _declared_dim(tbl, table: str, vector_column: str) -> int:
    """Reads the vector dimension off the table's Arrow schema, failing fast
    when the column is missing or is not a fixed-size float vector column."""
    import pyarrow as pa

    schema = tbl.schema
    try:
        field = schema.field(vector_column)
    except KeyError:
        raise ConnectorError(
            f"column '{vector_column}' not found in table '{table}' "
            f"(columns: {list(schema.names)})"
        ) from None
    value_type = getattr(field.type, "value_type", None)
    if not pa.types.is_fixed_size_list(field.type) or value_type not in (pa.float32(), pa.float64()):
        raise ConnectorError(
            f"column '{vector_column}' in table '{table}' has type {field.type}, expected a "
            "fixed-size vector column (FixedSizeList<Float32> or FixedSizeList<Float64>)"
        )
    return field.type.list_size


def _distance_metric(tbl, table: str, vector_column: str) -> str | None:
    """The distance metric of the vector index on `vector_column`, or None
    when the column has no index (a normal state for small or embedded
    tables) or the index does not report one."""
    try:
        index = next((i for i in tbl.list_indices() if vector_column in i.columns), None)
        if index is None:
            return None
        stats = tbl.index_stats(index.name)
    except Exception as exc:  # noqa: BLE001 - any SDK failure becomes a ConnectorError
        raise ConnectorError(f"could not read the vector index of table '{table}': {exc}") from exc
    return _DISTANCE_METRIC_NAMES.get(getattr(stats, "distance_type", None))


def fetch_all(
    uri: str,
    table: str,
    vector_column: str,
    batch_size: int = 1024,
) -> tuple[np.ndarray, str | None]:
    """Fetches every vector from a LanceDB table. Returns
    `(vectors, distance_metric)`, where `distance_metric` is one of
    `"cosine"`/`"dot"`/`"euclidean"`/`None`. Raises `vechealth.ConnectorError`
    on any connection, schema, or read problem.
    """
    lancedb = _require_lancedb()

    try:
        tbl = lancedb.connect(uri).open_table(table)
    except Exception as exc:  # noqa: BLE001 - any SDK failure becomes a ConnectorError
        raise ConnectorError(
            f"could not open table '{table}' in LanceDB database '{uri}': {exc}"
        ) from exc

    dim = _declared_dim(tbl, table, vector_column)
    distance_metric = _distance_metric(tbl, table, vector_column)

    chunks: list[np.ndarray] = []
    try:
        for batch in tbl.search().select([vector_column]).to_batches(batch_size=batch_size):
            column = batch.column(batch.schema.get_field_index(vector_column))
            if column.null_count:
                raise ConnectorError(
                    f"table '{table}' has null vectors in column '{vector_column}'"
                )
            values = column.flatten().to_numpy(zero_copy_only=False)
            chunks.append(np.asarray(values, dtype=np.float32).reshape(-1, dim))
    except ConnectorError:
        raise
    except Exception as exc:  # noqa: BLE001 - any SDK failure becomes a ConnectorError
        raise ConnectorError(f"reading table '{table}' failed: {exc}") from exc

    vectors = np.concatenate(chunks, axis=0) if chunks else np.empty((0, dim), dtype=np.float32)
    if vectors.shape[0] == 0:
        raise ConnectorError(f"table '{table}' has no rows")
    return vectors, distance_metric


def from_lancedb(
    uri: str,
    table: str,
    vector_column: str,
    batch_size: int = 1024,
) -> VecHealthEvaluator:
    """Fetches every vector from a LanceDB table via a plain columnar scan
    (never a vector search) and builds a
    `VecHealthEvaluator` from it. Pulls the *complete* table; `batch_size`
    only controls how many rows are read per Arrow batch. `uri` is most
    commonly a local directory path (LanceDB is embedded-first). Warns via
    `warnings.warn` if the table's vector index reports a distance metric
    other than cosine — same contract as `VecHealthEvaluator.from_qdrant` and
    the other connectors. Raises `vechealth.ConnectorError` on connection,
    schema, or read problems.
    """
    vectors, distance_metric = fetch_all(uri, table, vector_column, batch_size=batch_size)
    if distance_metric is not None and distance_metric != "cosine":
        warnings.warn(
            f"source collection was indexed with '{distance_metric}' distance, but "
            "VecHealthEvaluator assumes cosine similarity (it L2-normalizes vectors "
            "internally) — metric results may not reflect how this collection is "
            "actually queried"
        )
    return VecHealthEvaluator(vectors)
