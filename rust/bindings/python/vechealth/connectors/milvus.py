"""Milvus/Zilliz connector — a thin wrapper over the official `pymilvus`
Python SDK. Milvus has no maintained official Rust client, so this is
Python, same architectural call as Weaviate/Chroma. This module is intentionally the only place in the codebase
that imports `pymilvus`; it's a soft/optional dependency, only needed by
callers of `from_milvus`.

Uses `MilvusClient.query_iterator()` exclusively — a plain scalar
scan/pagination over stored records, officially documented as the right tool
for bulk export, never `.search()` (ANN). This connector never touches
Milvus's index-based search path, like every other connector in this
project.

No sampling, no cap on rows fetched: like the rest of this project's
connectors, this pulls the *complete* collection. `batch_size` only controls
how many rows are requested per
iterator page, not how many are ultimately fetched.

Exposed uniformly with the native connectors: importing `vechealth` adds
`VecHealthEvaluator.from_milvus` (see `vechealth/__init__.py`) so callers
never need to know this one happens to be a Python wrapper rather than Rust.
"""

from __future__ import annotations

import warnings

import numpy as np

from vechealth._core import ConnectorError, VecHealthEvaluator

__all__ = ["fetch_all", "from_milvus"]

# Milvus's `metric_type` index values, normalized to the same naming
# `vechealth-connectors` uses for Qdrant/LanceDB (`cosine`/`dot`/`euclidean`)
# so the warning message in `from_milvus` reads identically across every
# connector. Binary-vector metrics (`HAMMING`/`JACCARD`/...) have no
# equivalent and map to `None`, mirroring how the Rust connectors handle a
# distance metric they don't recognize.
_DISTANCE_METRIC_NAMES = {
    "COSINE": "cosine",
    "IP": "dot",
    "L2": "euclidean",
}


def _require_pymilvus():
    try:
        import pymilvus
    except ImportError as exc:
        raise ConnectorError(
            "the Milvus connector needs the official `pymilvus` SDK — install it "
            "with `pip install pymilvus`"
        ) from exc
    return pymilvus


def _resolve_vector_field(pymilvus, fields: list[dict], vector_field: str | None) -> str:
    dense = [f["name"] for f in fields if f["type"] == pymilvus.DataType.FLOAT_VECTOR]
    other_vector_types = {
        pymilvus.DataType.BINARY_VECTOR,
        pymilvus.DataType.FLOAT16_VECTOR,
        pymilvus.DataType.BFLOAT16_VECTOR,
        pymilvus.DataType.SPARSE_FLOAT_VECTOR,
        pymilvus.DataType.INT8_VECTOR,
    }
    unsupported = [f["name"] for f in fields if f["type"] in other_vector_types]

    if vector_field is not None:
        if vector_field in dense:
            return vector_field
        if vector_field in unsupported:
            raise ConnectorError(
                f"field '{vector_field}' is not a dense float vector — this connector "
                "only supports FLOAT_VECTOR fields"
            )
        raise ConnectorError(f"field '{vector_field}' not found in collection schema")

    if len(dense) == 1:
        return dense[0]
    if not dense:
        raise ConnectorError(
            "collection has no FLOAT_VECTOR field (this connector doesn't support "
            "binary/float16/bfloat16/sparse/int8 vector fields yet)"
        )
    raise ConnectorError(
        f"collection has multiple dense vector fields ({sorted(dense)}) — pass "
        "vector_field= to pick one"
    )


def fetch_all(
    uri: str,
    collection: str,
    vector_field: str | None = None,
    user: str = "",
    password: str = "",
    token: str = "",
    db_name: str = "",
    batch_size: int = 1000,
) -> tuple[np.ndarray, str | None]:
    """Fetches every vector from a Milvus collection. Returns
    `(vectors, distance_metric)`, where `distance_metric` is one of
    `"cosine"`/`"dot"`/`"euclidean"`/`None`. `uri` follows `pymilvus`'
    conventions: a `http(s)://host:port` server address, or a local `.db`
    file path to use embedded Milvus Lite. `vector_field` selects which
    field to read on a collection with more than one dense vector field
    (required only when there's more than one); ignored otherwise. Raises
    `vechealth.ConnectorError` on any connection, auth, or schema problem.
    """
    pymilvus = _require_pymilvus()
    from pymilvus.exceptions import MilvusException

    try:
        client = pymilvus.MilvusClient(
            uri=uri, user=user, password=password, token=token, db_name=db_name
        )
        if not client.has_collection(collection):
            raise ConnectorError(f"collection '{collection}' does not exist at '{uri}'")

        schema = client.describe_collection(collection)
        resolved_field = _resolve_vector_field(pymilvus, schema["fields"], vector_field)

        distance_metric = None
        index_names = client.list_indexes(collection, field_name=resolved_field)
        if index_names:
            index_info = client.describe_index(collection, index_name=index_names[0])
            distance_metric = _DISTANCE_METRIC_NAMES.get(index_info.get("metric_type"))

        rows: list[list[float]] = []
        dim: int | None = None
        iterator = client.query_iterator(
            collection_name=collection, batch_size=batch_size, output_fields=[resolved_field]
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for record in batch:
                    vector = list(record[resolved_field])
                    if dim is None:
                        dim = len(vector)
                    elif len(vector) != dim:
                        raise ConnectorError(
                            f"inconsistent vector dimension in collection '{collection}': "
                            f"expected {dim}, got {len(vector)}"
                        )
                    rows.append(vector)
        finally:
            iterator.close()
    except ConnectorError:
        raise
    except MilvusException as exc:
        raise ConnectorError(f"Milvus request failed: {exc}") from exc

    if not rows:
        raise ConnectorError(f"collection '{collection}' has no rows")

    return np.asarray(rows, dtype=np.float32), distance_metric


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
    `MilvusClient.query_iterator()` (never `.search()`/ANN) and builds a
    `VecHealthEvaluator` from it. Pulls the
    *complete* collection; `batch_size` only controls the transfer page
    size, not how many rows are fetched. Warns via `warnings.warn` if the
    collection's distance metric isn't cosine — same contract as
    `VecHealthEvaluator.from_qdrant`/`from_pgvector`/`from_lancedb`/
    `from_weaviate`/`from_chroma`. Raises `vechealth.ConnectorError` on
    connection, auth, or schema problems.
    """
    vectors, distance_metric = fetch_all(
        uri,
        collection,
        vector_field=vector_field,
        user=user,
        password=password,
        token=token,
        db_name=db_name,
        batch_size=batch_size,
    )
    if distance_metric is not None and distance_metric != "cosine":
        warnings.warn(
            f"source collection was indexed with '{distance_metric}' distance, but "
            "VecHealthEvaluator assumes cosine similarity (it L2-normalizes vectors "
            "internally) — metric results may not reflect how this collection is "
            "actually queried"
        )
    return VecHealthEvaluator(vectors)
