"""Chroma connector — a thin wrapper over the official `chromadb` Python SDK.

Chroma is explicitly *not* given a native Rust file-parser in this project:
its on-disk local storage format has already changed once (`duckdb+parquet` -> `sqlite3` + per-collection binary
segments), so the only stable integration point is their own client library,
even for the embedded/local case. This module is intentionally the only
place in the codebase that imports `chromadb`; it's a soft/optional
dependency, only needed by callers of `from_chroma`.

Uses `Collection.get(include=["embeddings"])` with `limit`/`offset`
pagination exclusively — a plain scan over stored records, never
`.query()`/`near_vector` (ANN search). This connector never touches Chroma's
HNSW index, like every other connector in this project.

No sampling, no cap on rows fetched: like the rest of this project's
connectors, this pulls the *complete* collection. `page_size` only controls
how many records are requested per
`get()` call, not how many are ultimately fetched.

Exposed uniformly with the native connectors: importing `vechealth` adds
`VecHealthEvaluator.from_chroma` (see `vechealth/__init__.py`) so callers
never need to know this one happens to be a Python wrapper rather than Rust.
"""

from __future__ import annotations

import warnings

import numpy as np

from vechealth._core import ConnectorError, VecHealthEvaluator

__all__ = ["fetch_all", "from_chroma"]

# Chroma's `hnsw:space` metadata values, normalized to the same naming
# `vechealth-connectors` uses for Qdrant/LanceDB (`cosine`/`dot`/`euclidean`)
# so the warning message in `from_chroma` reads identically across every
# connector. Chroma defaults a collection with no `hnsw:space` set to `"l2"`
# (confirmed in `chromadb`'s own test strategies, not just assumed).
_DISTANCE_METRIC_NAMES = {
    "l2": "euclidean",
    "cosine": "cosine",
    "ip": "dot",
}


def _require_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise ConnectorError(
            "the Chroma connector needs the official `chromadb` SDK — install it "
            "with `pip install chromadb`"
        ) from exc
    return chromadb


def fetch_all(
    collection: str,
    path: str | None = None,
    host: str | None = None,
    port: int = 8000,
    ssl: bool = False,
    api_key: str | None = None,
    tenant: str = "default_tenant",
    database: str = "default_database",
    page_size: int = 1000,
) -> tuple[np.ndarray, str | None]:
    """Fetches every embedding from a Chroma collection. Returns
    `(vectors, distance_metric)`, where `distance_metric` is one of
    `"cosine"`/`"dot"`/`"euclidean"`/`None`. Exactly one of `path` (embedded,
    on-disk `PersistentClient`) or `host` (`HttpClient`, e.g. a self-hosted
    server or Chroma Cloud) must be given. `api_key` is passed as an
    `x-chroma-token` header for a server that requires auth; ignored with
    `path`. Raises `vechealth.ConnectorError` on any connection, auth, or
    schema problem.
    """
    if (path is None) == (host is None):
        raise ConnectorError("pass exactly one of path= (local) or host= (server)")

    chromadb = _require_chromadb()
    from chromadb.errors import ChromaError

    try:
        if path is not None:
            client = chromadb.PersistentClient(path=path, tenant=tenant, database=database)
        else:
            headers = {"x-chroma-token": api_key} if api_key else None
            client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
                headers=headers,
                tenant=tenant,
                database=database,
            )
        coll = client.get_collection(collection)
    except (ValueError, ChromaError) as exc:
        raise ConnectorError(f"could not open Chroma collection '{collection}': {exc}") from exc

    metric_raw = (coll.metadata or {}).get("hnsw:space", "l2")
    distance_metric = _DISTANCE_METRIC_NAMES.get(metric_raw)

    rows: list[list[float]] = []
    dim: int | None = None
    offset = 0
    try:
        while True:
            page = coll.get(include=["embeddings"], limit=page_size, offset=offset)
            ids = page["ids"]
            if not ids:
                break
            embeddings = page["embeddings"]
            for record_id, vector in zip(ids, embeddings):
                if vector is None:
                    raise ConnectorError(f"record '{record_id}' has no embedding")
                vector = list(vector)
                if dim is None:
                    dim = len(vector)
                elif len(vector) != dim:
                    raise ConnectorError(
                        f"inconsistent vector dimension in collection '{collection}': "
                        f"expected {dim}, got {len(vector)} (record '{record_id}')"
                    )
                rows.append(vector)
            offset += len(ids)
            if len(ids) < page_size:
                break
    except ChromaError as exc:
        raise ConnectorError(f"reading collection '{collection}' failed: {exc}") from exc

    if not rows:
        raise ConnectorError(f"collection '{collection}' has no records")

    return np.asarray(rows, dtype=np.float32), distance_metric


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
    `Collection.get()` (never `.query()`/ANN search) and builds a
    `VecHealthEvaluator` from it. Pulls the
    *complete* collection; `page_size` only controls the transfer page size,
    not how many records are fetched. Warns via `warnings.warn` if the
    collection's distance metric isn't cosine — same contract as
    `VecHealthEvaluator.from_qdrant`/`from_pgvector`/`from_lancedb`/
    `from_weaviate`. Raises `vechealth.ConnectorError` on connection, auth, or
    schema problems.
    """
    vectors, distance_metric = fetch_all(
        collection,
        path=path,
        host=host,
        port=port,
        ssl=ssl,
        api_key=api_key,
        tenant=tenant,
        database=database,
        page_size=page_size,
    )
    if distance_metric is not None and distance_metric != "cosine":
        warnings.warn(
            f"source collection was indexed with '{distance_metric}' distance, but "
            "VecHealthEvaluator assumes cosine similarity (it L2-normalizes vectors "
            "internally) — metric results may not reflect how this collection is "
            "actually queried"
        )
    return VecHealthEvaluator(vectors)
