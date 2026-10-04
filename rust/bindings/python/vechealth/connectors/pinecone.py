"""Pinecone connector — a thin wrapper over the official `pinecone` Python
SDK. Pinecone has no maintained official Rust client and is managed-SaaS-only
(no local/offline scenario at all), so this is Python, same architectural
call as Weaviate/Chroma/Milvus. This module is intentionally the only place
in the codebase that imports `pinecone`; it's a soft/optional dependency,
only needed by callers of `from_pinecone`.

**This is the least-well-fitted connector in the project, and deliberately
so**: Pinecone has no mature, documented bulk-export path. Unlike every other
connector here, there is no cheap sequential scan/scroll/cursor endpoint
that hands back vectors directly. The only route is:

1. `Index.list()` — paginate raw vector IDs, **capped at 100 IDs per page**
   by the API itself (not a choice made here).
2. `Index.fetch(ids=...)` — a per-ID lookup, called once per page of IDs, to
   get the actual vector values back.

So pulling a large index means many small sequential HTTP round-trips no
matter what `batch_size` is set to (up to the hard cap of 100) — this is
several orders of magnitude chattier than Qdrant's `scroll`, pgvector's
cursor, or Milvus's `query_iterator`, all of which return vectors directly
in one call per page. There is still no ANN/search path involved anywhere
here (`.query()` is never called), it's simply that Pinecone's
non-search read path is unusually expensive. Expect this to be slow on
anything beyond a small index; running it against a Pinecone backup/snapshot
rather than a live production index is strongly recommended.

No sampling, no cap on rows fetched beyond what's described above: like the
rest of this project's connectors, this pulls the *complete* index/namespace.

Exposed uniformly with the native connectors: importing `vechealth` adds
`VecHealthEvaluator.from_pinecone` (see `vechealth/__init__.py`) so callers
never need to know this one happens to be a Python wrapper rather than Rust.
"""

from __future__ import annotations

import warnings

import numpy as np

from vechealth._core import ConnectorError, VecHealthEvaluator

__all__ = ["fetch_all", "from_pinecone"]

# Pinecone's dense-field `metric` values, normalized to the same naming
# `vechealth-connectors` uses for Qdrant/LanceDB (`cosine`/`dot`/`euclidean`)
# so the warning message in `from_pinecone` reads identically across every
# connector.
_DISTANCE_METRIC_NAMES = {
    "cosine": "cosine",
    "dotproduct": "dot",
    "euclidean": "euclidean",
}

# Hard API limit: `Index.list()`/`list_paginated()` reject a `limit` outside
# 1-100 before making any request (see the installed `pinecone` SDK's
# `Index.list_paginated` docstring) — not a default this project chose.
_MAX_LIST_PAGE_SIZE = 100


def _require_pinecone():
    try:
        import pinecone
    except ImportError as exc:
        raise ConnectorError(
            "the Pinecone connector needs the official `pinecone` SDK — install it "
            "with `pip install pinecone`"
        ) from exc
    return pinecone


def fetch_all(
    api_key: str,
    index_name: str,
    host: str | None = None,
    namespace: str = "",
    batch_size: int = _MAX_LIST_PAGE_SIZE,
) -> tuple[np.ndarray, str | None]:
    """Fetches every vector from a Pinecone index/namespace. Returns
    `(vectors, distance_metric)`, where `distance_metric` is one of
    `"cosine"`/`"dot"`/`"euclidean"`/`None` (`None` when the index has no
    single unambiguous dense vector field to read a metric off). `host`
    skips the control-plane host lookup (e.g. for a private-endpoint
    deployment) but the metric/dimension are still read via
    `describe_index(index_name)` regardless. `batch_size` is clamped to
    Pinecone's own hard cap of 100 IDs per `list()` page — see this module's
    docstring for why that makes this connector unusually chatty on a large
    index. Raises `vechealth.ConnectorError` on any connection, auth, or
    schema problem.
    """
    pinecone = _require_pinecone()
    from pinecone.errors import PineconeError

    page_size = max(1, min(batch_size, _MAX_LIST_PAGE_SIZE))

    try:
        pc = pinecone.Pinecone(api_key=api_key)
        info = pc.describe_index(index_name)
        try:
            distance_metric = _DISTANCE_METRIC_NAMES.get(info.metric)
        except AttributeError:
            # `IndexModel.metric` is a deprecated convenience property that only
            # resolves when the schema has exactly one vector field; ambiguous
            # (or non-dense) schemas raise AttributeError. Treated the same as
            # "unrecognized metric" elsewhere in this project: unknown, not fatal.
            distance_metric = None

        idx = pc.index(name=index_name, host=host) if host else pc.index(name=index_name)

        rows: list[list[float]] = []
        dim: int | None = None
        for page in idx.list(namespace=namespace, limit=page_size):
            ids = [item.id for item in page.vectors]
            if not ids:
                continue
            fetched = idx.fetch(ids=ids, namespace=namespace)
            for vector_id, vector in fetched.vectors.items():
                values = vector.values
                if not values:
                    raise ConnectorError(
                        f"record '{vector_id}' has no dense vector values (sparse-only "
                        "records aren't supported by this connector)"
                    )
                if dim is None:
                    dim = len(values)
                elif len(values) != dim:
                    raise ConnectorError(
                        f"inconsistent vector dimension in index '{index_name}': "
                        f"expected {dim}, got {len(values)} (record '{vector_id}')"
                    )
                rows.append(list(values))
    except ConnectorError:
        raise
    except PineconeError as exc:
        raise ConnectorError(f"Pinecone request failed: {exc}") from exc

    if not rows:
        raise ConnectorError(f"index '{index_name}' (namespace '{namespace}') has no vectors")

    return np.asarray(rows, dtype=np.float32), distance_metric


def from_pinecone(
    api_key: str,
    index_name: str,
    host: str | None = None,
    namespace: str = "",
    batch_size: int = _MAX_LIST_PAGE_SIZE,
) -> VecHealthEvaluator:
    """Fetches every vector from a Pinecone index/namespace via
    `Index.list()` + `Index.fetch()` (never `.query()`/ANN search) and builds a
    `VecHealthEvaluator` from it. Pulls the
    *complete* index/namespace. **Read this module's docstring** — Pinecone
    has no cheap bulk-export path, so this is meaningfully slower than the
    other connectors in this project on anything beyond a small index; a
    backup/snapshot is a better source than a live production index here.
    Warns via `warnings.warn` if the index's distance metric isn't cosine —
    same contract as `VecHealthEvaluator.from_qdrant`/`from_pgvector`/
    `from_lancedb`/`from_weaviate`/`from_chroma`/`from_milvus`. Raises
    `vechealth.ConnectorError` on connection, auth, or schema problems.
    """
    vectors, distance_metric = fetch_all(
        api_key,
        index_name,
        host=host,
        namespace=namespace,
        batch_size=batch_size,
    )
    if distance_metric is not None and distance_metric != "cosine":
        warnings.warn(
            f"source index was indexed with '{distance_metric}' distance, but "
            "VecHealthEvaluator assumes cosine similarity (it L2-normalizes vectors "
            "internally) — metric results may not reflect how this index is "
            "actually queried"
        )
    return VecHealthEvaluator(vectors)
