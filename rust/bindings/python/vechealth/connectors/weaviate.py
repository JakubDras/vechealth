"""Weaviate connector — a thin wrapper over the official `weaviate-client`
Python SDK. Weaviate has no maintained official Rust client, so the only
stable integration point is Python, not Rust. This module is intentionally
the only place in the codebase that
imports `weaviate`; it's a soft/optional dependency, only needed by callers
of `from_weaviate`.

Uses `Collection.iterator()` exclusively: a plain cursor-based scan (an
`after`-UUID cursor over the REST object-listing endpoint), never
`near_vector`/`near_text`/GraphQL search. This connector never touches
Weaviate's ANN path, matching every other connector in this project.

No sampling, no cap on rows fetched: like the rest of this project's
connectors, this pulls the *complete* collection. `page_size` only controls
the client-side read-ahead buffer
(`cache_size`) the SDK keeps between gRPC round-trips, not how many objects
are ultimately fetched.

Exposed uniformly with the native connectors: importing `vechealth` adds
`VecHealthEvaluator.from_weaviate` (see `vechealth/__init__.py`) so callers
never need to know this one happens to be a Python wrapper rather than Rust.
"""

from __future__ import annotations

import warnings
from urllib.parse import urlparse

import numpy as np

from vechealth._core import ConnectorError, VecHealthEvaluator

__all__ = ["fetch_all", "from_weaviate"]

# Weaviate's `VectorDistances` enum values, normalized to the same naming
# `vechealth-connectors` uses for Qdrant/LanceDB (`cosine`/`dot`/`euclidean`)
# so the warning message in `from_weaviate` reads identically across every
# connector. `manhattan`/`hamming` have no equivalent and map to `None`,
# mirroring how the Rust connectors handle a distance metric they don't
# recognize.
_DISTANCE_METRIC_NAMES = {
    "cosine": "cosine",
    "dot": "dot",
    "l2-squared": "euclidean",
}


def _require_weaviate_client():
    try:
        import weaviate
    except ImportError as exc:
        raise ConnectorError(
            "the Weaviate connector needs the official `weaviate-client` SDK — install it "
            "with `pip install weaviate-client`"
        ) from exc
    return weaviate


def _parse_http_url(url: str) -> tuple[str, int, bool]:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if not parsed.hostname:
        raise ConnectorError(f"could not parse a host out of Weaviate url '{url}'")
    secure = parsed.scheme == "https"
    port = parsed.port or (443 if secure else 8080)
    return parsed.hostname, port, secure


def _distance_metric_for(config, vector_name: str | None) -> tuple[str | None, str | None]:
    """Resolves which named vector to read and its distance metric off a
    collection's config. Returns `(resolved_vector_name, distance_metric)`;
    `resolved_vector_name` is `None` for a legacy/single-vector collection
    (no named vectors at all), in which case every object's `.vector` dict
    has exactly one entry and its key doesn't need to be known up front.
    """
    named = getattr(config, "vector_config", None) or {}
    if named:
        if vector_name is not None:
            if vector_name not in named:
                raise ConnectorError(
                    f"named vector '{vector_name}' not found in collection config — "
                    f"available: {sorted(named)}"
                )
            resolved = vector_name
        elif len(named) == 1:
            resolved = next(iter(named))
        else:
            raise ConnectorError(
                f"collection has multiple named vectors ({sorted(named)}) — pass "
                "vector_name= to pick one"
            )
        idx_cfg = named[resolved].vector_index_config
        metric = getattr(idx_cfg, "distance_metric", None)
        metric_value = getattr(metric, "value", None)
        return resolved, _DISTANCE_METRIC_NAMES.get(metric_value)

    idx_cfg = getattr(config, "vector_index_config", None)
    metric = getattr(idx_cfg, "distance_metric", None) if idx_cfg is not None else None
    metric_value = getattr(metric, "value", None)
    return vector_name, _DISTANCE_METRIC_NAMES.get(metric_value)


def fetch_all(
    url: str,
    collection: str,
    api_key: str | None = None,
    grpc_port: int = 50051,
    vector_name: str | None = None,
    page_size: int = 1000,
) -> tuple[np.ndarray, str | None]:
    """Fetches every object's vector from a Weaviate collection. Returns
    `(vectors, distance_metric)`, where `distance_metric` is one of
    `"cosine"`/`"dot"`/`"euclidean"`/`None`. `vector_name` selects which
    named vector to read on a collection configured with multiple named
    vectors (required only when there's more than one); ignored otherwise.
    Raises `vechealth.ConnectorError` on any connection, auth, or schema
    problem.
    """
    weaviate = _require_weaviate_client()
    from weaviate.classes.init import Auth
    from weaviate.exceptions import WeaviateBaseError

    host, http_port, secure = _parse_http_url(url)
    try:
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=http_port,
            http_secure=secure,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=secure,
            auth_credentials=Auth.api_key(api_key) if api_key else None,
        )
    except WeaviateBaseError as exc:
        raise ConnectorError(f"could not connect to Weaviate at '{url}': {exc}") from exc

    try:
        try:
            if not client.collections.exists(collection):
                raise ConnectorError(f"collection '{collection}' does not exist at '{url}'")
            coll = client.collections.get(collection)
            resolved_vector_name, distance_metric = _distance_metric_for(
                coll.config.get(), vector_name
            )

            rows: list[list[float]] = []
            dim: int | None = None
            for obj in coll.iterator(
                include_vector=True, return_properties=False, cache_size=page_size
            ):
                vectors_by_name = obj.vector
                if resolved_vector_name is not None:
                    vector = vectors_by_name.get(resolved_vector_name)
                elif len(vectors_by_name) == 1:
                    vector = next(iter(vectors_by_name.values()))
                else:
                    raise ConnectorError(
                        f"object {obj.uuid} carries multiple named vectors "
                        f"({sorted(vectors_by_name)}) — pass vector_name= to pick one"
                    )
                if not vector:
                    raise ConnectorError(f"object {obj.uuid} has no vector")
                if isinstance(vector[0], list):
                    raise ConnectorError(
                        f"object {obj.uuid} has a multi-vector (e.g. ColBERT-style), which "
                        "this connector doesn't support — pass a single-vector collection"
                    )
                if dim is None:
                    dim = len(vector)
                elif len(vector) != dim:
                    raise ConnectorError(
                        f"inconsistent vector dimension in collection '{collection}': "
                        f"expected {dim}, got {len(vector)} (object {obj.uuid})"
                    )
                rows.append(vector)
        except ConnectorError:
            raise
        except WeaviateBaseError as exc:
            raise ConnectorError(f"reading collection '{collection}' failed: {exc}") from exc
    finally:
        client.close()

    if not rows:
        raise ConnectorError(f"collection '{collection}' has no objects")

    return np.asarray(rows, dtype=np.float32), distance_metric


def from_weaviate(
    url: str,
    collection: str,
    api_key: str | None = None,
    grpc_port: int = 50051,
    vector_name: str | None = None,
    page_size: int = 1000,
) -> VecHealthEvaluator:
    """Fetches every object's vector from a Weaviate collection via
    `Collection.iterator()` (never `near_vector`/ANN search) and builds a
    `VecHealthEvaluator` from it. Pulls the
    *complete* collection; `page_size` only controls the client-side
    read-ahead buffer, not how many objects are fetched. Warns via
    `warnings.warn` if the collection's distance metric isn't cosine — same
    contract as `VecHealthEvaluator.from_qdrant`/`from_pgvector`/
    `from_lancedb`. Raises `vechealth.ConnectorError` on connection, auth, or
    schema problems.
    """
    vectors, distance_metric = fetch_all(
        url,
        collection,
        api_key=api_key,
        grpc_port=grpc_port,
        vector_name=vector_name,
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
