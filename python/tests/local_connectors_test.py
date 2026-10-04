"""Manual smoke test for the connector-backed evaluator constructors
(`VecHealthEvaluator.from_local/from_qdrant/from_pgvector/from_lancedb/
from_weaviate/from_chroma/from_milvus/from_pinecone`), covering the local
loaders, the native (Rust) connectors and the SDK-backed ones. Run it
directly (`python python/tests/local_connectors_test.py`); CI does.

Qdrant/pgvector/Weaviate/Pinecone aren't exercised end-to-end here — there's
no live server (or, for Pinecone, account) in this environment — but their
`from_*` constructors are still called against addresses/credentials nothing
is listening on or valid for, to check that a dead source fails fast with a
typed `vechealth.ConnectorError` instead of hanging or raising some
generic/unrelated exception.

LanceDB, Chroma and Milvus *are* exercised end-to-end (round trip, not just
the error path): all three are embedded/local by construction (LanceDB is a
plain on-disk columnar format; Chroma's `PersistentClient` and Milvus Lite
both run in-process with no server), so a real local store can be built with
each vendor's own Python package (a test-only convenience here, same
relationship this file already has with `pyarrow` — none of `lancedb`/
`chromadb`/`pymilvus[milvus_lite]` are a runtime dependency of `vechealth`
itself).
"""

import csv
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import vechealth as vh


def expect_connector_error(label: str, fn):
    try:
        fn()
    except vh.ConnectorError as e:
        print(f"[ok] {label}: raised ConnectorError as expected ({e})")
    except Exception as e:  # noqa: BLE001 - this *is* the check
        raise AssertionError(
            f"{label}: expected vh.ConnectorError, got {type(e).__name__}: {e}"
        ) from e
    else:
        raise AssertionError(f"{label}: expected ConnectorError, nothing was raised")


def test_from_local_npy():
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((200, 32)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "vectors.npy"
        np.save(path, vectors)
        evaluator = vh.VecHealthEvaluator.from_local(str(path))
        assert evaluator.n_vectors == 200
        assert evaluator.dim == 32
        result = evaluator.compute_hubness(k=5)
        print(f"[ok] from_local .npy: n_vectors={evaluator.n_vectors}, {result}")


def test_from_local_csv():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((150, 16)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "vectors.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"dim{i}" for i in range(16)])
            writer.writerows(vectors.tolist())
        evaluator = vh.VecHealthEvaluator.from_local(str(path), has_header=True)
        assert evaluator.n_vectors == 150
        assert evaluator.dim == 16
        print(f"[ok] from_local .csv: n_vectors={evaluator.n_vectors}, dim={evaluator.dim}")


def test_from_local_parquet():
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((100, 8)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "vectors.parquet"
        table = pa.table({f"dim{i}": vectors[:, i] for i in range(8)})
        pq.write_table(table, path)

        evaluator = vh.VecHealthEvaluator.from_local(str(path))
        assert evaluator.n_vectors == 100
        assert evaluator.dim == 8

        # Column subset/reorder.
        evaluator_subset = vh.VecHealthEvaluator.from_local(
            str(path), columns=["dim2", "dim0"]
        )
        assert evaluator_subset.dim == 2
        print(
            f"[ok] from_local .parquet: n_vectors={evaluator.n_vectors}, "
            f"dim={evaluator.dim}, subset_dim={evaluator_subset.dim}"
        )


def test_from_local_missing_file_raises_connector_error():
    expect_connector_error(
        "from_local missing file",
        lambda: vh.VecHealthEvaluator.from_local("/nonexistent/path/vectors.npy"),
    )


def test_from_qdrant_unreachable_fails_fast():
    start = time.monotonic()
    expect_connector_error(
        "from_qdrant unreachable host",
        lambda: vh.VecHealthEvaluator.from_qdrant(
            "http://127.0.0.1:1", "does-not-exist", timeout_secs=5
        ),
    )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"from_qdrant took {elapsed:.1f}s to fail — not failing fast"


def test_from_pgvector_unreachable_fails_fast():
    start = time.monotonic()
    expect_connector_error(
        "from_pgvector unreachable host",
        lambda: vh.VecHealthEvaluator.from_pgvector(
            "postgresql://user:pass@127.0.0.1:1/db", "items", "embedding", "id"
        ),
    )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"from_pgvector took {elapsed:.1f}s to fail — not failing fast"


def test_from_lancedb_roundtrip():
    import lancedb

    rng = np.random.default_rng(3)
    n, dim = 300, 16  # >=256 rows: IVF_PQ's minimum training set size.
    vectors = rng.standard_normal((n, dim)).astype(np.float32)

    with tempfile.TemporaryDirectory() as d:
        cosine_uri = str(Path(d) / "cosine_db")
        db = lancedb.connect(cosine_uri)
        table = pa.Table.from_pydict(
            {"vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.flatten()), dim)}
        )
        tbl = db.create_table("vectors", table)
        tbl.create_index(metric="cosine", vector_column_name="vector")

        evaluator = vh.VecHealthEvaluator.from_lancedb(cosine_uri, "vectors", "vector")
        assert evaluator.n_vectors == n
        assert evaluator.dim == dim
        result = evaluator.compute_hubness(k=5)
        print(f"[ok] from_lancedb roundtrip (cosine index): n_vectors={n}, {result}")

        # A non-cosine index should warn, same contract as from_qdrant.
        l2_uri = str(Path(d) / "l2_db")
        db2 = lancedb.connect(l2_uri)
        tbl2 = db2.create_table("vectors", table)
        tbl2.create_index(metric="l2", vector_column_name="vector")

        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vh.VecHealthEvaluator.from_lancedb(l2_uri, "vectors", "vector")
        assert any("euclidean" in str(w.message) for w in caught), (
            f"expected a euclidean-distance warning, got: {[str(w.message) for w in caught]}"
        )
        print("[ok] from_lancedb warns on non-cosine index")

        # No index at all: a normal, silent case (no warning).
        no_index_uri = str(Path(d) / "no_index_db")
        db3 = lancedb.connect(no_index_uri)
        db3.create_table("vectors", table)
        with warnings.catch_warnings(record=True) as caught_none:
            warnings.simplefilter("always")
            evaluator3 = vh.VecHealthEvaluator.from_lancedb(no_index_uri, "vectors", "vector")
        assert caught_none == []
        assert evaluator3.n_vectors == n
        print("[ok] from_lancedb with no vector index: no warning, reads fine")


def test_from_lancedb_missing_table_fails_fast():
    import lancedb

    with tempfile.TemporaryDirectory() as d:
        uri = str(Path(d) / "empty_db")
        lancedb.connect(uri)  # creates the database directory, no tables in it
        start = time.monotonic()
        expect_connector_error(
            "from_lancedb missing table",
            lambda: vh.VecHealthEvaluator.from_lancedb(uri, "does-not-exist", "vector"),
        )
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"from_lancedb took {elapsed:.1f}s to fail — not failing fast"


def test_weaviate_parse_http_url():
    from vechealth.connectors.weaviate import _parse_http_url

    assert _parse_http_url("localhost:8080") == ("localhost", 8080, False)
    assert _parse_http_url("http://myhost:9000") == ("myhost", 9000, False)
    assert _parse_http_url("https://my.cloud.weaviate.io") == ("my.cloud.weaviate.io", 443, True)
    print("[ok] weaviate _parse_http_url")


def test_weaviate_distance_metric_resolution():
    from types import SimpleNamespace

    from vechealth.connectors.weaviate import _distance_metric_for

    def index_cfg(metric_value):
        return SimpleNamespace(distance_metric=SimpleNamespace(value=metric_value))

    # Legacy / single default vector (no named-vector config at all).
    legacy_config = SimpleNamespace(vector_config=None, vector_index_config=index_cfg("cosine"))
    assert _distance_metric_for(legacy_config, None) == (None, "cosine")

    # Exactly one named vector: auto-resolved without vector_name=.
    single_named_config = SimpleNamespace(
        vector_config={"text": SimpleNamespace(vector_index_config=index_cfg("l2-squared"))},
        vector_index_config=None,
    )
    assert _distance_metric_for(single_named_config, None) == ("text", "euclidean")

    # Multiple named vectors: requires an explicit vector_name=.
    multi_named_config = SimpleNamespace(
        vector_config={
            "text": SimpleNamespace(vector_index_config=index_cfg("dot")),
            "image": SimpleNamespace(vector_index_config=index_cfg("cosine")),
        },
        vector_index_config=None,
    )
    expect_connector_error(
        "weaviate multi-named-vector collection without vector_name=",
        lambda: _distance_metric_for(multi_named_config, None),
    )
    assert _distance_metric_for(multi_named_config, "image") == ("image", "cosine")
    print("[ok] weaviate _distance_metric_for")


def test_from_weaviate_unreachable_fails_fast():
    start = time.monotonic()
    expect_connector_error(
        "from_weaviate unreachable host",
        lambda: vh.VecHealthEvaluator.from_weaviate("http://127.0.0.1:1", "DoesNotExist"),
    )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"from_weaviate took {elapsed:.1f}s to fail — not failing fast"


def test_from_chroma_roundtrip():
    import chromadb

    rng = np.random.default_rng(4)
    n, dim = 250, 12

    with tempfile.TemporaryDirectory() as d:
        client = chromadb.PersistentClient(path=d)
        coll = client.create_collection("items", metadata={"hnsw:space": "cosine"})
        coll.add(
            ids=[str(i) for i in range(n)],
            embeddings=rng.standard_normal((n, dim)).astype(np.float32).tolist(),
        )

        evaluator = vh.VecHealthEvaluator.from_chroma("items", path=d, page_size=64)
        assert evaluator.n_vectors == n
        assert evaluator.dim == dim
        result = evaluator.compute_hubness(k=5)
        print(f"[ok] from_chroma roundtrip (cosine): n_vectors={n}, {result}")

        # Chroma defaults an unspecified space to "l2" — should warn.
        coll2 = client.create_collection("items_default")
        coll2.add(
            ids=[str(i) for i in range(30)],
            embeddings=rng.standard_normal((30, 4)).astype(np.float32).tolist(),
        )
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vh.VecHealthEvaluator.from_chroma("items_default", path=d)
        assert any("euclidean" in str(w.message) for w in caught), (
            f"expected a euclidean-distance warning, got: {[str(w.message) for w in caught]}"
        )
        print("[ok] from_chroma warns on default (l2) space")


def test_from_chroma_missing_collection_fails_fast():
    with tempfile.TemporaryDirectory() as d:
        start = time.monotonic()
        expect_connector_error(
            "from_chroma missing collection",
            lambda: vh.VecHealthEvaluator.from_chroma("does-not-exist", path=d),
        )
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"from_chroma took {elapsed:.1f}s to fail — not failing fast"


def test_from_chroma_requires_exactly_one_of_path_or_host():
    expect_connector_error(
        "from_chroma with neither path nor host",
        lambda: vh.VecHealthEvaluator.from_chroma("items"),
    )
    expect_connector_error(
        "from_chroma with both path and host",
        lambda: vh.VecHealthEvaluator.from_chroma("items", path="/tmp", host="localhost"),
    )
    print("[ok] from_chroma requires exactly one of path=/host=")


def test_from_milvus_roundtrip():
    from pymilvus import DataType, MilvusClient

    rng = np.random.default_rng(5)
    n, dim = 320, 6

    with tempfile.TemporaryDirectory() as d:
        uri = str(Path(d) / "milvus_demo.db")
        client = MilvusClient(uri=uri)
        schema = client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection("items", schema=schema, index_params=index_params)
        client.insert(
            collection_name="items",
            data=[
                {"id": i, "vector": rng.standard_normal(dim).astype(np.float32).tolist()}
                for i in range(n)
            ],
        )
        client.close()

        evaluator = vh.VecHealthEvaluator.from_milvus(uri, "items", batch_size=50)
        assert evaluator.n_vectors == n
        assert evaluator.dim == dim
        result = evaluator.compute_hubness(k=5)
        print(f"[ok] from_milvus roundtrip (cosine index): n_vectors={n}, {result}")

        # No index at all: a normal, silent case (no warning), same as LanceDB.
        uri2 = str(Path(d) / "milvus_no_index.db")
        client2 = MilvusClient(uri=uri2)
        schema2 = client2.create_schema(auto_id=False)
        schema2.add_field("id", DataType.INT64, is_primary=True)
        schema2.add_field("vector", DataType.FLOAT_VECTOR, dim=4)
        client2.create_collection("items", schema=schema2)
        client2.insert(
            collection_name="items",
            data=[
                {"id": i, "vector": rng.standard_normal(4).astype(np.float32).tolist()}
                for i in range(10)
            ],
        )
        client2.close()
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            evaluator2 = vh.VecHealthEvaluator.from_milvus(uri2, "items")
        assert caught == []
        assert evaluator2.n_vectors == 10
        print("[ok] from_milvus with no vector index: no warning, reads fine")

        # Multiple dense vector fields: requires an explicit vector_field=.
        uri3 = str(Path(d) / "milvus_multi.db")
        client3 = MilvusClient(uri=uri3)
        schema3 = client3.create_schema(auto_id=False)
        schema3.add_field("id", DataType.INT64, is_primary=True)
        schema3.add_field("vec_a", DataType.FLOAT_VECTOR, dim=3)
        schema3.add_field("vec_b", DataType.FLOAT_VECTOR, dim=5)
        client3.create_collection("multi", schema=schema3)
        client3.insert(
            collection_name="multi",
            data=[{"id": 0, "vec_a": [0.1, 0.2, 0.3], "vec_b": [0.1] * 5}],
        )
        client3.close()
        expect_connector_error(
            "from_milvus multiple dense vector fields without vector_field=",
            lambda: vh.VecHealthEvaluator.from_milvus(uri3, "multi"),
        )
        evaluator3 = vh.VecHealthEvaluator.from_milvus(uri3, "multi", vector_field="vec_b")
        assert evaluator3.dim == 5
        print("[ok] from_milvus resolves multiple dense vector fields via vector_field=")


def test_from_milvus_missing_collection_fails_fast():
    from pymilvus import MilvusClient

    with tempfile.TemporaryDirectory() as d:
        uri = str(Path(d) / "empty.db")
        MilvusClient(uri=uri).close()
        start = time.monotonic()
        expect_connector_error(
            "from_milvus missing collection",
            lambda: vh.VecHealthEvaluator.from_milvus(uri, "does-not-exist"),
        )
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"from_milvus took {elapsed:.1f}s to fail — not failing fast"


def test_from_pinecone_bad_credentials_fails_fast():
    start = time.monotonic()
    expect_connector_error(
        "from_pinecone invalid api key",
        lambda: vh.VecHealthEvaluator.from_pinecone("fake-api-key", "does-not-exist-index"),
    )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"from_pinecone took {elapsed:.1f}s to fail — not failing fast"


def test_optional_connector_sdks_are_lazily_imported():
    """`import vechealth` must not require chromadb/pymilvus/pinecone (or
    weaviate-client) to be installed. Simulates the SDKs being absent via a blocking
    meta-path finder rather than actually uninstalling them, then checks
    that each `from_*` still fails with a typed `ConnectorError` pointing at
    the right `pip install`, not an unrelated ImportError/traceback.
    """
    import sys

    class _BlockOptionalConnectorSdks:
        def find_spec(self, name, path, target=None):
            if name in ("chromadb", "pymilvus", "pinecone"):
                raise ImportError(f"blocked for this test: {name}")
            return None

    # Earlier tests in this same process already imported chromadb/pymilvus/
    # pinecone, so `import chromadb` etc. inside `fetch_all` would just hit
    # `sys.modules`'s cache and never consult the meta-path finder at all.
    # Evict every module whose top-level package is one of the three so the
    # blocker actually gets exercised, then restore the originals afterward
    # so later tests (or a second call to this one) aren't left broken.
    blocked_prefixes = ("chromadb", "pymilvus", "pinecone")
    saved_modules = {
        name: mod
        for name, mod in sys.modules.items()
        if name.partition(".")[0] in blocked_prefixes
    }
    for name in saved_modules:
        del sys.modules[name]

    sys.meta_path.insert(0, _BlockOptionalConnectorSdks())
    try:
        for label, fn in [
            ("from_chroma", lambda: vh.VecHealthEvaluator.from_chroma("x", path="/tmp")),
            ("from_milvus", lambda: vh.VecHealthEvaluator.from_milvus("x", "y")),
            ("from_pinecone", lambda: vh.VecHealthEvaluator.from_pinecone("x", "y")),
        ]:
            expect_connector_error(f"{label} without its SDK installed", fn)
    finally:
        sys.meta_path.pop(0)
        sys.modules.update(saved_modules)
    print("[ok] from_chroma/from_milvus/from_pinecone lazily import their SDKs")


if __name__ == "__main__":
    test_from_local_npy()
    test_from_local_csv()
    test_from_local_parquet()
    test_from_local_missing_file_raises_connector_error()
    test_from_qdrant_unreachable_fails_fast()
    test_from_pgvector_unreachable_fails_fast()
    test_from_lancedb_roundtrip()
    test_from_lancedb_missing_table_fails_fast()
    test_weaviate_parse_http_url()
    test_weaviate_distance_metric_resolution()
    test_from_weaviate_unreachable_fails_fast()
    test_from_chroma_roundtrip()
    test_from_chroma_missing_collection_fails_fast()
    test_from_chroma_requires_exactly_one_of_path_or_host()
    test_from_milvus_roundtrip()
    test_from_milvus_missing_collection_fails_fast()
    test_from_pinecone_bad_credentials_fails_fast()
    test_optional_connector_sdks_are_lazily_imported()
    print("\nAll connector smoke tests passed.")
