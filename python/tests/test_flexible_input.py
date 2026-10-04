"""Manual smoke test for flexible array-like input on `VecHealthEvaluator`
and every `compute_*` method that takes `queries` — the automatic `float64`
to `float32` conversion. Same style as
`local_connectors_test.py`: run directly
(`python python/tests/test_flexible_input.py`), not via pytest (no pytest
in this project's venv).

Note: `.npy`/`.csv`/`.parquet` file loading with automatic float64->float32
conversion was already covered by `VecHealthEvaluator.from_local` (see
`connectors/src/local.rs`) before this change — this test is specifically
about the direct in-memory constructor and the `queries` parameters, which
previously required the caller to pre-cast to `float32` by hand.
"""

import numpy as np

import vechealth as vh


def test_constructor_accepts_float64():
    rng = np.random.default_rng(0)
    vectors64 = rng.standard_normal((100, 8)).astype(np.float64)
    evaluator = vh.VecHealthEvaluator(vectors64)
    assert evaluator.n_vectors == 100
    assert evaluator.dim == 8
    print(f"[ok] constructor accepts float64: n_vectors={evaluator.n_vectors}")


def test_constructor_float64_and_float32_give_same_result():
    rng = np.random.default_rng(1)
    vectors64 = rng.standard_normal((200, 16)).astype(np.float64)
    vectors32 = vectors64.astype(np.float32)

    result64 = vh.VecHealthEvaluator(vectors64).compute_hubness(k=10)
    result32 = vh.VecHealthEvaluator(vectors32).compute_hubness(k=10)

    assert abs(result64.hubness_skewness - result32.hubness_skewness) < 1e-4
    print("[ok] float64 and float32 constructors agree on results")


def test_constructor_accepts_python_nested_list():
    vectors = [[float(i), float(i) * 2.0, float(i) * -1.0] for i in range(30)]
    evaluator = vh.VecHealthEvaluator(vectors)
    assert evaluator.n_vectors == 30
    assert evaluator.dim == 3
    print(f"[ok] constructor accepts a plain Python list: n_vectors={evaluator.n_vectors}")


def test_constructor_accepts_int_array():
    rng = np.random.default_rng(2)
    vectors_int = (rng.standard_normal((50, 4)) * 100).astype(np.int32)
    evaluator = vh.VecHealthEvaluator(vectors_int)
    assert evaluator.n_vectors == 50
    assert evaluator.dim == 4
    print(f"[ok] constructor accepts int32: n_vectors={evaluator.n_vectors}")


def test_constructor_already_float32_is_unchanged_fast_path():
    # Same object identity check isn't meaningful across the Rust boundary,
    # but this at minimum guards against the float32 path regressing.
    rng = np.random.default_rng(3)
    vectors32 = rng.standard_normal((40, 4)).astype(np.float32)
    evaluator = vh.VecHealthEvaluator(vectors32)
    assert evaluator.n_vectors == 40
    print("[ok] already-float32 input still works (fast path)")


def test_constructor_rejects_non_numeric_input():
    try:
        vh.VecHealthEvaluator([["a", "b"], ["c", "d"]])
    except ValueError as e:
        print(f"[ok] non-numeric input rejected with ValueError: {e}")
    else:
        raise AssertionError("expected ValueError, nothing was raised")


def test_queries_accept_float64_on_every_entry_point():
    rng = np.random.default_rng(4)
    docs = rng.standard_normal((150, 16)).astype(np.float32)
    queries64 = rng.standard_normal((10, 16)).astype(np.float64)
    evaluator = vh.VecHealthEvaluator(docs)

    qmas = evaluator.compute_qmas(queries64, k=5)
    assert qmas.orphans_fraction >= 0.0

    all_metrics = evaluator.compute_all(queries=queries64, k=5, k_intrinsic_dim=5)
    assert all_metrics.qmas is not None

    report = evaluator.compute_report(queries=queries64, k=5, k_intrinsic_dim=5)
    assert report.metrics.qmas is not None

    print("[ok] compute_qmas/compute_all/compute_report all accept float64 queries")


if __name__ == "__main__":
    test_constructor_accepts_float64()
    test_constructor_float64_and_float32_give_same_result()
    test_constructor_accepts_python_nested_list()
    test_constructor_accepts_int_array()
    test_constructor_already_float32_is_unchanged_fast_path()
    test_constructor_rejects_non_numeric_input()
    test_queries_accept_float64_on_every_entry_point()
    print("\nAll flexible-input smoke tests passed.")
