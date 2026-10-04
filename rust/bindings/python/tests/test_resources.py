"""
tests/test_resources.py

Memory use of the exact k-NN search: the automatic batch size, the estimate,
and the warning raised before the work starts.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vechealth as vh

THRESHOLD_ENV = "VECHEALTH_MEMORY_WARNING_GB"


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((300, 24)).astype(np.float32)
    queries = rng.standard_normal((40, 24)).astype(np.float32)
    return vectors, queries


def test_the_automatic_batch_shrinks_as_the_collection_grows():
    assert vh.default_batch_size(20_000, threads=8) > vh.default_batch_size(200_000, threads=8)
    assert vh.default_batch_size(200_000, threads=4) > vh.default_batch_size(200_000, threads=44)
    assert 16 <= vh.default_batch_size(5_000_000, threads=64) <= 2048


def test_the_automatic_batch_never_exceeds_the_collection():
    assert vh.default_batch_size(15, threads=4) == 15
    assert vh.default_batch_size(1, threads=1) == 1


def test_the_estimate_matches_a_measured_run():
    """40,000 x 1024 on 44 threads at batch_size=256 peaked at 2.59 GB."""
    estimate = vh.estimate_peak_memory_bytes(40_000, 1024, batch_size=256, threads=44)
    assert estimate == pytest.approx(2.59e9, rel=0.12)


def test_the_estimate_defaults_to_the_automatic_batch():
    auto = vh.default_batch_size(100_000, threads=16)
    assert vh.estimate_peak_memory_bytes(100_000, 1024, threads=16) == (
        vh.estimate_peak_memory_bytes(100_000, 1024, batch_size=auto, threads=16)
    )


def test_results_do_not_depend_on_the_batch_size(data):
    vectors, queries = data
    flat = {}
    for label, batch in (("auto", None), ("tiny", 7), ("huge", 4096)):
        report = vh.VecHealthEvaluator(vectors).compute_report(queries=queries, batch_size=batch)
        flat[label] = report.flatten()
    assert flat["tiny"] == pytest.approx(flat["auto"], rel=1e-6)
    assert flat["huge"] == pytest.approx(flat["auto"], rel=1e-6)


def test_the_report_records_the_batch_size_that_was_used(data):
    vectors, _ = data
    evaluator = vh.VecHealthEvaluator(vectors)
    assert evaluator.compute_report().to_dict()["config"]["batch_size"] == vh.default_batch_size(300)
    assert evaluator.compute_report(batch_size=64).to_dict()["config"]["batch_size"] == 64


def test_a_batch_size_of_zero_is_rejected(data):
    vectors, _ = data
    evaluator = vh.VecHealthEvaluator(vectors)
    with pytest.raises(ValueError, match="at least 1"):
        evaluator.compute_hubness(batch_size=0)
    with pytest.raises(ValueError, match="at least 1"):
        evaluator.get_knn(5, batch_size=0)


def test_get_knn_works_without_a_batch_size(data):
    vectors, _ = data
    distances, indices = vh.VecHealthEvaluator(vectors).get_knn(5)
    assert distances.shape == (300, 5) and indices.shape == (300, 5)


def test_a_small_collection_raises_no_warning(data):
    vectors, queries = data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vh.VecHealthEvaluator(vectors).compute_all(queries=queries)


def test_the_warning_is_raised_before_the_work_starts(data, monkeypatch):
    vectors, _ = data
    monkeypatch.setenv(THRESHOLD_ENV, "0")  # every estimate exceeds zero
    evaluator = vh.VecHealthEvaluator(vectors)
    with pytest.warns(vh.MemoryUsageWarning, match="automatic batch_size="):
        evaluator.compute_all()
    with pytest.warns(vh.MemoryUsageWarning, match=r"batch_size=64 on \d+ threads"):
        evaluator.compute_hubness(batch_size=64)


def test_the_warning_can_be_turned_into_an_error(data, monkeypatch):
    vectors, _ = data
    monkeypatch.setenv(THRESHOLD_ENV, "0")
    evaluator = vh.VecHealthEvaluator(vectors)
    with warnings.catch_warnings():
        warnings.simplefilter("error", vh.MemoryUsageWarning)
        with pytest.raises(vh.MemoryUsageWarning):
            evaluator.compute_hubness()


def test_a_raised_limit_silences_the_warning(data, monkeypatch):
    vectors, _ = data
    monkeypatch.setenv(THRESHOLD_ENV, "1000")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vh.VecHealthEvaluator(vectors).compute_all()


def test_the_warning_category_can_be_filtered(data, monkeypatch):
    vectors, _ = data
    monkeypatch.setenv(THRESHOLD_ENV, "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=vh.MemoryUsageWarning)
        vh.VecHealthEvaluator(vectors).compute_hubness()
    assert caught == []


def test_the_warning_is_a_user_warning():
    assert issubclass(vh.MemoryUsageWarning, UserWarning)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
