"""Manual smoke test for `Report`/`Comparison` — serialization
(`to_dict`/`to_json`) and baseline tracking (`compute_report`, `save`,
`load`, `compare`). Same style as `local_connectors_test.py`: run directly
(`python python/tests/test_report.py`), not via pytest (no pytest in this
project's venv).
"""

import json
import tempfile
from pathlib import Path

import numpy as np

import vechealth as vh


def test_metric_result_to_dict_and_to_json():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((300, 16)).astype(np.float32)
    evaluator = vh.VecHealthEvaluator(vectors)

    hubness = evaluator.compute_hubness(k=10)
    as_dict = hubness.to_dict()
    assert as_dict["hubness_skewness"] == hubness.hubness_skewness
    assert as_dict["max_occurrences"] == hubness.max_occurrences

    as_json = json.loads(hubness.to_json())
    assert as_json == as_dict
    print(f"[ok] HubnessResult.to_dict()/to_json(): {as_dict}")


def test_all_metrics_result_to_dict_is_nested():
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((300, 16)).astype(np.float32)
    evaluator = vh.VecHealthEvaluator(vectors)

    result = evaluator.compute_all(k=10)
    as_dict = result.to_dict()
    assert as_dict["hubness"]["hubness_skewness"] == result.hubness.hubness_skewness
    assert as_dict["qmas"] is None
    print("[ok] AllMetricsResult.to_dict() nests every metric")


def test_compute_report_has_dataset_metadata():
    rng = np.random.default_rng(2)
    vectors = rng.standard_normal((500, 32)).astype(np.float32)
    evaluator = vh.VecHealthEvaluator(vectors)

    report = evaluator.compute_report(k=10, tags={"embedding_model": "test-model"})
    assert report.n_vectors == 500
    assert report.dim == 32
    assert report.schema_version == 1
    assert report.tags == {"embedding_model": "test-model"}
    assert isinstance(report.metrics, vh.AllMetricsResult)
    print(f"[ok] compute_report: {report}")


def test_report_save_load_round_trip():
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((200, 8)).astype(np.float32)
    evaluator = vh.VecHealthEvaluator(vectors)
    report = evaluator.compute_report(k=10)

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "report.json")
        report.save(path)
        loaded = vh.Report.load(path)

    assert loaded.n_vectors == report.n_vectors
    assert loaded.content_hash == report.content_hash
    assert loaded.metrics.hubness.hubness_skewness == report.metrics.hubness.hubness_skewness
    print(f"[ok] Report.save()/load() round trip: content_hash={loaded.content_hash}")


def test_report_load_missing_file_raises_report_error():
    try:
        vh.Report.load("/nonexistent/report.json")
    except vh.ReportError as e:
        print(f"[ok] Report.load missing file: raised ReportError as expected ({e})")
    else:
        raise AssertionError("expected ReportError, nothing was raised")


def test_flatten_matches_nested_values():
    rng = np.random.default_rng(4)
    vectors = rng.standard_normal((300, 16)).astype(np.float32)
    evaluator = vh.VecHealthEvaluator(vectors)
    report = evaluator.compute_report(k=10)

    flat = report.flatten()
    assert flat["hubness.hubness_skewness"] == report.metrics.hubness.hubness_skewness
    assert flat["dispersion.mean_1nn_distance"] == report.metrics.dispersion.mean_1nn_distance
    assert not any(key.startswith("qmas.") for key in flat)
    print(f"[ok] Report.flatten(): {len(flat)} scalar metrics")


def test_compare_detects_duplicate_regression():
    rng = np.random.default_rng(5)
    baseline_vectors = rng.standard_normal((500, 16)).astype(np.float32)
    evaluator = vh.VecHealthEvaluator(baseline_vectors)
    baseline = evaluator.compute_report(k=10, duplicate_epsilon=0.05)

    # Collapses a block of points onto a single vector — a "near-duplicate
    # flood" regression (see README's pathology table), robust to the
    # evaluator's internal L2-normalization (unlike e.g. a uniform shift,
    # which just moves everything to a new but still-mutually-distinct
    # direction on the unit sphere).
    current_vectors = baseline_vectors.copy()
    current_vectors[:50] = current_vectors[0]
    evaluator_current = vh.VecHealthEvaluator(current_vectors)
    current = evaluator_current.compute_report(k=10, duplicate_epsilon=0.05)

    comparison = current.compare(baseline)
    assert comparison.warnings == []

    ndds_delta = comparison.deltas["duplicates.ndds_fraction"]
    assert ndds_delta.current > ndds_delta.baseline
    assert ndds_delta.delta > 0
    print(f"[ok] compare(): duplicates.ndds_fraction delta={ndds_delta.delta:.4f}")


def test_compare_warns_on_k_mismatch():
    rng = np.random.default_rng(6)
    vectors = rng.standard_normal((200, 8)).astype(np.float32)
    evaluator = vh.VecHealthEvaluator(vectors)
    baseline = evaluator.compute_report(k=5)
    current = evaluator.compute_report(k=10)

    comparison = current.compare(baseline)
    assert any("k differs" in w for w in comparison.warnings)
    print(f"[ok] compare() warns on k mismatch: {comparison.warnings}")


if __name__ == "__main__":
    test_metric_result_to_dict_and_to_json()
    test_all_metrics_result_to_dict_is_nested()
    test_compute_report_has_dataset_metadata()
    test_report_save_load_round_trip()
    test_report_load_missing_file_raises_report_error()
    test_flatten_matches_nested_values()
    test_compare_detects_duplicate_regression()
    test_compare_warns_on_k_mismatch()
    print("\nAll report smoke tests passed.")
