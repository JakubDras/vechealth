"""
tests/test_hub_inspector.py

Second version — tests hub_inspector.py, which builds on the REAL
evaluator.identify_hubs()/get_knn(), not on a temporary numpy fallback
(see the hub_inspector.py docstring). The stub (`FakeEvaluator`) mimics
exactly the interface verified by a smoke test against the real environment
(2026-09-16): `identify_hubs(k, top_n, batch_size)` returns objects with
`.index`/`.occurrence_count`, `get_knn(k, batch_size)` returns
`(distances, indices)` as numpy arrays of shape (n, k).
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from vechealth.hub_inspector import (
    inspect_hubs,
    hub_inspector_as_example_provider,
    compare_hub_snapshots,
    HubEntry,
    HubInspectionReport,
)


@dataclass
class _FakeRustHubEntry:
    index: int
    occurrence_count: int


class _FakeEvaluator:
    """A stub mimicking the real VecHealthEvaluator interface, verified by a
    smoke test: identify_hubs()/get_knn()."""

    def __init__(self, hubs_by_k=None, knn_by_k=None):
        self._hubs_by_k = hubs_by_k or {}
        self._knn_by_k = knn_by_k or {}

    def identify_hubs(self, k, top_n, batch_size=2048):
        return self._hubs_by_k[k][:top_n]

    def get_knn(self, k, batch_size=2048):
        return self._knn_by_k[k]


# The A/B/C triangle verified end-to-end against the real environment
# (smoke test 2026-09-16): occurrences = A:1, B:2, C:0.
_TRIANGLE_EVALUATOR = _FakeEvaluator(
    hubs_by_k={1: [_FakeRustHubEntry(1, 2), _FakeRustHubEntry(0, 1), _FakeRustHubEntry(2, 0)]}
)


def test_maps_raw_index_to_id_array():
    report = inspect_hubs(_TRIANGLE_EVALUATOR, k=1, top_n=3, id_array=["A", "B", "C"])
    assert report.hubs[0].id == "B" and report.hubs[0].occurrence_count == 2
    assert report.hubs[1].id == "A" and report.hubs[1].occurrence_count == 1
    assert report.hubs[2].id == "C" and report.hubs[2].occurrence_count == 0


def test_id_defaults_to_index_without_id_array():
    report = inspect_hubs(_TRIANGLE_EVALUATOR, k=1, top_n=3)
    for h in report.hubs:
        assert h.id == h.index


def test_no_category_labels_gives_none_dispersion():
    report = inspect_hubs(_TRIANGLE_EVALUATOR, k=1, top_n=3)
    assert report.category_dispersion is None
    assert report.likely_pathology is None


def test_high_dispersion_correctly_flags_likely_pathology():
    """A hand-verified case (see the project conversation): a hub cited evenly
    by 4 different categories -> dispersion=1.0."""
    indices_k1 = np.array([[1], [0], [0], [0], [0], [6], [5], [5]])
    distances_k1 = np.zeros((8, 1), dtype=np.float32)
    evaluator = _FakeEvaluator(
        hubs_by_k={1: [_FakeRustHubEntry(0, 4)]},
        knn_by_k={1: (distances_k1, indices_k1)},
    )
    categories = ["A", "A", "B", "C", "D", "B", "C", "D"]

    report = inspect_hubs(evaluator, k=1, top_n=1, category_labels=categories)
    assert abs(report.category_dispersion - 1.0) < 1e-6
    assert report.likely_pathology is True


def test_low_dispersion_correctly_does_not_flag():
    indices_k1 = np.array([[0], [0], [0], [0]])
    distances_k1 = np.zeros((4, 1), dtype=np.float32)
    evaluator = _FakeEvaluator(
        hubs_by_k={1: [_FakeRustHubEntry(0, 3)]},
        knn_by_k={1: (distances_k1, indices_k1)},
    )
    categories = ["A", "A", "A", "A"]

    report = inspect_hubs(evaluator, k=1, top_n=1, category_labels=categories)
    assert report.category_dispersion == 0.0
    assert report.likely_pathology is False


def test_example_provider_returns_none_for_unrelated_metrics():
    provider = hub_inspector_as_example_provider(_TRIANGLE_EVALUATOR, id_array=["A", "B", "C"], k=1)
    assert provider("snc_score") is None
    assert provider("anisotropy_mean_norm") is None


def test_example_provider_returns_text_for_hubness_metrics():
    provider = hub_inspector_as_example_provider(_TRIANGLE_EVALUATOR, id_array=["A", "B", "C"], k=1)
    result = provider("hubness_max")
    assert result is not None and "B" in result


def test_example_provider_integrates_with_report_layer():
    from vechealth.interpretation.report import build_metric_report

    provider = hub_inspector_as_example_provider(_TRIANGLE_EVALUATOR, id_array=["A", "B", "C"], k=1)
    without = build_metric_report("hubness_max", diagnostic_strength=0.8)
    with_provider = build_metric_report("hubness_max", diagnostic_strength=0.8, example_provider=provider)
    assert "Hub Inspector" in without.example_note
    assert "Hub Inspector" not in with_provider.example_note
    assert "B" in with_provider.example_note


def test_compare_snapshots_identifies_new_persisted_resolved():
    prev = inspect_hubs(
        _FakeEvaluator(hubs_by_k={1: [_FakeRustHubEntry(0, 50), _FakeRustHubEntry(1, 30)]}),
        k=1, top_n=2, id_array=["doc_A", "doc_B"],
    )
    curr = inspect_hubs(
        _FakeEvaluator(hubs_by_k={1: [_FakeRustHubEntry(0, 55), _FakeRustHubEntry(2, 40)]}),
        k=1, top_n=2, id_array=["doc_A", "doc_B", "doc_C"],
    )
    result = compare_hub_snapshots(prev, curr)
    assert {h.id for h in result.new_hubs} == {"doc_C"}
    assert {h.id for h in result.persisted_hubs} == {"doc_A"}
    assert {h.id for h in result.resolved_hubs} == {"doc_B"}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))