"""
tests/test_cli.py

The `vechealth` command-line tool: defaults that follow the preprint, and the
reading note printed after the numbers.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vechealth.cli import build_parser, main


@pytest.fixture
def vectors_file(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "vectors.npy"
    np.save(path, rng.standard_normal((200, 16)).astype(np.float32))
    return path


def test_analyze_uses_the_preprint_outlier_threshold_by_default(vectors_file, tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["analyze", str(vectors_file), "--output", str(out)]) == 0
    config = json.loads(out.read_text())["config"]
    assert config["outlier_distance_threshold"] == pytest.approx(1.2)
    assert "Reading these numbers" in capsys.readouterr().out


def test_analyze_chooses_the_batch_size_automatically_unless_told_otherwise(vectors_file, tmp_path):
    import vechealth as vh

    auto_out, chosen_out = tmp_path / "auto.json", tmp_path / "chosen.json"
    assert main(["analyze", str(vectors_file), "--output", str(auto_out)]) == 0
    assert main(["analyze", str(vectors_file), "--output", str(chosen_out), "--batch-size", "64"]) == 0
    assert json.loads(auto_out.read_text())["config"]["batch_size"] == vh.default_batch_size(200)
    assert json.loads(chosen_out.read_text())["config"]["batch_size"] == 64


def test_analyze_adaptive_threshold_is_an_explicit_opt_in(vectors_file, tmp_path):
    out = tmp_path / "report.json"
    code = main(["analyze", str(vectors_file), "--output", str(out),
                 "--outlier-distance-threshold", "adaptive"])
    assert code == 0
    assert json.loads(out.read_text())["config"]["outlier_distance_threshold"] is None


def test_threshold_option_rejects_garbage():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyze", "v.npy", "--outlier-distance-threshold", "high"])


def test_compare_prints_the_reading_note(vectors_file, tmp_path, capsys):
    baseline, current = tmp_path / "a.json", tmp_path / "b.json"
    assert main(["analyze", str(vectors_file), "--output", str(baseline)]) == 0
    assert main(["analyze", str(vectors_file), "--output", str(current)]) == 0
    capsys.readouterr()
    assert main(["compare", str(baseline), str(current)]) == 0
    out = capsys.readouterr().out
    assert "Reading these numbers" in out and "Legend:" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
