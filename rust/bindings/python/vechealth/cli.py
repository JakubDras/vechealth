"""
vechealth.cli — the `vechealth` command-line entry point.

A thin `argparse` wrapper, for quick ad-hoc checks and for hooking into CI/CD
pipelines without writing Python glue, over the already-implemented,
already-tested Python API (`VecHealthEvaluator.from_local`, `compute_report`,
`Report.save/load/compare`, `interpretation.assess_comparison`) — no new
metric/interpretation logic lives here, only plumbing suitable for a terminal
or CI step.

Two subcommands:
  `vechealth analyze vectors.parquet --output report.json`
      Computes every metric for a local vectors file, prints a summary,
      optionally saves the full `Report` as JSON.
  `vechealth compare baseline.json current.json --fail-on-significant`
      Loads two saved reports, prints the delta per metric annotated with
      the noise-filter verdict (`interpretation.assess_comparison`), and
      exits non-zero when `--fail-on-significant` is given and at least one
      metric moved beyond its category's measured noise floor — the
      "hubness rose by 30%, is that real?" CI-gating scenario.

Deliberately local-file-only for `analyze` in this first version — every
`from_*` connector (Qdrant, pgvector, ...) takes different keyword
arguments, and mapping all of them onto CLI flags is a separate, larger
design task, not needed for a first version. Someone who
wants a live connector today can do it in three lines of Python; scripting
a CI check against a live database is exactly the case where a real Python
script also composes better than a growing pile of CLI flags anyway.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np

from . import ConnectorError, Report, ReportError, VecHealthError, VecHealthEvaluator
from .interpretation import SHORT_READING_NOTE, SignificanceVerdict, assess_comparison

__all__ = ["main", "build_parser"]


def _parse_tags(pairs: list[str]) -> dict[str, str]:
    tags = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"error: --tag expects KEY=VALUE, got {pair!r}")
        tags[key] = value
    return tags


def _load_queries(path: str) -> np.ndarray:
    # Only .npy for now: unlike `analyze`'s SOURCE (which goes through the
    # Rust from_local loader, supporting .npy/.csv/.parquet with no extra
    # Python dependency), a *second*, separate array here would need its
    # own loader. Supporting .csv/.parquet too would mean either adding
    # pandas/pyarrow as a CLI dependency or duplicating from_local's parsing
    # in Python — out of scope for this first version. Documented, not
    # silently limited.
    suffix = Path(path).suffix
    if suffix != ".npy":
        raise SystemExit(
            f"error: --queries only supports .npy files today, got '{suffix}' "
            "(convert with numpy, or use the Python API directly for "
            "compute_report(queries=...) with a .csv/.parquet source)"
        )
    return np.load(path)


def _threshold(value: str) -> Optional[float]:
    """`--outlier-distance-threshold`: a chord distance, or `adaptive` to derive
    it from the data (3x the mean nearest-neighbour distance)."""
    if value.lower() == "adaptive":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number or 'adaptive', got {value!r}"
        ) from None


def _print_report_summary(report: Report) -> None:
    print(f"VecHealth report — {report.n_vectors} vectors, dim={report.dim}")
    print(f"  generated_at: {report.generated_at}")
    if report.tags:
        print(f"  tags: {dict(report.tags)}")
    print()
    flat = report.flatten()
    for key in sorted(flat):
        print(f"  {key:<35} {flat[key]:.6g}")
    print()
    print(textwrap.fill(SHORT_READING_NOTE, width=88))


def _cmd_analyze(args: argparse.Namespace) -> int:
    try:
        evaluator = VecHealthEvaluator.from_local(
            args.source,
            has_header=not args.no_header,
            columns=args.columns.split(",") if args.columns else None,
        )
    except (VecHealthError, ConnectorError) as exc:
        print(f"error: could not load '{args.source}': {exc}", file=sys.stderr)
        return 1

    queries = _load_queries(args.queries) if args.queries else None

    try:
        report = evaluator.compute_report(
            queries=queries,
            k=args.k,
            k_intrinsic_dim=args.k_intrinsic_dim,
            batch_size=args.batch_size,
            duplicate_epsilon=args.duplicate_epsilon,
            outlier_distance_threshold=args.outlier_distance_threshold,
            tags=_parse_tags(args.tag),
        )
    except VecHealthError as exc:
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        return 1

    _print_report_summary(report)

    if args.output:
        try:
            report.save(args.output)
        except ReportError as exc:
            print(f"error: could not write '{args.output}': {exc}", file=sys.stderr)
            return 1
        print(f"\nSaved full report to {args.output}")

    return 0


_VERDICT_MARKER = {
    SignificanceVerdict.SIGNIFICANT: "!!",
    SignificanceVerdict.LIKELY_NOISE: "..",
    SignificanceVerdict.UNASSESSED: " ?",
}


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        baseline = Report.load(args.baseline)
        current = Report.load(args.current)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    comparison = current.compare(baseline)
    assessments = assess_comparison(comparison, multiplier=args.multiplier)

    print(f"Comparing {args.current} against baseline {args.baseline}")
    print(f"  baseline generated_at: {comparison.baseline_generated_at}")
    print(f"  current  generated_at: {comparison.current_generated_at}")
    print()

    any_significant = False
    for key in sorted(comparison.deltas):
        delta = comparison.deltas[key]
        assessment = assessments[key]
        if assessment.verdict == SignificanceVerdict.SIGNIFICANT:
            any_significant = True
        pct = f"{delta.delta_pct:+.1f}%" if delta.delta_pct is not None else "n/a"
        marker = _VERDICT_MARKER[assessment.verdict]
        print(f"{marker} {key:<35} {pct:>9}  [{assessment.verdict.value}]")

    if comparison.warnings:
        print("\nWarnings:")
        for warning in comparison.warnings:
            print(f"  - {warning}")

    print(
        "\nLegend: '!!' = beyond this metric's measured noise floor "
        "(interpretation.assess_comparison), '..' = within it, ' ?' = no "
        "reliability category to judge against (see noise_filter.py)."
    )
    print()
    print(textwrap.fill(SHORT_READING_NOTE, width=88))

    if args.fail_on_significant and any_significant:
        print("\nFAIL: at least one metric moved beyond its noise floor (see '!!' above).")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vechealth",
        description="Geometric health diagnostics for embedding spaces and vector stores.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze", help="Compute every metric for a local vectors file."
    )
    analyze.add_argument("source", help="Path to a .npy, .csv, or .parquet vectors file.")
    analyze.add_argument("--output", "-o", help="Write the full report as JSON to this path.")
    analyze.add_argument(
        "--queries", help="Optional .npy queries file — enables the QMAS metric."
    )
    analyze.add_argument("--k", type=int, default=10, help="Neighborhood size (default: 10).")
    analyze.add_argument(
        "--k-intrinsic-dim", type=int, default=20, dest="k_intrinsic_dim",
        help="Neighborhood size for intrinsic dimensionality (default: 20).",
    )
    analyze.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size",
        help="Rows per block of the k-NN search (default: chosen automatically from the "
             "size of the problem so that the search fits in memory). Peak memory grows "
             "with it; raise it only on a machine with memory to spare.",
    )
    analyze.add_argument(
        "--duplicate-epsilon", type=float, default=0.05, dest="duplicate_epsilon",
        help="Distance threshold for near-duplicate detection (default: 0.05).",
    )
    analyze.add_argument(
        "--outlier-distance-threshold", type=_threshold, default=1.2,
        dest="outlier_distance_threshold", metavar="VALUE|adaptive",
        help="Outlier threshold, a chord distance on the unit sphere (default: 1.2, the "
             "value used for the preprint's results; absolute, so recalibrate it for "
             "other encoders or dimensionalities). 'adaptive' derives it from the data "
             "as 3x the mean nearest-neighbor distance, which on typical text "
             "embeddings lands close to the largest possible distance (2.0) and "
             "flags essentially nothing.",
    )
    analyze.add_argument(
        "--tag", action="append", default=[], metavar="KEY=VALUE",
        help="Attach a tag to the report (repeatable).",
    )
    analyze.add_argument(
        "--no-header", action="store_true", help="CSV source has no header row.",
    )
    analyze.add_argument(
        "--columns", help="Comma-separated column subset/order (.parquet source only).",
    )
    analyze.set_defaults(func=_cmd_analyze)

    compare = subparsers.add_parser(
        "compare", help="Compare two saved reports, with noise-vs-signal filtering."
    )
    compare.add_argument("baseline", help="Path to the baseline report JSON (from `analyze -o`).")
    compare.add_argument("current", help="Path to the current report JSON.")
    compare.add_argument(
        "--multiplier", type=float, default=1.0,
        help="Scales the noise-floor threshold per category (default: 1.0 — "
             "the raw value measured under subsampling, see interpretation/noise_filter.py).",
    )
    compare.add_argument(
        "--fail-on-significant", action="store_true", dest="fail_on_significant",
        help="Exit with status 1 if any metric moved beyond its noise floor — "
             "for CI gating.",
    )
    compare.set_defaults(func=_cmd_compare)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
