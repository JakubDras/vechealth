"""VecHealth — observability for embedding spaces and vector stores.

The heavy lifting (k-NN search, all metrics) runs in a compiled Rust
extension (`vechealth._core`); this package only re-exports its public,
typed API. Every ``compute_*`` method on :class:`VecHealthEvaluator`
releases the GIL while it runs, so it is safe to call from a
multi-threaded service (e.g. behind a web framework's thread pool).

Quickstart
----------
>>> import numpy as np
>>> import vechealth as vh
>>> vectors = np.random.randn(1000, 128).astype(np.float32)
>>> evaluator = vh.VecHealthEvaluator(vectors)
>>> hubness = evaluator.compute_hubness(k=10)
>>> hubness.hubness_skewness
0.42...

Or run every implemented metric at once:

>>> report = evaluator.compute_all()
>>> report.hubness.hubness_skewness
0.42...
"""

from vechealth._core import (
    AllMetricsResult as AllMetricsResult,
    AllVectorsDegenerateError as AllVectorsDegenerateError,
    AnisotropyResult as AnisotropyResult,
    Comparison as Comparison,
    ConnectorError as ConnectorError,
    DimensionMismatchError as DimensionMismatchError,
    DispersionResult as DispersionResult,
    DuplicatesResult as DuplicatesResult,
    EmptyInputError as EmptyInputError,
    HubnessResult as HubnessResult,
    IntrinsicDimResult as IntrinsicDimResult,
    KTooLargeError as KTooLargeError,
    KTooSmallError as KTooSmallError,
    MetricDelta as MetricDelta,
    OutliersResult as OutliersResult,
    QmasResult as QmasResult,
    Report as Report,
    ReportError as ReportError,
    SncResult as SncResult,
    VecHealthError as VecHealthError,
    VecHealthEvaluator as VecHealthEvaluator,
)

__all__ = [
    "VecHealthEvaluator",
    "AllMetricsResult",
    "HubnessResult",
    "DispersionResult",
    "AnisotropyResult",
    "OutliersResult",
    "DuplicatesResult",
    "IntrinsicDimResult",
    "QmasResult",
    "SncResult",
    "Report",
    "MetricDelta",
    "Comparison",
    "VecHealthError",
    "DimensionMismatchError",
    "KTooLargeError",
    "KTooSmallError",
    "EmptyInputError",
    "AllVectorsDegenerateError",
    "ConnectorError",
    "ReportError",
    "__version__",
]

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("vechealth")
except PackageNotFoundError:  # pragma: no cover - editable/dev checkout without metadata
    __version__ = "0.0.0+unknown"
