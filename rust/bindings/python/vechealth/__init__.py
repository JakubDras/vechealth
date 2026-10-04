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
    MemoryUsageWarning as MemoryUsageWarning,
    MetricDelta as MetricDelta,
    OutliersResult as OutliersResult,
    QmasResult as QmasResult,
    Report as Report,
    ReportError as ReportError,
    SncResult as SncResult,
    VecHealthError as VecHealthError,
    VecHealthEvaluator as VecHealthEvaluator,
    default_batch_size as default_batch_size,
    estimate_peak_memory_bytes as estimate_peak_memory_bytes,
)

from vechealth import _core
from vechealth.connectors.lancedb import from_lancedb as _from_lancedb
from vechealth.connectors.weaviate import from_weaviate as _from_weaviate
from vechealth.connectors.chroma import from_chroma as _from_chroma
from vechealth.connectors.milvus import from_milvus as _from_milvus
from vechealth.connectors.pinecone import from_pinecone as _from_pinecone


def compiled_features() -> list[str]:
    """Names of the optional native connectors compiled into this build of the
    extension module (a subset of ``"qdrant"``, ``"pgvector"``, ``"lancedb"``)."""
    native = getattr(_core, "compiled_features", None)
    if native is None:  # a build that predates the feature split shipped them all
        return ["qdrant", "pgvector", "lancedb"]
    return list(native())


# `from_local`/`from_qdrant`/`from_pgvector` are implemented in Rust, in
# `vechealth-connectors`. Weaviate/Chroma/Milvus/Pinecone have no maintained
# official Rust client, so those four are pure Python
# (`vechealth/connectors/*.py`) — thin wrappers over each vendor's official
# SDK. They are attached here, after the class already exists, so callers see
# one uniform contract
# (`VecHealthEvaluator.from_weaviate(...)`, etc.) regardless of which
# language implements a given connector.
#
# LanceDB has a native Rust connector too, but only builds compiled with the
# `lancedb` Cargo feature include it (it is the heaviest dependency tree, so
# the PyPI wheels leave it out); every other build serves `from_lancedb`
# through the official SDK, with the same contract.
if "lancedb" not in compiled_features():
    VecHealthEvaluator.from_lancedb = staticmethod(_from_lancedb)
VecHealthEvaluator.from_weaviate = staticmethod(_from_weaviate)
VecHealthEvaluator.from_chroma = staticmethod(_from_chroma)
VecHealthEvaluator.from_milvus = staticmethod(_from_milvus)
VecHealthEvaluator.from_pinecone = staticmethod(_from_pinecone)

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
    "MemoryUsageWarning",
    "default_batch_size",
    "estimate_peak_memory_bytes",
    "compiled_features",
    "__version__",
]

from . import interpretation
from . import hub_inspector

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("vechealth")
except PackageNotFoundError:  # pragma: no cover - editable/dev checkout without metadata
    __version__ = "0.0.0+unknown"
