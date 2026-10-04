//! PyO3 bindings for `vechealth-core`, compiled as the `vechealth._core` extension
//! module. This crate is intentionally a thin translation layer: all metric math
//! lives in `vechealth-core`. Here we only take care of the Python-facing contract —
//! typed results, a proper exception hierarchy, and releasing the GIL around every
//! CPU-bound call so callers can use this safely from multi-threaded services.

use std::collections::BTreeMap;
use std::path::PathBuf;

use numpy::{AllowTypeChange, PyArray2, PyArrayLike2};
use pyo3::exceptions::{PyException, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{create_exception, PyErr};
use serde::Serialize;
#[cfg(feature = "lancedb")]
use vechealth_connectors::lancedb as connectors_lancedb;
#[cfg(feature = "pgvector")]
use vechealth_connectors::pgvector as connectors_pgvector;
#[cfg(feature = "qdrant")]
use vechealth_connectors::qdrant as connectors_qdrant;
use vechealth_connectors::{
    local as connectors_local, ConnectorError as CoreConnectorError, DistanceMetric,
    FetchedVectors,
};
use vechealth_core::hub_inspector;
use vechealth_core::knn::{VecHealthError as CoreError, VecHealthEvaluator as CoreEvaluator};
use vechealth_core::metrics::{
    all, anisotropy, duplicates, fragmentation, hubness, intrinsic_dim, outliers, qmas, snc,
};
use vechealth_core::resources;
use vechealth_report::{self as report, ReportError as CoreReportError};

create_exception!(
    vechealth._core,
    VecHealthError,
    PyException,
    "Base class for all errors raised by vechealth."
);
create_exception!(
    vechealth._core,
    DimensionMismatchError,
    VecHealthError,
    "Query/vector dimensionality does not match the indexed vectors."
);
create_exception!(
    vechealth._core,
    KTooLargeError,
    VecHealthError,
    "Requested k is >= the number of available vectors."
);
create_exception!(
    vechealth._core,
    KTooSmallError,
    VecHealthError,
    "Requested k is smaller than this metric requires."
);
create_exception!(
    vechealth._core,
    EmptyInputError,
    VecHealthError,
    "The input matrix of vectors is empty."
);
create_exception!(
    vechealth._core,
    AllVectorsDegenerateError,
    VecHealthError,
    "Every input vector has zero norm, so none can be normalized."
);
create_exception!(
    vechealth._core,
    ConnectorError,
    VecHealthError,
    "Fetching vectors from an external source (a local file, Qdrant, Postgres) failed — \
     I/O, parsing, a schema mismatch, or a network/auth problem. The message says which."
);
create_exception!(
    vechealth._core,
    MemoryUsageWarning,
    PyUserWarning,
    "Warns, before the work starts, that the estimated peak memory of an analysis exceeds \
     8 GB (or the value of VECHEALTH_MEMORY_WARNING_GB)."
);
create_exception!(
    vechealth._core,
    ReportError,
    VecHealthError,
    "Saving, loading, or parsing a `Report` failed — I/O or JSON (de)serialization. The \
     message says which."
);

fn map_err(err: CoreError) -> PyErr {
    let msg = err.to_string();
    match err {
        CoreError::DimensionMismatch { .. } => DimensionMismatchError::new_err(msg),
        CoreError::KTooLarge { .. } => KTooLargeError::new_err(msg),
        CoreError::KTooSmall { .. } => KTooSmallError::new_err(msg),
        CoreError::EmptyInput => EmptyInputError::new_err(msg),
        CoreError::AllVectorsDegenerate => AllVectorsDegenerateError::new_err(msg),
    }
}

fn map_connector_err(err: CoreConnectorError) -> PyErr {
    ConnectorError::new_err(err.to_string())
}

/// Names of the optional network connectors compiled into this build of the
/// extension (a subset of `"qdrant"`, `"pgvector"`, `"lancedb"`). The Python
/// package uses this to decide whether `from_lancedb` is native or has to
/// fall back to the pure-Python wrapper over the official SDK.
#[pyfunction]
fn compiled_features() -> Vec<&'static str> {
    [
        #[cfg(feature = "qdrant")]
        "qdrant",
        #[cfg(feature = "pgvector")]
        "pgvector",
        #[cfg(feature = "lancedb")]
        "lancedb",
    ]
    .to_vec()
}

/// The error raised by a `from_*` constructor whose connector was left out of
/// this build.
#[cfg(not(all(feature = "qdrant", feature = "pgvector", feature = "lancedb")))]
fn connector_not_compiled(feature: &str) -> PyErr {
    ConnectorError::new_err(format!(
        "this build of vechealth was compiled without the native '{feature}' connector \
         (rebuild with `maturin build --features {feature}`)"
    ))
}

fn map_report_err(err: CoreReportError) -> PyErr {
    ReportError::new_err(err.to_string())
}

/// Picks the batch size for a k-NN search — automatically when the caller
/// gave none — and, before any work starts, raises a `MemoryUsageWarning` if
/// the estimated peak memory is high. Must run with the GIL held.
fn resolve_batch_size(
    py: Python<'_>,
    n_vectors: usize,
    dim: usize,
    requested: Option<usize>,
) -> PyResult<usize> {
    if requested == Some(0) {
        return Err(PyValueError::new_err(
            "batch_size must be at least 1 (or None to choose it automatically)",
        ));
    }
    let threads = resources::worker_threads();
    let automatic = requested.is_none();
    let batch_size = requested.unwrap_or_else(|| resources::auto_batch_size(n_vectors, threads));
    if let Some(message) =
        resources::memory_warning(n_vectors, dim, batch_size, threads, automatic)
    {
        PyErr::warn_bound(py, &py.get_type_bound::<MemoryUsageWarning>(), &message, 1)?;
    }
    Ok(batch_size)
}

/// Estimated peak memory, in bytes, of the exact k-NN search on `n_vectors`
/// vectors of `dim` dimensions. `batch_size=None` uses the automatic choice
/// and `threads=None` the number of worker threads of this process; the
/// result is good to about 10% on the sizes it was measured on.
#[pyfunction]
#[pyo3(signature = (n_vectors, dim, batch_size=None, threads=None))]
fn estimate_peak_memory_bytes(
    n_vectors: usize,
    dim: usize,
    batch_size: Option<usize>,
    threads: Option<usize>,
) -> u64 {
    let threads = threads.unwrap_or_else(resources::worker_threads);
    let batch_size =
        batch_size.unwrap_or_else(|| resources::auto_batch_size(n_vectors, threads));
    resources::estimate_peak_memory_bytes(n_vectors, dim, batch_size, threads)
}

/// The batch size the k-NN search uses when none is given: as large as keeps
/// its similarity blocks within about 2 GB on `threads` worker threads
/// (default: this process's), between 16 and 2048 rows.
#[pyfunction]
#[pyo3(signature = (n_vectors, threads=None))]
fn default_batch_size(n_vectors: usize, threads: Option<usize>) -> usize {
    resources::auto_batch_size(n_vectors, threads.unwrap_or_else(resources::worker_threads))
}

// ---------------------------------------------------------------------------
// Serialization helpers, shared by every result/report class below.
// `vechealth-report` owns the actual JSON schema (its DTOs are the single
// source of truth); this just bridges `serde::Serialize` to Python objects.
// ---------------------------------------------------------------------------

fn to_pydict<'py, T: Serialize>(py: Python<'py>, value: &T) -> PyResult<Py<PyDict>> {
    let obj = pythonize::pythonize(py, value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    obj.into_bound(py)
        .downcast_into::<PyDict>()
        .map_err(|_| PyValueError::new_err("failed to serialize result into a Python dict"))
        .map(Bound::unbind)
}

fn to_json_string<T: Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string_pretty(value).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Builds an evaluator from freshly fetched vectors, warning the caller (via
/// Python's `warnings` module, matching how a library is expected to surface
/// non-fatal issues) when the source collection's distance metric isn't
/// cosine — `VecHealthEvaluator` L2-normalizes and assumes cosine internally,
/// so a Euclidean/dot-indexed collection would otherwise be silently
/// misinterpreted.
fn evaluator_from_fetched(
    py: Python<'_>,
    fetched: FetchedVectors,
) -> PyResult<PyVecHealthEvaluator> {
    if let Some(metric) = fetched.info.distance_metric {
        if metric != DistanceMetric::Cosine {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1(
                "warn",
                (format!(
                    "source collection was indexed with '{metric}' distance, but \
                     VecHealthEvaluator assumes cosine similarity (it L2-normalizes vectors \
                     internally) — metric results may not reflect how this collection is \
                     actually queried",
                ),),
            )?;
        }
    }
    let evaluator = CoreEvaluator::new(fetched.vectors).map_err(map_err)?;
    Ok(PyVecHealthEvaluator { evaluator })
}

#[pyclass(frozen, get_all, module = "vechealth._core")]
#[derive(Clone)]
pub struct HubnessResult {
    pub hubness_skewness: f64,
    pub orphans_fraction: f64,
    pub max_occurrences: u32,
}

#[pymethods]
impl HubnessResult {
    fn __repr__(&self) -> String {
        format!(
            "HubnessResult(hubness_skewness={:.4}, orphans_fraction={:.4}, max_occurrences={})",
            self.hubness_skewness, self.orphans_fraction, self.max_occurrences
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, &report::HubnessResult::from(self.clone()))
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(&report::HubnessResult::from(self.clone()))
    }
}

impl From<hubness::HubnessResult> for HubnessResult {
    fn from(r: hubness::HubnessResult) -> Self {
        Self {
            hubness_skewness: r.hubness_skewness,
            orphans_fraction: r.orphans_fraction,
            max_occurrences: r.max_occurrences,
        }
    }
}

impl From<HubnessResult> for report::HubnessResult {
    fn from(r: HubnessResult) -> Self {
        Self {
            hubness_skewness: r.hubness_skewness,
            orphans_fraction: r.orphans_fraction,
            max_occurrences: r.max_occurrences,
        }
    }
}

impl From<report::HubnessResult> for HubnessResult {
    fn from(r: report::HubnessResult) -> Self {
        Self {
            hubness_skewness: r.hubness_skewness,
            orphans_fraction: r.orphans_fraction,
            max_occurrences: r.max_occurrences,
        }
    }
}
#[pyclass(frozen, get_all, module = "vechealth._core")]
#[derive(Clone, Serialize)]
pub struct HubEntry {
    pub index: usize,
    pub occurrence_count: u32,
}

#[pymethods]
impl HubEntry {
    fn __repr__(&self) -> String {
        format!(
            "HubEntry(index={}, occurrence_count={})",
            self.index, self.occurrence_count
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, self)
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(self)
    }
}

impl From<hub_inspector::HubEntry> for HubEntry {
    fn from(h: hub_inspector::HubEntry) -> Self {
        Self {
            index: h.index,
            occurrence_count: h.occurrence_count,
        }
    }
}

#[pyclass(frozen, get_all, module = "vechealth._core")]
#[derive(Clone)]
pub struct DispersionResult {
    pub mean_1nn_distance: f32,
    pub mean_knn_distance: f32,
}

#[pymethods]
impl DispersionResult {
    fn __repr__(&self) -> String {
        format!(
            "DispersionResult(mean_1nn_distance={:.4}, mean_knn_distance={:.4})",
            self.mean_1nn_distance, self.mean_knn_distance
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, &report::DispersionResult::from(self.clone()))
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(&report::DispersionResult::from(self.clone()))
    }
}

impl From<fragmentation::DispersionResult> for DispersionResult {
    fn from(r: fragmentation::DispersionResult) -> Self {
        Self {
            mean_1nn_distance: r.mean_1nn_distance,
            mean_knn_distance: r.mean_knn_distance,
        }
    }
}

impl From<DispersionResult> for report::DispersionResult {
    fn from(r: DispersionResult) -> Self {
        Self {
            mean_1nn_distance: r.mean_1nn_distance,
            mean_knn_distance: r.mean_knn_distance,
        }
    }
}

impl From<report::DispersionResult> for DispersionResult {
    fn from(r: report::DispersionResult) -> Self {
        Self {
            mean_1nn_distance: r.mean_1nn_distance,
            mean_knn_distance: r.mean_knn_distance,
        }
    }
}

#[pyclass(frozen, get_all, module = "vechealth._core")]
#[derive(Clone)]
pub struct AnisotropyResult {
    pub mean_vector_norm: f32,
    pub top1_variance_ratio: f32,
    pub top10_variance_ratio: f32,
}

#[pymethods]
impl AnisotropyResult {
    fn __repr__(&self) -> String {
        format!(
            "AnisotropyResult(mean_vector_norm={:.4}, top1_variance_ratio={:.4}, top10_variance_ratio={:.4})",
            self.mean_vector_norm, self.top1_variance_ratio, self.top10_variance_ratio
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, &report::AnisotropyResult::from(self.clone()))
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(&report::AnisotropyResult::from(self.clone()))
    }
}

impl From<anisotropy::AnisotropyResult> for AnisotropyResult {
    fn from(r: anisotropy::AnisotropyResult) -> Self {
        Self {
            mean_vector_norm: r.mean_vector_norm,
            top1_variance_ratio: r.top1_variance_ratio,
            top10_variance_ratio: r.top10_variance_ratio,
        }
    }
}

impl From<AnisotropyResult> for report::AnisotropyResult {
    fn from(r: AnisotropyResult) -> Self {
        Self {
            mean_vector_norm: r.mean_vector_norm,
            top1_variance_ratio: r.top1_variance_ratio,
            top10_variance_ratio: r.top10_variance_ratio,
        }
    }
}

impl From<report::AnisotropyResult> for AnisotropyResult {
    fn from(r: report::AnisotropyResult) -> Self {
        Self {
            mean_vector_norm: r.mean_vector_norm,
            top1_variance_ratio: r.top1_variance_ratio,
            top10_variance_ratio: r.top10_variance_ratio,
        }
    }
}

#[pyclass(frozen, get_all, module = "vechealth._core")]
#[derive(Clone)]
pub struct OutliersResult {
    pub outlier_fraction: f32,
    pub max_1nn_distance: f32,
    pub std_1nn_distance: f32,
}

#[pymethods]
impl OutliersResult {
    fn __repr__(&self) -> String {
        format!(
            "OutliersResult(outlier_fraction={:.4}, max_1nn_distance={:.4}, std_1nn_distance={:.4})",
            self.outlier_fraction, self.max_1nn_distance, self.std_1nn_distance
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, &report::OutliersResult::from(self.clone()))
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(&report::OutliersResult::from(self.clone()))
    }
}

impl From<outliers::OutliersResult> for OutliersResult {
    fn from(r: outliers::OutliersResult) -> Self {
        Self {
            outlier_fraction: r.outlier_fraction,
            max_1nn_distance: r.max_1nn_distance,
            std_1nn_distance: r.std_1nn_distance,
        }
    }
}

impl From<OutliersResult> for report::OutliersResult {
    fn from(r: OutliersResult) -> Self {
        Self {
            outlier_fraction: r.outlier_fraction,
            max_1nn_distance: r.max_1nn_distance,
            std_1nn_distance: r.std_1nn_distance,
        }
    }
}

impl From<report::OutliersResult> for OutliersResult {
    fn from(r: report::OutliersResult) -> Self {
        Self {
            outlier_fraction: r.outlier_fraction,
            max_1nn_distance: r.max_1nn_distance,
            std_1nn_distance: r.std_1nn_distance,
        }
    }
}

#[pyclass(frozen, get_all, module = "vechealth._core")]
#[derive(Clone)]
pub struct DuplicatesResult {
    pub ndds_fraction: f32,
    pub mean_1nn_distance: f32,
    pub min_distance_global: f32,
}

#[pymethods]
impl DuplicatesResult {
    fn __repr__(&self) -> String {
        format!(
            "DuplicatesResult(ndds_fraction={:.4}, mean_1nn_distance={:.4}, min_distance_global={:.4})",
            self.ndds_fraction, self.mean_1nn_distance, self.min_distance_global
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, &report::DuplicatesResult::from(self.clone()))
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(&report::DuplicatesResult::from(self.clone()))
    }
}

impl From<duplicates::DuplicatesResult> for DuplicatesResult {
    fn from(r: duplicates::DuplicatesResult) -> Self {
        Self {
            ndds_fraction: r.ndds_fraction,
            mean_1nn_distance: r.mean_1nn_distance,
            min_distance_global: r.min_distance_global,
        }
    }
}

impl From<DuplicatesResult> for report::DuplicatesResult {
    fn from(r: DuplicatesResult) -> Self {
        Self {
            ndds_fraction: r.ndds_fraction,
            mean_1nn_distance: r.mean_1nn_distance,
            min_distance_global: r.min_distance_global,
        }
    }
}

impl From<report::DuplicatesResult> for DuplicatesResult {
    fn from(r: report::DuplicatesResult) -> Self {
        Self {
            ndds_fraction: r.ndds_fraction,
            mean_1nn_distance: r.mean_1nn_distance,
            min_distance_global: r.min_distance_global,
        }
    }
}

#[pyclass(frozen, get_all, module = "vechealth._core")]
#[derive(Clone)]
pub struct IntrinsicDimResult {
    pub mean_id: f32,
    pub median_id: f32,
}

#[pymethods]
impl IntrinsicDimResult {
    fn __repr__(&self) -> String {
        format!(
            "IntrinsicDimResult(mean_id={:.4}, median_id={:.4})",
            self.mean_id, self.median_id
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, &report::IntrinsicDimResult::from(self.clone()))
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(&report::IntrinsicDimResult::from(self.clone()))
    }
}

impl From<intrinsic_dim::IntrinsicDimResult> for IntrinsicDimResult {
    fn from(r: intrinsic_dim::IntrinsicDimResult) -> Self {
        Self {
            mean_id: r.mean_id,
            median_id: r.median_id,
        }
    }
}

impl From<IntrinsicDimResult> for report::IntrinsicDimResult {
    fn from(r: IntrinsicDimResult) -> Self {
        Self {
            mean_id: r.mean_id,
            median_id: r.median_id,
        }
    }
}

impl From<report::IntrinsicDimResult> for IntrinsicDimResult {
    fn from(r: report::IntrinsicDimResult) -> Self {
        Self {
            mean_id: r.mean_id,
            median_id: r.median_id,
        }
    }
}

#[pyclass(frozen, get_all, module = "vechealth._core")]
#[derive(Clone)]
pub struct QmasResult {
    pub mean_1nn_distance: f32,
    pub mean_knn_distance: f32,
    pub orphans_fraction: f32,
}

#[pymethods]
impl QmasResult {
    fn __repr__(&self) -> String {
        format!(
            "QmasResult(mean_1nn_distance={:.4}, mean_knn_distance={:.4}, orphans_fraction={:.4})",
            self.mean_1nn_distance, self.mean_knn_distance, self.orphans_fraction
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, &report::QmasResult::from(self.clone()))
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(&report::QmasResult::from(self.clone()))
    }
}

impl From<qmas::QmasResult> for QmasResult {
    fn from(r: qmas::QmasResult) -> Self {
        Self {
            mean_1nn_distance: r.mean_1nn_distance,
            mean_knn_distance: r.mean_knn_distance,
            orphans_fraction: r.orphans_fraction,
        }
    }
}

impl From<QmasResult> for report::QmasResult {
    fn from(r: QmasResult) -> Self {
        Self {
            mean_1nn_distance: r.mean_1nn_distance,
            mean_knn_distance: r.mean_knn_distance,
            orphans_fraction: r.orphans_fraction,
        }
    }
}

impl From<report::QmasResult> for QmasResult {
    fn from(r: report::QmasResult) -> Self {
        Self {
            mean_1nn_distance: r.mean_1nn_distance,
            mean_knn_distance: r.mean_knn_distance,
            orphans_fraction: r.orphans_fraction,
        }
    }
}

#[pyclass(frozen, get_all, module = "vechealth._core")]
#[derive(Clone)]
pub struct SncResult {
    pub mean_snc_score: f32,
}

#[pymethods]
impl SncResult {
    fn __repr__(&self) -> String {
        format!("SncResult(mean_snc_score={:.4})", self.mean_snc_score)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, &report::SncResult::from(self.clone()))
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(&report::SncResult::from(self.clone()))
    }
}

impl From<snc::SncResult> for SncResult {
    fn from(r: snc::SncResult) -> Self {
        Self {
            mean_snc_score: r.mean_snc_score,
        }
    }
}

impl From<SncResult> for report::SncResult {
    fn from(r: SncResult) -> Self {
        Self {
            mean_snc_score: r.mean_snc_score,
        }
    }
}

impl From<report::SncResult> for SncResult {
    fn from(r: report::SncResult) -> Self {
        Self {
            mean_snc_score: r.mean_snc_score,
        }
    }
}

/// Combined output of `VecHealthEvaluator.compute_all()` — one field per
/// metric, using the same typed result classes as the individual
/// `compute_*` methods. `qmas` is `None` when no `queries` were passed in.
#[pyclass(frozen, get_all, module = "vechealth._core")]
#[derive(Clone)]
pub struct AllMetricsResult {
    pub hubness: HubnessResult,
    pub dispersion: DispersionResult,
    pub anisotropy: AnisotropyResult,
    pub outliers: OutliersResult,
    pub duplicates: DuplicatesResult,
    pub intrinsic_dim: IntrinsicDimResult,
    pub snc: SncResult,
    pub qmas: Option<QmasResult>,
}

#[pymethods]
impl AllMetricsResult {
    fn __repr__(&self) -> String {
        format!(
            "AllMetricsResult(\n  hubness={},\n  dispersion={},\n  anisotropy={},\n  outliers={},\
            \n  duplicates={},\n  intrinsic_dim={},\n  snc={},\n  qmas={},\n)",
            self.hubness.__repr__(),
            self.dispersion.__repr__(),
            self.anisotropy.__repr__(),
            self.outliers.__repr__(),
            self.duplicates.__repr__(),
            self.intrinsic_dim.__repr__(),
            self.snc.__repr__(),
            self.qmas
                .as_ref()
                .map(|q| q.__repr__())
                .unwrap_or_else(|| "None".to_string()),
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, &report::AllMetricsResult::from(self.clone()))
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(&report::AllMetricsResult::from(self.clone()))
    }
}

impl From<all::AllMetricsResult> for AllMetricsResult {
    fn from(r: all::AllMetricsResult) -> Self {
        Self {
            hubness: r.hubness.into(),
            dispersion: r.dispersion.into(),
            anisotropy: r.anisotropy.into(),
            outliers: r.outliers.into(),
            duplicates: r.duplicates.into(),
            intrinsic_dim: r.intrinsic_dim.into(),
            snc: r.snc.into(),
            qmas: r.qmas.map(QmasResult::from),
        }
    }
}

impl From<AllMetricsResult> for report::AllMetricsResult {
    fn from(r: AllMetricsResult) -> Self {
        Self {
            hubness: r.hubness.into(),
            dispersion: r.dispersion.into(),
            anisotropy: r.anisotropy.into(),
            outliers: r.outliers.into(),
            duplicates: r.duplicates.into(),
            intrinsic_dim: r.intrinsic_dim.into(),
            snc: r.snc.into(),
            qmas: r.qmas.map(report::QmasResult::from),
        }
    }
}

impl From<report::AllMetricsResult> for AllMetricsResult {
    fn from(r: report::AllMetricsResult) -> Self {
        Self {
            hubness: r.hubness.into(),
            dispersion: r.dispersion.into(),
            anisotropy: r.anisotropy.into(),
            outliers: r.outliers.into(),
            duplicates: r.duplicates.into(),
            intrinsic_dim: r.intrinsic_dim.into(),
            snc: r.snc.into(),
            qmas: r.qmas.map(QmasResult::from),
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// Stateful evaluator over a fixed set of vectors. KNN and normalization
/// results are cached internally, so calling several `compute_*` methods on
/// the same instance re-uses the same k-NN search instead of recomputing it.
///
/// All `compute_*` methods release the GIL for the duration of the Rust-side
/// computation, so other Python threads keep running while this crunches.
#[pyclass(name = "VecHealthEvaluator", module = "vechealth._core")]
pub struct PyVecHealthEvaluator {
    evaluator: CoreEvaluator,
}

#[pymethods]
impl PyVecHealthEvaluator {
    /// `vectors` accepts any NumPy array-like of numbers — most commonly a
    /// `float32` or `float64` array, but also plain Python nested
    /// sequences. Non-`float32` input is cast via NumPy's own `asarray`
    /// (same conversion `from_local`/`from_qdrant`/`from_pgvector` already
    /// do internally); an already-`float32`, C-contiguous array is used
    /// as-is with no copy.
    #[new]
    fn new(vectors: PyArrayLike2<'_, f32, AllowTypeChange>) -> PyResult<Self> {
        let array = vectors.as_array().to_owned();
        let evaluator = CoreEvaluator::new(array).map_err(map_err)?;
        Ok(Self { evaluator })
    }

    /// Loads vectors from a local `.npy`, `.csv`, or `.parquet` file — the
    /// safest connector category, zero network involvement by construction.
    /// `.npy`: any 2D float32/float64 array. `.csv`: one row per vector, one
    /// field per dimension; `has_header` controls whether the first line is
    /// skipped. `.parquet`: one numeric column per dimension, one row per
    /// vector; `columns=None` uses every column in schema order, or pass an
    /// explicit list to select/reorder a subset.
    #[staticmethod]
    #[pyo3(signature = (path, has_header=true, columns=None))]
    fn from_local(
        py: Python<'_>,
        path: &str,
        has_header: bool,
        columns: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let path_buf = std::path::PathBuf::from(path);
        let fetched = py
            .allow_threads(|| match path_buf.extension().and_then(|e| e.to_str()) {
                Some("csv") => connectors_local::load_csv(&path_buf, has_header),
                Some("parquet") => connectors_local::load_parquet(&path_buf, columns.as_deref()),
                _ => connectors_local::load_local(&path_buf),
            })
            .map_err(map_connector_err)?;
        evaluator_from_fetched(py, fetched)
    }

    /// Fetches every point's vector from a Qdrant collection via the
    /// `scroll` API (never `search`/ANN). Pulls the *complete* collection;
    /// `page_size` only controls how many points are requested per network
    /// round-trip, not how many are fetched in total. Warns (via
    /// `warnings.warn`) if the collection's declared distance metric isn't
    /// cosine.
    #[staticmethod]
    #[pyo3(signature = (url, collection, api_key=None, page_size=1000, timeout_secs=30))]
    fn from_qdrant(
        py: Python<'_>,
        url: &str,
        collection: &str,
        api_key: Option<String>,
        page_size: u32,
        timeout_secs: u64,
    ) -> PyResult<Self> {
        #[cfg(feature = "qdrant")]
        {
            let mut config =
                connectors_qdrant::QdrantConfig::new(url, collection).with_page_size(page_size);
            config.timeout_secs = timeout_secs;
            if let Some(key) = api_key {
                config = config.with_api_key(key);
            }
            let fetched = py
                .allow_threads(|| connectors_qdrant::fetch_all(&config))
                .map_err(map_connector_err)?;
            evaluator_from_fetched(py, fetched)
        }
        #[cfg(not(feature = "qdrant"))]
        {
            let _ = (py, url, collection, api_key, page_size, timeout_secs);
            Err(connector_not_compiled("qdrant"))
        }
    }

    /// Fetches every row from a pgvector-backed Postgres table via keyset
    /// pagination on `id_column` (never `OFFSET`). Pulls the *complete*
    /// table; `page_size` only controls rows per round-trip. `id_column`
    /// must be an integer primary/unique key.
    /// Connection is unencrypted (`NoTls`) — use an SSH tunnel or a trusted
    /// network if your Postgres requires TLS.
    #[staticmethod]
    #[pyo3(signature = (connection_string, table, vector_column, id_column, page_size=5000))]
    fn from_pgvector(
        py: Python<'_>,
        connection_string: &str,
        table: &str,
        vector_column: &str,
        id_column: &str,
        page_size: i64,
    ) -> PyResult<Self> {
        #[cfg(feature = "pgvector")]
        {
            let config = connectors_pgvector::PgVectorConfig::new(
                connection_string,
                table,
                vector_column,
                id_column,
            )
            .with_page_size(page_size);
            let fetched = py
                .allow_threads(|| connectors_pgvector::fetch_all(&config))
                .map_err(map_connector_err)?;
            evaluator_from_fetched(py, fetched)
        }
        #[cfg(not(feature = "pgvector"))]
        {
            let _ = (py, connection_string, table, vector_column, id_column, page_size);
            Err(connector_not_compiled("pgvector"))
        }
    }

    /// Fetches every row's vector from a LanceDB table via a plain columnar
    /// scan (`Table::query()`, never `vector_search()`/ANN). Pulls the
    /// *complete* table; `batch_size` only controls the maximum rows per
    /// `RecordBatch` read at a time, not how many rows are fetched in
    /// total. `uri` is most commonly a local directory path (LanceDB is
    /// embedded-first). Warns via
    /// `warnings.warn` if the table's vector index reports a distance
    /// metric other than cosine. This native path only exists in builds
    /// compiled with the `lancedb` feature; the Python package replaces it
    /// with a wrapper over the official `lancedb` SDK otherwise.
    #[staticmethod]
    #[pyo3(signature = (uri, table, vector_column, batch_size=1024))]
    fn from_lancedb(
        py: Python<'_>,
        uri: &str,
        table: &str,
        vector_column: &str,
        batch_size: u32,
    ) -> PyResult<Self> {
        #[cfg(feature = "lancedb")]
        {
            let config = connectors_lancedb::LanceDbConfig::new(uri, table, vector_column)
                .with_batch_size(batch_size);
            let fetched = py
                .allow_threads(|| connectors_lancedb::fetch_all(&config))
                .map_err(map_connector_err)?;
            evaluator_from_fetched(py, fetched)
        }
        #[cfg(not(feature = "lancedb"))]
        {
            let _ = (py, uri, table, vector_column, batch_size);
            Err(connector_not_compiled("lancedb"))
        }
    }

    /// Number of indexed vectors.
    #[getter]
    fn n_vectors(&self) -> usize {
        self.evaluator.n_vectors()
    }

    /// Dimensionality of the indexed vectors.
    #[getter]
    fn dim(&self) -> usize {
        self.evaluator.dim
    }

    fn __repr__(&self) -> String {
        format!(
            "VecHealthEvaluator(n_vectors={}, dim={})",
            self.evaluator.n_vectors(),
            self.evaluator.dim
        )
    }

    /// Returns (distances, indices) for the k nearest neighbors of every vector.
    /// The self-match is filtered out automatically — column 0 is always the
    /// true nearest *other* neighbor. `batch_size=None` (the default) picks
    /// the number of rows per block from the size of the problem so that the
    /// search fits in memory; an explicit value overrides it. Results do not
    /// depend on it.
    #[pyo3(signature = (k, batch_size=None))]
    fn get_knn<'py>(
        &mut self,
        py: Python<'py>,
        k: usize,
        batch_size: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u32>>)> {
        let batch_size =
            resolve_batch_size(py, self.evaluator.n_vectors(), self.evaluator.dim, batch_size)?;
        let evaluator = &mut self.evaluator;
        let (dists, indices) = py
            .allow_threads(|| -> Result<_, CoreError> {
                let (d, i) = evaluator.get_knn(k, batch_size)?;
                Ok((d.to_owned(), i.to_owned()))
            })
            .map_err(map_err)?;
        Ok((
            PyArray2::from_array_bound(py, &dists),
            PyArray2::from_array_bound(py, &indices),
        ))
    }

    /// Hubness Score - detects "black holes": vectors that dominate the
    /// k-NN graph out of proportion to a uniform embedding space.
    #[pyo3(signature = (k=10, batch_size=None))]
    fn compute_hubness(
        &mut self,
        py: Python<'_>,
        k: usize,
        batch_size: Option<usize>,
    ) -> PyResult<HubnessResult> {
        let batch_size =
            resolve_batch_size(py, self.evaluator.n_vectors(), self.evaluator.dim, batch_size)?;
        let evaluator = &mut self.evaluator;
        py.allow_threads(|| hubness::compute_hubness_score(evaluator, k, batch_size))
            .map(HubnessResult::from)
            .map_err(map_err)
    }

    /// Hub Inspector — identifies the `top_n` points with the highest
    /// occurrence count across every point's k-NN list ("hubs"), for
    /// manual inspection. Unlike `compute_hubness` (an aggregate
    /// statistic), this returns the actual points responsible for a high
    /// hubness reading.
    #[pyo3(signature = (k=10, top_n=20, batch_size=None))]
    fn identify_hubs(
        &mut self,
        py: Python<'_>,
        k: usize,
        top_n: usize,
        batch_size: Option<usize>,
    ) -> PyResult<Vec<HubEntry>> {
        let batch_size =
            resolve_batch_size(py, self.evaluator.n_vectors(), self.evaluator.dim, batch_size)?;
        let evaluator = &mut self.evaluator;
        py.allow_threads(|| hub_inspector::identify_hubs(evaluator, k, top_n, batch_size))
            .map(|hubs| hubs.into_iter().map(HubEntry::from).collect())
            .map_err(map_err)
    }

    /// Neighborhood Dispersion Score - diagnoses cluster fragmentation via
    /// the mean distance to nearest neighbors.
    #[pyo3(signature = (k=10, batch_size=None))]
    fn compute_dispersion(
        &mut self,
        py: Python<'_>,
        k: usize,
        batch_size: Option<usize>,
    ) -> PyResult<DispersionResult> {
        let batch_size =
            resolve_batch_size(py, self.evaluator.n_vectors(), self.evaluator.dim, batch_size)?;
        let evaluator = &mut self.evaluator;
        py.allow_threads(|| fragmentation::compute_dispersion_score(evaluator, k, batch_size))
            .map(DispersionResult::from)
            .map_err(map_err)
    }

    /// Detects anisotropy (the "cone effect" / directional collapse) via the
    /// spectrum of the covariance matrix.
    fn compute_anisotropy(&self, py: Python<'_>) -> PyResult<AnisotropyResult> {
        let evaluator = &self.evaluator;
        py.allow_threads(|| anisotropy::compute_anisotropy_score(evaluator))
            .map(AnisotropyResult::from)
            .map_err(map_err)
    }

    /// Outlier Fraction Score - detects garbage representations via
    /// abnormally distant nearest neighbors.
    #[pyo3(signature = (distance_threshold, batch_size=None))]
    fn compute_outliers(
        &mut self,
        py: Python<'_>,
        distance_threshold: f32,
        batch_size: Option<usize>,
    ) -> PyResult<OutliersResult> {
        let batch_size =
            resolve_batch_size(py, self.evaluator.n_vectors(), self.evaluator.dim, batch_size)?;
        let evaluator = &mut self.evaluator;
        py.allow_threads(|| {
            outliers::compute_outlier_score(evaluator, distance_threshold, batch_size)
        })
        .map(OutliersResult::from)
        .map_err(map_err)
    }

    /// Near-Duplicate Density Score (NDDS) - detects space pollution from
    /// near-duplicate vectors.
    #[pyo3(signature = (epsilon=0.05, batch_size=None))]
    fn compute_duplicates(
        &mut self,
        py: Python<'_>,
        epsilon: f32,
        batch_size: Option<usize>,
    ) -> PyResult<DuplicatesResult> {
        let batch_size =
            resolve_batch_size(py, self.evaluator.n_vectors(), self.evaluator.dim, batch_size)?;
        let evaluator = &mut self.evaluator;
        py.allow_threads(|| duplicates::compute_ndds_score(evaluator, epsilon, batch_size))
            .map(DuplicatesResult::from)
            .map_err(map_err)
    }

    /// MLE (Levina-Bickel) estimator of the local intrinsic dimensionality.
    /// Default k=20, since ID estimation needs a wider neighborhood window
    /// than plain nearest-neighbor search.
    #[pyo3(signature = (k=20, batch_size=None))]
    fn compute_intrinsic_dim(
        &mut self,
        py: Python<'_>,
        k: usize,
        batch_size: Option<usize>,
    ) -> PyResult<IntrinsicDimResult> {
        let batch_size =
            resolve_batch_size(py, self.evaluator.n_vectors(), self.evaluator.dim, batch_size)?;
        let evaluator = &mut self.evaluator;
        py.allow_threads(|| intrinsic_dim::compute_intrinsic_dim_score(evaluator, k, batch_size))
            .map(IntrinsicDimResult::from)
            .map_err(map_err)
    }

    /// Query Manifold Alignment Score (QMAS) - measures how well query
    /// vectors align with the document space. `queries` accepts the same
    /// array-like input as the `VecHealthEvaluator` constructor (any
    /// numeric dtype, cast to `float32` as needed).
    #[pyo3(signature = (queries, k=10, batch_size=None))]
    fn compute_qmas(
        &mut self,
        py: Python<'_>,
        queries: PyArrayLike2<'_, f32, AllowTypeChange>,
        k: usize,
        batch_size: Option<usize>,
    ) -> PyResult<QmasResult> {
        let batch_size =
            resolve_batch_size(py, self.evaluator.n_vectors(), self.evaluator.dim, batch_size)?;
        let queries_owned = queries.as_array().to_owned();
        let evaluator = &mut self.evaluator;
        py.allow_threads(|| {
            qmas::compute_qmas_score(evaluator, queries_owned.view(), k, batch_size)
        })
        .map(QmasResult::from)
        .map_err(map_err)
    }

    /// Semantic Neighborhood Consistency (SNC) - measures 2-hop topology
    /// consistency via Jaccard similarity between k-NN neighborhoods.
    #[pyo3(signature = (k=10, batch_size=None))]
    fn compute_snc(
        &mut self,
        py: Python<'_>,
        k: usize,
        batch_size: Option<usize>,
    ) -> PyResult<SncResult> {
        let batch_size =
            resolve_batch_size(py, self.evaluator.n_vectors(), self.evaluator.dim, batch_size)?;
        let evaluator = &mut self.evaluator;
        py.allow_threads(|| snc::compute_snc_score(evaluator, k, batch_size))
            .map(SncResult::from)
            .map_err(map_err)
    }

    /// Orchestrator: runs every implemented metric on this evaluator in one
    /// call, with the same defaults each metric uses on its own. QMAS is
    /// included only if `queries` is given (it needs a query set to measure
    /// alignment against). `outlier_distance_threshold` is a chord distance
    /// on the unit sphere and defaults to 1.2, the value used for the
    /// results in the accompanying preprint; it is an absolute threshold, so
    /// recalibrate it for other encoders or dimensionalities. Passing `None`
    /// derives a threshold from the data instead (3x the mean
    /// nearest-neighbor distance) — for typical high-dimensional text
    /// embeddings that lands close to or above 2.0, the largest possible
    /// chord distance, where essentially no point can be flagged.
    /// `batch_size=None` picks the rows per k-NN block from the size of the
    /// problem so that the search fits in memory (results do not depend on
    /// it). A `MemoryUsageWarning` is raised before the search when the
    /// estimated peak memory exceeds 8 GB (`VECHEALTH_MEMORY_WARNING_GB`
    /// changes the limit).
    #[pyo3(signature = (
        queries=None,
        k=10,
        k_intrinsic_dim=20,
        batch_size=None,
        duplicate_epsilon=0.05,
        outlier_distance_threshold=Some(1.2),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn compute_all(
        &mut self,
        py: Python<'_>,
        queries: Option<PyArrayLike2<'_, f32, AllowTypeChange>>,
        k: usize,
        k_intrinsic_dim: usize,
        batch_size: Option<usize>,
        duplicate_epsilon: f32,
        outlier_distance_threshold: Option<f32>,
    ) -> PyResult<AllMetricsResult> {
        let batch_size =
            resolve_batch_size(py, self.evaluator.n_vectors(), self.evaluator.dim, batch_size)?;
        let queries_owned = queries.map(|q| q.as_array().to_owned());
        let config = all::AllMetricsConfig {
            k,
            k_intrinsic_dim,
            batch_size,
            duplicate_epsilon,
            outlier_distance_threshold,
        };
        let evaluator = &mut self.evaluator;
        py.allow_threads(|| {
            all::compute_all_metrics(evaluator, &config, queries_owned.as_ref().map(|q| q.view()))
        })
        .map(AllMetricsResult::from)
        .map_err(map_err)
    }

    /// Same as `compute_all`, but wraps the result in a `Report` — a
    /// self-contained, versioned snapshot (config used, dataset fingerprint,
    /// timestamp, arbitrary `tags`) that can be saved, reloaded, and
    /// compared against another `Report` later. Use `compute_all` instead
    /// if you only want the numbers, with no metadata attached.
    #[pyo3(signature = (
        queries=None,
        k=10,
        k_intrinsic_dim=20,
        batch_size=None,
        duplicate_epsilon=0.05,
        outlier_distance_threshold=Some(1.2),
        tags=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn compute_report(
        &mut self,
        py: Python<'_>,
        queries: Option<PyArrayLike2<'_, f32, AllowTypeChange>>,
        k: usize,
        k_intrinsic_dim: usize,
        batch_size: Option<usize>,
        duplicate_epsilon: f32,
        outlier_distance_threshold: Option<f32>,
        tags: Option<BTreeMap<String, String>>,
    ) -> PyResult<Report> {
        let batch_size =
            resolve_batch_size(py, self.evaluator.n_vectors(), self.evaluator.dim, batch_size)?;
        let queries_owned = queries.map(|q| q.as_array().to_owned());
        let config = all::AllMetricsConfig {
            k,
            k_intrinsic_dim,
            batch_size,
            duplicate_epsilon,
            outlier_distance_threshold,
        };
        let evaluator = &mut self.evaluator;
        let dataset = report::fingerprint_vectors(evaluator.vectors());
        let metrics = py
            .allow_threads(|| {
                all::compute_all_metrics(
                    evaluator,
                    &config,
                    queries_owned.as_ref().map(|q| q.view()),
                )
            })
            .map_err(map_err)?;

        let inner = report::Report::new(
            metrics.into(),
            config.into(),
            dataset,
            env!("CARGO_PKG_VERSION"),
            tags.unwrap_or_default(),
        );
        Ok(Report(inner))
    }
}

// ---------------------------------------------------------------------------
// Report / Comparison: persistence and baseline comparison, built on top of
// `vechealth-report`. `Report` is a thin newtype wrapper — the actual
// schema and logic (save/load/compare/flatten) lives in that crate; see it
// for the reasoning behind each design choice.
// ---------------------------------------------------------------------------

#[pyclass(module = "vechealth._core")]
pub struct Report(report::Report);

#[pymethods]
impl Report {
    fn __repr__(&self) -> String {
        format!(
            "Report(generated_at={}, n_vectors={}, dim={})",
            self.0.generated_at.to_rfc3339(),
            self.0.dataset.n_vectors,
            self.0.dataset.dim,
        )
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.0.schema_version
    }
    
    #[getter]
    fn generated_at(&self) -> String {
        self.0.generated_at.to_rfc3339()
    }

    #[getter]
    fn vechealth_version(&self) -> String {
        self.0.vechealth_version.clone()
    }

    #[getter]
    fn n_vectors(&self) -> usize {
        self.0.dataset.n_vectors
    }

    #[getter]
    fn dim(&self) -> usize {
        self.0.dataset.dim
    }
    
    #[getter]
    fn content_hash(&self) -> String {
        self.0.dataset.content_hash.clone()
    }

    #[getter]
    fn tags(&self) -> BTreeMap<String, String> {
        self.0.tags.clone()
    }

    #[getter]
    fn metrics(&self) -> AllMetricsResult {
        AllMetricsResult::from(self.0.metrics.clone())
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, &self.0)
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(&self.0)
    }
    
    fn flatten(&self) -> BTreeMap<String, f64> {
        self.0.flatten()
    }
    
    fn save(&self, path: &str) -> PyResult<()> {
        self.0.save(&PathBuf::from(path)).map_err(map_report_err)
    }
    
    #[staticmethod]
    fn load(path: &str) -> PyResult<Report> {
        report::Report::load(&PathBuf::from(path))
            .map(Report)
            .map_err(map_report_err)
    }

    /// Compares this report (treated as "current") against `baseline`,
    /// metric by metric. Raises no error and applies no thresholds — see
    /// `Comparison.warnings` for anything that would make the comparison
    /// unreliable (e.g. the two reports used a different `k`).
    fn compare(&self, baseline: &Report) -> Comparison {
        Comparison(report::compare(&baseline.0, &self.0))
    }
}

#[pyclass(frozen, get_all, module = "vechealth._core")]
#[derive(Clone)]
pub struct MetricDelta {
    pub baseline: f64,
    pub current: f64,
    pub delta: f64,
    pub delta_pct: Option<f64>,
}

#[pymethods]
impl MetricDelta {
    fn __repr__(&self) -> String {
        format!(
            "MetricDelta(baseline={:.4}, current={:.4}, delta={:.4}, delta_pct={})",
            self.baseline,
            self.current,
            self.delta,
            self.delta_pct
                .map(|p| format!("{p:.2}%"))
                .unwrap_or_else(|| "None".to_string()),
        )
    }
}

impl From<report::MetricDelta> for MetricDelta {
    fn from(d: report::MetricDelta) -> Self {
        Self {
            baseline: d.baseline,
            current: d.current,
            delta: d.delta,
            delta_pct: d.delta_pct,
        }
    }
}

#[pyclass(module = "vechealth._core")]
pub struct Comparison(report::ComparisonResult);

#[pymethods]
impl Comparison {
    fn __repr__(&self) -> String {
        format!(
            "Comparison({} metrics, {} warnings)",
            self.0.deltas.len(),
            self.0.warnings.len()
        )
    }

    #[getter]
    fn baseline_generated_at(&self) -> String {
        self.0.baseline_generated_at.to_rfc3339()
    }

    #[getter]
    fn current_generated_at(&self) -> String {
        self.0.current_generated_at.to_rfc3339()
    }

    #[getter]
    fn deltas(&self) -> BTreeMap<String, MetricDelta> {
        self.0
            .deltas
            .iter()
            .map(|(k, v)| (k.clone(), MetricDelta::from(v.clone())))
            .collect()
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.0.warnings.clone()
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        to_pydict(py, &self.0)
    }

    fn to_json(&self) -> PyResult<String> {
        to_json_string(&self.0)
    }
}

/// Compiled Rust core of `vechealth`. Import from `vechealth` instead of this
/// module directly — the pure-Python `vechealth` package re-exports the
/// stable public API.
#[pymodule]
fn _core(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compiled_features, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_peak_memory_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(default_batch_size, m)?)?;
    m.add_class::<PyVecHealthEvaluator>()?;
    m.add_class::<HubnessResult>()?;
    m.add_class::<HubEntry>()?;
    m.add_class::<DispersionResult>()?;
    m.add_class::<AnisotropyResult>()?;
    m.add_class::<OutliersResult>()?;
    m.add_class::<DuplicatesResult>()?;
    m.add_class::<IntrinsicDimResult>()?;
    m.add_class::<QmasResult>()?;
    m.add_class::<SncResult>()?;
    m.add_class::<AllMetricsResult>()?;
    m.add_class::<Report>()?;
    m.add_class::<MetricDelta>()?;
    m.add_class::<Comparison>()?;

    m.add("VecHealthError", py.get_type_bound::<VecHealthError>())?;
    m.add(
        "DimensionMismatchError",
        py.get_type_bound::<DimensionMismatchError>(),
    )?;
    m.add("KTooLargeError", py.get_type_bound::<KTooLargeError>())?;
    m.add("KTooSmallError", py.get_type_bound::<KTooSmallError>())?;
    m.add("EmptyInputError", py.get_type_bound::<EmptyInputError>())?;
    m.add(
        "AllVectorsDegenerateError",
        py.get_type_bound::<AllVectorsDegenerateError>(),
    )?;
    m.add("ConnectorError", py.get_type_bound::<ConnectorError>())?;
    m.add("ReportError", py.get_type_bound::<ReportError>())?;
    m.add("MemoryUsageWarning", py.get_type_bound::<MemoryUsageWarning>())?;
    Ok(())
}