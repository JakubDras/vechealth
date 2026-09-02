//! Serde-serializable mirrors of `vechealth-core`'s metric result types.
//!
//! These are deliberately plain field-for-field copies, not `#[derive]`s
//! bolted onto `core`'s own structs — `core` stays free of any
//! serialization dependency (see the crate-level docs in `lib.rs`). Each
//! type here is connected to its `core` counterpart via `From`, mirroring
//! the pattern `bindings` already uses for its own pyo3 result classes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use vechealth_core::metrics::{
    all, anisotropy, duplicates, fragmentation, hubness, intrinsic_dim, outliers, qmas, snc,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubnessResult {
    pub hubness_skewness: f64,
    pub orphans_fraction: f64,
    pub max_occurrences: u32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispersionResult {
    pub mean_1nn_distance: f32,
    pub mean_knn_distance: f32,
}

impl From<fragmentation::DispersionResult> for DispersionResult {
    fn from(r: fragmentation::DispersionResult) -> Self {
        Self {
            mean_1nn_distance: r.mean_1nn_distance,
            mean_knn_distance: r.mean_knn_distance,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnisotropyResult {
    pub mean_vector_norm: f32,
    pub top1_variance_ratio: f32,
    pub top10_variance_ratio: f32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutliersResult {
    pub outlier_fraction: f32,
    pub max_1nn_distance: f32,
    pub std_1nn_distance: f32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicatesResult {
    pub ndds_fraction: f32,
    pub mean_1nn_distance: f32,
    pub min_distance_global: f32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrinsicDimResult {
    pub mean_id: f32,
    pub median_id: f32,
}

impl From<intrinsic_dim::IntrinsicDimResult> for IntrinsicDimResult {
    fn from(r: intrinsic_dim::IntrinsicDimResult) -> Self {
        Self {
            mean_id: r.mean_id,
            median_id: r.median_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QmasResult {
    pub mean_1nn_distance: f32,
    pub mean_knn_distance: f32,
    pub orphans_fraction: f32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SncResult {
    pub mean_snc_score: f32,
}

impl From<snc::SncResult> for SncResult {
    fn from(r: snc::SncResult) -> Self {
        Self {
            mean_snc_score: r.mean_snc_score,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllMetricsConfig {
    pub k: usize,
    pub k_intrinsic_dim: usize,
    pub batch_size: usize,
    pub duplicate_epsilon: f32,
    pub outlier_distance_threshold: Option<f32>,
}

impl From<all::AllMetricsConfig> for AllMetricsConfig {
    fn from(c: all::AllMetricsConfig) -> Self {
        Self {
            k: c.k,
            k_intrinsic_dim: c.k_intrinsic_dim,
            batch_size: c.batch_size,
            duplicate_epsilon: c.duplicate_epsilon,
            outlier_distance_threshold: c.outlier_distance_threshold,
        }
    }
}

/// Combined output of `compute_all_metrics`, mirroring
/// `vechealth_core::metrics::all::AllMetricsResult`.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl AllMetricsResult {
    /// Flattens every scalar field into a `"{group}.{field}" -> value` map.
    /// The group prefix is required, not cosmetic: field names repeat
    /// across metrics (`mean_1nn_distance` appears in dispersion,
    /// duplicates, and qmas), so the bare field name alone is ambiguous.
    /// `qmas` is omitted entirely when absent, rather than emitting a null
    /// — the same convention MLflow/Evidently use for metrics that weren't
    /// computed in a given run.
    pub fn flatten(&self) -> BTreeMap<String, f64> {
        let mut out = BTreeMap::new();
        out.insert(
            "hubness.hubness_skewness".to_string(),
            self.hubness.hubness_skewness,
        );
        out.insert(
            "hubness.orphans_fraction".to_string(),
            self.hubness.orphans_fraction,
        );
        out.insert(
            "hubness.max_occurrences".to_string(),
            self.hubness.max_occurrences as f64,
        );
        out.insert(
            "dispersion.mean_1nn_distance".to_string(),
            self.dispersion.mean_1nn_distance as f64,
        );
        out.insert(
            "dispersion.mean_knn_distance".to_string(),
            self.dispersion.mean_knn_distance as f64,
        );
        out.insert(
            "anisotropy.mean_vector_norm".to_string(),
            self.anisotropy.mean_vector_norm as f64,
        );
        out.insert(
            "anisotropy.top1_variance_ratio".to_string(),
            self.anisotropy.top1_variance_ratio as f64,
        );
        out.insert(
            "anisotropy.top10_variance_ratio".to_string(),
            self.anisotropy.top10_variance_ratio as f64,
        );
        out.insert(
            "outliers.outlier_fraction".to_string(),
            self.outliers.outlier_fraction as f64,
        );
        out.insert(
            "outliers.max_1nn_distance".to_string(),
            self.outliers.max_1nn_distance as f64,
        );
        out.insert(
            "outliers.std_1nn_distance".to_string(),
            self.outliers.std_1nn_distance as f64,
        );
        out.insert(
            "duplicates.ndds_fraction".to_string(),
            self.duplicates.ndds_fraction as f64,
        );
        out.insert(
            "duplicates.mean_1nn_distance".to_string(),
            self.duplicates.mean_1nn_distance as f64,
        );
        out.insert(
            "duplicates.min_distance_global".to_string(),
            self.duplicates.min_distance_global as f64,
        );
        out.insert(
            "intrinsic_dim.mean_id".to_string(),
            self.intrinsic_dim.mean_id as f64,
        );
        out.insert(
            "intrinsic_dim.median_id".to_string(),
            self.intrinsic_dim.median_id as f64,
        );
        out.insert(
            "snc.mean_snc_score".to_string(),
            self.snc.mean_snc_score as f64,
        );
        if let Some(q) = &self.qmas {
            out.insert(
                "qmas.mean_1nn_distance".to_string(),
                q.mean_1nn_distance as f64,
            );
            out.insert(
                "qmas.mean_knn_distance".to_string(),
                q.mean_knn_distance as f64,
            );
            out.insert(
                "qmas.orphans_fraction".to_string(),
                q.orphans_fraction as f64,
            );
        }
        out
    }
}
