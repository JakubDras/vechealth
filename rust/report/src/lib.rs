//! Serialization, persistence, and baseline comparison for
//! `vechealth-core` metric results.
//!
//! Kept as a crate separate from `vechealth-core` on purpose, for the same
//! reason `vechealth-connectors` is separate from it (see the rationale in
//! `connectors/src/lib.rs`): the pure metrics engine has no business
//! depending on `serde`/`chrono`/filesystem access, so those concerns live
//! here instead. `core` stays free of this crate's dependencies entirely.

pub mod compare;
pub mod fingerprint;
pub mod metrics;

pub use compare::{compare, ComparisonResult, MetricDelta};
pub use fingerprint::{fingerprint_vectors, DatasetFingerprint};
pub use metrics::{
    AllMetricsConfig, AllMetricsResult, AnisotropyResult, DispersionResult, DuplicatesResult,
    HubnessResult, IntrinsicDimResult, OutliersResult, QmasResult, SncResult,
};

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Bumped whenever a change to `Report`'s shape could break an existing
/// consumer (e.g. a field is removed or its meaning changes) — additive
/// changes (new optional fields) don't need a bump, since serde ignores
/// unknown fields on read by default and `Report` never sets
/// `deny_unknown_fields`.
pub const REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug)]
pub enum ReportError {
    Io(String),
    Serde(String),
}

impl fmt::Display for ReportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::Serde(msg) => write!(f, "Failed to (de)serialize report: {msg}"),
        }
    }
}

impl std::error::Error for ReportError {}

/// A self-contained, versioned snapshot of `compute_all_metrics`'s output,
/// plus the metadata needed to make it meaningful on its own later: when it
/// was computed, what config produced it, and which dataset it describes.
/// Save it, commit it, diff it, or hand it to `compare()` against another
/// one — a `Report` never depends on anything outside itself to be
/// interpreted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: u32,
    pub generated_at: DateTime<Utc>,
    pub vechealth_version: String,
    pub dataset: DatasetFingerprint,
    pub config: AllMetricsConfig,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
    pub metrics: AllMetricsResult,
}

impl Report {
    pub fn new(
        metrics: AllMetricsResult,
        config: AllMetricsConfig,
        dataset: DatasetFingerprint,
        vechealth_version: impl Into<String>,
        tags: BTreeMap<String, String>,
    ) -> Self {
        Self {
            schema_version: REPORT_SCHEMA_VERSION,
            generated_at: Utc::now(),
            vechealth_version: vechealth_version.into(),
            dataset,
            config,
            tags,
            metrics,
        }
    }

    pub fn to_json(&self) -> Result<String, ReportError> {
        serde_json::to_string_pretty(self).map_err(|e| ReportError::Serde(e.to_string()))
    }

    pub fn from_json(s: &str) -> Result<Self, ReportError> {
        serde_json::from_str(s).map_err(|e| ReportError::Serde(e.to_string()))
    }

    pub fn save(&self, path: &Path) -> Result<(), ReportError> {
        let json = self.to_json()?;
        fs::write(path, json).map_err(|e| ReportError::Io(e.to_string()))
    }

    pub fn load(path: &Path) -> Result<Self, ReportError> {
        let contents = fs::read_to_string(path).map_err(|e| ReportError::Io(e.to_string()))?;
        Self::from_json(&contents)
    }

    /// Flat `"{group}.{field}" -> value` view of `self.metrics` — see
    /// `AllMetricsResult::flatten` for why the group prefix is required.
    /// This is the shape a future metric-store/Prometheus exporter would
    /// consume; building it directly into `Report` now avoids having to
    /// redesign the schema when that exporter is built.
    pub fn flatten(&self) -> BTreeMap<String, f64> {
        self.metrics.flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vechealth_core::knn::VecHealthEvaluator;
    use vechealth_core::metrics::all::{compute_all_metrics, AllMetricsConfig as CoreConfig};

    fn sample_report(duplicate_epsilon: f32) -> Report {
        let vectors: Vec<[f32; 4]> = (0..15)
            .map(|i| {
                let x = i as f32;
                [x, x * 0.5, -x * 0.2, (x % 3.0)]
            })
            .collect();
        let vectors =
            ndarray::Array2::from_shape_vec((15, 4), vectors.into_iter().flatten().collect())
                .unwrap();
        let dataset = fingerprint_vectors(vectors.view());

        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let config = CoreConfig {
            k: 5,
            k_intrinsic_dim: 5,
            duplicate_epsilon,
            ..CoreConfig::default()
        };
        let metrics = compute_all_metrics(&mut evaluator, &config, None).unwrap();

        Report::new(
            metrics.into(),
            config.into(),
            dataset,
            "0.1.0-test",
            BTreeMap::new(),
        )
    }

    #[test]
    fn json_round_trip_preserves_content() {
        let report = sample_report(0.05);
        let json = report.to_json().unwrap();
        let restored = Report::from_json(&json).unwrap();
        assert_eq!(restored.schema_version, REPORT_SCHEMA_VERSION);
        assert_eq!(restored.dataset, report.dataset);
        assert_eq!(
            restored.metrics.hubness.hubness_skewness,
            report.metrics.hubness.hubness_skewness
        );
    }

    #[test]
    fn save_load_round_trip() {
        let report = sample_report(0.05);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.json");

        report.save(&path).unwrap();
        let loaded = Report::load(&path).unwrap();

        assert_eq!(loaded.dataset, report.dataset);
        assert_eq!(loaded.config.k, report.config.k);
    }

    #[test]
    fn load_missing_file_is_io_error() {
        let err = Report::load(Path::new("/nonexistent/report.json")).unwrap_err();
        assert!(matches!(err, ReportError::Io(_)));
    }

    #[test]
    fn flatten_omits_qmas_when_absent() {
        let report = sample_report(0.05);
        let flat = report.flatten();
        assert!(flat.contains_key("hubness.hubness_skewness"));
        assert!(flat.keys().all(|k| !k.starts_with("qmas.")));
    }

    #[test]
    fn compare_identical_reports_has_zero_deltas_and_no_warnings() {
        let report = sample_report(0.05);
        let result = compare(&report, &report);
        assert!(result.warnings.is_empty());
        for delta in result.deltas.values() {
            assert_eq!(delta.delta, 0.0);
        }
    }

    #[test]
    fn compare_warns_on_config_mismatch() {
        let baseline = sample_report(0.05);
        let current = sample_report(0.10);
        let result = compare(&baseline, &current);
        assert!(result
            .warnings
            .iter()
            .any(|w| w.contains("duplicate_epsilon")));
    }

    #[test]
    fn compare_delta_pct_is_none_when_baseline_is_zero() {
        let baseline = sample_report(0.05);
        let current = sample_report(0.05);
        let result = compare(&baseline, &current);
        for (key, delta) in &result.deltas {
            if delta.baseline == 0.0 {
                assert!(delta.delta_pct.is_none(), "{key} should have no delta_pct");
            }
        }
    }
}
