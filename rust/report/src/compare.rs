//! Baseline comparison: a per-metric delta between two `Report`s, plus
//! warnings about anything that would make the comparison misleading.
//!
//! Deliberately stops at raw deltas — no health thresholds, no
//! pass/fail verdict. That classification belongs to the separate
//! "interpretation layer" work (TODO_List.md's 🔴 item), which doesn't
//! exist yet; baking judgment calls in here would be scope creep into a
//! task that hasn't been designed.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::Report;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDelta {
    pub baseline: f64,
    pub current: f64,
    /// `current - baseline`.
    pub delta: f64,
    /// `None` when `baseline` is `0.0` — a percentage change from zero is
    /// undefined, not infinite-and-worth-reporting.
    pub delta_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub baseline_generated_at: DateTime<Utc>,
    pub current_generated_at: DateTime<Utc>,
    pub deltas: BTreeMap<String, MetricDelta>,
    pub warnings: Vec<String>,
}

/// Compares `current` against `baseline`. A metric present in only one of
/// the two reports (e.g. `qmas.*`, computed only when queries were passed)
/// is skipped from `deltas` and noted in `warnings` instead of silently
/// dropped.
pub fn compare(baseline: &Report, current: &Report) -> ComparisonResult {
    let baseline_flat = baseline.metrics.flatten();
    let current_flat = current.metrics.flatten();

    let mut deltas = BTreeMap::new();
    let mut warnings = Vec::new();

    for (key, &current_value) in &current_flat {
        match baseline_flat.get(key) {
            Some(&baseline_value) => {
                let delta = current_value - baseline_value;
                let delta_pct = if baseline_value != 0.0 {
                    Some(delta / baseline_value * 100.0)
                } else {
                    None
                };
                deltas.insert(
                    key.clone(),
                    MetricDelta {
                        baseline: baseline_value,
                        current: current_value,
                        delta,
                        delta_pct,
                    },
                );
            }
            None => warnings.push(format!(
                "metric '{key}' is present in the current report but not in the baseline \
                 — skipped from the comparison"
            )),
        }
    }
    for key in baseline_flat.keys() {
        if !current_flat.contains_key(key) {
            warnings.push(format!(
                "metric '{key}' is present in the baseline report but not in the current one \
                 — skipped from the comparison"
            ));
        }
    }

    if baseline.dataset.dim != current.dataset.dim {
        warnings.push(format!(
            "dataset dimensionality differs (baseline dim={}, current dim={}) — the two \
             reports likely describe different embedding models, so this comparison may not \
             be meaningful",
            baseline.dataset.dim, current.dataset.dim
        ));
    }
    if baseline.config.k != current.config.k {
        warnings.push(format!(
            "k differs between reports (baseline k={}, current k={}) — hubness/dispersion/snc \
             were computed with different neighborhood sizes and aren't directly comparable",
            baseline.config.k, current.config.k
        ));
    }
    if baseline.config.k_intrinsic_dim != current.config.k_intrinsic_dim {
        warnings.push(format!(
            "k_intrinsic_dim differs between reports (baseline={}, current={}) — \
             intrinsic_dim values aren't directly comparable",
            baseline.config.k_intrinsic_dim, current.config.k_intrinsic_dim
        ));
    }
    if (baseline.config.duplicate_epsilon - current.config.duplicate_epsilon).abs() > f32::EPSILON {
        warnings.push(format!(
            "duplicate_epsilon differs between reports (baseline={}, current={}) — \
             duplicates.ndds_fraction values aren't directly comparable",
            baseline.config.duplicate_epsilon, current.config.duplicate_epsilon
        ));
    }

    ComparisonResult {
        baseline_generated_at: baseline.generated_at,
        current_generated_at: current.generated_at,
        deltas,
        warnings,
    }
}
