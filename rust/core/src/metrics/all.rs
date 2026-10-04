use crate::knn::{VecHealthError, VecHealthEvaluator};
use crate::metrics::{
    anisotropy, anisotropy::AnisotropyResult, duplicates, duplicates::DuplicatesResult,
    fragmentation, fragmentation::DispersionResult, hubness, hubness::HubnessResult,
    intrinsic_dim, intrinsic_dim::IntrinsicDimResult, outliers, outliers::OutliersResult, qmas,
    qmas::QmasResult, snc, snc::SncResult,
};
use crate::resources::DEFAULT_BATCH_SIZE;
use ndarray::ArrayView2;

/// Default outlier threshold (chord distance) — the value passed explicitly
/// for every result reported in the accompanying preprint.
pub const DEFAULT_OUTLIER_DISTANCE_THRESHOLD: f32 = 1.2;

/// Parameters for `compute_all_metrics`. `Default` mirrors the defaults each
/// metric has when called on its own (see `rust/bindings/src/lib.rs`), so
/// `compute_all_metrics` with the default config is identical to calling
/// every metric by hand with its own defaults.
#[derive(Debug, Clone)]
pub struct AllMetricsConfig {
    /// Neighbourhood size for hubness / dispersion / snc.
    pub k: usize,
    /// Neighbourhood size for intrinsic_dim — inherently needs a wider window.
    pub k_intrinsic_dim: usize,
    /// Rows per block of the k-NN search. Peak memory grows with it (see
    /// `crate::resources`); results do not depend on it.
    pub batch_size: usize,
    pub duplicate_epsilon: f32,
    /// Outlier threshold: a point is an outlier when its nearest neighbour
    /// is farther than this chord distance (distances live on the unit
    /// sphere, so they never exceed 2). `Some(1.2)` is the value used for
    /// the results in the accompanying preprint; it is absolute, so it must
    /// be recalibrated for other encoders or dimensionalities. `None`
    /// derives the threshold from the data as 3x the mean nearest-neighbour
    /// distance — for typical high-dimensional text embeddings that lands
    /// close to or above 2.0, the largest possible chord distance, where
    /// essentially no point can be flagged.
    pub outlier_distance_threshold: Option<f32>,
}

impl Default for AllMetricsConfig {
    fn default() -> Self {
        Self {
            k: 10,
            k_intrinsic_dim: 20,
            batch_size: DEFAULT_BATCH_SIZE,
            duplicate_epsilon: 0.05,
            outlier_distance_threshold: Some(DEFAULT_OUTLIER_DISTANCE_THRESHOLD),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AllMetricsResult {
    pub hubness: HubnessResult,
    pub dispersion: DispersionResult,
    pub anisotropy: AnisotropyResult,
    pub outliers: OutliersResult,
    pub duplicates: DuplicatesResult,
    pub intrinsic_dim: IntrinsicDimResult,
    pub snc: SncResult,
    /// `None` if no `queries` were given — QMAS measures how well queries
    /// align with the document space, and without queries there is nothing to
    /// compute.
    pub qmas: Option<QmasResult>,
}

/// Orchestrator: computes every implemented geometric metric on one evaluator
/// in a single pass. The evaluator's internal k-NN cache (`get_knn`) is shared
/// between the metrics that use the same k, so e.g. hubness/dispersion/snc at
/// the default k=10 do not recompute the k-NN three times.
pub fn compute_all_metrics(
    evaluator: &mut VecHealthEvaluator,
    config: &AllMetricsConfig,
    queries: Option<ArrayView2<f32>>,
) -> Result<AllMetricsResult, VecHealthError> {
    let dispersion = fragmentation::compute_dispersion_score(evaluator, config.k, config.batch_size)?;
    let hubness = hubness::compute_hubness_score(evaluator, config.k, config.batch_size)?;
    let anisotropy = anisotropy::compute_anisotropy_score(evaluator)?;

    let outlier_threshold = config
        .outlier_distance_threshold
        .unwrap_or(dispersion.mean_1nn_distance * 3.0);
    let outliers = outliers::compute_outlier_score(evaluator, outlier_threshold, config.batch_size)?;

    let duplicates =
        duplicates::compute_ndds_score(evaluator, config.duplicate_epsilon, config.batch_size)?;
    let intrinsic_dim = intrinsic_dim::compute_intrinsic_dim_score(
        evaluator,
        config.k_intrinsic_dim,
        config.batch_size,
    )?;
    let snc = snc::compute_snc_score(evaluator, config.k, config.batch_size)?;

    let qmas = match queries {
        Some(q) => Some(qmas::compute_qmas_score(evaluator, q, config.k, config.batch_size)?),
        None => None,
    };

    Ok(AllMetricsResult {
        hubness,
        dispersion,
        anisotropy,
        outliers,
        duplicates,
        intrinsic_dim,
        snc,
        qmas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn runs_every_metric_with_defaults() {
        // 15 vectors, so that every k used below stays smaller than
        // n_vectors and no KTooLarge error is raised.
        let vectors: Vec<[f32; 4]> = (0..15)
            .map(|i| {
                let x = i as f32;
                [x, x * 0.5, -x * 0.2, (x % 3.0)]
            })
            .collect();
        let vectors = ndarray::Array2::from_shape_vec(
            (15, 4),
            vectors.into_iter().flatten().collect(),
        )
        .unwrap();

        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let config = AllMetricsConfig {
            k: 5,
            k_intrinsic_dim: 5,
            ..AllMetricsConfig::default()
        };

        let result = compute_all_metrics(&mut evaluator, &config, None).unwrap();
        assert!(result.qmas.is_none());
        assert!(result.hubness.max_occurrences > 0);
        assert!(result.dispersion.mean_1nn_distance >= 0.0);
        assert!(result.outliers.outlier_fraction >= 0.0);
    }

    #[test]
    fn includes_qmas_when_queries_given() {
        let docs = array![[1.0f32, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]];
        let queries = array![[0.8f32, 0.6]];
        let mut evaluator = VecHealthEvaluator::new(docs).unwrap();
        let config = AllMetricsConfig {
            k: 2,
            k_intrinsic_dim: 2,
            ..AllMetricsConfig::default()
        };

        let result = compute_all_metrics(&mut evaluator, &config, Some(queries.view())).unwrap();
        assert!(result.qmas.is_some());
    }

    #[test]
    fn default_config_uses_the_preprint_outlier_threshold() {
        assert_eq!(DEFAULT_OUTLIER_DISTANCE_THRESHOLD, 1.2);
        assert_eq!(
            AllMetricsConfig::default().outlier_distance_threshold,
            Some(DEFAULT_OUTLIER_DISTANCE_THRESHOLD)
        );
    }

    /// In high dimensions random vectors are nearly orthogonal, so every
    /// nearest neighbour sits at a chord distance of about 1.3. The default
    /// threshold (1.2) sees that; the data-derived one (3x the mean 1-NN
    /// distance, here ~3.9) lies above the largest possible chord distance
    /// (2.0), so it flags nothing. This is why the adaptive mode is opt-in
    /// (`None`) rather than the default.
    #[test]
    fn adaptive_threshold_is_blind_on_high_dimensional_data_but_the_default_is_not() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let (n, dim) = (100usize, 256usize);
        let mut rng = StdRng::seed_from_u64(1);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let vectors = ndarray::Array2::from_shape_vec((n, dim), data).unwrap();

        let mut with_default = VecHealthEvaluator::new(vectors.clone()).unwrap();
        let flagged = compute_all_metrics(&mut with_default, &AllMetricsConfig::default(), None)
            .unwrap()
            .outliers
            .outlier_fraction;

        let adaptive_config = AllMetricsConfig {
            outlier_distance_threshold: None,
            ..AllMetricsConfig::default()
        };
        let mut with_adaptive = VecHealthEvaluator::new(vectors).unwrap();
        let blind = compute_all_metrics(&mut with_adaptive, &adaptive_config, None)
            .unwrap()
            .outliers
            .outlier_fraction;

        assert!(flagged > 0.9, "default threshold flagged only {flagged}");
        assert_eq!(blind, 0.0);
    }
}
