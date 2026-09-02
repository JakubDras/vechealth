use crate::knn::{VecHealthError, VecHealthEvaluator};
use crate::metrics::{
    anisotropy, anisotropy::AnisotropyResult, duplicates, duplicates::DuplicatesResult,
    fragmentation, fragmentation::DispersionResult, hubness, hubness::HubnessResult,
    intrinsic_dim, intrinsic_dim::IntrinsicDimResult, outliers, outliers::OutliersResult, qmas,
    qmas::QmasResult, snc, snc::SncResult,
};
use ndarray::ArrayView2;

/// Parametry dla `compute_all_metrics`. `Default` odzwierciedla te same
/// wartości domyślne, które mają poszczególne metryki wywoływane osobno
/// (patrz `rust/bindings/src/lib.rs`), więc wynik `compute_all_metrics` z
/// domyślnym configiem jest identyczny z ręcznym wywołaniem każdej metryki
/// z jej własnymi domyślnymi parametrami.
#[derive(Debug, Clone)]
pub struct AllMetricsConfig {
    /// k dla hubness / dispersion / snc.
    pub k: usize,
    /// k dla intrinsic_dim — z natury wymaga szerszego okna sąsiedztwa.
    pub k_intrinsic_dim: usize,
    pub batch_size: usize,
    pub duplicate_epsilon: f32,
    /// Próg dystansu dla outliers. `None` => heurystyka: 3x średni dystans
    /// do najbliższego sąsiada (z metryki dispersion), czyli punkt jest
    /// "outlierem" gdy jego najbliższy sąsiad jest > 3x dalej niż typowo
    /// w tym zbiorze. Ustaw jawnie, jeśli znasz sensowny próg dla swoich danych.
    pub outlier_distance_threshold: Option<f32>,
}

impl Default for AllMetricsConfig {
    fn default() -> Self {
        Self {
            k: 10,
            k_intrinsic_dim: 20,
            batch_size: 2048,
            duplicate_epsilon: 0.05,
            outlier_distance_threshold: None,
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
    /// `None` jeśli `queries` nie zostały podane — QMAS mierzy dopasowanie
    /// zapytań do przestrzeni dokumentów i bez zapytań nie ma czego liczyć.
    pub qmas: Option<QmasResult>,
}

/// Orkiestrator: liczy komplet już zaimplementowanych metryk geometrycznych
/// na jednym evaluatorze, w jednym przebiegu. Wewnętrzny cache KNN
/// evaluatora (`get_knn`) jest współdzielony między metrykami, które używają
/// tego samego k, więc np. hubness/dispersion/snc przy domyślnym k=10 nie
/// przeliczają KNN trzykrotnie.
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
        // 15 wektorów, żeby default k=10 (hubness/dispersion/snc) i
        // k_intrinsic_dim=20->clamp wymagały k < n_vectors bez błędu KTooLarge.
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
}
