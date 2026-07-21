use crate::knn::{VecHealthError, VecHealthEvaluator};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct OutliersResult {
    pub outlier_fraction: f32,
    pub max_1nn_distance: f32,
    pub std_1nn_distance: f32,
}

pub fn compute_outlier_score(
    evaluator: &mut VecHealthEvaluator,
    distance_threshold: f32,
    batch_size: usize,
) -> Result<OutliersResult, VecHealthError> {
    let (distances, _) = evaluator.get_knn(1, batch_size)?;
    let closest_distances = distances.column(0);
    let n = closest_distances.len();

    if n == 0 {
        return Ok(OutliersResult {
            outlier_fraction: 0.0,
            max_1nn_distance: 0.0,
            std_1nn_distance: 0.0,
        });
    }

    let (outliers_count, max_1nn_distance, sum_distance, sum_sq_distance) = closest_distances
        .into_par_iter()
        .map(|&dist| {
            let is_outlier = if dist > distance_threshold { 1u32 } else { 0u32 };
            (is_outlier, dist, dist, dist * dist)
        })
        .reduce(
            || (0u32, f32::MIN, 0.0f32, 0.0f32),
            |a, b| (a.0 + b.0, a.1.max(b.1), a.2 + b.2, a.3 + b.3),
        );

    let mean_distance = sum_distance / n as f32;
    let outlier_fraction = outliers_count as f32 / n as f32;
    let max_1nn_distance = if max_1nn_distance == f32::MIN { 0.0 } else { max_1nn_distance };

    let variance = (sum_sq_distance / n as f32 - mean_distance * mean_distance).max(0.0);
    let std_1nn_distance = variance.sqrt();

    Ok(OutliersResult {
        outlier_fraction,
        max_1nn_distance,
        std_1nn_distance,
    })
}