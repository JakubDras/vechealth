use crate::knn::{VecHealthError, VecHealthEvaluator};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct DuplicatesResult {
    pub ndds_fraction: f32,
    pub mean_1nn_distance: f32,
    pub min_distance_global: f32,
}

pub fn compute_ndds_score(
    evaluator: &mut VecHealthEvaluator,
    epsilon: f32,
    batch_size: usize,
) -> Result<DuplicatesResult, VecHealthError> {
    let (distances, _) = evaluator.get_knn(1, batch_size)?;
    let closest_distances = distances.column(0);
    let n = closest_distances.len();

    if n == 0 {
        return Ok(DuplicatesResult {
            ndds_fraction: 0.0,
            mean_1nn_distance: 0.0,
            min_distance_global: 0.0,
        });
    }

    let (ndds_count, sum_distance, min_dist) = closest_distances
        .into_par_iter()
        .map(|&dist| {
            let is_duplicate = if dist < epsilon { 1u32 } else { 0u32 };
            (is_duplicate, dist, dist)
        })
        .reduce(
            || (0u32, 0.0f32, f32::MAX),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2.min(b.2)),
        );

    let ndds_fraction = ndds_count as f32 / n as f32;
    let mean_1nn_distance = sum_distance / n as f32;
    let min_distance_global = if min_dist == f32::MAX { 0.0 } else { min_dist };

    Ok(DuplicatesResult {
        ndds_fraction,
        mean_1nn_distance,
        min_distance_global,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_near_duplicates_detection() {
        let vectors = array![
            [1.0f32, 0.0, 0.0],
            [0.999, 0.01, 0.0],
            [0.0, 1.0, 0.0],
        ];

        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let result = compute_ndds_score(&mut evaluator, 0.05, 10).unwrap();

        assert!(result.ndds_fraction > 0.0);
        assert!(result.min_distance_global < 0.05);
    }

    #[test]
    fn test_no_duplicates_returns_zero_fraction() {
        let vectors = array![
            [1.0f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let result = compute_ndds_score(&mut evaluator, 0.05, 10).unwrap();

        assert_eq!(result.ndds_fraction, 0.0);
    }
}