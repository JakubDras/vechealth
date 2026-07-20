use crate::knn::{VecHealthError, VecHealthEvaluator};

#[derive(Debug, Clone)]
pub struct DispersionResult {
    pub mean_1nn_distance: f32,
    pub mean_knn_distance: f32,
}

pub fn compute_dispersion_score(
    evaluator: &mut VecHealthEvaluator,
    k: usize,
    batch_size: usize,
) -> Result<DispersionResult, VecHealthError> {
    let (distances, _) = evaluator.get_knn(k, batch_size)?;

    let mean_1nn_distance = distances.column(0).mean().unwrap_or(0.0);
    let mean_knn_distance = distances.mean().unwrap_or(0.0);

    Ok(DispersionResult {
        mean_1nn_distance,
        mean_knn_distance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn hand_computed_triangle_dispersion() {
        let vectors = array![
            [1.0f32, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let result = compute_dispersion_score(&mut evaluator, 2, 10).unwrap();

        assert!((result.mean_1nn_distance - 0.719779).abs() < 1e-3);
        assert!((result.mean_knn_distance - 0.980365).abs() < 1e-3);
    }

    #[test]
    fn identical_neighbors_have_zero_dispersion() {
        let vectors = array![
            [1.0f32, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let result = compute_dispersion_score(&mut evaluator, 2, 10).unwrap();

        assert!(result.mean_1nn_distance.abs() < 1e-5);
        assert!(result.mean_knn_distance.abs() < 1e-5);
    }
}