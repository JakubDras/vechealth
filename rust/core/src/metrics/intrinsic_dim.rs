use crate::knn::{VecHealthError, VecHealthEvaluator};
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct IntrinsicDimResult {
    pub mean_id: f32,
    pub median_id: f32,
}

pub fn compute_intrinsic_dim_score(
    evaluator: &mut VecHealthEvaluator,
    k: usize,
    batch_size: usize,
) -> Result<IntrinsicDimResult, VecHealthError> {
    if k < 2 {
        return Err(VecHealthError::KTooLarge {
            k,
            n_vectors: evaluator.n_vectors(),
        });
    }
    
    let (distances, _) = evaluator.get_knn(k, batch_size)?;
    let n = distances.nrows();

    if n == 0 {
        return Ok(IntrinsicDimResult {
            mean_id: 0.0,
            median_id: 0.0,
        });
    }
    
    let mut id_per_point: Vec<f32> = distances
        .axis_iter(ndarray::Axis(0))
        .into_par_iter()
        .map(|row| {
            let tk = row[k - 1].max(1e-10); // Dystans do k-tego sąsiada
            let mut sum_log_ratio = 0.0f32;

            for j in 0..(k - 1) {
                let tj = row[j].max(1e-10); // Dystans do j-tego sąsiada (1..k-1)
                sum_log_ratio += (tk / tj).ln();
            }

            if sum_log_ratio > 1e-5 {
                (k - 1) as f32 / sum_log_ratio
            } else {
                0.0
            }
        })
        .collect();
    
    let sum_id: f32 = id_per_point.par_iter().sum();
    let mean_id = sum_id / n as f32;
    
    id_per_point.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median_id = if n % 2 == 1 {
        id_per_point[n / 2]
    } else {
        (id_per_point[n / 2 - 1] + id_per_point[n / 2]) / 2.0
    };

    Ok(IntrinsicDimResult { mean_id, median_id })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_intrinsic_dimensionality_computation() {
        let vectors = array![
            [1.0f32, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.7, 0.3, 0.0, 0.0],
        ];

        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let result = compute_intrinsic_dim_score(&mut evaluator, 3, 10).unwrap();

        assert!(result.mean_id > 0.0);
        assert!(result.median_id > 0.0);
    }
}