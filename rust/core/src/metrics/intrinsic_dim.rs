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
    
    let mut id_per_point: Vec<f64> = distances
        .axis_iter(ndarray::Axis(0))
        .into_par_iter()
        .map(|row| {
            let tk = (row[k - 1] as f64).max(1e-10);
            let mut sum_log_ratio = 0.0f64;

            for j in 0..(k - 1) {
                let tj = (row[j] as f64).max(1e-10);
                sum_log_ratio += (tk / tj).ln();
            }

            if sum_log_ratio > 1e-5 {
                (k - 1) as f64 / sum_log_ratio
            } else {
                0.0
            }
        })
        .collect();
    
    let sum_id: f64 = id_per_point.iter().sum();
    let mean_id = (sum_id / n as f64) as f32;
    
    id_per_point.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median_id = if n % 2 == 1 {
        id_per_point[n / 2]
    } else {
        (id_per_point[n / 2 - 1] + id_per_point[n / 2]) / 2.0
    } as f32;

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
    
    #[test]
    fn exactly_equidistant_neighbourhood_gives_zero_id_for_the_centre() {
        let (dim, m) = (32usize, 20usize);
        let delta = 2.0f32.powi(-7);
        let mut data = vec![0.0f32; (m + 1) * dim];
        data[0] = 1.0;
        for i in 1..=m {
            data[i * dim] = 1.0;
            data[i * dim + i] = delta;
        }
        let vectors = ndarray::Array2::from_shape_vec((m + 1, dim), data).unwrap();
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let r = compute_intrinsic_dim_score(&mut evaluator, 20, 8).unwrap();
        let dl = delta as f64;
        let sc = (1.0 + dl * dl).sqrt();
        let d_cp = ((1.0 - 1.0 / sc).powi(2) + (dl / sc).powi(2)).sqrt();
        let d_pp = 2.0f64.sqrt() * dl / sc;
        let expected = (m as f64 / (m + 1) as f64) * 19.0 / (d_pp / d_cp).ln();
        assert!(
            ((r.mean_id as f64 - expected) / expected).abs() < 1e-5,
            "mean_id {} expected {}",
            r.mean_id,
            expected
        );
    }
    
    #[test]
    fn intrinsic_dim_matches_f64_brute_force() {
        use rand::{Rng, SeedableRng};
        let (n, dim, k) = (300usize, 16usize, 20usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let vectors = ndarray::Array2::from_shape_vec((n, dim), data).unwrap();
        let rows: Vec<Vec<f64>> = vectors
            .rows()
            .into_iter()
            .map(|r| {
                let nr = r.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                r.iter().map(|&x| x as f64 / nr).collect()
            })
            .collect();
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let mut d: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| rows[i].iter().zip(rows[j].iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt())
                .collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let tk = d[k - 1].max(1e-10);
            let s: f64 = (0..k - 1).map(|j| (tk / d[j].max(1e-10)).ln()).sum();
            ids.push(if s > 1e-5 { (k - 1) as f64 / s } else { 0.0 });
        }
        let reference = ids.iter().sum::<f64>() / n as f64;
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let r = compute_intrinsic_dim_score(&mut evaluator, k, 32).unwrap();
        assert!(((r.mean_id as f64 - reference) / reference).abs() < 1e-5, "{} vs {}", r.mean_id, reference);
    }
}
