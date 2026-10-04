use crate::knn::{VecHealthError, VecHealthEvaluator};

#[derive(Debug, Clone)]
pub struct DispersionResult {
    pub mean_1nn_distance: f32,
    pub mean_knn_distance: f32,
}

pub(crate) fn mean_f64(values: impl Iterator<Item = f32>) -> f64 {
    let (sum, count) = values.fold((0.0f64, 0usize), |(s, c), v| (s + v as f64, c + 1));
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

pub fn compute_dispersion_score(
    evaluator: &mut VecHealthEvaluator,
    k: usize,
    batch_size: usize,
) -> Result<DispersionResult, VecHealthError> {
    let (distances, _) = evaluator.get_knn(k, batch_size)?;
    
    let mean_1nn_distance = mean_f64(distances.column(0).iter().copied()) as f32;
    let mean_knn_distance = mean_f64(distances.iter().copied()) as f32;

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
    
    #[test]
    fn narrow_distribution_mean_matches_f64_reference() {
        let n = 2_000_000usize;
        let values: Vec<f32> = (0..n)
            .map(|i| (1.3 + 0.007 * ((i as f64) * 0.618_033_988_7).sin()) as f32)
            .collect();
        let (mut s, mut c) = (0.0f64, 0.0f64);
        for &v in &values {
            let y = v as f64 - c;
            let t = s + y;
            c = (t - s) - y;
            s = t;
        }
        let reference = s / n as f64;
        let got = mean_f64(values.iter().copied());
        assert!(((got - reference) / reference).abs() < 1e-12, "got {got}, reference {reference}");
        let naive_f32 = values.iter().fold(0.0f32, |a, &v| a + v) / n as f32;
        assert!(((naive_f32 as f64 - reference) / reference).abs() > 1e-5);
    }

    #[test]
    fn dispersion_matches_f64_brute_force() {
        use rand::{Rng, SeedableRng};
        let (n, dim, k) = (400usize, 24usize, 10usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0) * 1.002).collect();
        let vectors = ndarray::Array2::from_shape_vec((n, dim), data).unwrap();
        let rows: Vec<Vec<f64>> = vectors
            .rows()
            .into_iter()
            .map(|r| {
                let nr = r.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                r.iter().map(|&x| x as f64 / nr).collect()
            })
            .collect();
        let (mut s1, mut sk) = (0.0f64, 0.0f64);
        for i in 0..n {
            let mut d: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| rows[i].iter().zip(rows[j].iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt())
                .collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s1 += d[0];
            sk += d[..k].iter().sum::<f64>();
        }
        let (ref1, refk) = (s1 / n as f64, sk / (n * k) as f64);
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let r = compute_dispersion_score(&mut evaluator, k, 64).unwrap();
        assert!(((r.mean_1nn_distance as f64 - ref1) / ref1).abs() < 1e-6);
        assert!(((r.mean_knn_distance as f64 - refk) / refk).abs() < 1e-6);
    }
}
