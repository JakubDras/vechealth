use crate::knn::{VecHealthError, VecHealthEvaluator};
use ndarray::Axis;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct SncResult {
    pub mean_snc_score: f32,
}

pub fn compute_snc_score(
    evaluator: &mut VecHealthEvaluator,
    k: usize,
    batch_size: usize,
) -> Result<SncResult, VecHealthError> {
    let (_, indices) = evaluator.get_knn(k, batch_size)?;
    let n = indices.nrows();

    if n == 0 || k == 0 {
        return Ok(SncResult { mean_snc_score: 0.0 });
    }

    let mut sorted_neighbors = indices.to_owned();
    sorted_neighbors
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let slice = row.as_slice_mut().expect("The row is continuous.");
            slice.sort_unstable();
        });

    let full_slice = sorted_neighbors
        .as_slice()
        .expect("The matrix is contiguous in memory.");

    let sum_snc: f32 = (0..n)
        .into_par_iter()
        .map(|idx| {
            let my_start = idx * k;
            let my_neighbors = &full_slice[my_start..my_start + k];

            if my_neighbors.is_empty() {
                return 0.0f32;
            }

            let mut local_jaccard_sum = 0.0f32;

            for &neighbor_idx in my_neighbors {
                let n_idx = neighbor_idx as usize;
                if n_idx >= n {
                    continue;
                }

                let n_start = n_idx * k;
                let neighbor_neighbors = &full_slice[n_start..n_start + k];

                let intersection = count_intersection(my_neighbors, neighbor_neighbors);
                let union = my_neighbors.len() + neighbor_neighbors.len() - intersection;

                if union > 0 {
                    local_jaccard_sum += intersection as f32 / union as f32;
                }
            }

            local_jaccard_sum / my_neighbors.len() as f32
        })
        .sum();

    Ok(SncResult {
        mean_snc_score: sum_snc / n as f32,
    })
}

#[inline(always)]
fn count_intersection(a: &[u32], b: &[u32]) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_snc_small_cluster_exact_jaccard() {
        let vectors = array![
            [1.0f32, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
        ];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let result = compute_snc_score(&mut evaluator, 2, 10).unwrap();

        assert!((result.mean_snc_score - (1.0 / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn test_snc_larger_cluster_higher_consistency() {
        let vectors = array![
            [1.0f32, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.96, 0.04],
        ];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let result = compute_snc_score(&mut evaluator, 4, 10).unwrap();

        assert!((result.mean_snc_score - 0.6).abs() < 1e-4);
    }

    #[test]
    fn test_snc_disjoint_clusters_have_lower_consistency() {
        let vectors = array![
            [1.0f32, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let result = compute_snc_score(&mut evaluator, 2, 10).unwrap();

        assert!(result.mean_snc_score >= 0.0 && result.mean_snc_score <= 1.0);
    }
}