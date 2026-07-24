use crate::knn::{normalize_l2_with_report, VecHealthError, VecHealthEvaluator};
use ndarray::{ArrayView2, Axis};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct QmasResult {
    pub mean_1nn_distance: f32,
    pub mean_knn_distance: f32,
    pub orphans_fraction: f32,
}

#[inline(always)]
fn insert_top_k(top_k: &mut [f32], val: f32) {
    if val <= top_k[0] {
        return;
    }
    top_k[0] = val;
    let mut i = 0;
    while i + 1 < top_k.len() && top_k[i] > top_k[i + 1] {
        top_k.swap(i, i + 1);
        i += 1;
    }
}

pub fn compute_qmas_score(
    evaluator: &mut VecHealthEvaluator,
    queries: ArrayView2<f32>,
    k: usize,
    batch_size: usize,
) -> Result<QmasResult, VecHealthError> {
    let num_queries = queries.nrows();
    let num_docs = evaluator.n_vectors();

    if num_queries == 0 || num_docs == 0 {
        return Err(VecHealthError::EmptyInput);
    }
    if queries.ncols() != evaluator.dim {
        return Err(VecHealthError::DimensionMismatch {
            expected: evaluator.dim,
            found: queries.ncols(),
        });
    }
    if k == 0 {
        return Err(VecHealthError::KTooSmall { k, minimum: 1 });
    }
    if k > num_docs {
        return Err(VecHealthError::KTooLarge { k, n_vectors: num_docs });
    }

    let (norm_queries, _report) = normalize_l2_with_report(queries, 1e-3)?;
    let docs = evaluator.normalized_vectors()?.clone();

    let effective_batch_size = if batch_size == 0 { 512 } else { batch_size };

    let (sum_1nn, sum_knn, orphan_count) = norm_queries
        .axis_chunks_iter(Axis(0), effective_batch_size)
        .into_par_iter()
        .map(|query_batch| {
            let mut batch_1nn_sum = 0.0f64;
            let mut batch_knn_sum = 0.0f64;
            let mut batch_orphans = 0usize;

            let similarities = query_batch.dot(&docs.t());
            let mut top_k = vec![f32::NEG_INFINITY; k];

            for row in similarities.rows() {
                top_k.fill(f32::NEG_INFINITY);
                for &sim in row.iter() {
                    insert_top_k(&mut top_k, sim);
                }

                let best_sim = top_k[k - 1];
                let dist_1nn = (1.0 - best_sim).clamp(0.0, 2.0);

                let knn_dist_sum: f32 = top_k
                    .iter()
                    .map(|&sim| (1.0 - sim).clamp(0.0, 2.0))
                    .sum();

                batch_1nn_sum += dist_1nn as f64;
                batch_knn_sum += (knn_dist_sum / k as f32) as f64;

                if dist_1nn > 0.3 {
                    batch_orphans += 1;
                }
            }

            (batch_1nn_sum, batch_knn_sum, batch_orphans)
        })
        .reduce(
            || (0.0, 0.0, 0),
            |(a1, b1, c1), (a2, b2, c2)| (a1 + a2, b1 + b2, c1 + c2),
        );

    Ok(QmasResult {
        mean_1nn_distance: (sum_1nn / num_queries as f64) as f32,
        mean_knn_distance: (sum_knn / num_queries as f64) as f32,
        orphans_fraction: orphan_count as f32 / num_queries as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn hand_computed_query_alignment() {
        let docs = array![[1.0f32, 0.0], [0.0, 1.0]];
        let queries = array![[0.8f32, 0.6]];

        let mut evaluator = VecHealthEvaluator::new(docs).unwrap();
        let result = compute_qmas_score(&mut evaluator, queries.view(), 1, 10).unwrap();

        assert!((result.mean_1nn_distance - 0.2).abs() < 1e-4);
        assert_eq!(result.orphans_fraction, 0.0);
    }

    #[test]
    fn orphaned_query_far_from_all_docs() {
        let docs = array![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let queries = array![[0.0f32, 0.0, 1.0]];

        let mut evaluator = VecHealthEvaluator::new(docs).unwrap();
        let result = compute_qmas_score(&mut evaluator, queries.view(), 1, 10).unwrap();

        assert!((result.mean_1nn_distance - 1.0).abs() < 1e-4);
        assert_eq!(result.orphans_fraction, 1.0);
    }

    #[test]
    fn k_larger_than_docs_returns_error() {
        let docs = array![[1.0f32, 0.0]];
        let queries = array![[1.0f32, 0.0]];
        let mut evaluator = VecHealthEvaluator::new(docs).unwrap();

        assert!(matches!(
            compute_qmas_score(&mut evaluator, queries.view(), 5, 10),
            Err(VecHealthError::KTooLarge { .. })
        ));
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let docs = array![[1.0f32, 0.0, 0.0]];
        let queries = array![[1.0f32, 0.0]]; // 2 wymiary zamiast 3
        let mut evaluator = VecHealthEvaluator::new(docs).unwrap();

        assert!(matches!(
            compute_qmas_score(&mut evaluator, queries.view(), 1, 10),
            Err(VecHealthError::DimensionMismatch { .. })
        ));
    }
}