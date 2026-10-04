use crate::knn::{l2_norms_f64, normalize_l2_with_report, VecHealthError, VecHealthEvaluator};
use ndarray::{ArrayView2, Axis};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct QmasResult {
    pub mean_1nn_distance: f32,
    pub mean_knn_distance: f32,
    pub orphans_fraction: f32,
}

#[inline(always)]
fn insert_top_k(top_k: &mut [(f32, u32)], val: f32, idx: u32) {
    if val <= top_k[0].0 {
        return;
    }
    top_k[0] = (val, idx);
    let mut i = 0;
    while i + 1 < top_k.len() && top_k[i].0 > top_k[i + 1].0 {
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
    // f64 norms of the queries and the documents, for an exact 1 - cos.
    let query_norms = l2_norms_f64(queries);
    let doc_vectors = evaluator.vectors();
    let doc_norms = l2_norms_f64(doc_vectors);

    let effective_batch_size = if batch_size == 0 { 512 } else { batch_size };

    let batch_results: Vec<(f64, f64, usize)> = norm_queries
        .axis_chunks_iter(Axis(0), effective_batch_size)
        .into_par_iter()
        .enumerate()
        .map(|(chunk_idx, query_batch)| {
            let mut batch_1nn_sum = 0.0f64;
            let mut batch_knn_sum = 0.0f64;
            let mut batch_orphans = 0usize;
            let batch_start = chunk_idx * effective_batch_size;

            let similarities = query_batch.dot(&docs.t());
            let mut top_k = vec![(f32::NEG_INFINITY, u32::MAX); k];

            for (local_row, row) in similarities.rows().into_iter().enumerate() {
                top_k.fill((f32::NEG_INFINITY, u32::MAX));
                for (idx, &sim) in row.iter().enumerate() {
                    insert_top_k(&mut top_k, sim, idx as u32);
                }

                let q_idx = batch_start + local_row;
                let q = queries.row(q_idx);
                let nq = query_norms[q_idx];
                let mut dist_1nn = f64::INFINITY;
                let mut knn_dist_sum = 0.0f64;
                for &(_sim, doc_idx) in top_k.iter() {
                    let j = doc_idx as usize;
                    let nd = doc_norms[j];
                    let cos = if nq == 0.0 || nd == 0.0 {
                        0.0
                    } else {
                        q.iter()
                            .zip(doc_vectors.row(j).iter())
                            .map(|(&a, &b)| (a as f64) * (b as f64))
                            .sum::<f64>()
                            / (nq * nd)
                    };
                    let dist = (1.0 - cos).clamp(0.0, 2.0);
                    knn_dist_sum += dist;
                    if dist < dist_1nn {
                        dist_1nn = dist;
                    }
                }

                batch_1nn_sum += dist_1nn;
                batch_knn_sum += knn_dist_sum / k as f64;

                if dist_1nn > 0.3 {
                    batch_orphans += 1;
                }
            }

            (batch_1nn_sum, batch_knn_sum, batch_orphans)
        })
        .collect();
    let (sum_1nn, sum_knn, orphan_count) = batch_results
        .iter()
        .fold((0.0f64, 0.0f64, 0usize), |(a1, b1, c1), &(a2, b2, c2)| (a1 + a2, b1 + b2, c1 + c2));

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
        let queries = array![[1.0f32, 0.0]]; // 2 dimensions instead of 3
        let mut evaluator = VecHealthEvaluator::new(docs).unwrap();

        assert!(matches!(
            compute_qmas_score(&mut evaluator, queries.view(), 1, 10),
            Err(VecHealthError::DimensionMismatch { .. })
        ));
    }


    #[test]
    fn near_identical_query_distance_matches_f64() {
        let docs = array![[0.6f32, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let queries = array![[0.6f32, 0.800_1, 0.0, 0.0]];
        let (q0, q1) = (0.6f32 as f64, 0.800_1f32 as f64);
        let (d0, d1) = (0.6f32 as f64, 0.8f32 as f64);
        let cos = (q0 * d0 + q1 * d1) / ((q0 * q0 + q1 * q1).sqrt() * (d0 * d0 + d1 * d1).sqrt());
        let expected = 1.0 - cos;
        let mut evaluator = VecHealthEvaluator::new(docs).unwrap();
        let r = compute_qmas_score(&mut evaluator, queries.view(), 2, 8).unwrap();
        assert!(
            ((r.mean_1nn_distance as f64) - expected).abs() <= 1e-6 * expected + 1e-12,
            "got {}, expected {}",
            r.mean_1nn_distance,
            expected
        );
    }
}
