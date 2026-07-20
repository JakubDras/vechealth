use ndarray::{s, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fmt;

#[derive(Debug)]
pub enum VecHealthError {
    EmptyInput,
    DimensionMismatch { expected: usize, found: usize },
    KTooLarge { k: usize, n_vectors: usize },
    AllVectorsDegenerate,
}

impl fmt::Display for VecHealthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "The input matrix of vectors is empty."),
            Self::DimensionMismatch { expected, found } => {
                write!(f, "Dimension mismatch: expected {}, received {}.", expected, found)
            }
            Self::KTooLarge { k, n_vectors } => {
                write!(f, "k={} was requested, but only {} vectors are available.", k, n_vectors)
            }
            Self::AllVectorsDegenerate => write!(f, "All vectors have zero norm."),
        }
    }
}

impl std::error::Error for VecHealthError {}

#[derive(Debug)]
pub struct NormalizationReport {
    pub is_fully_normalized: bool,
    pub fraction_non_normalized: f32,
    pub min_norm: f32,
    pub max_norm: f32,
    pub mean_norm: f32,
    pub degenerate_indices: Vec<usize>,
    pub fraction_degenerate: f32,
}

pub fn normalize_l2_with_report(
    vectors: ArrayView2<f32>,
    tolerance: f32,
) -> Result<(Array2<f32>, NormalizationReport), VecHealthError> {
    if vectors.nrows() == 0 {
        return Err(VecHealthError::EmptyInput);
    }
    let n = vectors.nrows();
    let mut normalized = vectors.to_owned();

    let per_row_stats: Vec<(f32, bool)> = normalized
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .map(|mut row| {
            let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm == 0.0 {
                (0.0, true)
            } else {
                row.mapv_inplace(|x| x / norm);
                (norm, false)
            }
        })
        .collect();

    let mut non_normalized_count = 0usize;
    let mut min_norm = f32::MAX;
    let mut max_norm = f32::MIN;
    let mut sum_norm = 0.0f32;
    let mut degenerate_indices = Vec::new();

    for (idx, &(norm, is_degenerate)) in per_row_stats.iter().enumerate() {
        if is_degenerate {
            degenerate_indices.push(idx);
            continue;
        }
        if (norm - 1.0).abs() > tolerance {
            non_normalized_count += 1;
        }
        min_norm = min_norm.min(norm);
        max_norm = max_norm.max(norm);
        sum_norm += norm;
    }

    let valid_count = n - degenerate_indices.len();
    if valid_count == 0 {
        return Err(VecHealthError::AllVectorsDegenerate);
    }

    let mean_norm = sum_norm / valid_count as f32;
    let is_fully_normalized = non_normalized_count == 0 && degenerate_indices.is_empty();
    let fraction_non_normalized = non_normalized_count as f32 / n as f32;
    let fraction_degenerate = degenerate_indices.len() as f32 / n as f32;

    Ok((
        normalized,
        NormalizationReport {
            is_fully_normalized,
            fraction_non_normalized,
            min_norm,
            max_norm,
            mean_norm,
            degenerate_indices,
            fraction_degenerate,
        },
    ))
}

struct KnnCache {
    k: usize,
    distances: Array2<f32>,
    indices: Array2<u32>,
}

pub struct VecHealthEvaluator {
    vectors: Array2<f32>,
    normalized_cache: Option<Array2<f32>>,
    normalization_report: Option<NormalizationReport>,
    n_vectors: usize,
    pub dim: usize,
    knn_cache: Option<KnnCache>,
}

impl VecHealthEvaluator {
    pub fn new(vectors: Array2<f32>) -> Result<Self, VecHealthError> {
        if vectors.nrows() == 0 {
            return Err(VecHealthError::EmptyInput);
        }
        let n_vectors = vectors.nrows();
        let dim = vectors.ncols();

        Ok(Self {
            vectors,
            normalized_cache: None,
            normalization_report: None,
            n_vectors,
            dim,
            knn_cache: None,
        })
    }

    fn ensure_normalized(&mut self) -> Result<(&Array2<f32>, &NormalizationReport), VecHealthError> {
        if self.normalized_cache.is_none() {
            let (normalized, report) = normalize_l2_with_report(self.vectors.view(), 1e-3)?;
            self.normalized_cache = Some(normalized);
            self.normalization_report = Some(report);
        }
        Ok((
            self.normalized_cache.as_ref().unwrap(),
            self.normalization_report.as_ref().unwrap(),
        ))
    }

    pub fn normalization_report(&mut self) -> Result<&NormalizationReport, VecHealthError> {
        let (_, report) = self.ensure_normalized()?;
        Ok(report)
    }

    pub fn get_original_vector(&self, index: usize) -> ArrayView1<'_, f32> {
        self.vectors.row(index)
    }

    pub fn n_vectors(&self) -> usize {
        self.n_vectors
    }

    pub fn vectors(&self) -> ArrayView2<'_, f32> {
        self.vectors.view()
    }

    pub fn get_knn(
        &mut self,
        k: usize,
        batch_size: usize,
    ) -> Result<(ArrayView2<'_, f32>, ArrayView2<'_, u32>), VecHealthError> {
        if k >= self.n_vectors {
            return Err(VecHealthError::KTooLarge { k, n_vectors: self.n_vectors });
        }

        let need_recompute = match &self.knn_cache {
            Some(cache) => cache.k < k,
            None => true,
        };

        if need_recompute {
            let (normalized, _report) = self.ensure_normalized()?;
            let normalized = normalized.clone();
            let (distances, indices) = blocked_topk_cosine(normalized.view(), k, batch_size)?;
            self.knn_cache = Some(KnnCache { k, distances, indices });
        }

        let cache = self.knn_cache.as_ref().unwrap();
        Ok((
            cache.distances.slice(s![.., ..k]),
            cache.indices.slice(s![.., ..k]),
        ))
    }
}

fn blocked_topk_cosine(
    normalized_vectors: ArrayView2<f32>,
    k: usize,
    batch_size: usize,
) -> Result<(Array2<f32>, Array2<u32>), VecHealthError> {
    let n = normalized_vectors.nrows();

    let mut all_distances = Array2::<f32>::zeros((n, k));
    let mut all_indices = Array2::<u32>::zeros((n, k));

    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let query_batch = normalized_vectors.slice(s![batch_start..batch_end, ..]);
        let sim_batch = query_batch.dot(&normalized_vectors.t());

        let mut dist_batch_out = all_distances.slice_mut(s![batch_start..batch_end, ..]);
        let mut idx_batch_out = all_indices.slice_mut(s![batch_start..batch_end, ..]);

        dist_batch_out
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(idx_batch_out.axis_iter_mut(Axis(0)).into_par_iter())
            .zip(sim_batch.axis_iter(Axis(0)).into_par_iter())
            .enumerate()
            .for_each(|(local_row, ((mut dist_row, mut idx_row), sim_row))| {
                let global_row = batch_start + local_row;

                let mut sims_with_idx: Vec<(f32, u32)> = sim_row
                    .iter()
                    .enumerate()
                    .filter(|&(idx, _)| idx != global_row)
                    .map(|(idx, &sim)| (sim, idx as u32))
                    .collect();

                let k_actual = k.min(sims_with_idx.len());

                if k_actual < sims_with_idx.len() {
                    sims_with_idx.select_nth_unstable_by(k_actual, |a, b| {
                        b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
                    });
                }

                let top = &mut sims_with_idx[..k_actual];
                top.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

                for i in 0..k_actual {
                    let sim = top[i].0;
                    let euclidean_dist = f32::max(0.0, 2.0 - 2.0 * sim).sqrt();
                    dist_row[i] = euclidean_dist;
                    idx_row[i] = top[i].1;
                }
            });
    }

    Ok((all_distances, all_indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn top1_is_actually_the_nearest_neighbor() {
        let vectors = array![
            [1.0f32, 0.0, 0.0, 0.0],
            [0.9, 0.436, 0.0, 0.0],
            [0.7, 0.714, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let (distances, indices) = evaluator.get_knn(2, 10).unwrap();

        assert!(distances[[0, 0]] <= distances[[0, 1]]);
        assert_eq!(indices[[0, 0]], 1);
    }

    #[test]
    fn degenerate_vector_does_not_fail_whole_batch() {
        let vectors = array![
            [1.0f32, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let report = evaluator.normalization_report().unwrap();
        assert_eq!(report.degenerate_indices, vec![1]);

        assert!(evaluator.get_knn(1, 10).is_ok());
    }

    #[test]
    fn all_degenerate_returns_explicit_error() {
        let vectors = array![[0.0f32, 0.0], [0.0, 0.0]];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        assert!(matches!(
            evaluator.normalization_report(),
            Err(VecHealthError::AllVectorsDegenerate)
        ));
    }

    #[test]
    fn get_original_vector_is_unnormalized() {
        let vectors = array![[3.0f32, 4.0]];
        let evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let original = evaluator.get_original_vector(0);
        assert_eq!(original[0], 3.0);
        assert_eq!(original[1], 4.0);
    }
}