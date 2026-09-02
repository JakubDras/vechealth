use ndarray::{s, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::fmt;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;

#[derive(Debug)]
pub enum VecHealthError {
    EmptyInput,
    DimensionMismatch { expected: usize, found: usize },
    KTooLarge { k: usize, n_vectors: usize },
    KTooSmall { k: usize, minimum: usize },
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
            Self::KTooSmall { k, minimum } => {
                write!(f, "k={} was requested, but this metric requires k >= {}.", k, minimum)
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

    pub fn normalized_vectors(&mut self) -> Result<&Array2<f32>, VecHealthError> {
        let (normalized, _report) = self.ensure_normalized()?;
        Ok(normalized)
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
            return Err(VecHealthError::KTooLarge {
                k,
                n_vectors: self.n_vectors,
            });
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

// Wstawia (val, idx) do top_k utrzymywanego posortowanego rosnąco wg `val`
// (top_k[0] = najmniejsza wartość w bieżącym top-k = pierwsza do wyrzucenia).
// O(1) gdy val nie łapie się do top-k, O(k) gdy się łapie — bez alokacji.
// Ten sam wzorzec co insert_top_k w metrics/qmas.rs.
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

fn blocked_topk_cosine(
    normalized_vectors: ArrayView2<f32>,
    k: usize,
    batch_size: usize,
) -> Result<(Array2<f32>, Array2<u32>), VecHealthError> {
    let n = normalized_vectors.nrows();

    let mut all_distances = Array2::<f32>::zeros((n, k));
    let mut all_indices = Array2::<u32>::zeros((n, k));

    // Zrównoleglenie po paczkach (batchach) na poziomie Rayona
    all_distances
        .axis_chunks_iter_mut(Axis(0), batch_size)
        .into_par_iter()
        .zip(
            all_indices
                .axis_chunks_iter_mut(Axis(0), batch_size)
                .into_par_iter(),
        )
        .enumerate()
        .for_each(|(chunk_idx, (mut dist_chunk, mut idx_chunk))| {
            let batch_start = chunk_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(n);
            let query_batch = normalized_vectors.slice(s![batch_start..batch_end, ..]);
            let sim_batch = query_batch.dot(&normalized_vectors.t());

            // Bufor top-k alokowany RAZ na cały batch (nie na wiersz) i
            // resetowany między wierszami — zamiast Vec<(f32,u32)> rozmiaru
            // n alokowanego dla każdego z n wierszy (poprzednio O(n) alokacji
            // po O(n) elementów każda).
            let mut top_k: Vec<(f32, u32)> = vec![(f32::NEG_INFINITY, u32::MAX); k];

            dist_chunk
                .axis_iter_mut(Axis(0))
                .zip(idx_chunk.axis_iter_mut(Axis(0)))
                .zip(sim_batch.axis_iter(Axis(0)))
                .enumerate()
                .for_each(|(local_row, ((mut dist_row, mut idx_row), sim_row))| {
                    let global_row = batch_start + local_row;

                    top_k.fill((f32::NEG_INFINITY, u32::MAX));
                    for (idx, &sim) in sim_row.iter().enumerate() {
                        if idx == global_row {
                            continue;
                        }
                        insert_top_k(&mut top_k, sim, idx as u32);
                    }

                    // top_k jest posortowane rosnąco wg podobieństwa, więc
                    // najbliższy sąsiad (najwyższe podobieństwo) to top_k[k-1].
                    // k < n_vectors jest wymuszone w get_knn, więc mamy co
                    // najmniej k kandydatów (n-1 >= k) i żaden slot nie
                    // zostaje sentinelem NEG_INFINITY.
                    for i in 0..k {
                        let (sim, neighbor_idx) = top_k[k - 1 - i];
                        let euclidean_dist = f32::max(0.0, 2.0 - 2.0 * sim).sqrt();
                        dist_row[i] = euclidean_dist;
                        idx_row[i] = neighbor_idx;
                    }
                });
        });

    Ok((all_distances, all_indices))
}

impl From<VecHealthError> for PyErr {
    fn from(err: VecHealthError) -> Self {
        PyValueError::new_err(err.to_string())
    }
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
        let vectors = array![[1.0f32, 0.0], [0.0, 0.0], [0.0, 1.0]];
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

    #[test]
    fn get_knn_does_not_panic_at_maximum_k() {
        let vectors = array![[1.0f32, 0.0], [0.0, 1.0], [-1.0, 0.0]];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let result = evaluator.get_knn(2, 10);
        assert!(result.is_ok());
    }

    // Test tymczasowy: weryfikuje, że nowy insert_top_k w blocked_topk_cosine
    // daje identyczne wyniki co niezależna, naiwna referencja O(n^2 log n)
    // (pełne sortowanie każdego wiersza), dla różnych n, k, batch_size.
    #[test]
    fn blocked_topk_matches_naive_full_sort_reference() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        for &(n, dim, k, batch_size) in &[
            (5usize, 3usize, 2usize, 2usize),
            (50, 8, 5, 7),
            (200, 16, 10, 32),
            (200, 16, 1, 200),
            (37, 4, 36, 5), // k = n - 1, przypadek graniczny
        ] {
            let mut rng = StdRng::seed_from_u64(42 + n as u64);
            let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let vectors = Array2::from_shape_vec((n, dim), data).unwrap();

            let mut evaluator = VecHealthEvaluator::new(vectors.clone()).unwrap();
            let (distances, indices) = evaluator.get_knn(k, batch_size).unwrap();

            // Referencja: normalizuj ręcznie, policz pełną macierz cosine,
            // dla każdego wiersza posortuj malejąco po podobieństwie.
            let (normalized, _) = normalize_l2_with_report(vectors.view(), 1e-3).unwrap();
            for i in 0..n {
                let row_i = normalized.row(i);
                let mut sims: Vec<(f32, u32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let row_j = normalized.row(j);
                        let sim: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
                        (sim, j as u32)
                    })
                    .collect();
                sims.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                for rank in 0..k {
                    let expected_sim = sims[rank].0;
                    let expected_dist = f32::max(0.0, 2.0 - 2.0 * expected_sim).sqrt();
                    let got_dist = distances[[i, rank]];
                    assert!(
                        (got_dist - expected_dist).abs() < 1e-4,
                        "n={n} k={k} row={i} rank={rank}: got dist {got_dist}, expected {expected_dist}"
                    );

                    // Indeks może się różnić tylko przy dokładnych remisach
                    // podobieństwa — sprawdzamy więc, że zwrócony indeks ma
                    // dokładnie takie samo podobieństwo jak oczekiwane.
                    let got_idx = indices[[i, rank]] as usize;
                    let got_sim: f32 = row_i
                        .iter()
                        .zip(normalized.row(got_idx).iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    assert!(
                        (got_sim - expected_sim).abs() < 1e-4,
                        "n={n} k={k} row={i} rank={rank}: got_idx={got_idx} sim={got_sim}, expected sim={expected_sim}"
                    );
                }
            }
        }
    }
}