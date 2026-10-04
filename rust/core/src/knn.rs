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

    let per_row_stats: Vec<(f64, bool)> = normalized
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .map(|mut row| {
            let norm = row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
            if norm == 0.0 {
                (0.0, true)
            } else {
                row.mapv_inplace(|x| ((x as f64) / norm) as f32);
                (norm, false)
            }
        })
        .collect();

    let mut non_normalized_count = 0usize;
    let mut min_norm = f64::MAX;
    let mut max_norm = f64::MIN;
    let mut sum_norm = 0.0f64;
    let mut degenerate_indices = Vec::new();

    for (idx, &(norm, is_degenerate)) in per_row_stats.iter().enumerate() {
        if is_degenerate {
            degenerate_indices.push(idx);
            continue;
        }
        if (norm - 1.0).abs() > tolerance as f64 {
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

    let mean_norm = (sum_norm / valid_count as f64) as f32;
    let min_norm = min_norm as f32;
    let max_norm = max_norm as f32;
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

pub fn l2_norms_f64(vectors: ArrayView2<f32>) -> Vec<f64> {
    vectors
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt())
        .collect()
}

#[inline]
pub fn exact_chord_distance(a: ArrayView1<f32>, norm_a: f64, b: ArrayView1<f32>, norm_b: f64) -> f64 {
    if norm_a == 0.0 || norm_b == 0.0 {
        return 2.0f64.sqrt();
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x as f64) / norm_a - (y as f64) / norm_b;
            d * d
        })
        .sum::<f64>()
        .sqrt()
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
    norms_cache: Option<Vec<f64>>,
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
            norms_cache: None,
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
            self.norms_cache = Some(l2_norms_f64(self.vectors.view()));
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
            let norms = self.norms_cache.as_ref().expect("norms computed with normalization");
            let (distances, indices) =
                blocked_topk_cosine(normalized.view(), self.vectors.view(), norms, k, batch_size)?;
            self.knn_cache = Some(KnnCache { k, distances, indices });
        }

        let cache = self.knn_cache.as_ref().unwrap();
        Ok((
            cache.distances.slice(s![.., ..k]),
            cache.indices.slice(s![.., ..k]),
        ))
    }
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

fn blocked_topk_cosine(
    normalized_vectors: ArrayView2<f32>,
    original_vectors: ArrayView2<f32>,
    norms: &[f64],
    k: usize,
    batch_size: usize,
) -> Result<(Array2<f32>, Array2<u32>), VecHealthError> {
    let n = normalized_vectors.nrows();

    let mut all_distances = Array2::<f32>::zeros((n, k));
    let mut all_indices = Array2::<u32>::zeros((n, k));

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
            let k_cand = (2 * k).min(n - 1).max(k);
            let mut top_k: Vec<(f32, u32)> = vec![(f32::NEG_INFINITY, u32::MAX); k_cand];
            // Buffer for the exactly recomputed (f64) distances.
            let mut exact: Vec<(f64, u32)> = vec![(0.0, 0); k_cand];

            dist_chunk
                .axis_iter_mut(Axis(0))
                .zip(idx_chunk.axis_iter_mut(Axis(0)))
                .zip(sim_batch.axis_iter(Axis(0)))
                .enumerate()
                .for_each(|(local_row, ((mut dist_row,
                    mut idx_row), sim_row))| {
                    let global_row = batch_start + local_row;

                    top_k.fill((f32::NEG_INFINITY, u32::MAX));
                    for (idx, &sim) in sim_row.iter().enumerate() {
                        if idx == global_row {
                            continue;
                        }
                        insert_top_k(&mut top_k, sim, idx as u32);
                    }

                    let row_i = original_vectors.row(global_row);
                    let norm_i = norms[global_row];
                    for i in 0..k_cand {
                        let (_sim, neighbor_idx) = top_k[i];
                        let j = neighbor_idx as usize;
                        let d = exact_chord_distance(row_i, norm_i, original_vectors.row(j), norms[j]);
                        exact[i] = (d, neighbor_idx);
                    }
                    exact.sort_by(|a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1))
                    });
                    for i in 0..k {
                        dist_row[i] = exact[i].0 as f32;
                        idx_row[i] = exact[i].1;
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

    #[test]
    fn blocked_topk_matches_naive_full_sort_reference() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        for &(n, dim, k, batch_size) in &[
            (5usize, 3usize, 2usize, 2usize),
            (50, 8, 5, 7),
            (200, 16, 10, 32),
            (200, 16, 1, 200),
            (37, 4, 36, 5),
        ] {
            let mut rng = StdRng::seed_from_u64(42 + n as u64);
            let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let vectors = Array2::from_shape_vec((n, dim), data).unwrap();

            let mut evaluator = VecHealthEvaluator::new(vectors.clone()).unwrap();
            let (distances, indices) = evaluator.get_knn(k, batch_size).unwrap();
            
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

    
    fn unit_random(rng: &mut rand::rngs::StdRng, dim: usize) -> Vec<f64> {
        use rand::Rng;
        let v: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0f64..1.0)).collect();
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    }
    
    fn pairs_at_distance(target: f64, pairs: usize, dim: usize, lo: f64, hi: f64, seed: u64) -> Array2<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut data: Vec<f32> = Vec::with_capacity(2 * pairs * dim);
        for _ in 0..pairs {
            let x = unit_random(&mut rng, dim);
            let mut u = unit_random(&mut rng, dim);
            let dot: f64 = x.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            for (ui, xi) in u.iter_mut().zip(x.iter()) {
                *ui -= dot * xi;
            }
            let nu = u.iter().map(|v| v * v).sum::<f64>().sqrt();
            let y: Vec<f64> = x.iter().zip(u.iter()).map(|(a, b)| a + target * b / nu).collect();
            let ny = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            let (rx, ry) = (rng.gen_range(lo..hi), rng.gen_range(lo..hi));
            data.extend(x.iter().map(|&v| (v * rx) as f32));
            data.extend(y.iter().map(|&v| (v / ny * ry) as f32));
        }
        Array2::from_shape_vec((2 * pairs, dim), data).unwrap()
    }

    fn reference_distance(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f64 {
        let na = a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let nb = b.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x as f64 / na - y as f64 / nb;
                d * d
            })
            .sum::<f64>()
            .sqrt()
    }

    fn check_pairs(target: f64, lo: f64, hi: f64, seed: u64) {
        let vectors = pairs_at_distance(target, 50, 1024, lo, hi, seed);
        let mut evaluator = VecHealthEvaluator::new(vectors.clone()).unwrap();
        let (distances, indices) = evaluator.get_knn(1, 16).unwrap();
        for i in 0..vectors.nrows() {
            let partner = i ^ 1;
            assert_eq!(indices[[i, 0]] as usize, partner, "target {target}: 1-NN of {i} is not its partner");
            let dstar = reference_distance(vectors.row(i), vectors.row(partner));
            let got = distances[[i, 0]] as f64;
            assert!(
                (got - dstar).abs() <= 1e-6 * dstar + 1e-9,
                "target {target}, point {i}: got {got:e}, reference {dstar:e}"
            );
        }
    }

    #[test]
    fn small_distances_match_f64_reference() {
        for (s, &t) in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4].iter().enumerate() {
            check_pairs(t, 1.0, 1.0 + 1e-12, 100 + s as u64);
        }
    }

    #[test]
    fn non_unit_inputs_are_normalized_before_the_distance() {
        // input norms like the stored B3 vectors: 0.996–1.004
        for (s, &t) in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4].iter().enumerate() {
            check_pairs(t, 0.996, 1.004, 200 + s as u64);
        }
    }

    #[test]
    fn neighbour_selection_matches_f64_brute_force_on_separated_data() {
        use rand::{Rng, SeedableRng};
        let (n, dim, k) = (300usize, 32usize, 10usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0) * 1.3).collect();
        let vectors = Array2::from_shape_vec((n, dim), data).unwrap();
        let mut evaluator = VecHealthEvaluator::new(vectors.clone()).unwrap();
        let (distances, indices) = evaluator.get_knn(k, 37).unwrap();
        for i in 0..n {
            let mut refs: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (reference_distance(vectors.row(i), vectors.row(j)), j))
                .collect();
            refs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for r in 0..k {
                assert_eq!(indices[[i, r]] as usize, refs[r].1, "row {i} rank {r}");
                let got = distances[[i, r]] as f64;
                assert!((got - refs[r].0).abs() <= 1e-6 * refs[r].0 + 1e-9, "row {i} rank {r}");
            }
        }
    }
    
    fn cone(n: usize, dim: usize, spread: f64, seed: u64) -> Array2<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let c = unit_random(&mut rng, dim);
        let mut data: Vec<f32> = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let g: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0f64..1.0) * 3.0f64.sqrt()).collect();
            data.extend(c.iter().zip(g.iter()).map(|(&ci, &gi)| (ci + spread * gi) as f32));
        }
        Array2::from_shape_vec((n, dim), data).unwrap()
    }

    fn brute_force_knn(vectors: &Array2<f32>, i: usize, k: usize) -> Vec<(f64, usize)> {
        let mut refs: Vec<(f64, usize)> = (0..vectors.nrows())
            .filter(|&j| j != i)
            .map(|j| (reference_distance(vectors.row(i), vectors.row(j)), j))
            .collect();
        refs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        refs.truncate(k);
        refs
    }

    #[test]
    fn neighbour_sets_on_a_tight_cone_match_f64_brute_force() {
        let (n, dim, k, rows) = (2000usize, 64usize, 20usize, 400usize);
        let vectors = cone(n, dim, 0.04 / (2.0 * dim as f64).sqrt(), 11);
        let mut evaluator = VecHealthEvaluator::new(vectors.clone()).unwrap();
        let (distances, indices) = evaluator.get_knn(k, 64).unwrap();
        let mut mismatched = Vec::new();
        for i in 0..rows {
            let refs = brute_force_knn(&vectors, i, k);
            let got: Vec<usize> = (0..k).map(|r| indices[[i, r]] as usize).collect();
            let want: Vec<usize> = refs.iter().map(|r| r.1).collect();
            if got != want {
                mismatched.push(i);
                continue;
            }
            for r in 0..k {
                let d = distances[[i, r]] as f64;
                assert!((d - refs[r].0).abs() <= 1e-6 * refs[r].0 + 1e-9, "row {i} rank {r}");
            }
        }
        assert!(mismatched.is_empty(), "{} of {rows} rows differ from brute force: {:?}",
                mismatched.len(), &mismatched[..mismatched.len().min(10)]);
    }

    #[test]
    fn exact_ties_are_broken_by_lower_index() {
        let vectors = array![[0.6f32, 0.8, 0.0], [0.6, 0.8, 0.0], [0.6, 0.8, 0.0], [0.6, 0.8, 0.0], [0.0, 0.8, 0.6]];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let (distances, indices) = evaluator.get_knn(2, 5).unwrap();
        assert_eq!((indices[[3, 0]], indices[[3, 1]]), (0, 1));
        assert_eq!((distances[[3, 0]], distances[[3, 1]]), (0.0, 0.0));
        assert_eq!((indices[[0, 0]], indices[[0, 1]]), (1, 2));
        assert_eq!((indices[[4, 0]], indices[[4, 1]]), (0, 1));
    }

    #[test]
    fn exact_duplicates_have_zero_distance_and_degenerate_rows_keep_sqrt2() {
        let vectors = array![[0.6f32, 0.8, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let (distances, indices) = evaluator.get_knn(1, 4).unwrap();
        assert_eq!(indices[[0, 0]], 1);
        assert_eq!(distances[[0, 0]], 0.0);
        // degenerate row: Π(0) = 0, so by definition d = sqrt(2 - 2·0)
        assert!((distances[[2, 0]] as f64 - 2.0f64.sqrt()).abs() < 1e-7);
    }
}
