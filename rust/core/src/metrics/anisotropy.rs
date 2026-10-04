use crate::knn::{VecHealthError, VecHealthEvaluator};
use faer::{mat, Parallelism, Scale, Side};
use ndarray::Axis;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct AnisotropyResult {
    pub mean_vector_norm: f32,
    pub top1_variance_ratio: f32,
    pub top10_variance_ratio: f32,
}

pub fn compute_anisotropy_score(
    evaluator: &VecHealthEvaluator,
) -> Result<AnisotropyResult, VecHealthError> {
    faer::set_global_parallelism(Parallelism::Rayon(0));

    let vectors = evaluator.vectors();
    let n_vectors = evaluator.n_vectors();
    let dim = evaluator.dim;

    let partial_sums: Vec<Vec<f64>> = vectors
        .axis_chunks_iter(Axis(0), 4096)
        .into_par_iter()
        .map(|chunk| {
            let mut acc = vec![0.0f64; dim];
            for row in chunk.rows() {
                for (a, &x) in acc.iter_mut().zip(row.iter()) {
                    *a += x as f64;
                }
            }
            acc
        })
        .collect();
    if n_vectors == 0 {
        return Err(VecHealthError::EmptyInput);
    }
    let mut mean64 = vec![0.0f64; dim];
    for part in &partial_sums {
        for (m, &p) in mean64.iter_mut().zip(part.iter()) {
            *m += p;
        }
    }
    for m in mean64.iter_mut() {
        *m /= n_vectors as f64;
    }
    let mean_vector_norm = mean64.iter().map(|&m| m * m).sum::<f64>().sqrt() as f32;

    if n_vectors < 2 {
        return Ok(AnisotropyResult {
            mean_vector_norm,
            top1_variance_ratio: 0.0,
            top10_variance_ratio: 0.0,
        });
    }

    let mut covariance = mat::Mat::<f64>::zeros(dim, dim);
    let block_rows = 4096usize;
    let mut block = vec![0.0f64; block_rows * dim];
    for chunk in vectors.axis_chunks_iter(Axis(0), block_rows) {
        let rows = chunk.nrows();
        for (r, row) in chunk.rows().into_iter().enumerate() {
            let dst = &mut block[r * dim..(r + 1) * dim];
            for ((d, &x), &m) in dst.iter_mut().zip(row.iter()).zip(mean64.iter()) {
                *d = x as f64 - m;
            }
        }
        let b = mat::from_row_major_slice(&block[..rows * dim], rows, dim);
        faer::linalg::matmul::matmul(
            covariance.as_mut(),
            b.transpose(),
            b,
            Some(1.0),
            1.0,
            Parallelism::Rayon(0),
        );
    }
    let covariance = Scale(1.0f64 / (n_vectors as f64 - 1.0)) * covariance;
    
    let eigendecomposition = covariance.selfadjoint_eigendecomposition(Side::Lower);

    let mut eigenvalues: Vec<f64> = (0..dim)
        .map(|i| eigendecomposition.s().column_vector().read(i))
        .collect();

    eigenvalues.reverse();

    for v in eigenvalues.iter_mut() {
        *v = v.max(0.0);
    }

    let total_variance: f64 = eigenvalues.iter().sum();

    let (top1_variance_ratio, top10_variance_ratio) = if total_variance > 0.0 {
        let top1 = (eigenvalues[0] / total_variance) as f32;
        let top10_count = dim.min(10);
        let top10 = (eigenvalues[..top10_count].iter().sum::<f64>() / total_variance) as f32;
        (top1, top10)
    } else {
        (0.0, 0.0)
    };

    Ok(AnisotropyResult {
        mean_vector_norm,
        top1_variance_ratio,
        top10_variance_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    
    #[test]
    fn mean_vector_norm_matches_f64_reference() {
        use rand::{Rng, SeedableRng};
        let (n, dim) = (20_000usize, 64usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-0.2f32..1.0)).collect();
        let vectors = ndarray::Array2::from_shape_vec((n, dim), data).unwrap();
        let mut mean = vec![0.0f64; dim];
        for row in vectors.rows() {
            for (m, &x) in mean.iter_mut().zip(row.iter()) {
                *m += x as f64;
            }
        }
        let reference = mean.iter().map(|m| (m / n as f64).powi(2)).sum::<f64>().sqrt();
        let evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let r = compute_anisotropy_score(&evaluator).unwrap();
        assert!(((r.mean_vector_norm as f64 - reference) / reference).abs() < 1e-6);
    }
}
