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

    let mean_vec = vectors
        .mean_axis(Axis(0))
        .ok_or(VecHealthError::EmptyInput)?;
    let mean_vector_norm = mean_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if n_vectors < 2 {
        return Ok(AnisotropyResult {
            mean_vector_norm,
            top1_variance_ratio: 0.0,
            top10_variance_ratio: 0.0,
        });
    }
    
    let mut centered = vectors.to_owned();
    centered
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            row.zip_mut_with(&mean_vec, |x, &m| *x -= m);
        });

    let slice = centered
        .as_slice_memory_order()
        .ok_or(VecHealthError::EmptyInput)?;
    
    let centered_mat = mat::from_row_major_slice(slice, n_vectors, dim);
    
    let inv_n_minus_1 = 1.0f32 / (n_vectors as f32 - 1.0);
    let covariance = Scale(inv_n_minus_1) * (centered_mat.transpose() * centered_mat);
    
    let eigendecomposition = covariance.selfadjoint_eigendecomposition(Side::Lower);

    let mut eigenvalues: Vec<f32> = (0..dim)
        .map(|i| eigendecomposition.s().column_vector().read(i))
        .collect();

    eigenvalues.reverse();

    for v in eigenvalues.iter_mut() {
        *v = v.max(0.0);
    }

    let total_variance: f32 = eigenvalues.iter().sum();

    let (top1_variance_ratio, top10_variance_ratio) = if total_variance > 0.0 {
        let top1 = eigenvalues[0] / total_variance;
        let top10_count = dim.min(10);
        let top10 = eigenvalues[..top10_count].iter().sum::<f32>() / total_variance;
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