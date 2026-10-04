use crate::knn::{VecHealthError, VecHealthEvaluator};

#[derive(Debug, Clone)]
pub struct HubnessResult {
    pub hubness_skewness: f64,
    pub orphans_fraction: f64,
    pub max_occurrences: u32,
}

pub fn compute_hubness_score(
    evaluator: &mut VecHealthEvaluator,
    k: usize,
    batch_size: usize,
) -> Result<HubnessResult, VecHealthError> {
    let n = evaluator.n_vectors();
    let (_, indices) = evaluator.get_knn(k, batch_size)?;

    let mut occurrences = vec![0u32; n];
    for &idx in indices.iter() {
        occurrences[idx as usize] += 1;
    }

    // One pass over occurrences: Welford/Terriberry online moments (mean,
    // M2, M3) together with the max and the orphans. Numerically stable (no
    // subtraction of large raw moments) and cheap (a single pass over memory).
    let mut count = 0.0f64;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    let mut max_occurrences = 0u32;
    let mut orphans_count = 0usize;
    for &c in &occurrences {
        if c > max_occurrences {
            max_occurrences = c;
        }
        if c == 0 {
            orphans_count += 1;
        }

        let x = c as f64;
        let n1 = count;
        count += 1.0;
        let delta = x - mean;
        let delta_n = delta / count;
        let term1 = delta * delta_n * n1;
        mean += delta_n;
        m3 += term1 * delta_n * (count - 2.0) - 3.0 * delta_n * m2;
        m2 += term1;
    }
    let n_f64 = n as f64;
    let orphans_fraction = orphans_count as f64 / n_f64;

    let hubness_skewness = if m2 <= 0.0 {
        0.0
    } else {
        (n_f64.sqrt() * m3) / m2.powf(1.5)
    };

    Ok(HubnessResult {
        hubness_skewness,
        orphans_fraction,
        max_occurrences,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn hand_computed_triangle_no_dominant_hub() {
        // A=(1,0), B=(0.8,0.6), C=(0,1) — a triangle without ties.
        // sim(A,B)=0.8, sim(A,C)=0.0, sim(B,C)=0.6
        // NN(A)=B, NN(B)=A, NN(C)=B  =>  occurrences = [A:1, B:2, C:0]
        // mean=1, m2=0.6667, m3=0.0  =>  skewness = 0.0 exactly
        let vectors = array![
            [1.0f32, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let result = compute_hubness_score(&mut evaluator, 1, 10).unwrap();

        assert!((result.hubness_skewness - 0.0).abs() < 1e-4);
        assert_eq!(result.max_occurrences, 2);
        assert!((result.orphans_fraction - 1.0 / 3.0).abs() < 1e-4);
    }

    #[test]
    fn hand_computed_hub_with_three_satellites() {
        // Hub=(1,0,0,0). Three satellites, each closer to the hub (sim≈0.99)
        // than to one another (sim≈0.98) — a deliberately designed hub.
        let vectors = array![
            [1.0f32, 0.0, 0.0, 0.0],
            [0.99, 0.14, 0.0, 0.0],
            [0.99, 0.0, 0.14, 0.0],
            [0.99, 0.0, 0.0, 0.14],
        ];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let result = compute_hubness_score(&mut evaluator, 1, 10).unwrap();

        // the hub (idx 0) stays the NN of all 3 satellites; exactly one
        // satellite becomes the NN of the hub itself (a tie between the
        // satellites, which does not affect the assertions below)
        assert_eq!(result.max_occurrences, 3);
        assert!((result.orphans_fraction - 0.5).abs() < 1e-4);
        assert!(result.hubness_skewness > 0.5); // clearly positive skewness
    }
}