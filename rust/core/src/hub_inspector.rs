use crate::knn::{VecHealthError, VecHealthEvaluator};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubEntry {
    pub index: usize,
    pub occurrence_count: u32,
}

pub fn identify_hubs(
    evaluator: &mut VecHealthEvaluator,
    k: usize,
    top_n: usize,
    batch_size: usize,
) -> Result<Vec<HubEntry>, VecHealthError> {
    let n = evaluator.n_vectors();
    let (_, indices) = evaluator.get_knn(k, batch_size)?;

    let mut occurrences = vec![0u32; n];
    for &idx in indices.iter() {
        occurrences[idx as usize] += 1;
    }

    let mut hubs: Vec<HubEntry> = occurrences
        .into_iter()
        .enumerate()
        .map(|(index, occurrence_count)| HubEntry { index, occurrence_count })
        .collect();

    hubs.sort_unstable_by(|a, b| {
        b.occurrence_count
            .cmp(&a.occurrence_count)
            .then(a.index.cmp(&b.index))
    });
    hubs.truncate(top_n.min(hubs.len()));

    Ok(hubs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identify_hubs_matches_hand_computed_triangle() {
        let vectors = array![
            [1.0f32, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let hubs = identify_hubs(&mut evaluator, 1, 3, 10).unwrap();

        assert_eq!(hubs.len(), 3);
        assert_eq!(hubs[0], HubEntry { index: 1, occurrence_count: 2 }); // B
        assert_eq!(hubs[1], HubEntry { index: 0, occurrence_count: 1 }); // A
        assert_eq!(hubs[2], HubEntry { index: 2, occurrence_count: 0 }); // C
    }

    #[test]
    fn identify_hubs_respects_top_n() {
        let vectors = array![
            [1.0f32, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let hubs = identify_hubs(&mut evaluator, 1, 1, 10).unwrap();
        assert_eq!(hubs.len(), 1);
        assert_eq!(hubs[0].index, 1);
    }

    #[test]
    fn identify_hubs_top_n_larger_than_n_vectors_does_not_panic() {
        let vectors = array![[1.0f32, 0.0], [0.0, 1.0], [-1.0, 0.0]];
        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let hubs = identify_hubs(&mut evaluator, 1, 999, 10).unwrap();
        assert_eq!(hubs.len(), 3); // truncated to n_vectors, does not panic
    }

    #[test]
    fn identify_hubs_max_occurrence_matches_compute_hubness_score_on_random_data() {
        // A consistency test BETWEEN TWO MODULES (hub_inspector.rs vs
        // hubness.rs), not a hand-computed one: both functions count from
        // the SAME get_knn call (same k, same batch_size), so the
        // occurrence_count of the strongest hub MUST equal max_occurrences
        // from compute_hubness_score, whatever the geometry of the data.
        // Keeping them in separate files must not break that agreement.
        use crate::metrics::hubness::compute_hubness_score;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(7);
        let n = 60;
        let dim = 5;
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let vectors = ndarray::Array2::from_shape_vec((n, dim), data).unwrap();

        let mut evaluator_a = VecHealthEvaluator::new(vectors.clone()).unwrap();
        let hubness = compute_hubness_score(&mut evaluator_a, 5, 16).unwrap();

        let mut evaluator_b = VecHealthEvaluator::new(vectors).unwrap();
        let hubs = identify_hubs(&mut evaluator_b, 5, 1, 16).unwrap();

        assert_eq!(hubs[0].occurrence_count, hubness.max_occurrences);
    }

    #[test]
    fn identify_hubs_sorted_descending() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(3);
        let n = 40;
        let dim = 4;
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let vectors = ndarray::Array2::from_shape_vec((n, dim), data).unwrap();

        let mut evaluator = VecHealthEvaluator::new(vectors).unwrap();
        let hubs = identify_hubs(&mut evaluator, 5, 40, 16).unwrap();

        for pair in hubs.windows(2) {
            assert!(pair[0].occurrence_count >= pair[1].occurrence_count);
        }
    }
}