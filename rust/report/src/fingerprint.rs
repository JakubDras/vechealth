//! Cheap, deterministic identity for the vector set a `Report` was computed
//! on — needed so `compare()` can tell whether two reports are even talking
//! about the same data before comparing their metric values.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetFingerprint {
    pub n_vectors: usize,
    pub dim: usize,
    /// Hex-encoded `std::hash::Hasher` digest over the raw vector bytes.
    /// Not cryptographic — this is an equality check ("is this the same
    /// data?"), not a security boundary. Uses `std::hash`'s `DefaultHasher`
    /// on purpose: a fingerprint is computed once per report, so hashing
    /// speed doesn't matter, and this way `vechealth-report` doesn't need a
    /// dedicated hashing dependency.
    pub content_hash: String,
}

pub fn fingerprint_vectors(vectors: ArrayView2<f32>) -> DatasetFingerprint {
    let n_vectors = vectors.nrows();
    let dim = vectors.ncols();

    let mut hasher = DefaultHasher::new();
    n_vectors.hash(&mut hasher);
    dim.hash(&mut hasher);
    for value in vectors.iter() {
        value.to_bits().hash(&mut hasher);
    }
    let content_hash = format!("{:016x}", hasher.finish());

    DatasetFingerprint {
        n_vectors,
        dim,
        content_hash,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn same_data_hashes_identically() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[1.0f32, 2.0], [3.0, 4.0]];
        assert_eq!(
            fingerprint_vectors(a.view()).content_hash,
            fingerprint_vectors(b.view()).content_hash
        );
    }

    #[test]
    fn different_data_hashes_differently() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[1.0f32, 2.0], [3.0, 5.0]];
        assert_ne!(
            fingerprint_vectors(a.view()).content_hash,
            fingerprint_vectors(b.view()).content_hash
        );
    }
}
