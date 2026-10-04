//! Memory use of the exact k-NN search, and the batch size that keeps it in
//! check.
//!
//! The search computes similarities in blocks of `batch_size` rows against
//! every vector, one block per worker thread at a time. Its peak memory is
//! therefore dominated by `threads x batch_size x N x 4` bytes of similarities
//! on top of a few copies of the vectors. A fixed batch size that is
//! comfortable on a laptop exhausts the memory of a many-core workstation or
//! of a large collection, so the helpers here choose one from the size of the
//! problem, estimate what a given choice will cost, and word the warning a
//! caller can raise before it starts.
//!
//! Everything here is a pure function of its arguments (the one environment
//! variable is read in a single place), so it is cheap to test.

/// Fixed batch size for Rust callers that do not use [`auto_batch_size`]. The
/// Python API picks the batch automatically instead.
pub const DEFAULT_BATCH_SIZE: usize = 256;

/// Smallest batch the automatic choice will go down to.
pub const MIN_BATCH_SIZE: usize = 16;

/// Largest batch the automatic choice will go up to.
pub const MAX_BATCH_SIZE: usize = 2048;

/// Memory the automatic choice allows the similarity blocks, in bytes (decimal
/// gigabytes, like every figure in the messages).
pub const BLOCK_BUDGET_BYTES: u64 = 2_000_000_000;

/// Estimated peak memory above which a warning is raised, in bytes. A normal
/// developer machine has 16 GB; this leaves it half. The environment variable
/// `VECHEALTH_MEMORY_WARNING_GB` overrides it.
pub const DEFAULT_MEMORY_WARNING_BYTES: u64 = 8_000_000_000;

/// Name of the environment variable that overrides the warning threshold.
pub const MEMORY_WARNING_ENV: &str = "VECHEALTH_MEMORY_WARNING_GB";

const BYTES_PER_F32: u64 = 4;

/// Copies of the vectors alive at the peak: the caller's array, the
/// evaluator's own copy, its normalised copy, and the clone made while
/// searching. Counting the caller's array is conservative when the vectors
/// were loaded from a file.
const VECTOR_COPIES: u64 = 4;

/// Number of worker threads the search runs on.
pub fn worker_threads() -> usize {
    rayon::current_num_threads().max(1)
}

/// Estimated peak memory, in bytes, of the exact k-NN search.
///
/// `copies of the vectors + min(threads x batch_size, N) rows of N
/// similarities`. The second term is the number of similarity rows that exist
/// at once: every worker holds one block of `batch_size` rows, and there can
/// never be more rows in flight than there are vectors. Calibrated against
/// measured peaks (see the unit tests); good to about 10%.
pub fn estimate_peak_memory_bytes(
    n_vectors: usize,
    dim: usize,
    batch_size: usize,
    threads: usize,
) -> u64 {
    vector_bytes(n_vectors, dim).saturating_add(block_bytes(n_vectors, batch_size, threads))
}

/// Bytes the similarity blocks alone take for the given batch. Every figure
/// here saturates instead of overflowing, because the arguments come straight
/// from the caller.
fn block_bytes(n_vectors: usize, batch_size: usize, threads: usize) -> u64 {
    let rows_in_flight = (threads.max(1) as u64)
        .saturating_mul(batch_size.max(1) as u64)
        .min(n_vectors as u64);
    rows_in_flight
        .saturating_mul(n_vectors as u64)
        .saturating_mul(BYTES_PER_F32)
}

/// Bytes the copies of the vectors take.
fn vector_bytes(n_vectors: usize, dim: usize) -> u64 {
    VECTOR_COPIES
        .saturating_mul(n_vectors as u64)
        .saturating_mul(dim as u64)
        .saturating_mul(BYTES_PER_F32)
}

/// The batch size used when the caller does not choose one: as large as the
/// block budget allows (a bigger block costs memory the machine may not have,
/// a much smaller one costs time), between [`MIN_BATCH_SIZE`] and
/// [`MAX_BATCH_SIZE`], and never more than the number of vectors.
pub fn auto_batch_size(n_vectors: usize, threads: usize) -> usize {
    let n = n_vectors.max(1) as u64;
    let bytes_per_batch_row = n
        .saturating_mul(BYTES_PER_F32)
        .saturating_mul(threads.max(1) as u64);
    let rows = BLOCK_BUDGET_BYTES / bytes_per_batch_row;
    (rows as usize)
        .clamp(MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        .min(n_vectors.max(1))
}

/// Warning threshold in bytes: [`DEFAULT_MEMORY_WARNING_BYTES`], or
/// `VECHEALTH_MEMORY_WARNING_GB` (decimal gigabytes) when it is set to a
/// non-negative number.
pub fn memory_warning_threshold_bytes() -> u64 {
    parse_threshold_gb(std::env::var(MEMORY_WARNING_ENV).ok().as_deref())
}

fn parse_threshold_gb(value: Option<&str>) -> u64 {
    match value.and_then(|v| v.trim().parse::<f64>().ok()) {
        Some(gb) if gb.is_finite() && gb >= 0.0 => (gb * 1e9) as u64,
        _ => DEFAULT_MEMORY_WARNING_BYTES,
    }
}

/// The warning to raise before a search, or `None` when its estimated peak
/// memory is within the threshold. `automatic` says whether `batch_size` was
/// chosen by [`auto_batch_size`] or by the caller.
pub fn memory_warning(
    n_vectors: usize,
    dim: usize,
    batch_size: usize,
    threads: usize,
    automatic: bool,
) -> Option<String> {
    memory_warning_with_threshold(
        n_vectors,
        dim,
        batch_size,
        threads,
        automatic,
        memory_warning_threshold_bytes(),
    )
}

/// [`memory_warning`] with an explicit threshold.
pub fn memory_warning_with_threshold(
    n_vectors: usize,
    dim: usize,
    batch_size: usize,
    threads: usize,
    automatic: bool,
    threshold_bytes: u64,
) -> Option<String> {
    let estimate = estimate_peak_memory_bytes(n_vectors, dim, batch_size, threads);
    if estimate <= threshold_bytes {
        return None;
    }
    let gb = |bytes: u64| bytes as f64 / 1e9;
    let silence = "Silence it with warnings.filterwarnings('ignore', \
                   category=vechealth.MemoryUsageWarning), or raise the limit with the \
                   VECHEALTH_MEMORY_WARNING_GB environment variable.";

    if automatic {
        return Some(format!(
            "The exact k-NN search is estimated to need about {:.1} GB of memory at its peak \
             ({n_vectors} vectors x {dim} dimensions on {threads} threads, automatic \
             batch_size={batch_size}); the vectors alone, held in several copies, take about \
             {:.1} GB. That is more than {:.1} GB, which many machines do not have. Free memory \
             or analyse a smaller collection. {silence}",
            gb(estimate),
            gb(vector_bytes(n_vectors, dim)),
            gb(threshold_bytes),
        ));
    }

    let auto = auto_batch_size(n_vectors, threads);
    let auto_estimate = estimate_peak_memory_bytes(n_vectors, dim, auto, threads);
    let advice = if auto_estimate <= threshold_bytes {
        format!(
            "Lower batch_size — leaving it unset picks {auto} here, about {:.1} GB —",
            gb(auto_estimate)
        )
    } else {
        format!(
            "Even the automatic batch ({auto}) would need about {:.1} GB, because the \
             vectors alone take about {:.1} GB. Analyse a smaller collection,",
            gb(auto_estimate),
            gb(vector_bytes(n_vectors, dim))
        )
    };
    Some(format!(
        "batch_size={batch_size} on {threads} threads and {n_vectors} vectors is estimated to \
         need about {:.1} GB of memory at its peak (the similarity blocks alone take {:.1} GB). \
         That is more than {:.1} GB. {advice} or ignore this warning if the machine has the \
         memory. {silence}",
        gb(estimate),
        gb(block_bytes(n_vectors, batch_size, threads)),
        gb(threshold_bytes),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Peak resident memory measured for `compute_all` on random subsets of a
    /// 1024-dimensional corpus on a 44-thread machine: (vectors, batch_size,
    /// measured bytes). The first six runs used explicit batch sizes, the last
    /// five the automatic one (the table in the README).
    const MEASURED: [(usize, usize, f64); 11] = [
        (10_000, 2048, 0.76e9),
        (20_000, 2048, 1.94e9),
        (40_000, 2048, 6.93e9),
        (40_000, 256, 2.59e9),
        (100_000, 128, 4.27e9),
        (246_460, 64, 7.10e9),
        (10_000, 1136, 0.631e9),
        (20_000, 568, 2.005e9),
        (40_000, 284, 2.660e9),
        (100_000, 113, 3.927e9),
        (246_460, 46, 6.164e9),
    ];

    #[test]
    fn estimate_matches_the_measured_peaks() {
        for (n, batch, measured) in MEASURED {
            let estimate = estimate_peak_memory_bytes(n, 1024, batch, 44) as f64;
            let error = (estimate - measured).abs() / measured;
            // The 10,000-vector run is dominated by fixed costs of the
            // process itself, so it gets a wider margin.
            let tolerance = if n == 10_000 { 0.4 } else { 0.12 };
            assert!(
                error < tolerance,
                "n={n} batch={batch}: estimated {estimate:.3e}, measured {measured:.3e}"
            );
        }
    }

    #[test]
    fn absurd_sizes_saturate_instead_of_overflowing() {
        assert_eq!(
            estimate_peak_memory_bytes(usize::MAX, usize::MAX, usize::MAX, usize::MAX),
            u64::MAX
        );
        assert_eq!(
            estimate_peak_memory_bytes(1_000, 8, usize::MAX, usize::MAX),
            4 * 1_000 * 8 * 4 + 1_000 * 1_000 * 4
        );
        assert_eq!(auto_batch_size(usize::MAX, usize::MAX), MIN_BATCH_SIZE);
    }

    #[test]
    fn rows_in_flight_never_exceed_the_collection() {
        // One block per thread cannot outgrow the N x N similarity matrix.
        let huge_batch = estimate_peak_memory_bytes(1_000, 8, 1_000_000, 64);
        let n = 1_000u64;
        assert_eq!(huge_batch, 4 * n * 8 * 4 + n * n * 4);
    }

    #[test]
    fn auto_batch_shrinks_with_more_vectors_and_more_threads() {
        assert!(auto_batch_size(20_000, 8) > auto_batch_size(200_000, 8));
        assert!(auto_batch_size(200_000, 4) > auto_batch_size(200_000, 44));
    }

    #[test]
    fn auto_batch_is_clamped_and_never_exceeds_the_collection() {
        assert_eq!(auto_batch_size(5_000_000, 64), MIN_BATCH_SIZE);
        assert_eq!(auto_batch_size(1_000, 4), 1_000.min(MAX_BATCH_SIZE));
        assert_eq!(auto_batch_size(15, 4), 15);
        assert_eq!(auto_batch_size(1, 1), 1);
        assert_eq!(auto_batch_size(0, 1), 1);
        assert!(auto_batch_size(40_000, 44) <= MAX_BATCH_SIZE);
    }

    #[test]
    fn the_automatic_batch_keeps_the_blocks_within_the_budget() {
        for &(n, threads) in &[
            (20_000usize, 44usize),
            (100_000, 16),
            (246_460, 44),
            (50_000, 8),
        ] {
            let batch = auto_batch_size(n, threads);
            let blocks = block_bytes(n, batch, threads);
            // Rounding the batch down keeps it within the budget, except where
            // the lower clamp had to win.
            assert!(
                blocks <= BLOCK_BUDGET_BYTES || batch == MIN_BATCH_SIZE,
                "n={n} threads={threads} batch={batch}: blocks {blocks}"
            );
        }
    }

    #[test]
    fn a_collection_that_fits_gets_no_warning() {
        let threads = 44;
        let batch = auto_batch_size(246_460, threads);
        assert_eq!(
            memory_warning_with_threshold(246_460, 1024, batch, threads, true, 8_000_000_000),
            None
        );
    }

    #[test]
    fn a_large_explicit_batch_is_warned_about_with_the_alternative() {
        let message =
            memory_warning_with_threshold(100_000, 1024, 2048, 44, false, 8_000_000_000).unwrap();
        assert!(message.contains("batch_size=2048"));
        assert!(message.contains("44 threads"));
        assert!(message.contains("leaving it unset picks"));
        assert!(message.contains("MemoryUsageWarning"));
        assert!(message.contains(MEMORY_WARNING_ENV));
    }

    #[test]
    fn data_too_large_for_any_batch_is_warned_about_as_such() {
        // 600,000 x 1024 floats: the copies alone are about 9.8 GB.
        let batch = auto_batch_size(600_000, 8);
        let message =
            memory_warning_with_threshold(600_000, 1024, batch, 8, true, 8_000_000_000).unwrap();
        assert!(message.contains("automatic batch_size="));
        assert!(message.contains("vectors alone"));
        let explicit =
            memory_warning_with_threshold(600_000, 1024, 64, 8, false, 8_000_000_000).unwrap();
        assert!(explicit.contains("Even the automatic batch"));
    }

    #[test]
    fn the_threshold_can_be_overridden_with_the_environment_variable() {
        assert_eq!(parse_threshold_gb(None), DEFAULT_MEMORY_WARNING_BYTES);
        assert_eq!(parse_threshold_gb(Some("16")), 16_000_000_000);
        assert_eq!(parse_threshold_gb(Some(" 0.5 ")), 500_000_000);
        assert_eq!(parse_threshold_gb(Some("0")), 0);
        // Anything that is not a non-negative number falls back to the default.
        assert_eq!(
            parse_threshold_gb(Some("lots")),
            DEFAULT_MEMORY_WARNING_BYTES
        );
        assert_eq!(parse_threshold_gb(Some("-1")), DEFAULT_MEMORY_WARNING_BYTES);
        assert_eq!(
            parse_threshold_gb(Some("nan")),
            DEFAULT_MEMORY_WARNING_BYTES
        );
    }
}
