//! Connectors that pull raw vectors into an `ndarray::Array2<f32>` for
//! `vechealth-core` to analyze. Kept as a separate crate (not part of
//! `vechealth-core`) on purpose: these pull in networking/IO dependencies
//! (tokio, gRPC, a Postgres driver, arrow/parquet) that the pure metrics
//! engine has no business depending on.
//!
//! Design constraints every connector here follows (see `TODO_Conectors.md`
//! for the full rationale):
//! - Never use a similarity/ANN search path to bulk-read data — always the
//!   storage-level scroll/cursor/iterator path, which bypasses the index
//!   entirely and is what these databases use for their own backups/exports.
//! - No sampling here. Fala 1 pulls the *complete* vector set from the
//!   source; a geometry-aware reduction algorithm (planned separately) needs
//!   the full set as input, so trimming here would defeat it.
//! - Low, bounded concurrency and paginated requests — the opposite
//!   optimization goal from `vechealth-core`, which maximizes local CPU
//!   parallelism. A connector should be a polite, sequential citizen against
//!   someone else's production database.

pub mod local;
pub mod pgvector;
pub mod qdrant;

use ndarray::Array2;
use std::fmt;

/// Distance metric a source collection was configured/indexed with, when the
/// source can report it. `vechealth-core` assumes cosine similarity
/// internally (it L2-normalizes), so callers should surface a warning to the
/// user when this is `Some` and not `Cosine`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Dot,
    Euclidean,
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cosine => write!(f, "cosine"),
            Self::Dot => write!(f, "dot"),
            Self::Euclidean => write!(f, "euclidean"),
        }
    }
}

/// Metadata about a source collection, reported alongside fetched vectors so
/// callers can sanity-check/warn before trusting the result.
#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub dim: usize,
    pub count: usize,
    pub distance_metric: Option<DistanceMetric>,
}

#[derive(Debug, Clone)]
pub struct FetchedVectors {
    pub vectors: Array2<f32>,
    pub info: SourceInfo,
}

#[derive(Debug)]
pub enum ConnectorError {
    Io(String),
    Parse(String),
    SchemaMismatch(String),
    Network(String),
    Auth(String),
    EmptySource(String),
}

impl fmt::Display for ConnectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::Parse(msg) => write!(f, "Failed to parse source data: {msg}"),
            Self::SchemaMismatch(msg) => write!(f, "Schema mismatch: {msg}"),
            Self::Network(msg) => write!(f, "Network/connection error: {msg}"),
            Self::Auth(msg) => write!(f, "Authentication/authorization error: {msg}"),
            Self::EmptySource(msg) => write!(f, "Source has no vectors: {msg}"),
        }
    }
}

impl std::error::Error for ConnectorError {}
