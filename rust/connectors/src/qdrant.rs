//! Qdrant connector. Uses the `scroll` API exclusively — it iterates points
//! in ID order with no ranking/scoring, i.e. it never touches the HNSW
//! index. This is the same mechanism Qdrant's own backup/export tooling
//! uses, and is explicitly documented by Qdrant as the right API for bulk
//! reads and admin tasks (as opposed to `search`/`query`, which is for
//! low-latency ANN lookups and would be the wrong tool for this).
//!
//! No sampling, no `limit` on total points pulled: this fetches the
//! *complete* collection (see `TODO_Conectors.md` for why). `page_size` only
//! controls how many points are requested per network round-trip.

use crate::{ConnectorError, DistanceMetric, FetchedVectors, SourceInfo};
use ndarray::Array2;
use qdrant_client::qdrant::vector_output::Vector as VectorOutputEnum;
use qdrant_client::qdrant::vectors_config::Config as VectorsConfigEnum;
use qdrant_client::qdrant::{CollectionInfo, Distance, PointId, ScrollPointsBuilder};
use qdrant_client::Qdrant;

#[derive(Debug, Clone)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub collection: String,
    /// Points requested per scroll round-trip. Purely a network-chunking
    /// knob, not a cap on how many vectors are ultimately fetched.
    pub page_size: u32,
    pub timeout_secs: u64,
}

impl QdrantConfig {
    pub fn new(url: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            api_key: None,
            collection: collection.into(),
            page_size: 1000,
            timeout_secs: 30,
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_page_size(mut self, page_size: u32) -> Self {
        self.page_size = page_size;
        self
    }
}

/// Fetches every point's vector from a Qdrant collection. Blocking: builds a
/// small single-threaded Tokio runtime internally so this crate's public API
/// stays synchronous (matching `vechealth-core`, which callers wrap in
/// `py.allow_threads` on the Python side).
pub fn fetch_all(config: &QdrantConfig) -> Result<FetchedVectors, ConnectorError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| ConnectorError::Io(e.to_string()))?;
    runtime.block_on(fetch_all_async(config))
}

async fn fetch_all_async(config: &QdrantConfig) -> Result<FetchedVectors, ConnectorError> {
    let mut builder = Qdrant::from_url(&config.url);
    if let Some(key) = &config.api_key {
        builder = builder.api_key(key.clone());
    }
    let client = builder
        .build()
        .map_err(|e| ConnectorError::Network(format!("could not build Qdrant client: {e}")))?;

    let collection_info = client
        .collection_info(&config.collection)
        .await
        .map_err(|e| ConnectorError::Network(format!("could not read collection info: {e}")))?
        .result
        .ok_or_else(|| {
            ConnectorError::SchemaMismatch(format!(
                "collection '{}' has no config in its info response",
                config.collection
            ))
        })?;

    let (declared_dim, distance_metric) =
        parse_vector_params(&collection_info, &config.collection)?;

    let mut all_rows: Vec<f32> = Vec::new();
    let mut n_points = 0usize;
    let mut dim: Option<usize> = declared_dim;
    let mut offset = None;

    loop {
        let mut scroll = ScrollPointsBuilder::new(config.collection.clone())
            .limit(config.page_size)
            .with_vectors(true)
            .timeout(config.timeout_secs);
        if let Some(o) = offset.take() {
            scroll = scroll.offset(o);
        }

        let response = client
            .scroll(scroll.build())
            .await
            .map_err(|e| ConnectorError::Network(format!("scroll request failed: {e}")))?;

        for point in response.result {
            let dense = point.vectors.as_ref().and_then(|v| v.get_vector());
            let vector = extract_dense_vector(dense, point.id.as_ref(), &config.collection)?;

            let this_dim = vector.len();
            match dim {
                None => dim = Some(this_dim),
                Some(d) if d != this_dim => {
                    return Err(ConnectorError::SchemaMismatch(format!(
                        "inconsistent vector dimension in collection '{}': expected {d}, got {this_dim}",
                        config.collection
                    )));
                }
                _ => {}
            }

            all_rows.extend_from_slice(&vector);
            n_points += 1;
        }

        match response.next_page_offset {
            Some(next) => offset = Some(next),
            None => break,
        }
    }

    if n_points == 0 {
        return Err(ConnectorError::EmptySource(format!(
            "collection '{}' has no points",
            config.collection
        )));
    }
    let dim = dim.ok_or_else(|| {
        ConnectorError::SchemaMismatch(format!(
            "could not determine vector dimension for collection '{}'",
            config.collection
        ))
    })?;

    let vectors = Array2::from_shape_vec((n_points, dim), all_rows)
        .map_err(|e| ConnectorError::Parse(e.to_string()))?;

    Ok(FetchedVectors {
        vectors,
        info: SourceInfo {
            dim,
            count: n_points,
            distance_metric,
        },
    })
}

/// Reads the declared vector size and distance metric off a collection's
/// config, if present. Kept as a pure function (no network calls) so it can
/// be unit-tested by constructing a `CollectionInfo` by hand instead of
/// requiring a live Qdrant instance.
fn parse_vector_params(
    collection_info: &CollectionInfo,
    collection_name: &str,
) -> Result<(Option<usize>, Option<DistanceMetric>), ConnectorError> {
    let vectors_config = collection_info
        .config
        .as_ref()
        .and_then(|c| c.params.as_ref())
        .and_then(|p| p.vectors_config.as_ref())
        .and_then(|vc| vc.config.as_ref());

    match vectors_config {
        Some(VectorsConfigEnum::Params(params)) => Ok((
            Some(params.size as usize),
            match params.distance() {
                Distance::Cosine => Some(DistanceMetric::Cosine),
                Distance::Dot => Some(DistanceMetric::Dot),
                Distance::Euclid => Some(DistanceMetric::Euclidean),
                _ => None,
            },
        )),
        Some(VectorsConfigEnum::ParamsMap(_)) => Err(ConnectorError::SchemaMismatch(format!(
            "collection '{collection_name}' uses multiple named vectors per point, which this \
             connector doesn't support yet — pass a single-vector collection",
        ))),
        None => Ok((None, None)),
    }
}

/// Pulls the flat `Vec<f32>` out of a scroll response's per-point vector
/// output. Qdrant 1.x's `get_vector()` returns the dense/sparse/multi-dense
/// enum uniformly even for plain dense collections, so this is where that
/// gets narrowed — and where we reject shapes this connector doesn't (yet)
/// support. Pure function, unit-tested without a live server.
fn extract_dense_vector(
    vector: Option<VectorOutputEnum>,
    point_id: Option<&PointId>,
    collection_name: &str,
) -> Result<Vec<f32>, ConnectorError> {
    match vector {
        Some(VectorOutputEnum::Dense(dense)) => Ok(dense.data),
        Some(VectorOutputEnum::Sparse(_)) => Err(ConnectorError::SchemaMismatch(format!(
            "point {point_id:?} in collection '{collection_name}' has a sparse vector, which \
             this connector doesn't support — pass a dense-vector collection",
        ))),
        Some(VectorOutputEnum::MultiDense(_)) => Err(ConnectorError::SchemaMismatch(format!(
            "point {point_id:?} in collection '{collection_name}' has a multi-vector (e.g. \
             ColBERT-style), which this connector doesn't support — pass a single dense-vector \
             collection",
        ))),
        None => Err(ConnectorError::SchemaMismatch(format!(
            "point {point_id:?} in collection '{collection_name}' has no vector \
             (scroll was requested with with_vectors=true)",
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::{
        CollectionConfig, CollectionParams, DenseVector, MultiDenseVector, SparseVector,
        VectorParams, VectorsConfig,
    };

    fn collection_info_with(config: Option<VectorsConfigEnum>) -> CollectionInfo {
        CollectionInfo {
            config: Some(CollectionConfig {
                params: Some(CollectionParams {
                    vectors_config: config.map(|c| VectorsConfig { config: Some(c) }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn parse_vector_params_reads_dim_and_cosine() {
        let info = collection_info_with(Some(VectorsConfigEnum::Params(VectorParams {
            size: 384,
            distance: Distance::Cosine.into(),
            ..Default::default()
        })));
        let (dim, metric) = parse_vector_params(&info, "coll").unwrap();
        assert_eq!(dim, Some(384));
        assert_eq!(metric, Some(DistanceMetric::Cosine));
    }

    #[test]
    fn parse_vector_params_reads_euclidean() {
        let info = collection_info_with(Some(VectorsConfigEnum::Params(VectorParams {
            size: 128,
            distance: Distance::Euclid.into(),
            ..Default::default()
        })));
        let (dim, metric) = parse_vector_params(&info, "coll").unwrap();
        assert_eq!(dim, Some(128));
        assert_eq!(metric, Some(DistanceMetric::Euclidean));
    }

    #[test]
    fn parse_vector_params_rejects_named_vectors() {
        let info = collection_info_with(Some(VectorsConfigEnum::ParamsMap(Default::default())));
        assert!(matches!(
            parse_vector_params(&info, "coll"),
            Err(ConnectorError::SchemaMismatch(_))
        ));
    }

    #[test]
    fn parse_vector_params_missing_config_is_none() {
        let info = collection_info_with(None);
        let (dim, metric) = parse_vector_params(&info, "coll").unwrap();
        assert_eq!(dim, None);
        assert_eq!(metric, None);
    }

    #[test]
    fn extract_dense_vector_ok() {
        let out = extract_dense_vector(
            Some(VectorOutputEnum::Dense(DenseVector {
                data: vec![1.0, 2.0, 3.0],
            })),
            None,
            "coll",
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn extract_dense_vector_rejects_sparse() {
        let err = extract_dense_vector(
            Some(VectorOutputEnum::Sparse(SparseVector::default())),
            None,
            "coll",
        );
        assert!(matches!(err, Err(ConnectorError::SchemaMismatch(_))));
    }

    #[test]
    fn extract_dense_vector_rejects_multi_dense() {
        let err = extract_dense_vector(
            Some(VectorOutputEnum::MultiDense(MultiDenseVector::default())),
            None,
            "coll",
        );
        assert!(matches!(err, Err(ConnectorError::SchemaMismatch(_))));
    }

    #[test]
    fn extract_dense_vector_rejects_missing() {
        let err = extract_dense_vector(None, None, "coll");
        assert!(matches!(err, Err(ConnectorError::SchemaMismatch(_))));
    }
}
