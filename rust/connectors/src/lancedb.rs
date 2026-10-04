//! LanceDB connector. Uses the plain table scan path (`Table::query()` with
//! no vector) exclusively — LanceDB's ANN path is `Table::vector_search()` /
//! `Query::nearest_to()`, which this connector never calls. A plain `query()`
//! is a columnar scan over the underlying Lance dataset files, the same
//! mechanism LanceDB itself uses for exports and full-table reads.
//!
//! No sampling, no `limit` on total rows pulled: this fetches the *complete*
//! table. `batch_size` only controls the maximum
//! number of rows per `RecordBatch` yielded by the scan (`max_batch_length`),
//! not how many rows are ultimately fetched — same role as `page_size` in
//! the Qdrant/pgvector connectors.
//!
//! LanceDB is embedded-first: `uri` is most commonly a local directory path,
//! but can also be an object-store URI (`s3://`, `gs://`, ...) if the right
//! Cargo feature were enabled — this connector builds with only the local
//! filesystem backend to keep the dependency footprint down.

use crate::{ConnectorError, DistanceMetric, FetchedVectors, SourceInfo};
use futures::TryStreamExt;
use lancedb::arrow::arrow_array::{Array, FixedSizeListArray, Float32Array, Float64Array};
use lancedb::arrow::arrow_schema::DataType;
use lancedb::query::{ExecutableQuery, QueryBase, QueryExecutionOptions, Select};
use lancedb::{DistanceType, Table};
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct LanceDbConfig {
    pub uri: String,
    pub table: String,
    pub vector_column: String,
    pub batch_size: u32,
}

impl LanceDbConfig {
    pub fn new(
        uri: impl Into<String>,
        table: impl Into<String>,
        vector_column: impl Into<String>,
    ) -> Self {
        Self {
            uri: uri.into(),
            table: table.into(),
            vector_column: vector_column.into(),
            batch_size: 1024,
        }
    }

    pub fn with_batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = batch_size;
        self
    }
}

/// Fetches every row's vector from a LanceDB table via a plain columnar
/// scan. Blocking: builds a small single-threaded Tokio runtime internally
/// so this crate's public API stays synchronous, matching the Qdrant/pgvector
/// connectors.
pub fn fetch_all(config: &LanceDbConfig) -> Result<FetchedVectors, ConnectorError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| ConnectorError::Io(e.to_string()))?;
    runtime.block_on(fetch_all_async(config))
}

async fn fetch_all_async(config: &LanceDbConfig) -> Result<FetchedVectors, ConnectorError> {
    let connection = lancedb::connect(&config.uri).execute().await.map_err(|e| {
        ConnectorError::Network(format!(
            "could not open LanceDB database at '{}': {e}",
            config.uri
        ))
    })?;

    let table = connection
        .open_table(&config.table)
        .execute()
        .await
        .map_err(|e| {
            ConnectorError::Network(format!(
                "could not open table '{}' in LanceDB database '{}': {e}",
                config.table, config.uri
            ))
        })?;

    let declared_dim = declared_vector_dim(&table, &config.vector_column).await?;
    let distance_metric = distance_metric_for_column(&table, &config.vector_column).await?;

    // `QueryExecutionOptions` is `#[non_exhaustive]`, so it can't be built
    // with struct-literal syntax from outside the crate — start from
    // `default()` and mutate the one field we care about instead.
    let mut execution_options = QueryExecutionOptions::default();
    execution_options.max_batch_length = config.batch_size;

    let mut stream = table
        .query()
        .select(Select::columns(std::slice::from_ref(&config.vector_column)))
        .execute_with_options(execution_options)
        .await
        .map_err(|e| ConnectorError::Network(format!("table scan failed: {e}")))?;

    let mut all_rows: Vec<f32> = Vec::new();
    let mut n_rows = 0usize;
    let mut dim = declared_dim;

    while let Some(batch) = stream
        .try_next()
        .await
        .map_err(|e| ConnectorError::Network(format!("table scan failed: {e}")))?
    {
        let col_idx = batch
            .schema()
            .index_of(&config.vector_column)
            .map_err(|e| {
                ConnectorError::SchemaMismatch(format!(
                    "scanned batch is missing column '{}': {e}",
                    config.vector_column
                ))
            })?;
        let list_array = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                ConnectorError::SchemaMismatch(format!(
                    "column '{}' in table '{}' is not a fixed-size vector column — this \
                     connector only supports LanceDB's standard `FixedSizeList<Float32/Float64>` \
                     vector columns",
                    config.vector_column, config.table
                ))
            })?;

        let this_dim = list_array.value_length() as usize;
        match dim {
            None => dim = Some(this_dim),
            Some(d) if d != this_dim => {
                return Err(ConnectorError::SchemaMismatch(format!(
                    "inconsistent vector dimension in table '{}': expected {d}, got {this_dim}",
                    config.table
                )));
            }
            _ => {}
        }

        for row in 0..list_array.len() {
            if list_array.is_null(row) {
                return Err(ConnectorError::SchemaMismatch(format!(
                    "row {row} in table '{}' has a null vector in column '{}'",
                    config.table, config.vector_column
                )));
            }
            let values = numeric_array_to_f32(list_array.value(row).as_ref(), &config.table)?;
            all_rows.extend_from_slice(&values);
            n_rows += 1;
        }
    }

    if n_rows == 0 {
        return Err(ConnectorError::EmptySource(format!(
            "table '{}' has no rows",
            config.table
        )));
    }
    let dim = dim.ok_or_else(|| {
        ConnectorError::SchemaMismatch(format!(
            "could not determine vector dimension for table '{}'",
            config.table
        ))
    })?;

    let vectors = Array2::from_shape_vec((n_rows, dim), all_rows)
        .map_err(|e| ConnectorError::Parse(e.to_string()))?;

    Ok(FetchedVectors {
        vectors,
        info: SourceInfo {
            dim,
            count: n_rows,
            distance_metric,
        },
    })
}

/// Reads the declared list size straight off the table's Arrow schema —
/// cheaper than waiting for the first scanned batch, and lets us fail fast
/// with a clear error if the named column isn't a fixed-size vector column
/// at all.
async fn declared_vector_dim(
    table: &Table,
    vector_column: &str,
) -> Result<Option<usize>, ConnectorError> {
    let schema = table
        .schema()
        .await
        .map_err(|e| ConnectorError::Network(format!("could not read table schema: {e}")))?;
    let field = schema.field_with_name(vector_column).map_err(|_| {
        ConnectorError::SchemaMismatch(format!(
            "column '{vector_column}' not found in schema {:?}",
            schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        ))
    })?;
    match field.data_type() {
        DataType::FixedSizeList(_, size) => Ok(Some(*size as usize)),
        other => Err(ConnectorError::SchemaMismatch(format!(
            "column '{vector_column}' has type {other:?}, expected a fixed-size vector column \
             (FixedSizeList<Float32> or FixedSizeList<Float64>)"
        ))),
    }
}

/// Reads the distance metric off the vector index configured on
/// `vector_column`, if one exists. LanceDB tables (especially small/embedded
/// ones) often have no ANN index at all — that's a valid, common state, not
/// an error — so this returns `Ok(None)` rather than failing when no index
/// is found or when the index doesn't report a distance type (e.g. a scalar
/// index that happens to share the column name).
async fn distance_metric_for_column(
    table: &Table,
    vector_column: &str,
) -> Result<Option<DistanceMetric>, ConnectorError> {
    let indices = table
        .list_indices()
        .await
        .map_err(|e| ConnectorError::Network(format!("could not list table indices: {e}")))?;
    let Some(index) = indices
        .iter()
        .find(|idx| idx.columns.iter().any(|c| c == vector_column))
    else {
        return Ok(None);
    };
    let stats = table
        .index_stats(&index.name)
        .await
        .map_err(|e| ConnectorError::Network(format!("could not read index stats: {e}")))?;
    Ok(stats
        .and_then(|s| s.distance_type)
        .and_then(map_distance_type))
}

/// LanceDB's `Hamming` (and any future variant — `DistanceType` is
/// `#[non_exhaustive]`) has no equivalent in `crate::DistanceMetric` (binary
/// vectors aren't a scenario `VecHealthEvaluator` supports) — mapped to
/// `None` rather than an error, since a missing distance metric is already a
/// normal, handled case throughout this crate.
fn map_distance_type(dt: DistanceType) -> Option<DistanceMetric> {
    match dt {
        DistanceType::Cosine => Some(DistanceMetric::Cosine),
        DistanceType::Dot => Some(DistanceMetric::Dot),
        DistanceType::L2 => Some(DistanceMetric::Euclidean),
        _ => None,
    }
}

/// Pulls a `Vec<f32>` out of one row's vector value (a slice of the
/// FixedSizeList's child array). Accepts `Float32`/`Float64` child arrays —
/// `Float64` is cast down, matching how `local.rs` handles `.npy`/`.parquet`.
fn numeric_array_to_f32(values: &dyn Array, table: &str) -> Result<Vec<f32>, ConnectorError> {
    match values.data_type() {
        DataType::Float32 => Ok(values
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("checked Float32 data type")
            .values()
            .to_vec()),
        DataType::Float64 => Ok(values
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("checked Float64 data type")
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect()),
        other => Err(ConnectorError::SchemaMismatch(format!(
            "table '{table}': vector column has element type {other:?}, expected Float32 or \
             Float64"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_distance_type_covers_supported_metrics() {
        assert_eq!(
            map_distance_type(DistanceType::Cosine),
            Some(DistanceMetric::Cosine)
        );
        assert_eq!(
            map_distance_type(DistanceType::Dot),
            Some(DistanceMetric::Dot)
        );
        assert_eq!(
            map_distance_type(DistanceType::L2),
            Some(DistanceMetric::Euclidean)
        );
        assert_eq!(map_distance_type(DistanceType::Hamming), None);
    }

    #[test]
    fn numeric_array_to_f32_casts_float64() {
        let arr = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let out = numeric_array_to_f32(&arr, "t").unwrap();
        assert_eq!(out, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn numeric_array_to_f32_passes_through_float32() {
        let arr = Float32Array::from(vec![1.0f32, 2.0, 3.0]);
        let out = numeric_array_to_f32(&arr, "t").unwrap();
        assert_eq!(out, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn numeric_array_to_f32_rejects_unsupported_type() {
        use lancedb::arrow::arrow_array::Int32Array;
        let arr = Int32Array::from(vec![1, 2, 3]);
        assert!(matches!(
            numeric_array_to_f32(&arr, "t"),
            Err(ConnectorError::SchemaMismatch(_))
        ));
    }

    /// LanceDB is embedded/local by construction, so — unlike Qdrant and
    /// pgvector — this connector's happy path can actually be exercised
    /// end-to-end without a live external server: this test creates a real
    /// on-disk LanceDB table and reads it back through `fetch_all`.
    #[test]
    fn fetch_all_reads_a_real_lancedb_table() {
        use lancedb::arrow::arrow_array::types::Float32Type;
        use lancedb::arrow::arrow_array::{
            FixedSizeListArray as ArrowFixedSizeListArray, RecordBatch,
        };
        use lancedb::arrow::arrow_schema::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("db").to_str().unwrap().to_string();

        // Table setup uses its own runtime, built and dropped before
        // `fetch_all` runs — `fetch_all` builds its own internal runtime,
        // and Tokio panics on nested runtimes if the two overlap.
        let setup_rt = tokio::runtime::Runtime::new().unwrap();
        setup_rt.block_on(async {
            let db = lancedb::connect(&uri).execute().await.unwrap();
            let schema = Arc::new(Schema::new(vec![Field::new(
                "vector",
                ArrowDataType::FixedSizeList(
                    Arc::new(Field::new("item", ArrowDataType::Float32, true)),
                    3,
                ),
                true,
            )]));
            let batch = RecordBatch::try_new(
                schema,
                vec![Arc::new(ArrowFixedSizeListArray::from_iter_primitive::<
                    Float32Type,
                    _,
                    _,
                >(
                    vec![
                        Some(vec![Some(1.0f32), Some(2.0), Some(3.0)]),
                        Some(vec![Some(4.0), Some(5.0), Some(6.0)]),
                    ],
                    3,
                ))],
            )
            .unwrap();
            db.create_table("vectors", batch).execute().await.unwrap();
        });
        drop(setup_rt);

        let config = LanceDbConfig::new(uri, "vectors", "vector");
        let fetched = fetch_all(&config).unwrap();

        assert_eq!(fetched.vectors.shape(), &[2, 3]);
        assert_eq!(fetched.info.dim, 3);
        assert_eq!(fetched.info.count, 2);
        assert_eq!(fetched.info.distance_metric, None);
    }

    #[test]
    fn fetch_all_rejects_non_vector_column() {
        use lancedb::arrow::arrow_array::{Int32Array, RecordBatch};
        use lancedb::arrow::arrow_schema::{DataType as ArrowDataType, Field, Schema};
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("db").to_str().unwrap().to_string();

        let setup_rt = tokio::runtime::Runtime::new().unwrap();
        setup_rt.block_on(async {
            let db = lancedb::connect(&uri).execute().await.unwrap();
            let schema = Arc::new(Schema::new(vec![Field::new(
                "id",
                ArrowDataType::Int32,
                false,
            )]));
            let batch =
                RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
                    .unwrap();
            db.create_table("ids", batch).execute().await.unwrap();
        });
        drop(setup_rt);

        let config = LanceDbConfig::new(uri, "ids", "id");
        assert!(matches!(
            fetch_all(&config),
            Err(ConnectorError::SchemaMismatch(_))
        ));
    }
}
