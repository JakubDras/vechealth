//! Local file loaders: `.npy`, `.csv`, `.parquet`. Zero network/server
//! involvement by construction — this is the safest connector category
//! (see `TODO_Conectors.md`, Tier 1) and covers the "analyze a local backup
//! / export" workflow directly.

use crate::{ConnectorError, FetchedVectors, SourceInfo};
use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::DataType;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

/// Reads a 2D `.npy` array as vectors. Accepts float32 or float64 arrays
/// (numpy's default dtype is float64); float64 is cast down to float32.
pub fn load_npy(path: &Path) -> Result<FetchedVectors, ConnectorError> {
    let file = File::open(path).map_err(|e| ConnectorError::Io(e.to_string()))?;
    let vectors: Array2<f32> = match Array2::<f32>::read_npy(&file) {
        Ok(arr) => arr,
        Err(_) => {
            let file = File::open(path).map_err(|e| ConnectorError::Io(e.to_string()))?;
            let arr64 = Array2::<f64>::read_npy(file).map_err(|e| {
                ConnectorError::Parse(format!(
                    "'{}' is not a 2D float32 or float64 .npy array: {e}",
                    path.display()
                ))
            })?;
            arr64.mapv(|x| x as f32)
        }
    };

    let (n_rows, dim) = vectors.dim();
    if n_rows == 0 {
        return Err(ConnectorError::EmptySource(format!(
            "'{}' has zero rows",
            path.display()
        )));
    }
    Ok(FetchedVectors {
        vectors,
        info: SourceInfo {
            dim,
            count: n_rows,
            distance_metric: None,
        },
    })
}

/// Reads a CSV file where every row is one vector and every field is one
/// dimension (no id/label columns — use `load_parquet` with an explicit
/// column list if you need to select a subset of columns).
pub fn load_csv(path: &Path, has_header: bool) -> Result<FetchedVectors, ConnectorError> {
    let file = File::open(path).map_err(|e| ConnectorError::Io(e.to_string()))?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(has_header)
        // The csv crate rejects ragged rows itself by default, with a
        // generic message and no row number. We want our own check below to
        // fire instead, so it can report which row and what dimension.
        .flexible(true)
        .from_reader(file);

    let mut rows: Vec<f32> = Vec::new();
    let mut dim: Option<usize> = None;
    let mut n_rows = 0usize;

    for result in reader.records() {
        let record = result.map_err(|e| ConnectorError::Parse(e.to_string()))?;
        let this_dim = record.len();
        match dim {
            None => dim = Some(this_dim),
            Some(d) if d != this_dim => {
                return Err(ConnectorError::SchemaMismatch(format!(
                    "'{}': row {n_rows} has {this_dim} columns, expected {d} (from earlier rows)",
                    path.display()
                )));
            }
            _ => {}
        }
        for field in record.iter() {
            let value: f32 = field.trim().parse().map_err(|_| {
                ConnectorError::Parse(format!(
                    "'{}': could not parse '{field}' as a number (row {n_rows})",
                    path.display()
                ))
            })?;
            rows.push(value);
        }
        n_rows += 1;
    }

    let dim = dim.ok_or_else(|| {
        ConnectorError::EmptySource(format!("'{}' has zero rows", path.display()))
    })?;
    let vectors = Array2::from_shape_vec((n_rows, dim), rows)
        .map_err(|e| ConnectorError::Parse(e.to_string()))?;

    Ok(FetchedVectors {
        vectors,
        info: SourceInfo {
            dim,
            count: n_rows,
            distance_metric: None,
        },
    })
}

/// Reads a "wide" Parquet file — one numeric column per vector dimension,
/// one row per vector (matches how this project's own benchmarks already
/// write vectors: `df.to_numpy(dtype=np.float32)` over the whole frame).
/// `columns=None` uses every column in the file, in schema order;
/// `columns=Some([...])` selects and orders specific columns.
pub fn load_parquet(
    path: &Path,
    columns: Option<&[String]>,
) -> Result<FetchedVectors, ConnectorError> {
    let file = File::open(path).map_err(|e| ConnectorError::Io(e.to_string()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| ConnectorError::Parse(e.to_string()))?;
    let schema = builder.schema().clone();

    let selected: Vec<usize> = match columns {
        Some(names) => names
            .iter()
            .map(|name| {
                schema.index_of(name).map_err(|_| {
                    ConnectorError::SchemaMismatch(format!(
                        "'{}': column '{name}' not found in schema {:?}",
                        path.display(),
                        schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
                    ))
                })
            })
            .collect::<Result<_, _>>()?,
        None => (0..schema.fields().len()).collect(),
    };
    if selected.is_empty() {
        return Err(ConnectorError::SchemaMismatch(format!(
            "'{}': no columns selected",
            path.display()
        )));
    }
    let dim = selected.len();

    let reader = builder
        .build()
        .map_err(|e| ConnectorError::Parse(e.to_string()))?;

    let mut rows: Vec<f32> = Vec::new();
    let mut n_rows = 0usize;

    for batch_result in reader {
        let batch = batch_result.map_err(|e| ConnectorError::Parse(e.to_string()))?;
        let batch_rows = batch.num_rows();

        let mut columns_f32: Vec<Vec<f32>> = Vec::with_capacity(dim);
        for &col_idx in &selected {
            columns_f32.push(numeric_column_to_f32(batch.column(col_idx).as_ref(), path)?);
        }

        rows.reserve(batch_rows * dim);
        for row in 0..batch_rows {
            for col in columns_f32.iter().take(dim) {
                rows.push(col[row]);
            }
        }
        n_rows += batch_rows;
    }

    if n_rows == 0 {
        return Err(ConnectorError::EmptySource(format!(
            "'{}' has zero rows",
            path.display()
        )));
    }
    let vectors = Array2::from_shape_vec((n_rows, dim), rows)
        .map_err(|e| ConnectorError::Parse(e.to_string()))?;

    Ok(FetchedVectors {
        vectors,
        info: SourceInfo {
            dim,
            count: n_rows,
            distance_metric: None,
        },
    })
}

fn numeric_column_to_f32(col: &dyn Array, path: &Path) -> Result<Vec<f32>, ConnectorError> {
    if col.null_count() > 0 {
        return Err(ConnectorError::SchemaMismatch(format!(
            "'{}': column has {} null value(s) — vector components can't be missing",
            path.display(),
            col.null_count()
        )));
    }
    match col.data_type() {
        DataType::Float32 => Ok(col
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("checked Float32 data type")
            .values()
            .to_vec()),
        DataType::Float64 => Ok(col
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("checked Float64 data type")
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect()),
        DataType::Int64 => Ok(col
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("checked Int64 data type")
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect()),
        DataType::Int32 => Ok(col
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("checked Int32 data type")
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect()),
        other => Err(ConnectorError::SchemaMismatch(format!(
            "'{}': unsupported column type {other:?}, expected a numeric type",
            path.display()
        ))),
    }
}

/// Dispatches to the right loader based on file extension.
pub fn load_local(path: &Path) -> Result<FetchedVectors, ConnectorError> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("npy") => load_npy(path),
        Some("csv") => load_csv(path, true),
        Some("parquet") => load_parquet(path, None),
        other => Err(ConnectorError::SchemaMismatch(format!(
            "'{}': unrecognized extension {other:?} — expected .npy, .csv, or .parquet \
             (or call load_npy/load_csv/load_parquet directly)",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray_npy::WriteNpyExt;
    use std::io::Write;

    #[test]
    fn npy_roundtrip_f32() {
        let arr = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.npy");
        let file = File::create(&path).unwrap();
        arr.write_npy(file).unwrap();

        let result = load_npy(&path).unwrap();
        assert_eq!(result.vectors, arr);
        assert_eq!(result.info.dim, 3);
        assert_eq!(result.info.count, 2);
    }

    #[test]
    fn npy_f64_is_cast_down() {
        let arr = array![[1.0f64, 2.0], [3.0, 4.0]];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors_f64.npy");
        let file = File::create(&path).unwrap();
        arr.write_npy(file).unwrap();

        let result = load_npy(&path).unwrap();
        assert_eq!(result.vectors, array![[1.0f32, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn csv_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.csv");
        let mut file = File::create(&path).unwrap();
        writeln!(file, "a,b,c").unwrap();
        writeln!(file, "1.0,2.0,3.0").unwrap();
        writeln!(file, "4.0,5.0,6.0").unwrap();

        let result = load_csv(&path, true).unwrap();
        assert_eq!(result.vectors, array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(result.info.count, 2);
    }

    #[test]
    fn csv_inconsistent_width_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.csv");
        let mut file = File::create(&path).unwrap();
        writeln!(file, "1.0,2.0,3.0").unwrap();
        writeln!(file, "4.0,5.0").unwrap();

        assert!(matches!(
            load_csv(&path, false),
            Err(ConnectorError::SchemaMismatch(_))
        ));
    }

    #[test]
    fn load_local_dispatches_by_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.csv");
        let mut file = File::create(&path).unwrap();
        // load_local dispatches .csv through load_csv(path, has_header=true),
        // so the first line here is consumed as a header, not data.
        writeln!(file, "a,b").unwrap();
        writeln!(file, "1.0,2.0").unwrap();

        let result = load_local(&path).unwrap();
        assert_eq!(result.info.dim, 2);
    }

    #[test]
    fn load_local_unknown_extension_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.bin");
        File::create(&path).unwrap();

        assert!(matches!(
            load_local(&path),
            Err(ConnectorError::SchemaMismatch(_))
        ));
    }

    #[test]
    fn parquet_wide_format_roundtrip() {
        use arrow::array::Float32Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("dim0", DataType::Float32, false),
            Field::new("dim1", DataType::Float32, false),
            Field::new("dim2", DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from(vec![1.0, 4.0])),
                Arc::new(Float32Array::from(vec![2.0, 5.0])),
                Arc::new(Float32Array::from(vec![3.0, 6.0])),
            ],
        )
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.parquet");
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let result = load_parquet(&path, None).unwrap();
        assert_eq!(result.vectors, array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(result.info.dim, 3);
        assert_eq!(result.info.count, 2);

        // Wybór/kolejność podzbioru kolumn.
        let subset = load_parquet(&path, Some(&["dim2".to_string(), "dim0".to_string()])).unwrap();
        assert_eq!(subset.vectors, array![[3.0f32, 1.0], [6.0, 4.0]]);
    }

    /// `pandas.DataFrame.to_parquet()` — and every generator under
    /// `python/benchmarks/synthetic/` — writes Snappy-compressed Parquet by
    /// default (no `compression=` argument). Without the `parquet` crate's
    /// `snap` feature this fails to read at all; regression test for that.
    #[test]
    fn parquet_snappy_compressed_is_readable() {
        use arrow::array::Float32Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::basic::Compression;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("dim0", DataType::Float32, false),
            Field::new("dim1", DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from(vec![1.0, 3.0])),
                Arc::new(Float32Array::from(vec![2.0, 4.0])),
            ],
        )
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors_snappy.parquet");
        let file = File::create(&path).unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let result = load_parquet(&path, None).unwrap();
        assert_eq!(result.vectors, array![[1.0f32, 2.0], [3.0, 4.0]]);
    }
}
