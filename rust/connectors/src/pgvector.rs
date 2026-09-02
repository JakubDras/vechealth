//! pgvector connector. Reads through a plain SQL cursor with keyset
//! pagination (`WHERE id > $last_id ORDER BY id LIMIT $page_size`), never
//! `OFFSET` — `OFFSET` makes Postgres walk and discard every skipped row,
//! which gets linearly worse the deeper you page into a large table.
//!
//! No sampling: fetches every row (see `TODO_Conectors.md`). `page_size`
//! only controls how many rows are requested per round-trip.
//!
//! Known limitations of this first version: `id_column` must be an integer
//! (`i64`-compatible) primary/unique key — the common case for Postgres
//! tables — and connections are unencrypted (`NoTls`); use an SSH tunnel or
//! a trusted network if your Postgres requires TLS.

use crate::{ConnectorError, FetchedVectors, SourceInfo};
use ndarray::Array2;
use pgvector::Vector as PgVector;
use postgres::{Client, NoTls};

#[derive(Debug, Clone)]
pub struct PgVectorConfig {
    pub connection_string: String,
    pub table: String,
    pub vector_column: String,
    pub id_column: String,
    /// Rows requested per round-trip — a network-chunking knob, not a cap
    /// on how many rows are ultimately fetched.
    pub page_size: i64,
}

impl PgVectorConfig {
    pub fn new(
        connection_string: impl Into<String>,
        table: impl Into<String>,
        vector_column: impl Into<String>,
        id_column: impl Into<String>,
    ) -> Self {
        Self {
            connection_string: connection_string.into(),
            table: table.into(),
            vector_column: vector_column.into(),
            id_column: id_column.into(),
            page_size: 5000,
        }
    }

    pub fn with_page_size(mut self, page_size: i64) -> Self {
        self.page_size = page_size;
        self
    }
}

/// Safely quotes a Postgres identifier for interpolation into SQL text.
/// Table/column names can't go through bind parameters (those only work for
/// values), so this is the standard defense against a crafted identifier
/// breaking out of the query.
fn quote_ident(ident: &str) -> String {
    format!("\"{}\"", ident.replace('"', "\"\""))
}

pub fn fetch_all(config: &PgVectorConfig) -> Result<FetchedVectors, ConnectorError> {
    let mut client = Client::connect(&config.connection_string, NoTls)
        .map_err(|e| ConnectorError::Network(format!("could not connect to Postgres: {e}")))?;

    let table = quote_ident(&config.table);
    let id_col = quote_ident(&config.id_column);
    let vec_col = quote_ident(&config.vector_column);

    let first_page_query = format!("SELECT {id_col}, {vec_col} FROM {table} ORDER BY {id_col} ASC LIMIT $1");
    let next_page_query =
        format!("SELECT {id_col}, {vec_col} FROM {table} WHERE {id_col} > $1 ORDER BY {id_col} ASC LIMIT $2");

    let mut all_rows: Vec<f32> = Vec::new();
    let mut n_rows = 0usize;
    let mut dim: Option<usize> = None;
    let mut last_id: Option<i64> = None;

    loop {
        let rows = match last_id {
            None => client
                .query(&first_page_query, &[&config.page_size])
                .map_err(|e| query_error(&config.table, e))?,
            Some(id) => client
                .query(&next_page_query, &[&id, &config.page_size])
                .map_err(|e| query_error(&config.table, e))?,
        };

        if rows.is_empty() {
            break;
        }

        for row in &rows {
            let id: i64 = row.get(0);
            let vector: PgVector = row.get(1);
            let values: Vec<f32> = vector.into();

            let this_dim = values.len();
            match dim {
                None => dim = Some(this_dim),
                Some(d) if d != this_dim => {
                    return Err(ConnectorError::SchemaMismatch(format!(
                        "inconsistent vector dimension in '{}': expected {d}, got {this_dim} (id={id})",
                        config.table
                    )));
                }
                _ => {}
            }

            all_rows.extend_from_slice(&values);
            n_rows += 1;
            last_id = Some(id);
        }

        if (rows.len() as i64) < config.page_size {
            break;
        }
    }

    if n_rows == 0 {
        return Err(ConnectorError::EmptySource(format!(
            "'{}' has no rows",
            config.table
        )));
    }
    let dim = dim.expect("n_rows > 0 implies dim was set");
    let vectors = Array2::from_shape_vec((n_rows, dim), all_rows)
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

fn query_error(table: &str, e: postgres::Error) -> ConnectorError {
    ConnectorError::Network(format!(
        "query against '{table}' failed: {e} (note: id_column must be an integer-typed \
         primary/unique key for this connector's pagination to work)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quote_ident_wraps_and_escapes() {
        assert_eq!(quote_ident("embeddings"), "\"embeddings\"");
        assert_eq!(quote_ident("weird\"name"), "\"weird\"\"name\"");
    }

    #[test]
    fn quote_ident_prevents_injection_via_table_name() {
        let malicious = "items; DROP TABLE users; --";
        let quoted = quote_ident(malicious);
        // Cały napis ląduje jako JEDEN cudzysłowiony identyfikator SQL —
        // średnik i reszta nie mogą "wyjść" poza literał identyfikatora.
        assert_eq!(quoted, "\"items; DROP TABLE users; --\"");
        assert_eq!(quoted.matches('"').count(), 2);
    }
}
