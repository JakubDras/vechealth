# Changelog

All notable changes to VecHealth are recorded here. The project follows
[semantic versioning](https://semver.org/) with the usual caveat for `0.x`:
until `1.0` the API may change between minor versions, and a pre-release may
change it again before the final version.

The Cargo version `0.2.0-alpha.1` is published to PyPI as the PEP 440 version
`0.2.0a1`; install a pre-release by asking for its exact version:
`pip install vechealth==0.2.0a1`.

## 0.2.0a1 — first public test version

The version `0.1.0` that already exists on PyPI is an empty placeholder, so
the first real release starts at `0.2.0`.

### Included

- Eleven content-free geometric statistics (hubness, intrinsic dimension,
  near-duplicate density, dispersion, anisotropy, outliers, neighbourhood
  consistency and two query-side statistics) computed in Rust from an exact
  k-NN search, with distances and reductions accumulated in float64.
- `VecHealthEvaluator` for in-memory arrays, plus loaders for `.npy`, `.csv`
  and `.parquet` files and connectors for Qdrant, pgvector, LanceDB, Weaviate,
  Chroma, Milvus and Pinecone.
- Versioned `Report` snapshots (`save` / `load` / `compare`) and the
  `vechealth` command-line tool (`analyze`, `compare --fail-on-significant`).
- `vechealth.interpretation`: noise floors per statistic, pathology-family
  matching with a threshold and an ambiguity margin, fail-closed repair
  recommendations, index-conditioned context, a Hub Inspector.

### Behaviour that differs from earlier development builds

These builds were never published; the entries matter to anyone who ran one.

- **The default outlier threshold is now 1.2** (a chord distance on the unit
  sphere, the value used for the preprint's results). It used to be derived
  from the data as three times the mean nearest-neighbour distance, which on
  typical high-dimensional text embeddings lands close to the largest possible
  distance (2.0) and flags essentially nothing. Pass
  `outlier_distance_threshold=None` (or `--outlier-distance-threshold
  adaptive`) for the old behaviour. `Report.compare()` now warns when two
  reports used different thresholds, or a data-derived one.
- **`batch_size` is chosen automatically.** The k-NN search used to default to
  `batch_size=2048`, which on a many-core machine or a large collection asked
  for several gigabytes of similarity blocks before any result appeared
  (6.9 GB at 40,000 vectors of 1024 dimensions on 44 threads). The default is
  now `None`: the batch is sized so that the blocks of all worker threads stay
  around 2 GB together (between 16 and 2048 rows, never more than the number
  of vectors); `vechealth.default_batch_size(n_vectors)` shows the value. An
  explicit `batch_size` is used as given, so a machine with more memory can
  raise it. The results do not depend on the batch size. The command-line
  `--batch-size` follows the same default.
- **A `MemoryUsageWarning` comes before a computation that may not fit.**
  Before the search starts the package estimates its peak memory
  (`vechealth.estimate_peak_memory_bytes`, within about 10% of the peaks
  measured for the README) and, above 8 GB, warns with the estimate, the
  batch size and what to do about it. The computation is never stopped: the
  warning is a `UserWarning` subclass, `VECHEALTH_MEMORY_WARNING_GB` moves the
  threshold, and the `warnings` filters silence it or turn it into an error.
- **Changes are no longer labelled "improved" or "regressed".** No statistic
  has a direction of change that is validated as good or bad, so every
  profile is `KnownDirection.AMBIGUOUS` and `interpret_drift` describes a
  change neutrally.
- **Status `URGENT` no longer depends on whether a repair exists.** It now
  means that the benchmark measured a severe index-dependent loss for the
  matched pathology under the index type you declared (hubness and noise
  under HNSW or IVF-PQ). Every other change beyond the noise floor is
  `OBSERVE`.
- **Index-dependent context is reported as retention** (percent of
  exact-search Recall@10 kept under HNSW and IVF-PQ) for all nine benchmark
  families, with its scope, instead of "loss multipliers" for two of them.
  `AnnVulnerability` / `ANN_VULNERABILITY` are replaced by `AnnRetention` /
  `ANN_RETENTION`.
- Recommendations describe the repairs of the preprint instead of naming
  functions, and carry `implemented_in_package=False`: no repair ships in
  this package. Their texts now include the measured effect and its caveats.
- Every `FullReport` carries `limitations`, and the command-line tool prints a
  short reading note after its numbers.
- `Badge.icon` and the `*_badge_icon` fields were removed; reports contain no
  emoji.

### Packaging

- The extension is built against the CPython stable ABI (`abi3-py310`): one
  wheel per platform for every CPython 3.10 or newer.
- The release profile strips symbols and enables thin LTO.
- The Qdrant and pgvector connectors are Cargo features that are on by
  default. The native LanceDB connector is an opt-in feature
  (`maturin build --features lancedb`); without it `from_lancedb` is served by
  a wrapper over the official SDK (`pip install vechealth[lancedb]`).
  `vechealth.compiled_features()` lists what a build contains.

### Known limitations

- The k-NN search is exact and brute-force: time grows as O(N²·d) and nothing
  is sampled, so the practical range is tens to a few hundred thousand
  vectors. A geometry-preserving reduction for larger collections is in
  development and is not part of this release.
- The search holds about four copies of the vectors in memory (16 KB per
  1024-dimensional vector) on top of the similarity blocks, and there is no
  streaming mode: a collection whose copies alone do not fit in memory cannot
  be analysed.
- Qdrant, pgvector, Weaviate and Pinecone have not been exercised against a
  live server. The pgvector connection is unencrypted.
