# VecHealth reference

Everything the [README](../README.md) leaves out. Benchmark figures come from the
preprint behind the package (see [The benchmark](#the-benchmark)).

**Contents:** [Naming the likely defect](#naming-the-likely-defect) ·
[The eleven statistics](#the-eleven-statistics) ·
[Evidence and tests](#evidence-and-tests) ·
[What the evidence does not support](#what-the-evidence-does-not-support) ·
[Scale and memory](#scale-and-memory) · [Connectors](#connectors) ·
[Command line](#command-line) · [Building and testing from source](#building-and-testing-from-source) ·
[The benchmark](#the-benchmark)

## Naming the likely defect

`build_full_report` takes two snapshots keyed by statistic name. When the pattern
of changes clearly matches one defect family it says which; declaring the index
you run makes it add what the benchmark measured for that defect under that index:

```python
import vechealth as vh
from vechealth.interpretation import (
    FLATTEN_KEY_TO_METRIC_NAME, DeploymentContext, IndexType, build_full_report,
)

def snapshot(report):
    flat = report.flatten()
    return {name: flat[key] for key, name in FLATTEN_KEY_TO_METRIC_NAME.items() if key in flat}

full = build_full_report(snapshot(baseline), snapshot(current),
                         DeploymentContext(index_type=IndexType.HNSW))
print(full.grouped_message)                       # the defect family the pattern resembles, if clearly one
for r in full.metric_reports[:3]:
    print(r.status_label, r.level1_summary)       # per statistic, most notable first
print(*full.limitations, sep="\n")                # what this report does NOT say
```

Every report carries its limitations, and the command-line tool prints a short
version after its numbers. If two defects fit the pattern about equally well, the
report says so and names neither.

## The eleven statistics

`compute_all()` runs one exact k-NN search shared by all of them. `compute_all(queries=...)`
also computes the two query-side statistics from an array of query embeddings. The
defaults are `k=10`, `k_intrinsic_dim=20`, `duplicate_epsilon=0.05` and
`outlier_distance_threshold=1.2`; the last two are chord distances on the unit
sphere and were calibrated on one encoder.

| Statistic (report key) | What it reads | Designed for | Specific to it in the benchmark? |
|---|---|---|---|
| `hubness_max`, `hubness_skewness` | the largest and the skewness of how often a point appears in others' 10-NN lists | hubness | yes |
| `intrinsic_dim_mean` | mean Levina-Bickel local intrinsic dimension (k=20) | dimensional collapse | no: injected noise moves it 16.9x more, in the opposite direction |
| `ndds_fraction` | fraction of points whose nearest neighbour is closer than epsilon = 0.05 | near-duplicates | only by a practical tie (1.003x): anisotropy produces the same reading |
| `dispersion_1nn` | mean nearest-neighbour distance | fragmentation, noise | yes (through noise) |
| `dispersion_10nn` | mean distance over the 10-neighbourhood | fragmentation, noise | no: anisotropy moves it 1.19x more |
| `anisotropy_mean_norm` | norm of the mean vector | anisotropy | yes (1.057x over noise, which moves it the other way) |
| `outlier_fraction` | fraction of points whose nearest neighbour is farther than 1.2 | outliers | no: injected noise saturates it |
| `snc_score` | mean Jaccard overlap between a point's neighbourhood and its neighbours' | none | no designated defect |
| `qmas_mean_1nn`, `qmas_orphans_pct` | how close queries land to their nearest document; share with none within cosine distance 0.3 | none | no designated defect (needs `queries`) |

Void regions and cluster imbalance have no designated statistic: their largest
movements are in `qmas_orphans_pct`. Only the *movement* of `qmas_orphans_pct` is
interpretable, not its level: a third of the queries of the unmodified benchmark
corpus already have no document within 0.3.

## Evidence and tests

Each row pairs a use with the evidence behind it and with the tests that exercise
it in this repository.

| You can... | What the benchmark showed | Covered by |
|---|---|---|
| Compute the eleven statistics reproducibly | The implementation agrees with an independent float64 implementation on all 583 base-statistic cells (largest relative difference 6.2e-8); distances and reductions are accumulated in float64 | Rust unit tests check neighbour search, dispersion, intrinsic dimension, anisotropy and query alignment against float64 brute-force references |
| Recognise the pattern of an injected defect | Every designated statistic moves on its family in the expected direction; five of the eight designated statistics are specific to it, one of them by a practical tie | `test_signature_ambiguity.py` replays the 53-base benchmark and asserts that no level is matched to a foreign family |
| Tell a real move from noise, per statistic | Noise floors from sixty random subsamples of the corpus: 0.5, 2.8, 6.8 and 30.1 percentage points for the four reliability categories | `test_noise_filter.py`, `test_report.py`, `test_cli.py` |
| Gate a CI step on a baseline | `vechealth compare baseline.json current.json --fail-on-significant` exits non-zero when a statistic moves beyond its noise floor | `test_cli.py` and the CI smoke test |
| Avoid a repair that would hurt | The deduplication that raised Recall@10 by 15% on duplicated data cut it by 47.3% on anisotropic data with a near-identical `ndds_fraction` reading (0.9971 against 1.0000); recommendations are withheld unless the matched pattern is the one a repair was measured on | `test_fail_closed_recommendations.py` |
| See the index-dependent consequence | Under HNSW, hubness and noise kept 5.2% and 9.8% of their exact-search recall while every other family kept 98.4-100% (one severity level per family); under IVF-PQ the cost is graded. The grouping held on a second encoder | `test_deployment_context.py` |
| List the points behind a hubness reading | In a corpus with nothing injected, five of the twenty most frequent hubs named an immune checkpoint inhibitor across four cancer categories: hubness can be real semantic structure (a single case study) | `test_hub_inspector.py` |

## What the evidence does not support

The numbers above come from a controlled study. Read them with these limits.

- **It does not predict retrieval quality.** Across held-out defect families the
  statistics did not explain the variance of Recall@10 or nDCG@10, and the
  rank-order agreement that remained did not reach its pre-specified criterion
  (p = 0.0511). No positive predictive claim is made.
- **It is not a health score.** A single fixed-weight number would flatten the
  context that decides what a reading means, and the package does not offer one.
- **The direction of a change is not labelled good or bad.** The same statistic
  moves in opposite directions across defects, and several move toward what would
  read as improvement while recall falls.
- **A reading is a symptom, not a verdict.** Five of the eight statistics with a
  designated defect were specific to it, and the same reading can come from
  different causes, which is why repairs are gated on the matched pattern.
- **Thresholds do not travel.** The near-duplicate radius (0.05), the outlier
  threshold (1.2), the signature threshold and margin and the noise floors were
  calibrated on one benchmark: one corpus of 246,460 BEIR documents embedded with
  Qwen3-Embedding-0.6B (1024 dimensions) and injected defects, with a second
  encoder, two further corpora and a corpus with nothing injected used only for
  narrower checks. Recalibrate them for other encoders, dimensionalities and
  corpora; absolute thresholds are not comparable across dimensionalities.
- **Comparing a deployment with its own history is a hypothesis.** Whether changes
  of these statistics within one deployment track changes of its recall was not
  tested: the measurements compare different collections, not one collection over
  time.
- **The index comparison has its own limits.** One severity level per family, one
  HNSW and one IVF-PQ configuration on ten bases (a second configuration on four).
  The ratio of a base's recall loss to the baseline's varies by about 25% between
  index builds, which is why retention rather than a loss multiplier is reported.
  Other index families are untested.
- **No repair ships in this package.** Recommendations describe the repairs
  measured in the preprint, flagged `implemented_in_package=False`; the package
  never modifies your vectors.

## Scale and memory

**Current limitation.** Every neighbourhood-based statistic is computed from an
*exact*, brute-force k-NN search over all pairs, and nothing is sampled. Time
therefore grows as **O(N²·d)**.

**Memory.** The search holds about four copies of the vectors (the array you pass
in, the evaluator's own copy, a normalised one, and one more clone made while
searching: 16 KB per 1024-dimensional vector) plus one block of similarities per
worker thread, `threads × batch_size × N × 4` bytes in all.

**`batch_size` is chosen for you.** The default keeps the blocks of all threads
together at about 2 GB (between 16 and 2048 rows, never more than the number of
vectors), so a laptop and a many-core workstation get different batch sizes for
the same data; `vechealth.default_batch_size(n_vectors)` returns the value. That is
a trade of speed for memory: a bigger batch means fewer passes over the vectors and
is faster, until there are fewer blocks than threads. On 100,000 vectors, 256 rows
took 83 s against 111 s for the automatic 113 and 151 s for 64, at 4.5 GB of blocks
instead of 2 GB; on 40,000 vectors 2048 rows (only 20 blocks for 44 threads) were
slower than 284. If you have the memory to spare, pass `batch_size=` to any compute
method (or `--batch-size` on the command line). The results do not depend on it.

**A warning comes before the work starts.** The package estimates the peak memory
first (`vechealth.estimate_peak_memory_bytes(n_vectors, dim)`, within about 10% of
the peaks measured below). Above 8 GB it raises a `vechealth.MemoryUsageWarning`
that gives the estimate, the batch size and what to change, and then carries on.
Raise the limit with the environment variable `VECHEALTH_MEMORY_WARNING_GB`, silence
it with `warnings.filterwarnings("ignore", category=vechealth.MemoryUsageWarning)`,
or turn it into an error with `"error"` in place of `"ignore"`.

For orientation, `compute_all` (with 2,000 queries) on random subsets of the
benchmark corpus: 1024 dimensions, the default `batch_size`, a release build on a
44-thread Xeon E5-2699 v4 workstation:

| vectors | automatic `batch_size` | time | peak memory | estimate |
|---:|---:|---:|---:|---:|
| 10,000 | 1136 | 6 s | 0.6 GB | 0.6 GB |
| 20,000 | 568 | 9 s | 2.0 GB | 1.9 GB |
| 40,000 | 284 | 18 s | 2.7 GB | 2.7 GB |
| 100,000 | 113 | 111 s | 3.9 GB | 3.6 GB |
| 246,460 (the whole corpus) | 46 | 20 min | 6.2 GB | 6.0 GB |

At small sizes fixed costs (the d×d covariance for anisotropy, the query side)
dominate; the quadratic term takes over as N grows. On the whole corpus the 2 GB
budget forces small blocks: an earlier run with `batch_size=64` took about 16
minutes and peaked at 7.1 GB. Times are medians of one to three runs, which differ
by up to 30% on this shared workstation; your hardware will differ, and the
automatic batch size follows your number of threads.

The practical range today is tens to a few hundred thousand vectors. It does not
scale to multi-million-vector collections, and `compute_all` does not refuse a size
it cannot handle: it warns when the estimate is high, then runs, or runs out of
memory.

**Naive random subsampling is not a safe workaround.** In the benchmark's
subsampling study only `anisotropy_mean_norm` was stable on a subsample (maximum
error 0.53%); `ndds_fraction` reads about *p* times its full-corpus value on a
fraction *p*; and `hubness_max` and `qmas_orphans_pct` vary *more* at 50% sampling
than at 0.5%.

**We are working on this.** A geometry-preserving reduction, a coreset-style
geometry-aware sampling that keeps the structure these statistics depend on (local
density and degree for hubness, spectral structure for anisotropy, local dimension)
while shrinking the collection for monitoring, is in active development. It is not
part of this release.

## Connectors

Every connector reads through the store's bulk-scan path (scroll, cursor, iterator)
and never through its ANN search, so pulling data for a diagnostic does not compete
with your query traffic. Each pulls the *complete* collection; page and batch
parameters only control how much is requested per round trip.

```python
evaluator = vh.VecHealthEvaluator.from_qdrant("http://localhost:6333", "my_collection")
# or from_pgvector / from_lancedb / from_weaviate / from_chroma / from_milvus /
# from_pinecone / from_local("vectors.parquet"): the same API either way
```

| Source | Implemented in | Status |
|---|---|---|
| `.npy`, `.csv`, `.parquet` files (`from_local`) | Rust | tested |
| LanceDB (`from_lancedb`) | official SDK (`vechealth[lancedb]`); a native Rust connector is an opt-in build feature | tested end to end on a local database |
| Chroma (`from_chroma`), Milvus Lite (`from_milvus`) | official SDKs | tested end to end on a local store |
| Qdrant (`from_qdrant`), pgvector (`from_pgvector`) | Rust | implemented; error paths tested, **not exercised against a live server**. The pgvector connection is unencrypted: use a tunnel or a trusted network |
| Weaviate (`from_weaviate`) | official SDK | implemented; **not exercised against a live server** |
| Pinecone (`from_pinecone`) | official SDK | implemented; **not exercised against a live account**. Pinecone has no bulk-export endpoint, so a full pull is many small requests (100 IDs per page): prefer a snapshot over a live production index |

A warning is raised when the source collection was indexed with a distance other
than cosine, because VecHealth L2-normalises the vectors it reads.

## Command line

```bash
vechealth analyze vectors.npy --output baseline.json --tag stage=launch

# ...later...
vechealth analyze vectors_now.npy --output current.json
vechealth compare baseline.json current.json --fail-on-significant
```

`analyze` reads `.npy`, `.csv` and `.parquet` files. `compare` marks every statistic
`significant`, `likely_noise` or `unassessed` against its measured noise floor, and
`--fail-on-significant` exits with status 1 if anything moved beyond it, a ready-made
CI gate. `--outlier-distance-threshold` takes a value or `adaptive` (derive it from
the data; on typical text embeddings that lands close to the largest possible
distance, 2.0, and flags essentially nothing). Both commands end with a short note on
how to read the numbers.

## Building and testing from source

```bash
cd rust && cargo test --workspace                       # Rust unit tests
pytest rust/bindings/python/tests                       # Python-facing suite (after maturin develop)
cargo test -p vechealth-connectors --features lancedb   # native LanceDB connector (needs protoc)
```

`vechealth.compiled_features()` lists the optional native connectors in a build
(`qdrant`, `pgvector` and, with `maturin build --features lancedb`, `lancedb`). If
you built an earlier development version in place, delete the stale
`rust/bindings/python/vechealth/_core.cpython-*.so`: it would shadow the new
`_core.abi3.so`.

## The benchmark

The statistics, thresholds and figures come from a controlled benchmark of 53 vector
stores built from one corpus: 246,460 BEIR documents embedded with
Qwen3-Embedding-0.6B (d = 1024). It holds the unmodified baseline, nine injected
defect families (hubness, voids, imbalance, near-duplicates, anisotropy, dimensional
collapse, fragmentation, noise, outliers) at five to seven severity levels each, and
four mixed bases. A preprint describing it is in preparation.
