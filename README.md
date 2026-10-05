# VecHealth

**Find geometric defects in an embedding store, and see what they cost under your
ANN index, from the vectors alone.**

No documents, no query text, no relevance labels, no LLM calls. Rust core, Python
API, command-line tool.

Evaluation frameworks such as RAGAs and ARES score a system's outputs; they do not
compute signals from the geometry of the vector store. VecHealth does: eleven
statistics from one exact k-NN pass, a noise floor per statistic, and a match
against nine defect families measured in a benchmark.

> **Pre-release `0.2.0a1`, research software.** Statistics, noise floors and
> thresholds come from one benchmark: 53 vector stores built from 246,460 BEIR
> documents (Qwen3-Embedding-0.6B, d = 1024).

## Why use it

- **It sees defects that otherwise need labels.** In the benchmark's nine injected
  defect families (hubness, voids, imbalance, near-duplicates, anisotropy,
  dimensional collapse, fragmentation, noise, outliers), every statistic designed
  for a defect moved on it in the expected direction.
- **The cost depends on your index, and it reports it.** Under HNSW, hubness and
  noise cost 90-95% of exact-search recall (5.2% and 9.8% retained); the other seven
  families cost at most 1.6%. Under IVF-PQ the cost is graded. Declare your index
  and the report quotes the measured figure.
- **It recommends a repair only where the repair was measured.** The same
  deduplication that helps on duplicated data lowered Recall@10 by 47.3% on an
  anisotropic store with a near-identical duplicate reading, so a recommendation is
  withheld unless the matched pattern is the one the repair was measured on.
- **It fits a pipeline.** Versioned snapshots, per-statistic noise floors, and
  `vechealth compare --fail-on-significant` as a CI gate. It reads vectors from
  your store by bulk scan, never through its ANN search. 100,000 x 1024 takes about
  2 minutes.

## Example

The benchmark's own statistics: the unmodified baseline against the same store with
injected hubness, HNSW declared. Output, abridged:

```
look like a pattern close to: Hubness (dominant points in k-NN) (similarity 100%).

URGENT  'hubness_skewness' changed by +801.5% relative to your baseline (from 3.183
to 28.69). [...]

You declared an HNSW index. In the benchmark behind this package (one corpus,
Qwen3-Embedding-0.6B, HNSW M = 16, efConstruction 200, efSearch 128), a store with
the hubness pathology (hub attraction alpha = 0.6, the strongest level) kept 5.2% of
its exact-search Recall@10 under HNSW (BGE-large: 7.32%), against 99.4% for the
unperturbed baseline (BGE-large: 99.84%). [...] Measure exact-versus-approximate
recall on a sample of your own queries before drawing conclusions.

NICDM rescaling of the retrieval distance (k = 10): divide each cosine distance by
the geometric mean of the two points' local scales [...]
```

## What is inside

- **Statistics:** hubness (max, skewness), local intrinsic dimension, near-duplicate
  fraction, 1-NN and 10-NN dispersion, anisotropy, outlier fraction, neighbourhood
  consistency, query alignment. Float64 accumulation.
- **Hub inspection:** lists the points behind a hubness reading.
- **Data sources:** NumPy arrays; `.npy`, `.csv`, `.parquet`; LanceDB, Chroma,
  Milvus Lite (tested end to end); Qdrant, pgvector, Weaviate, Pinecone (implemented).
- **Scale:** exact search, O(N²·d); 246,460 x 1024 in about 20 minutes and 6 GB
  (44 threads). The batch size is chosen automatically, and a `MemoryUsageWarning`
  precedes searches estimated above 8 GB.

## Install

```bash
pip install vechealth==0.2.0a1
```

Python 3.10+. Wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64) and
Windows (x64). Optional connectors: `pip install "vechealth[lancedb]==0.2.0a1"`
(also `weaviate`, `chroma`, `milvus`, `pinecone`).

## Usage

```python
import numpy as np
import vechealth as vh
from vechealth.interpretation import assess_comparison

evaluator = vh.VecHealthEvaluator(np.load("embeddings.npy"))
metrics = evaluator.compute_all()                    # one reading, all statistics
evaluator.compute_report().save("baseline.json")     # snapshot to compare against later

current = vh.VecHealthEvaluator(np.load("embeddings_now.npy")).compute_report()
comparison = current.compare(vh.Report.load("baseline.json"))
for name, verdict in assess_comparison(comparison).items():
    print(name, comparison.deltas[name].delta_pct, verdict.verdict.value)
```

```bash
vechealth analyze vectors.npy --output baseline.json
vechealth analyze vectors_now.npy --output current.json
vechealth compare baseline.json current.json --fail-on-significant
```

## Documentation

[Reference](https://github.com/JakubDras/vechealth/blob/main/docs/reference.md):
defect matching and index context in code, the statistics, evidence and tests,
limits, scale and memory, connectors, building from source.
[Changelog](https://github.com/JakubDras/vechealth/blob/main/CHANGELOG.md).

## License

Apache 2.0.
