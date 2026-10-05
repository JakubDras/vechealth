# VecHealth

Label-free geometric diagnostics for embedding spaces and vector stores.

Eleven statistics computed from the vectors alone: no documents, no query text,
no relevance labels. Rust core, Python API, command-line tool.

> **Pre-release `0.2.0a1`, research software.** The statistics, noise floors and
> thresholds were derived on one benchmark: 53 vector stores built from 246,460
> BEIR documents (Qwen3-Embedding-0.6B, d = 1024) with nine injected defect
> families.

## Capabilities

- **Statistics.** Hubness (max, skewness), local intrinsic dimension,
  near-duplicate fraction, 1-NN and 10-NN dispersion, anisotropy, outlier
  fraction, neighbourhood consistency (SNC), query alignment (QMAS). One exact
  k-NN search, float64 accumulation.
- **Baseline comparison.** Versioned JSON snapshots. A later snapshot is compared
  with a baseline, and every change is judged against a per-statistic noise floor
  measured by subsampling (`significant` / `likely_noise`).
- **Defect matching.** Matches a pattern of changes to one of nine defect
  families (hubness, voids, imbalance, near-duplicates, anisotropy, dimensional
  collapse, fragmentation, noise, outliers), or reports it as ambiguous.
- **Index context.** For a declared index (`HNSW`, `IVFPQ`), reports the Recall@10
  retention measured in the benchmark for the matched defect. Under HNSW, hubness
  and noise retain 5.2% and 9.8% of exact-search recall; the other seven families
  98.4-100%.
- **Hub inspection.** Lists the points behind a hubness reading.
- **Data sources.** NumPy arrays; `.npy`, `.csv`, `.parquet`; LanceDB, Chroma,
  Milvus Lite (tested end to end); Qdrant, pgvector, Weaviate, Pinecone
  (implemented). Read through bulk scans, never through the store's ANN search.
- **CLI.** `vechealth analyze` and `vechealth compare --fail-on-significant`
  (exit code 1 when a statistic leaves its noise floor).
- **Scale.** Exact search, O(N²·d): 100,000 x 1024 in about 2 minutes, 246,460 x
  1024 in about 20 minutes and 6 GB (44 threads). The batch size is chosen
  automatically; a `MemoryUsageWarning` precedes searches estimated above 8 GB.

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
