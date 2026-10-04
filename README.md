# VecHealth

> A smoke detector for your embeddings.

VecHealth computes eleven geometric statistics of a set of embedding vectors and
tells you when the geometry has a known defect or has changed. It never reads
your documents, your query text or relevance labels, so it works on any vector
store you can read vectors from. The numerics are written in Rust; you use it
from Python or from the command line.

> **Pre-release (`0.2.0a1`).** The first public test version: the API may
> change, and the thresholds were derived on one benchmark (one corpus, one
> encoder). See [Good to know](#good-to-know).

## What it is for

**As a smoke detector.** Take a snapshot of the statistics while your system is
healthy, and compare later snapshots with it. VecHealth tells a real change from
ordinary noise, statistic by statistic, and when a pattern clearly matches one
known defect (hubness, near-duplicates, anisotropy, noise, ...) it says which. In
the benchmark behind it, every statistic designed for a defect moved on that
defect in the expected direction. Like a smoke detector, it says *something
changed, look here*; it does not say how much retrieval quality suffers.

**To choose a search index.** What a defect costs depends on the index. In the
benchmark, under HNSW the hubness and noise defects kept only 5% and 10% of
exact-search recall while every other defect kept 98-100%; under IVF-PQ the cost
was graded. If your vectors read like hubness or noise, HNSW is the index to test
first. VecHealth reports what was measured for the index you declare, and the
paper's advice is to confirm with a pilot: compare approximate with exact recall
on a sample, on the configuration you intend to deploy.

**What it is not.** It is not a quality predictor or a health score: on held-out
defect families the statistics did not predict retrieval quality. And watching
them over time is, so far, a well-motivated idea rather than a tested result: the
statistics are label-free and cheap to recompute, but whether changes within one
deployment track changes in its recall has not been tested yet.

## Install

Python 3.10 or newer and NumPy. One wheel per platform (Linux, macOS, Windows).

```bash
pip install vechealth==0.2.0a1      # a pre-release: ask for the exact version
```

Optional connectors: `pip install "vechealth[lancedb]==0.2.0a1"` (also `weaviate`,
`chroma`, `milvus`, `pinecone`). To build from source you need a Rust toolchain:
`pip install git+https://github.com/JakubDras/vechealth`.

## Quick start

```python
import numpy as np
import vechealth as vh
from vechealth.interpretation import assess_comparison

# 1. While the system is healthy: take and save a baseline snapshot.
baseline = vh.VecHealthEvaluator(np.load("embeddings.npy")).compute_report()
baseline.save("baseline.json")

# 2. Later: take a new snapshot and compare it with the baseline.
current = vh.VecHealthEvaluator(np.load("embeddings_now.npy")).compute_report()
comparison = current.compare(vh.Report.load("baseline.json"))

for name, verdict in assess_comparison(comparison).items():
    print(name, comparison.deltas[name].delta_pct, verdict.verdict.value)
    # verdict: significant / likely_noise / unassessed
```

The same from the command line, ready for a CI step:

```bash
vechealth analyze vectors.npy --output baseline.json
vechealth analyze vectors_now.npy --output current.json
vechealth compare baseline.json current.json --fail-on-significant   # exit 1 if something really moved
```

To name the likely defect and see what the benchmark measured for your index, see
[the reference](https://github.com/JakubDras/vechealth/blob/main/docs/reference.md#naming-the-likely-defect).

## Good to know

- **Not a quality predictor, not a score.** One fixed-weight number would hide what
  a reading means, so there is none, and a change is never labelled good or bad.
- **A reading is a symptom, not a verdict.** Five of the eight statistics tied to
  a defect were specific to it; the same reading can come from different causes.
- **One benchmark.** One corpus (246,460 BEIR documents), one encoder
  (Qwen3-Embedding-0.6B), injected defects. Thresholds such as the near-duplicate
  radius (0.05), the outlier distance (1.2) and the noise floors may not carry
  over: recalibrate them for your encoder and data.
- **Exact search, so it scales as O(N²·d).** Comfortable for tens to a few hundred
  thousand vectors: 100,000 vectors of 1024 dimensions take about 2 minutes, the
  whole 246,460-vector corpus about 20 minutes and 6 GB on a 44-thread
  workstation. It does not yet scale to millions; a geometry-aware sampling that
  keeps what the statistics read is in development. The batch size is chosen
  automatically, and a warning comes before a search that may not fit in memory.
- **Connectors.** Files (`.npy`, `.csv`, `.parquet`), LanceDB, Chroma and Milvus
  Lite are tested end to end; Qdrant, pgvector, Weaviate and Pinecone are
  implemented but not yet tried against a live server.
- **No repair ships.** Recommendations describe the repairs measured in the
  paper; the package never modifies your vectors.

## More

- [Reference](https://github.com/JakubDras/vechealth/blob/main/docs/reference.md):
  the eleven statistics, naming a defect, evidence and tests, scale and memory,
  connectors, building from source.
- [Changelog](https://github.com/JakubDras/vechealth/blob/main/CHANGELOG.md).
- The benchmark behind it, 53 vector stores built from one corpus (the
  unmodified baseline, nine injected defect families and four mixed bases), is
  described in a preprint that is in preparation.

## License

Apache 2.0.
