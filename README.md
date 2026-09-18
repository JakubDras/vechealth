# VecHealth

> Observability and diagnostic framework for embedding spaces and vector stores.

Like Prometheus monitors infrastructure and MLflow monitors models,
**VecHealth monitors the health of your embedding space**.

## The Problem

When RAG retrieval quality drops, teams typically try to fix it by:
- Changing the embedding model
- Rebuilding the vector store
- Modifying chunking strategy

This is expensive and time-consuming — and often doesn't answer:
**Where is the actual root cause?**

## What VecHealth Does

VecHealth answers *why* retrieval quality degrades by analyzing
the geometry of your embedding space directly — no labeled data,
no LLM calls, just the vectors you already have.

```python
import vechealth as vh

# Read straight from your vector store — no manual export needed.
evaluator = vh.VecHealthEvaluator.from_qdrant("http://localhost:6333", "my_collection")
# ...or from_pgvector / from_lancedb / from_weaviate / from_chroma / from_milvus /
# from_pinecone / from_local("vectors.parquet") — same API either way.

metrics = evaluator.compute_all()
print(metrics.hubness.hubness_skewness)   # 0.42
print(metrics.anisotropy.mean_vector_norm)
print(metrics.duplicates.ndds_fraction)
```

`compute_all()` runs 8 geometric diagnostics in one pass — the underlying
k-NN search is shared across metrics, so this is much cheaper than calling
each one separately.

## Detected Pathologies

| Pathology | What it means | Impact on retrieval |
|-----------|--------------|---------------------|
| Hubness | Few vectors dominate all k-NN results | Low diversity |
| Void regions | Dead semantic zones in embedding space | Coverage failures |
| Anisotropy | Vectors clustered in narrow cone | Cosine similarity breaks down |
| Embedding collapse | Low intrinsic dimensionality | ANN recall degrades |
| Near-duplicate flood | Redundant vectors dominate retrieval | Low diversity |

Every pathology above is backed by controlled experiments measuring its
actual effect on retrieval recall — not a heuristic guess. See `papers/`
for the research this is built on.

## Tracking health over time

The recommended way to use VecHealth in production isn't a single absolute
score — it's comparing against your *own* baseline, taken when the system
was known-healthy:

```python
baseline = evaluator_at_launch.compute_report(tags={"stage": "launch"})
baseline.save("baseline.json")

# ...weeks later...
current = evaluator_now.compute_report()
comparison = current.compare(vh.Report.load("baseline.json"))

for key, delta in comparison.deltas.items():
    print(key, delta.delta_pct)
```

On top of raw deltas, `vechealth.interpretation` adds a full interpretation
layer: which metrics are trustworthy at all (`metric_profiles`), whether a
given swing is likely real or just this metric's known noise floor
(`assess_comparison` — thresholds grounded in measured subsampling
variance, not guesses), causal-evidence-aware recommendations, and
deployment-context-aware warnings (e.g. hubness matters far more if you're
serving over HNSW than over exact search). It's a toolkit, not yet a single
`analyze()` call — see `vechealth.interpretation.build_full_report` for the
most complete entry point today.

## Command-line interface

For quick, ad-hoc checks or a CI/CD pipeline step — no Python required:

```bash
vechealth analyze vectors.npy --output baseline.json --tag stage=launch

# ...later...
vechealth analyze vectors_now.npy --output current.json
vechealth compare baseline.json current.json --fail-on-significant
```

`compare` annotates every metric's delta with the same noise-vs-signal
verdict `interpretation.assess_comparison` computes in Python
(`significant` / `likely_noise` / `unassessed`), and `--fail-on-significant`
exits with status 1 if anything moved beyond its measured noise floor — a
ready-made CI gate for exactly the "hubness rose 30%, is that real?"
question this project keeps coming back to.

`analyze` reads local `.npy`/`.csv`/`.parquet` files today (`from_local`);
checking a live connector is still a few lines of Python (see above) rather
than a CLI flag — every connector takes different arguments, so this is a
deliberate first-version scope limit, not an oversight.

## Connecting to a live vector store

Read directly from Qdrant, pgvector, LanceDB, Weaviate, Chroma, Milvus, or
Pinecone — every connector uses the vendor's bulk-scan/scroll path, never
the ANN search path, so pulling data for diagnostics never competes with
your production query traffic. See `rust/TODO_Conectors.md` for the design
rationale and current verification status per connector.

## Status

VecHealth's core is implemented and tested: 8 geometric metrics, baseline
tracking, an interpretation layer (causal-evidence-aware recommendations,
noise-vs-signal filtering, deployment-context calibration), and connectors
to 8 vector stores/file formats.

**Known current limitation — actively being worked on:** the k-NN engine
underneath these metrics is exact/brute-force, which is fine up to the
hundreds-of-thousands-of-vectors range but doesn't scale to multi-million
vector stores. Naive random subsampling doesn't safely fix this — we
measured that it distorts several of these metrics (see `papers/paper1` and
the subsampling-stability work referenced there). A geometry-aware
reduction algorithm, designed to shrink a dataset for monitoring purposes
*without* destroying the structure these metrics depend on, is in active
development — see `papers/future_papers/` for the research angle.

Not yet a stable, versioned PyPI release — API may still shift.
Star the repo to follow progress.

## Research

This project is developed as part of research into geometric and
topological analysis of embedding spaces. Papers coming soon.

## License

Apache 2.0
