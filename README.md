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
the geometry and topology of your embedding space directly.

```python
import vechealth as vh

report = vh.analyze("path/to/your/vectorstore")
print(report.health_score)      # 0.73
print(report.pathologies)       # ["hubness", "void_regions"]
print(report.recommendations)   # ["Consider re-indexing with higher M parameter"]
```

## Detected Pathologies

| Pathology | What it means | Impact on retrieval |
|-----------|--------------|---------------------|
| Hubness | Few vectors dominate all k-NN results | Low diversity |
| Void regions | Dead semantic zones in embedding space | Coverage failures |
| Anisotropy | Vectors clustered in narrow cone | Cosine similarity breaks down |
| Embedding collapse | Low intrinsic dimensionality | ANN recall degrades |
| Near-duplicate flood | Redundant vectors dominate retrieval | Low diversity |

## Status

> 🔬 Active research project — Paper #1 in preparation.
> Star the repo to follow progress.

## Research

This project is developed as part of research into geometric and
topological analysis of embedding spaces. Papers coming soon.

## License

Apache 2.0
