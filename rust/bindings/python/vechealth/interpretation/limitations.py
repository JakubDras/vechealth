"""
interpretation/limitations.py

The standing statements about what a VecHealth reading does NOT say.

They restate, in report form, the limits the accompanying preprint puts on
these statistics (Sections 4.3-4.10, 5 and 6). They are attached to every
`FullReport` and summarised in a short note at the end of the CLI output, so
a number is never shown without the caveats that travel with it.
"""

from __future__ import annotations

__all__ = ["LIMITATIONS", "SHORT_READING_NOTE", "limitations_text"]


LIMITATIONS: tuple[str, ...] = (
    # Section 4.5
    "These are content-free geometric statistics. In the benchmark they did not "
    "predict retrieval quality across held-out pathology families (leave-one-family-"
    "out validation; the rank-order agreement did not reach the pre-specified "
    "criterion, p = 0.0511), so no predictive claim is made.",
    # Section 6, item 5
    "A change against your own baseline is a symptom, not a verdict. Whether "
    "changes of these statistics within one deployment track changes of its recall "
    "was not tested.",
    # Sections 4.3-4.4
    "The direction of a change is not labelled good or bad: in the benchmark the same "
    "statistic moved in opposite directions across pathology families, and several "
    "moved toward what would read as improvement while Recall@10 fell (dimension "
    "truncation).",
    # Section 4.3
    "Of the eight statistics that have a designated pathology family only five were "
    "specific to it (one of them by a practical tie), and the same reading can come "
    "from different causes. Void regions and cluster imbalance have no designated "
    "statistic.",
    # Sections 4.8-4.9, 5
    "The evidence comes from one corpus (a BEIR mixture), one primary encoder "
    "(Qwen3-Embedding-0.6B, 1024 dimensions) and injected pathologies on one severity "
    "grid per family; a second encoder, two further corpora and a corpus with nothing "
    "injected were used only for narrower checks.",
    # Sections 4.4, 4.10, 6 item 4
    "Thresholds are absolute or calibrated on that benchmark: the near-duplicate "
    "radius (0.05) and the outlier threshold (1.2) are chord distances, and the "
    "signature threshold, its margin and the per-statistic noise floors were derived "
    "from it. Do not port them across encoders, dimensionalities or corpora without "
    "recalibrating.",
    # Section 4.10, limitation 9
    "Pathology-family matching was calibrated and evaluated on the same benchmark. A "
    "similarity of 1.00 means the pattern equals a benchmark signature, not that the "
    "family is confirmed in your data.",
    # Section 4.9
    "The noise floors come from repeated random subsampling of one healthy corpus; "
    "that is a proxy for snapshot-to-snapshot variation, not a direct measurement of "
    "it.",
    # Section 4.8, Section 6 item 1
    "Index-dependent figures are Recall@10 retention measured at one severity level "
    "per family under specific index settings. Measure exact-versus-approximate "
    "recall on a sample of your own queries before drawing conclusions.",
    # Section 4.7
    "Repairs named in recommendations are described in the preprint but are not "
    "implemented in this package, which never modifies your vectors. Their measured "
    "effects are scoped to the benchmark, and part of the gain of one of them depends "
    "on its evaluation protocol.",
)


# What the command-line tool prints after its numbers.
SHORT_READING_NOTE = (
    "Reading these numbers: they are content-free geometric statistics. In the "
    "benchmark they did not predict retrieval quality, and the direction of a change "
    "is not validated as good or bad. Use them to compare a deployment with its own "
    "history, calibrate thresholds to your data, and measure retrieval quality "
    "yourself. See the README ('Good to know')."
)


def limitations_text(limitations: tuple[str, ...] = LIMITATIONS) -> str:
    """The limitations as a numbered plain-text block."""
    return "\n".join(f"{i}. {text}" for i, text in enumerate(limitations, start=1))
