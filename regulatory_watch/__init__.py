"""regulatory_watch — ESMA Annex 2 Regulatory Watch (Stage 1, observational).

Stage 1 answers one narrow question, deterministically and offline:

    Has the authoritative ESMA Annex 2 regulatory specification changed relative
    to the version Trakt currently implements, and if so which existing Trakt
    Annex 2 components are likely affected?

It is **observational only**. Nothing in this package writes to an active
regime/delivery config, creates a regime version, generates XML, or changes the
canonical -> regime -> validation -> XML pathway. Every Trakt config it touches
is opened read-only.

Flow::

    authoritative snapshot -> normalized Annex 2 spec -> comparator -> impact

    manifest.py    the explicit ESMA source allowlist (no scraping)
    retrieval.py   optional, isolated network fetch (never used by tests)
    snapshots.py   immutable content-addressed snapshot store (sha256)
    annex2_spec.py deterministic normalizer: workbook + XSD -> NormalizedSpec
    changelog.py   ESMA's own published change log, parsed as evidence
    comparator.py  deterministic spec-vs-spec diff
    impact.py      delta -> likely Trakt component impact
    report.py      machine-readable JSON + human-readable Markdown
    cli.py         end-to-end runner

The data contract in :mod:`regulatory_watch.contracts` is UI-agnostic so a
Stage 2 OCC Regulatory Config view can consume the JSON report unchanged.
"""

from __future__ import annotations

#: Bumped whenever the normalizer's output shape or derivation rules change.
#: Recorded on every snapshot and every normalized spec so a stored spec can
#: never be silently compared across incompatible parser generations.
PARSER_VERSION = "annex2-normalizer/1.1.0"

#: Bumped whenever the JSON report contract changes.
REPORT_CONTRACT_VERSION = "1.1.0"

REGIME = "ESMA_Annex2"

__all__ = ["PARSER_VERSION", "REPORT_CONTRACT_VERSION", "REGIME"]
