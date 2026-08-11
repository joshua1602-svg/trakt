"""regulatory_watch.discovery — Stage 1B: ESMA publication discovery.

Stage 1A answers *"has the machine-readable Annex 2 specification changed?"*
by normalizing and diffing technical artefacts. It cannot see anything coming:
a consultation, a final report or a Q&A moves nothing in the workbook or the
XSD, so Stage 1A correctly reports `CURRENT` right up until the day a new
technical package lands.

Stage 1B is that missing upstream layer::

    ESMA publication discovery -> deduplication -> Annex 2 relevance triage
        -> regulatory-development record -> optional Stage 1A linkage

The distinction it exists to preserve:

* a **regulatory development** is a publication that *may* affect Annex 2. It
  is narrative. Stage 1B classifies it and asks a human to look.
* a **technical regulatory delta** is an exact, machine-readable difference in
  the Annex 2 field set. Only Stage 1A may determine one.

Stage 1B therefore **never** converts narrative text into a regime change, and
never computes a field delta. When a publication carries a technical artefact
it recognises, it says so and links to Stage 1A's own report — it does not
reimplement the comparison.

Like Stage 1A this is observational: no active config is written, no approval
workflow, no notifications, no scheduler of its own.
"""

from __future__ import annotations

#: Bumped when the discovery/parse/identity rules change in a way that could
#: alter a development_id or a triage outcome for the same input.
DISCOVERY_VERSION = "annex2-discovery/1.0.0"

#: Bumped when the weekly discovery report JSON contract changes.
DISCOVERY_REPORT_CONTRACT_VERSION = "1.0.0"

AUTHORITY = "ESMA"
REGIME = "ESMA_Annex2"

__all__ = ["DISCOVERY_VERSION", "DISCOVERY_REPORT_CONTRACT_VERSION",
           "AUTHORITY", "REGIME"]
