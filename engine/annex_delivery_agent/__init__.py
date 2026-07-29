"""
engine.annex_delivery_agent
===========================

The Annex Delivery Agent — the governed path from projected regulatory data to a
submission-ready artefact, for any annex report.

    projected data → delivery normalisation → artefact → schema validation
    → governed result → explicit approval → publication

One lifecycle, one governance model, one persistence layer, one set of evidence
contracts. Everything report-specific lives behind
:class:`~engine.annex_delivery_agent.providers.base.AnnexProvider`, so adding
Annex 3 or Annex 9 means writing a provider and registering it — not cloning the
agent.

ESMA Annex 2 is implemented by **wrapping** the proven repository route
(``regime_projector`` → ``annex2_delivery_normalizer`` → ``xml_builder_annex2``
→ ``DRAFT1auth.099.001.04_1.3.0.xsd`` → ``lxml`` validation). None of that
business logic is reimplemented or altered here; the agent composes it, times
it, instruments what it does to the data, and governs what happens next.

Two rules the package exists to hold:

* **XSD validity is not regulatory completeness.** A valid artefact is
  *prepared*, never published. See :mod:`.publication`.
* **Automatic values are disclosed.** Everything the builder filled in, coerced
  or omitted is surfaced as governed evidence. See :mod:`.evidence`.

Example::

    from trakt_core.context import ExecutionContext
    from engine.annex_delivery_agent import AnnexDeliveryAgent, DeliveryRequest

    agent = AnnexDeliveryAgent()
    result = agent.prepare(
        ExecutionContext.for_internal("ERE"),
        DeliveryRequest(annex_id="ESMA_Annex2",
                        reporting_date="2025-11-30",
                        projected_input="out/…_ESMA_Annex2_projected.csv",
                        output_root="out_annex_delivery"))
"""

from .agent import CAPABILITY, DEFAULT_STORE_ROOT, AnnexDeliveryAgent
from .contracts import (
    AnnexDefinition, ApprovalState, ArtefactRef, DeliveryRequest, DeliveryResult,
    DeliveryStatus, SchemaValidationResult,
)
from .evidence import TransformationEvidence
from .persistence import DeliveryRunStore
from .preflight import PreflightFinding, PreflightReport
from .providers.base import AnnexProvider
from .publication import FilesystemPublicationSink, PublicationGate, PublicationSink
from .registry import AnnexRegistry, default_registry

__all__ = [
    "AnnexDeliveryAgent", "CAPABILITY", "DEFAULT_STORE_ROOT",
    "AnnexDefinition", "DeliveryRequest", "DeliveryResult", "DeliveryStatus",
    "ApprovalState", "ArtefactRef", "SchemaValidationResult",
    "TransformationEvidence",
    "DeliveryRunStore",
    "PreflightReport", "PreflightFinding",
    "AnnexProvider",
    "PublicationGate", "PublicationSink", "FilesystemPublicationSink",
    "AnnexRegistry", "default_registry",
]
