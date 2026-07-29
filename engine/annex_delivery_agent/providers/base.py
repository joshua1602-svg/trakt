"""
engine.annex_delivery_agent.providers.base
==========================================

The annex plug-in interface.

Everything annex-specific lives behind this class: which projector output to
read, which normaliser to run, how to build the artefact, which schema to check
it against, and what the builder did to the data along the way. The shared
lifecycle calls these methods in a fixed order and never inspects an annex field
code.

The contract in one line per method:

``preflight``      — cheap blocking checks, before anything expensive.
``resolve_input``  — locate and describe the projected data.
``normalise``      — projected data → delivery-ready data (annex's own rules).
``build``          — delivery-ready data → the regulatory artefact.
``validate``       — artefact → schema result.
``inspect``        — what the builder filled, coerced, dropped or assumed.
``classify``       — which of those are blocking and which are warnings.
``publication_metadata`` — what a published package must reference.

A provider is constructed once per run and may hold per-run state between calls
(the Annex 2 provider keeps the mapping specs it loaded in ``build`` so
``inspect`` does not reload a 9.6 MB workbook).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..contracts import AnnexDefinition, ArtefactRef, DeliveryRequest, SchemaValidationResult
from ..evidence import TransformationEvidence
from ..preflight import PreflightReport


@dataclass
class ResolvedInput:
    """The projected data this run will deliver, located and described."""

    path: Path
    row_count: int
    field_count: int
    field_codes: Tuple[str, ...] = ()
    content_hash: str = ""
    #: Anything the provider wants to carry into later stages.
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "row_count": self.row_count,
            "field_count": self.field_count,
            "field_codes": list(self.field_codes),
            "content_hash": self.content_hash,
        }


@dataclass
class NormalisationOutcome:
    """The result of the annex's own delivery normalisation."""

    #: The delivery-ready dataset written to disk.
    artefact: ArtefactRef
    row_count: int
    field_count: int
    #: Everything the normaliser objected to, in its own vocabulary.
    issues: List[Dict[str, Any]] = field(default_factory=list)
    blocking_issue_count: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)
    side_artefacts: List[ArtefactRef] = field(default_factory=list)
    #: In-memory frame handed to :meth:`AnnexProvider.build`, so the lifecycle
    #: never round-trips a large dataset through disk between two stages that
    #: both already hold it.
    frame: Any = None

    @property
    def blocked(self) -> bool:
        return self.blocking_issue_count > 0


@dataclass
class BuildOutcome:
    """The produced regulatory artefact."""

    artefact: ArtefactRef
    record_count: int
    field_count: int
    #: The in-memory artefact (e.g. an lxml root), passed to the validator to
    #: avoid re-parsing a very large file.
    parsed: Any = None
    detail: Dict[str, Any] = field(default_factory=dict)


class AnnexProvider(ABC):
    """One annex report's delivery behaviour."""

    #: The declarative descriptor. Class attribute so the registry can describe
    #: an annex without constructing it.
    definition: AnnexDefinition

    # -- configuration / applicability -------------------------------------- #
    @abstractmethod
    def check_configuration(self, request: DeliveryRequest,
                            report: PreflightReport) -> Mapping[str, str]:
        """Verify the client and system configuration this annex needs.

        Adds blocking findings to ``report`` for anything missing, and returns
        ``{label: fingerprint}`` for the configuration actually used, so the
        governed result records the exact rule versions behind the artefact.
        """

    @abstractmethod
    def check_applicability(self, request: DeliveryRequest,
                            report: PreflightReport) -> bool:
        """Whether this report is applicable to this client/portfolio/period.

        A report that is not applicable is not a failure: the agent reports it
        as a blocked run with an explanation, and never produces a file.
        """

    # -- lifecycle ---------------------------------------------------------- #
    @abstractmethod
    def resolve_input(self, request: DeliveryRequest,
                      report: PreflightReport) -> Optional[ResolvedInput]:
        """Locate the projected data. ``None`` when it cannot be resolved."""

    @abstractmethod
    def preflight(self, request: DeliveryRequest, resolved: Optional[ResolvedInput],
                  report: PreflightReport) -> None:
        """Annex-specific blocking checks, run before any expensive work."""

    @abstractmethod
    def normalise(self, request: DeliveryRequest, resolved: ResolvedInput,
                  work_dir: Path) -> NormalisationOutcome:
        """Run the annex's delivery normalisation into ``work_dir``."""

    @abstractmethod
    def build(self, request: DeliveryRequest, normalised: NormalisationOutcome,
              work_dir: Path) -> BuildOutcome:
        """Build the regulatory artefact into ``work_dir``.

        Must finalise atomically: a consumer that sees the artefact path must
        never see a partially written file.
        """

    @abstractmethod
    def validate(self, request: DeliveryRequest,
                 built: BuildOutcome) -> SchemaValidationResult:
        """Validate the artefact against this annex's authoritative schema."""

    @abstractmethod
    def inspect(self, request: DeliveryRequest, resolved: ResolvedInput,
                normalised: NormalisationOutcome,
                built: BuildOutcome) -> TransformationEvidence:
        """Report what the build did to the data — fills, coercions, omissions."""

    # -- classification / publication --------------------------------------- #
    def classify(self, evidence: TransformationEvidence,
                 validation: SchemaValidationResult) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Split the run's findings into blocking errors and operator warnings.

        The default is the conservative one and is right for every annex whose
        schema is authoritative: a schema failure blocks, and everything the
        builder invented is a warning that requires review but does not by
        itself invalidate the artefact. Override only with a reason.
        """
        blocking: List[Dict[str, Any]] = []
        if validation.executed and not validation.valid:
            blocking.append({
                "code": "SCHEMA_INVALID",
                "message": (f"The report did not pass validation against "
                            f"{Path(validation.schema_reference).name}: "
                            f"{validation.error_count} schema error(s)."),
                "detail": [dict(e) for e in validation.errors[:10]],
            })
        return blocking, evidence.operator_warnings()

    @abstractmethod
    def publication_metadata(self, request: DeliveryRequest,
                             resolved: ResolvedInput,
                             normalised: NormalisationOutcome,
                             built: BuildOutcome,
                             validation: SchemaValidationResult) -> Dict[str, Any]:
        """The annex-specific references a published package must carry."""

    # -- descriptive -------------------------------------------------------- #
    def component_versions(self) -> Mapping[str, str]:
        """Fingerprints of the code that produced the artefact.

        Default: empty. Providers that wrap repository modules should return
        content hashes so an auditor can tell whether a rebuild would be
        byte-identical.
        """
        return {}


__all__ = [
    "AnnexProvider", "ResolvedInput", "NormalisationOutcome", "BuildOutcome",
]
