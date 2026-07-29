"""
engine.annex_delivery_agent.publication
=======================================

The gate between "prepared" and "submitted".

The design rule this module exists to enforce: **a passing XSD result never
publishes anything.** Schema validity says a parser will accept the file. It
says nothing about whether the figures are right, and nothing about whether the
values the builder inserted on the client's behalf are the ones the client would
have chosen. So the path is always:

    PREPARED / PREPARED_WITH_WARNINGS
        → request_approval()   → AWAITING_APPROVAL
        → approve() / reject() → APPROVED / REJECTED
        → publish()            → PUBLISHED

Each transition is checked, recorded with who did it and when, and refused if
the run is not in the right state. ``publish`` refuses anything that is not
``APPROVED``, and refuses a second time for a run already published — the
duplicate-publication guard.

Approval vocabulary matches ``apps.blob_trigger_app.approvals``
(``pending``/``approved``/``rejected``/``promoted``) so the operator console
speaks one language across source promotion and regulatory delivery.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from trakt_core.context import ExecutionContext
from trakt_core.errors import ErrorCode, TraktError

from .contracts import (
    APPROVAL_APPROVED, APPROVAL_PENDING, APPROVAL_PROMOTED, APPROVAL_REJECTED,
    ApprovalState, DeliveryResult, DeliveryStatus,
)
from .persistence import DeliveryRunStore, atomic_write_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Where a published package goes
# --------------------------------------------------------------------------- #


class PublicationSink(ABC):
    """Where an approved package is delivered.

    Kept abstract because the destination is a deployment concern, not an annex
    concern: a local run publishes to a directory, an Azure deployment publishes
    to a blob container under the regime prefix. The gating logic above is
    identical either way.
    """

    name: str = "sink"

    @abstractmethod
    def publish(self, *, package: Mapping[str, Any], artefact: Path,
                destination_key: str) -> Dict[str, Any]:
        """Deliver ``artefact`` plus its ``package`` manifest. Returns a receipt."""


class FilesystemPublicationSink(PublicationSink):
    """Publish into a directory tree. The default for local and on-premise runs."""

    name = "filesystem"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def publish(self, *, package: Mapping[str, Any], artefact: Path,
                destination_key: str) -> Dict[str, Any]:
        target_dir = self.root / destination_key
        target_dir.mkdir(parents=True, exist_ok=True)

        published_artefact = target_dir / Path(artefact).name
        # Copy to a temporary sibling and move into place: a consumer polling the
        # published location must never observe a partial regulatory file.
        staging = target_dir / f".{published_artefact.name}.part"
        shutil.copy2(artefact, staging)
        staging.replace(published_artefact)

        manifest = atomic_write_json(target_dir / "publication_manifest.json", dict(package))
        return {
            "sink": self.name,
            "destination": str(target_dir),
            "artefact": str(published_artefact),
            "manifest": str(manifest),
            "published_at": _now(),
        }


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


@dataclass
class PublicationGate:
    """Approval and publication for one run directory."""

    store: DeliveryRunStore
    run_dir: Path
    sink: Optional[PublicationSink] = None

    # -- approval ----------------------------------------------------------- #
    def request_approval(self, context: ExecutionContext) -> DeliveryResult:
        """Move a prepared run into the operator's queue."""
        result = self._load(context)
        if result.status not in (DeliveryStatus.PREPARED,
                                 DeliveryStatus.PREPARED_WITH_WARNINGS):
            raise TraktError(
                ErrorCode.ARTEFACT_INCOMPLETE,
                f"This report is not ready for approval — it is currently "
                f"'{result.status}'. Only a prepared report can be sent for approval.",
                request_id=context.request_id,
                details={"status": result.status})

        updated = _with(result,
                        status=DeliveryStatus.AWAITING_APPROVAL,
                        approval=ApprovalState(state=APPROVAL_PENDING,
                                               requested_at=_now()))
        return self._save(updated)

    def approve(self, context: ExecutionContext, *, note: str = "") -> DeliveryResult:
        """Record an operator's explicit approval to submit."""
        result = self._load(context)
        if result.status not in (DeliveryStatus.AWAITING_APPROVAL,
                                 DeliveryStatus.PREPARED,
                                 DeliveryStatus.PREPARED_WITH_WARNINGS):
            raise TraktError(
                ErrorCode.ARTEFACT_INCOMPLETE,
                f"This report cannot be approved — it is currently '{result.status}'.",
                request_id=context.request_id,
                details={"status": result.status})
        if result.status == DeliveryStatus.PUBLISHED:  # pragma: no cover - covered above
            raise TraktError(ErrorCode.ARTEFACT_NOT_APPROVED,
                             "This report has already been published.",
                             request_id=context.request_id)

        updated = _with(result,
                        status=DeliveryStatus.APPROVED,
                        approval=ApprovalState(state=APPROVAL_APPROVED,
                                               decided_by=context.actor_id,
                                               decided_at=_now(),
                                               reason=note or None,
                                               requested_at=result.approval.requested_at))
        return self._save(updated)

    def reject(self, context: ExecutionContext, *, reason: str) -> DeliveryResult:
        """Record an operator's refusal. A rejected run is never publishable."""
        if not str(reason or "").strip():
            raise TraktError(ErrorCode.INVALID_INPUT,
                             "A reason is required when rejecting a report.",
                             request_id=context.request_id)
        result = self._load(context)
        if result.status == DeliveryStatus.PUBLISHED:
            raise TraktError(
                ErrorCode.ARTEFACT_NOT_APPROVED,
                "This report has already been published and cannot be rejected.",
                request_id=context.request_id)

        updated = _with(result,
                        status=DeliveryStatus.REJECTED,
                        publication_eligible=False,
                        publication_blockers=tuple(
                            [*result.publication_blockers,
                             f"Rejected by {context.actor_id}: {reason}"]),
                        approval=ApprovalState(state=APPROVAL_REJECTED,
                                               decided_by=context.actor_id,
                                               decided_at=_now(),
                                               reason=reason,
                                               requested_at=result.approval.requested_at))
        return self._save(updated)

    # -- publication -------------------------------------------------------- #
    def publish(self, context: ExecutionContext, *,
                destination_key: Optional[str] = None) -> DeliveryResult:
        """Deliver an approved package. Refuses anything not explicitly approved."""
        result = self._load(context)

        if result.status == DeliveryStatus.PUBLISHED:
            raise TraktError(
                ErrorCode.ARTEFACT_NOT_APPROVED,
                "This report has already been published. Publishing it again would "
                "create a duplicate submission.",
                request_id=context.request_id,
                details={"published": result.publication})
        if result.status != DeliveryStatus.APPROVED or not result.approval.approved:
            raise TraktError(
                ErrorCode.ARTEFACT_NOT_APPROVED,
                "This report has not been approved for submission. A report that "
                "passes schema validation is prepared, not approved — an operator "
                "must approve it explicitly before it can be published.",
                request_id=context.request_id,
                details={"status": result.status, "approval": result.approval.to_dict()})
        if result.artefact is None or not Path(result.artefact.path).is_file():
            raise TraktError(
                ErrorCode.ARTEFACT_NOT_FOUND,
                "The approved report file is no longer available. Prepare the report "
                "again before publishing.",
                request_id=context.request_id)

        sink = self.sink or FilesystemPublicationSink(self.run_dir / "published")
        package = build_publication_package(result, context=context)
        key = destination_key or _destination_key(result)
        receipt = sink.publish(package=package, artefact=Path(result.artefact.path),
                               destination_key=key)

        record = {"package": package, "receipt": receipt}
        self.store.write_publication(self.run_dir, record)

        updated = _with(result,
                        status=DeliveryStatus.PUBLISHED,
                        publication=record,
                        publication_eligible=False,
                        approval=ApprovalState(state=APPROVAL_PROMOTED,
                                               decided_by=result.approval.decided_by,
                                               decided_at=result.approval.decided_at,
                                               reason=result.approval.reason,
                                               requested_at=result.approval.requested_at))
        return self._save(updated)

    # -- internals ---------------------------------------------------------- #
    def _load(self, context: ExecutionContext) -> DeliveryResult:
        result = self.store.read_result(self.run_dir)
        if result is None:
            raise TraktError(ErrorCode.ARTEFACT_NOT_FOUND,
                             "There is no prepared report at this location.",
                             request_id=context.request_id)
        # Tenant isolation: the run directory is addressed by tenant, but the
        # stored result carries the tenant it was produced for, so a mismatch is
        # a refusal rather than a silent cross-tenant read.
        if result.tenant_id and result.tenant_id != context.tenant_id:
            raise TraktError(
                ErrorCode.TENANT_MISMATCH,
                "This report belongs to a different client.",
                request_id=context.request_id)
        return result

    def _save(self, result: DeliveryResult) -> DeliveryResult:
        self.store.write_result(self.run_dir, result)
        return result


# --------------------------------------------------------------------------- #
# The published package
# --------------------------------------------------------------------------- #


def build_publication_package(result: DeliveryResult, *,
                              context: ExecutionContext) -> Dict[str, Any]:
    """Everything a submitted package must be able to point back at.

    A regulator's question a year from now is "how was this number produced?".
    That is answerable only if the package names the source data, the projected
    and normalised datasets, the artefact, the schema, the validation outcome,
    the values the system supplied on the client's behalf, who approved it, and
    the exact rule and code versions in force at the time.
    """
    evidence = result.transformation_evidence or {}
    return {
        "package_version": "1.0",
        "created_at": _now(),
        "run_id": result.run_id,
        "annex": {"annex_id": result.annex_id, "report_version": result.annex_version},
        "tenant_id": result.tenant_id,
        "portfolio_id": result.portfolio_id,
        "reporting_date": result.reporting_date,

        "source_references": [dict(s) for s in result.source_references],
        "projected_dataset": next(
            (dict(s) for s in result.source_references
             if s.get("kind") == "projected_dataset"), None),
        "normalised_dataset": next(
            (a.to_dict() for a in result.intermediate_artefacts
             if a.kind.endswith("_csv") and "delivery_ready" in a.kind), None),
        "artefact": result.artefact.to_dict() if result.artefact else None,
        "schema": (result.schema_validation.schema_reference
                   if result.schema_validation else None),
        "schema_sha256": (result.schema_validation.schema_sha256
                          if result.schema_validation else None),
        "validation_report": (result.schema_validation.to_dict()
                              if result.schema_validation else None),

        "automatic_value_evidence": {
            "automatic_fill_count": evidence.get("automatic_fill_count", 0),
            "coercion_count": evidence.get("coercion_count", 0),
            "omission_count": evidence.get("omission_count", 0),
            "review_required": evidence.get("review_required", False),
            "operator_warnings": evidence.get("operator_warnings", []),
            "nd_reconciliation": evidence.get("nd_reconciliation", {}),
        },

        "operator_approval": {
            **result.approval.to_dict(),
            "published_by": context.actor_id,
            "channel": context.channel,
        },
        "configuration_versions": dict(result.configuration_versions),
        "component_versions": dict(result.component_versions),
        "record_count": result.record_count,
        "delivered_field_count": result.delivered_field_count,
        "timings": [t.to_dict() for t in result.timings],
        "disclaimer": (
            "Schema validation confirms the file is structurally acceptable to the "
            "regulator's parser. It does not confirm the figures are economically or "
            "regulatorily correct."),
    }


def _destination_key(result: DeliveryResult) -> str:
    parts = [result.tenant_id or "unknown", result.annex_id,
             result.portfolio_id or "default", result.reporting_date or "undated",
             result.run_id]
    return "/".join(
        "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(p)) for p in parts)


def _with(result: DeliveryResult, **changes: Any) -> DeliveryResult:
    from dataclasses import replace
    return replace(result, **changes)


__all__ = [
    "PublicationGate", "PublicationSink", "FilesystemPublicationSink",
    "build_publication_package",
]
