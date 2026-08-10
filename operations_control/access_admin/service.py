"""The OCC access-administration service: propose, review, confirm, activate.

    document / operator request
            ↓
    OCC extracts or receives facts
            ↓
    propose()      structured change set -> candidate document -> DRAFT version
            ↓
    deterministic validation (the real Stage 1-3 registries; nothing restated)
            ↓
    operator reviews  (proposal(), a plain-English summary + the exact diff)
            ↓
    confirm()      activates the version. A HUMAN does this, never the agent.
            ↓
    materialise()  publishes the four files where the runtime loaders read them
            ↓
    future ExecutionContexts resolve against the new configuration

What is reused rather than rebuilt
----------------------------------
Nearly all of it. ``ConfigPackageStore`` already provides versioned,
content-hashed, immutable draft → validate → activate → rollback with a
superseded trail; ``OpsStore.append_audit`` already provides a hash-chained,
tamper-evident audit; ``operations_control.api.auth`` already provides
admin-only operator principals. This service adds the access **layer** to the
first, structured proposals in front of it, and nothing at all to the other two.

Deliberately NOT added to ``ADMIN_LAYERS``
------------------------------------------
The generic ``/ops/admin/config/{layer}/draft`` route takes raw YAML edits. That
is right for the system/regime/asset packages and wrong for access, where the
whole point is that changes arrive as reviewable actions rather than a diff. The
access layer therefore has its own routes and stays out of the generic ones.

Why activation and publication are separate
-------------------------------------------
Activating records the decision: this version is the approved one, by this
person, at this time, superseding that one. Publishing writes the files where
``load_organisation_registry`` and friends will read them. They are separate
because they fail differently — a filesystem that is read-only or full must not
be able to lose the *decision*, and re-publishing an already-approved version
must not need a second approval. :meth:`materialise` is idempotent and can be
re-run at deploy time.

The OCC Agent is not in the authorisation path
-----------------------------------------------
It proposes. Every runtime access decision continues to run through
``ExecutionContext`` → ``EntitlementStore`` → ``ResourceCatalogue`` →
``authorise_resource_access``, reading published configuration. No MI, risk or
forecast request asks this service anything, and nothing here is imported by
``mi_agent_api``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..configuration.packages import (
    LAYER_ACCESS,
    STATUS_ACTIVE,
    STATUS_DRAFT,
    ConfigPackageStore,
)
from ..contracts import now_iso
from ..stores import OpsStore
from .changeset import (
    PROPOSAL_CONFIRMED,
    PROPOSAL_DRAFT,
    PROPOSAL_REJECTED,
    SOURCE_OCC_AGENT,
    AccessChange,
    AccessChangeSet,
    ChangeSetError,
    apply_change_set,
)
from .document import ACCESS_FILES, AccessDocument
from .validation import validate_document

logger = logging.getLogger("operations_control.access_admin")

#: The audit stream administrator configuration already uses. Access changes
#: join it rather than starting a parallel trail, so one query answers "what
#: changed about this deployment's configuration, and who did it".
ADMIN_AUDIT_CLIENT = "_admin"

EVENT_PROPOSED = "access_change_proposed"
EVENT_REJECTED = "access_change_rejected"
EVENT_CONFIRMED = "access_change_confirmed"
EVENT_PUBLISHED = "access_config_published"


class ConfirmationError(PermissionError):
    """A confirmation that must not proceed — including self-approval."""


@dataclass(frozen=True)
class Proposal:
    """A validated candidate, waiting for a human."""

    change_set: AccessChangeSet
    version: int
    status: str
    problems: Tuple[Dict[str, Any], ...] = ()
    #: What the configuration looks like now, and would look like after.
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    confirmed_by: Optional[str] = None
    confirmed_at: Optional[str] = None

    @property
    def valid(self) -> bool:
        return not self.problems

    @property
    def blocked(self) -> bool:
        return bool(self.problems)

    def summary(self) -> Dict[str, Any]:
        """What the OCC puts in front of an approver.

        Deliberately includes the fingerprint: confirmation quotes it back, so
        an operator cannot approve one proposal and have a different one land.
        """
        return {
            "change_set_id": self.change_set.change_set_id,
            "title": self.change_set.title,
            "rationale": self.change_set.rationale,
            "proposed_by": self.change_set.proposed_by,
            "source": self.change_set.source,
            "proposed_at": self.change_set.proposed_at,
            "fingerprint": self.change_set.fingerprint,
            "version": self.version,
            "status": self.status,
            "requires_confirmation": self.change_set.requires_confirmation,
            "actions": self.change_set.describe(),
            "changes": [c.to_dict() for c in self.change_set.changes],
            "valid": self.valid,
            "problems": list(self.problems),
            "confirmed_by": self.confirmed_by,
            "confirmed_at": self.confirmed_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self.summary(), "before": self.before, "after": self.after}


class AccessAdminService:
    """Administer organisations, principals, resources and grants."""

    def __init__(self, store: OpsStore,
                 packages: Optional[ConfigPackageStore] = None) -> None:
        self.store = store
        self.packages = packages or ConfigPackageStore(store)

    # -- reading ------------------------------------------------------------ #
    def active_document(self) -> AccessDocument:
        """The configuration currently in force.

        An unseeded deployment yields an empty document — nothing configured,
        which is exactly today's ERE behaviour and the right starting point.
        """
        active = self.packages.active_version(LAYER_ACCESS)
        if active is None:
            active = self.packages.ensure_seeded(LAYER_ACCESS)
        return AccessDocument.from_files(active.get("files") or {})

    def overview(self) -> Dict[str, Any]:
        """Who exists, who they are, what they can reach — for the console."""
        active = self.packages.active_version(LAYER_ACCESS) or {}
        document = self.active_document()
        organisations = []
        for row in document.sorted_organisations():
            organisation_id = str(row.get("organisation_id") or "")
            organisations.append({
                **row,
                "principals": [p for p in document.sorted_principals()
                               if p.get("organisation_id") == organisation_id],
                "grants": [g for g in document.sorted_entitlements()
                           if g.get("organisation_id") == organisation_id],
            })
        return {
            "active_version": active.get("version"),
            "activated_by": active.get("activated_by"),
            "activated_at": active.get("activated_at"),
            "organisations": organisations,
            "resources": document.sorted_resources(),
            "drafts": [v for v in self.packages.list_versions(LAYER_ACCESS)
                       if v.get("status") == STATUS_DRAFT],
        }

    def proposal(self, version: int) -> Optional[Proposal]:
        doc = self.packages.get_version(LAYER_ACCESS, version)
        if doc is None:
            return None
        raw = (doc.get("access_change_set") or {})
        if not raw:
            return None
        change_set = AccessChangeSet.from_dict(raw)
        validation = doc.get("validation") or {}
        return Proposal(
            change_set=change_set,
            version=version,
            status=doc.get("access_proposal_status") or PROPOSAL_DRAFT,
            problems=tuple(validation.get("problem_details") or ()),
            before=doc.get("access_before"),
            after=doc.get("access_after"),
            confirmed_by=doc.get("access_confirmed_by") or doc.get("activated_by"),
            confirmed_at=doc.get("access_confirmed_at") or doc.get("activated_at"),
        )

    def open_proposals(self) -> List[Proposal]:
        out = []
        for summary in self.packages.list_versions(LAYER_ACCESS):
            if summary.get("status") != STATUS_DRAFT:
                continue
            proposal = self.proposal(int(summary["version"]))
            if proposal is not None and proposal.status == PROPOSAL_DRAFT:
                out.append(proposal)
        return out

    # -- proposing ---------------------------------------------------------- #
    def propose(self, change_set: AccessChangeSet) -> Proposal:
        """Turn a change set into a validated DRAFT version.

        Raises :class:`ChangeSetError` when the actions cannot be applied at all
        (an unknown organisation, a duplicate grant). Records a draft with its
        problems when they apply but produce an invalid configuration — an
        operator needs to SEE why a proposal is unsafe, and a proposal that
        vanished on a validation failure teaches nobody anything.

        Nothing becomes active here. Ever.
        """
        current = self.active_document()
        candidate = apply_change_set(current, change_set)          # may raise
        problems = validate_document(candidate)

        files = candidate.to_files()
        draft = self.packages.create_draft(
            LAYER_ACCESS, by=change_set.proposed_by, edits=files,
            notes=change_set.title or "access change")

        draft["access_change_set"] = change_set.to_dict()
        draft["access_proposal_status"] = PROPOSAL_DRAFT
        # Stored, not recomputed on read: the approver must see the diff as it
        # stood when the proposal was made, even if another change has been
        # activated since. Recomputing would quietly re-base the review.
        draft["access_before"] = current.to_dict()
        draft["access_after"] = candidate.to_dict()
        draft["validated"] = not problems
        draft["validation"] = {
            "ok": not problems,
            "problems": [f"{p['area']}: {p['message']}" for p in problems],
            "problem_details": problems,
            "validated_at": now_iso(),
        }
        self._write(draft)

        self._audit(EVENT_PROPOSED, change_set.proposed_by, {
            "version": draft["version"],
            "change_set_id": change_set.change_set_id,
            "fingerprint": change_set.fingerprint,
            "source": change_set.source,
            "actions": change_set.describe(),
            "valid": not problems,
            "problems": [p["message"] for p in problems],
        })

        return Proposal(change_set=change_set, version=int(draft["version"]),
                        status=PROPOSAL_DRAFT, problems=tuple(problems),
                        before=current.to_dict(), after=candidate.to_dict())

    # -- confirming --------------------------------------------------------- #
    def confirm(self, version: int, *, actor: str,
                fingerprint: Optional[str] = None) -> Proposal:
        """Activate a proposal. This is the human decision.

        Refuses when:

          * the version is not an open access proposal;
          * it did not validate — an invalid configuration cannot be approved
            into existence;
          * ``fingerprint`` is supplied and does not match, so an operator
            cannot approve one proposal and have a newer one applied;
          * the confirming actor is the OCC Agent itself. The agent may
            recommend; it must never be both proposer and approver.
        """
        actor = str(actor or "").strip()
        if not actor:
            raise ConfirmationError("A confirmation must record who made it.")

        proposal = self.proposal(version)
        if proposal is None:
            raise ConfirmationError(f"No access proposal at version {version}.")
        if proposal.status != PROPOSAL_DRAFT:
            raise ConfirmationError(
                f"That proposal is already {proposal.status}.")
        if fingerprint and fingerprint != proposal.change_set.fingerprint:
            raise ConfirmationError(
                "This proposal has changed since it was reviewed. Read it "
                "again before confirming.")
        if proposal.blocked:
            raise ConfirmationError(
                "This proposal did not pass its checks and cannot be "
                "activated: " + "; ".join(p["message"] for p in proposal.problems))
        if _is_agent(actor) or (
                proposal.change_set.source == SOURCE_OCC_AGENT
                and actor == proposal.change_set.proposed_by):
            raise ConfirmationError(
                "The OCC Agent proposed this change and cannot approve it. A "
                "named operator must confirm.")

        doc = self.packages.get_version(LAYER_ACCESS, version)
        doc["access_proposal_status"] = PROPOSAL_CONFIRMED
        doc["access_confirmed_by"] = actor
        doc["access_confirmed_at"] = now_iso()
        self._write(doc)

        activated = self.packages.activate_version(LAYER_ACCESS, version, by=actor)

        self._audit(EVENT_CONFIRMED, actor, {
            "version": version,
            "change_set_id": proposal.change_set.change_set_id,
            "fingerprint": proposal.change_set.fingerprint,
            "proposed_by": proposal.change_set.proposed_by,
            "source": proposal.change_set.source,
            "actions": proposal.change_set.describe(),
            "supersedes": doc.get("based_on_version"),
        })

        return Proposal(change_set=proposal.change_set, version=version,
                        status=PROPOSAL_CONFIRMED,
                        confirmed_by=actor,
                        confirmed_at=activated.get("activated_at"))

    def reject(self, version: int, *, actor: str, reason: str = "") -> Proposal:
        """Close a proposal without activating it."""
        actor = str(actor or "").strip()
        if not actor:
            raise ConfirmationError("A rejection must record who made it.")
        proposal = self.proposal(version)
        if proposal is None:
            raise ConfirmationError(f"No access proposal at version {version}.")
        if proposal.status != PROPOSAL_DRAFT:
            raise ConfirmationError(f"That proposal is already {proposal.status}.")

        doc = self.packages.get_version(LAYER_ACCESS, version)
        doc["access_proposal_status"] = PROPOSAL_REJECTED
        doc["access_rejected_by"] = actor
        doc["access_rejected_at"] = now_iso()
        doc["access_rejection_reason"] = reason
        self._write(doc)

        self._audit(EVENT_REJECTED, actor, {
            "version": version,
            "change_set_id": proposal.change_set.change_set_id,
            "reason": reason})
        return Proposal(change_set=proposal.change_set, version=version,
                        status=PROPOSAL_REJECTED)

    # -- publishing --------------------------------------------------------- #
    def materialise(self, destination: str | Path, *,
                    actor: str = "system") -> Dict[str, Path]:
        """Write the ACTIVE access configuration where the runtime reads it.

        Idempotent, and separate from confirmation on purpose (see the module
        docstring). Point ``TRAKT_ORGANISATIONS_CONFIG`` and its three siblings
        at the files this writes, and the Stage 1-3 loaders pick the change up.
        """
        active = self.packages.active_version(LAYER_ACCESS)
        if active is None:
            raise ConfirmationError("There is no active access configuration.")
        written = self.packages.materialise(
            LAYER_ACCESS, int(active["version"]), Path(destination))
        self._audit(EVENT_PUBLISHED, actor, {
            "version": active["version"],
            "files": sorted(str(p) for p in written.values())})
        return written

    # -- internals ---------------------------------------------------------- #
    def _write(self, doc: Mapping[str, Any]) -> None:
        self.packages.save_version(dict(doc))

    def _audit(self, event: str, actor: str, detail: Dict[str, Any]) -> None:
        """Append to the existing hash-chained administrator trail.

        Identifiers and outcomes only — never a token, a bearer credential or a
        payload. A change set's actions ARE the material fact and are recorded
        in full, because an audit that cannot say what was approved is not an
        audit.
        """
        try:
            self.store.append_audit(ADMIN_AUDIT_CLIENT, event, actor=actor,
                                    detail=detail)
        except Exception:  # noqa: BLE001 - auditing must not fail the change
            logger.exception("access admin audit failed for %s", event)


def _is_agent(actor: str) -> bool:
    return str(actor or "").strip().lower() in {SOURCE_OCC_AGENT, "agent",
                                                "occ-agent", "system"}
