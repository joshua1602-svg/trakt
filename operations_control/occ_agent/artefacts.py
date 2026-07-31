"""operations_control.occ_agent.artefacts — synthetic client responses.

An artefact provided to a synthetic case takes exactly the route a real
delivery takes, minus the one step that would touch production:

1. the file name is sanitised by the SAME function a manual delivery uses
   (:func:`operations_control.manual_intake.sanitise_filename`), so traversal,
   control characters, the legacy readiness sentinel and unreadable file types
   are refused identically;
2. the intended live destination is derived by
   :func:`operations_control.manual_intake.derive_raw_prefix`, which re-parses
   its own output with the production path parser — so the URI recorded on the
   case is one the automated intake route would accept;
3. **the file is written into the case sandbox and nowhere else.** The intended
   URI is recorded as text. :meth:`ArtefactService.register` never calls a
   storage write against it, and the synthetic policy would refuse it if it did;
4. the bytes are inspected and classified with the existing header-first role
   classifier (:mod:`apps.blob_trigger_app.file_roles`), the same vocabulary
   ``config/system/workflow_input_requirements.yaml`` is written in.

Readiness against the required roles is assessed with the administrator-governed
requirements, including the configured minimum recognition confidence — so a
weakly recognised file parks for confirmation here exactly as it would live.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from apps.blob_trigger_app.file_roles import classify_pack

from ..contracts import (
    BATCH_DATASET_DEFAULT,
    BATCH_FREQUENCY_DEFAULT,
    new_id,
    now_iso,
)
from ..engine import OpsError
from ..manual_intake import (
    ManualIntakeError,
    derive_blob_uri,
    derive_raw_prefix,
    sanitise_filename,
)
from .case import SyntheticArtefact, SyntheticCase
from .policy import CAP_LIVE_BLOB_WRITE, SyntheticPolicy
from .store import SyntheticCaseStore
from .vocabulary import artefact_vocabulary

#: Upper bound on one synthetic artefact. Generous for a tape, small enough that
#: a mistaken upload cannot fill the sandbox.
MAX_ARTEFACT_BYTES = 64 * 1024 * 1024

#: How many data rows are read to profile a file. The whole file is stored; only
#: the profile is bounded, so a large tape does not turn registration into a
#: full parse.
PROFILE_ROWS = 2000


class ArtefactRejected(OpsError):
    """A synthetic artefact could not be accepted."""

    def __init__(self, message: str):
        super().__init__("OCC_AGENT_ARTEFACT_REJECTED", message,
                         http_status=400)


@dataclass
class RoleReadiness:
    """Whether the received artefacts satisfy the configured required roles."""

    outcome: str
    required: List[str]
    optional: List[str]
    satisfied: List[str]
    missing: List[str]
    low_confidence: List[Dict[str, Any]]

    @property
    def ready(self) -> bool:
        return not self.missing and not self.low_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {"outcome": self.outcome, "required_roles": self.required,
                "optional_roles": self.optional,
                "satisfied_roles": self.satisfied,
                "missing_roles": self.missing,
                "low_confidence": self.low_confidence, "ready": self.ready}


class ArtefactService:
    """Registration, classification and readiness for synthetic artefacts."""

    def __init__(self, store: SyntheticCaseStore, policy: SyntheticPolicy):
        self.store = store
        self.policy = policy

    # ------------------------------------------------------------------ #
    def register(self, case: SyntheticCase, *, filename: str, data: bytes,
                 provided_by: str, fixture_id: str = "",
                 declared_type: str = "") -> SyntheticArtefact:
        """Accept one synthetic artefact into the case sandbox."""
        if len(data) > MAX_ARTEFACT_BYTES:
            raise ArtefactRejected(
                "That file is too large for a synthetic case.")
        if not data:
            raise ArtefactRejected("That file is empty.")
        try:
            leaf = sanitise_filename(filename)
        except ManualIntakeError as exc:
            raise ArtefactRejected(str(exc)) from exc

        path = self.store.write_artefact_bytes(case.tenant, case.case_id, leaf,
                                               data)
        columns, rows = _inspect(path)
        artefact = SyntheticArtefact(
            artefact_id=new_id("sart"),
            source_file=leaf,
            artefact_type=declared_type,
            synthetic_location=self.store.relative(case.tenant, case.case_id,
                                                   path),
            intended_live_uri=self.intended_uri(case, leaf),
            execution_status="simulated_only",
            sha256="sha256:" + hashlib.sha256(data).hexdigest(),
            size=len(data),
            columns=columns,
            row_count=rows,
            provided_by=provided_by,
            provided_at=now_iso(),
            fixture_id=fixture_id)
        return artefact

    def intended_uri(self, case: SyntheticCase, filename: str) -> str:
        """The live Blob URI this artefact WOULD occupy. Never written to.

        Derived with the production path rules; when identity or period is not
        yet confirmed the URI is deliberately empty rather than guessed — a
        wrong intended path is worse than a missing one, because readiness
        checks it.
        """
        req = case.confirmed or case.extracted
        client = case.client_id or req.proposed_client_id
        portfolio = case.portfolio_id or req.proposed_portfolio_id
        period = req.reporting_date
        if not (client and portfolio and period):
            return ""
        try:
            prefix = derive_raw_prefix(
                client_id=client, portfolio_id=portfolio,
                reporting_period=period, dataset=BATCH_DATASET_DEFAULT,
                frequency=(req.reporting_frequency or BATCH_FREQUENCY_DEFAULT))
            return derive_blob_uri(prefix, filename)
        except ManualIntakeError:
            return ""

    def assert_never_written(self, uri: str, *, case_id: str = "",
                             tenant: str = "", actor: str = "") -> None:
        """The guard a future live-handoff implementation must pass.

        Called wherever an intended URI is produced, so the refusal is exercised
        rather than merely documented: in synthetic mode this always raises.
        """
        self.policy.require(CAP_LIVE_BLOB_WRITE, detail=uri, case_id=case_id,
                            tenant=tenant, actor=actor)

    # ------------------------------------------------------------------ #
    def classify(self, case: SyntheticCase,
                 artefacts: Sequence[SyntheticArtefact]
                 ) -> Tuple[List[SyntheticArtefact], List[str]]:
        """Assign a semantic input role to each artefact.

        Uses the production classifier over ``(filename, columns)`` pairs — the
        same evidence the live intake route classifies on — and returns its
        cross-file findings alongside, so an ambiguous pack surfaces as an
        observation rather than being silently accepted.
        """
        pack: List[Tuple[str, Sequence[str]]] = [
            (a.source_file, list(a.columns)) for a in artefacts]
        try:
            result = classify_pack(pack)
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            raise ArtefactRejected(
                "Trakt could not read the file layouts well enough to "
                f"recognise them. ({type(exc).__name__})") from exc

        by_name = {a.filename: a for a in result.assignments}
        vocab = artefact_vocabulary()
        known_roles = {r.role for r in vocab.roles}
        out: List[SyntheticArtefact] = []
        for artefact in artefacts:
            assignment = by_name.get(artefact.source_file)
            if assignment is None:  # pragma: no cover — classify_pack is total
                out.append(artefact)
                continue
            role = str(assignment.assigned_role or "")
            # The classifier's fallback role is a stable digest of the file
            # name, not a semantic role. Recording it as the artefact type
            # would make an unrecognised file look recognised, so it is kept
            # out of the type and left visible as an unrecognised artefact.
            artefact.artefact_type = role if role in known_roles else ""
            artefact.recognition_basis = str(assignment.role_basis or "")
            artefact.recognition_confidence = float(assignment.confidence or 0.0)
            out.append(artefact)

        findings: List[str] = []
        if result.ambiguous_role_conflict:
            names = ", ".join(vocab.label(r) for r in result.conflicting_roles)
            findings.append(
                f"More than one file was recognised as the same kind of data "
                f"({names}). Confirm which file Trakt should use.")
        if result.drift_suspected:
            findings.append(
                "One or more files did not match a known layout and were "
                "recognised from their name only.")
        return out, findings

    # ------------------------------------------------------------------ #
    def readiness(self, case: SyntheticCase, outcome: str) -> RoleReadiness:
        """Are the configured required roles satisfied for this outcome?"""
        vocab = artefact_vocabulary()
        required = vocab.required_roles(outcome)
        optional = vocab.optional_roles(outcome)
        satisfied: List[str] = []
        low: List[Dict[str, Any]] = []
        for raw in case.received_artefacts:
            artefact = SyntheticArtefact.from_dict(raw)
            if artefact.artefact_type not in required:
                continue
            confidence = artefact.recognition_confidence
            if confidence is not None and confidence < vocab.min_confidence:
                low.append({
                    "role": artefact.artefact_type,
                    "source_file": artefact.source_file,
                    "confidence": confidence,
                    "minimum": vocab.min_confidence,
                    "question": f"Is '{artefact.source_file}' the "
                                f"{vocab.label(artefact.artefact_type)}?"})
                continue
            if artefact.artefact_type not in satisfied:
                satisfied.append(artefact.artefact_type)
        missing = [r for r in required if r not in satisfied]
        return RoleReadiness(outcome=outcome, required=required,
                             optional=optional, satisfied=satisfied,
                             missing=missing, low_confidence=low)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _inspect(path: Path) -> Tuple[List[str], int]:
    """Column headings and row count, without loading a whole tape.

    Failure to parse is not an error here: an unreadable file simply has no
    columns, which the classifier then treats as absent header evidence and the
    readiness check reports as unrecognised.
    """
    suffix = path.suffix.lower()
    try:
        import pandas as pd
        if suffix == ".csv":
            head = pd.read_csv(path, nrows=PROFILE_ROWS, low_memory=False)
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                rows = max(sum(1 for _ in fh) - 1, 0)
        else:
            head = pd.read_excel(path, nrows=PROFILE_ROWS)
            rows = int(len(head))
        return [str(c) for c in head.columns], rows
    except Exception:  # noqa: BLE001 — an unreadable file is a finding, not a crash
        return [], 0


