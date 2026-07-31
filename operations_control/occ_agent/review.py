"""operations_control.occ_agent.review — the package a human approves.

Approving an activation is a real decision with real consequences, so the thing
put in front of the approver has to be complete enough to make it on. This
module assembles that package. It **derives** everything from records that
already exist — the onboarding case, the governed catalogue, the run, the pack
and its issue history, the readiness verdict, the audit trail — and adds no
facts of its own.

Four things it states outright rather than leaving to be noticed:

* **where every answer came from.** Each collected field carries its
  provenance, in the catalogue's own labels, so an approver can see which
  answers the client gave, which Trakt worked out, and which an operator typed.
* **that field mappings are not collected here.** They are learned from the
  first representative delivery and approved through the existing mapping path.
  :data:`~operations_control.occ_agent.pack.MAPPING_STATEMENT` says so to the
  client; :data:`MAPPING_NOTE` says so to the approver.
* **what is still outstanding**, from the catalogue's own required-field rules
  rather than from a judgement made here.
* **what has not been provisioned.** User access collected during onboarding is
  a *requirement*, not a grant: this environment still reads its operators from
  configuration, so the package emits structured operator actions and never
  claims access exists. See :func:`access_actions`.

The package is a projection. Nothing here writes, validates or decides.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import canonical_json, stable_hash
from ..onboarding.case import STATUS_LABELS, OnboardingCase
from ..onboarding.catalogue import Catalogue, Field, Section
from . import classification as _classification
from . import states as _states
from .derive import ExecutionFacts, product_label
from .pack import MAPPING_STATEMENT
from .run import SyntheticRun

#: What the approver is told about mappings, in as many words.
MAPPING_NOTE = (
    "Field mappings are NOT part of this configuration and were not collected. "
    "They are proposed by Trakt from the first representative delivery, "
    "reviewed and approved by an operator during that first ingestion, and "
    "then fingerprinted and fixed. Approving this activation does not approve "
    "any mapping."
)

#: What the approver is told about user access.
ACCESS_NOTE = (
    "User access recorded during onboarding is a REQUIREMENT, not a grant. "
    "Trakt reads its operators from environment configuration in this "
    "environment, so nothing below has been provisioned. Each row is an "
    "action for an administrator to carry out separately."
)

#: How each provenance value reads to a human.
#:
#: A pre-populated value is not one thing, and an approver has to be able to
#: tell which kind it is: an operator's decision, a value read out of an
#: existing client record, something Trakt computed and a governed default are
#: four different claims about how much scrutiny the number deserves.
PROVENANCE_LABELS = {
    "human_supplied": "an operator told Trakt",
    "client_supplied": "the client told Trakt",
    "existing_record": "read from this client's existing record",
    "trakt_derived": "Trakt derived it",
    "inherited_default": "a governed default, inherited",
    "artefact_derived": "read from a file the client sent",
    "agent_proposed": "the agent proposed it",
    "human_approved": "the agent proposed it and an operator approved it",
    "inherited_configuration": "inherited from existing configuration",
}


@dataclass
class AnswerRow:
    """One collected value, with where it came from."""

    section: str
    section_label: str
    field: str
    label: str
    value: Any = None
    item: str = ""
    index: Optional[int] = None
    provenance: str = ""
    provenance_label: str = "not recorded"
    writes_to: str = ""
    sensitive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorAction:
    """Something a human must do outside Trakt before this is usable."""

    kind: str
    subject: str
    detail: str
    status: str = "not_provisioned"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewPackage:
    """Everything the approver needs, in one document."""

    case_ref: str
    client_name: str = ""
    sections: List[Dict[str, Any]] = field(default_factory=list)
    outstanding: List[Dict[str, Any]] = field(default_factory=list)
    data_definitions: List[Dict[str, Any]] = field(default_factory=list)
    access_requirements: List[Dict[str, Any]] = field(default_factory=list)
    operator_actions: List[Dict[str, Any]] = field(default_factory=list)
    artefacts: List[Dict[str, Any]] = field(default_factory=list)
    communication: Dict[str, Any] = field(default_factory=dict)
    configuration_preview: Dict[str, Any] = field(default_factory=dict)
    rehearsal: Dict[str, Any] = field(default_factory=dict)
    readiness: Dict[str, Any] = field(default_factory=dict)
    activation: Dict[str, Any] = field(default_factory=dict)
    approvals: Dict[str, Any] = field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    mapping_note: str = MAPPING_NOTE
    mapping_statement_to_client: str = MAPPING_STATEMENT
    access_note: str = ACCESS_NOTE
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def document(self) -> str:
        """The package as something a person can actually read."""
        lines = [f"# Review — {self.client_name or self.case_ref}", "",
                 f"Reference: {self.case_ref}", ""]
        lines += ["## What Trakt holds", ""]
        for section in self.sections:
            if not section["rows"]:
                continue
            lines += [f"### {section['label']}", ""]
            for row in section["rows"]:
                where = f" ({row['item']})" if row["item"] else ""
                lines.append(f"- **{row['label']}**{where}: "
                             f"{_render(row['value'])} "
                             f"— {row['provenance_label']}")
            lines.append("")
        if self.outstanding:
            lines += ["## Still outstanding", ""]
            lines += [f"- {row['label']}" for row in self.outstanding]
            lines.append("")
        lines += ["## Field mappings", "", self.mapping_note, ""]
        if self.access_requirements:
            lines += ["## User access", "", self.access_note, ""]
            for row in self.access_requirements:
                lines.append(f"- {row.get('user_name') or 'unnamed'} "
                             f"<{row.get('user_email') or 'no email'}> — "
                             f"{row.get('user_role') or 'no role'}")
            lines.append("")
        if self.operator_actions:
            lines += ["## Actions for an administrator", ""]
            lines += [f"- [{a['status']}] {a['subject']} — {a['detail']}"
                      for a in self.operator_actions]
            lines.append("")
        if self.activation:
            lines += ["## What activation would do", ""]
            lines += [f"- {a}" for a in self.activation.get("actions") or []]
            lines.append("")
            if self.activation.get("statement"):
                lines += [self.activation["statement"], ""]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Building it
# --------------------------------------------------------------------------- #

def build(case: OnboardingCase, run: SyntheticRun, facts: ExecutionFacts, *,
          cat: Catalogue, readiness: Optional[Dict[str, Any]] = None,
          preview: Optional[Dict[str, Any]] = None,
          onboarding: Optional[Dict[str, Any]] = None,
          intent: Optional[Dict[str, Any]] = None,
          audit: Optional[List[Dict[str, Any]]] = None) -> ReviewPackage:
    """Assemble the review package from records that already exist."""
    onboarding = onboarding or {}
    client = case.answers.get("client") or {}
    package = ReviewPackage(
        case_ref=case.case_id,
        client_name=str(client.get("client_name") or case.client_name or ""))

    for section in cat.sections:
        rows = _rows(case, section)
        if rows:
            package.sections.append({
                "key": section.key, "label": section.label,
                "rows": [r.to_dict() for r in rows]})

    package.outstanding = list(onboarding.get("client_checklist") or [])
    package.data_definitions = [dict(i) for i in case.items("data_definitions")]
    package.access_requirements = [dict(i) for i in case.items("access")]
    package.operator_actions = [a.to_dict() for a in
                                access_actions(package.access_requirements)]
    package.artefacts = [
        {"source_file": a.source_file, "artefact_type": a.artefact_type,
         "sha256": a.sha256, "row_count": a.row_count,
         "execution_status": a.execution_status}
        for a in run.artefacts()]
    package.communication = {
        "pack_status": run.pack_status,
        "history": list(run.pack_history),
        "receipt": dict(run.pack_receipt),
        "sent": bool((run.pack_receipt or {}).get("sent")),
        "statement": str((run.pack_receipt or {}).get("statement") or ""),
        "outstanding_questions": int((run.pack or {}).get("outstanding") or 0),
    }
    package.configuration_preview = {
        "artefacts": list((preview or {}).get("artefacts") or []),
        "changes": list((preview or {}).get("changes") or []),
        "next_version": (preview or {}).get("next_version", 0),
        "defaults_used": list((preview or {}).get("defaults_used") or []),
        "unrepresented": list((preview or {}).get("unrepresented") or []),
        "written": False,
    }
    package.rehearsal = {
        "mode": run.mode,
        "state": run.state,
        "state_label": _states.spec_label(run.state),
        "stage_outcomes": dict(run.stage_outcomes),
        "mapping_report": list(run.mapping_report),
        "resolved_decisions": [d for d in run.open_decisions
                               if d.get("status") != "open"],
        "open_decisions": run.blocking_decisions(),
        "observations": list(run.observations),
        "products": [{"product_id": p, "label": product_label(p, cat)}
                     for p in facts.products],
    }
    package.readiness = dict(readiness or {})
    package.activation = dict(intent or {})
    package.approvals = {
        "onboarding": {"status": case.status,
                       "status_label": STATUS_LABELS.get(case.status,
                                                         case.status),
                       "approved_by": case.approved_by,
                       "approved_at": case.approved_at,
                       "reason": case.approval_reason},
        "execution": list(run.approvals),
    }
    package.audit_trail = list(audit or [])
    package.content_hash = stable_hash(canonical_json({
        "sections": package.sections,
        "outstanding": package.outstanding,
        "access": package.access_requirements,
        "artefacts": package.artefacts,
        "readiness": package.readiness,
    }))
    return package


def _rows(case: OnboardingCase, section: Section) -> List[AnswerRow]:
    migrated = bool(case.base_documents)
    out: List[AnswerRow] = []
    if section.repeatable:
        for index, item in enumerate(case.items(section.key)):
            label = str(item.get(section.item_label_field) or "").strip()
            for f in section.fields:
                row = _row(case, section, f, item, index, label, migrated)
                if row is not None:
                    out.append(row)
    else:
        block = case.answers.get(section.key) or {}
        for f in section.fields:
            row = _row(case, section, f, block, None, "", migrated)
            if row is not None:
                out.append(row)
    return out


def _row(case: OnboardingCase, section: Section, f: Field,
         holder: Dict[str, Any], index: Optional[int],
         item: str, migrated: bool = False) -> Optional[AnswerRow]:
    value = holder.get(f.key)
    if not _present(value):
        return None
    path = f"{section.key}.{f.key}"
    indexed = f"{section.key}[{index}].{f.key}" if index is not None else path
    # What something recorded, and — where nothing did — the origin the field's
    # own `source` implies. Every value on a review package says which of the
    # five kinds of "already known" it is, rather than half of them reading as
    # "not recorded" because no code path happened to classify them.
    recorded = str(case.provenance_class.get(indexed)
                   or case.provenance_class.get(path) or "")
    provenance = _classification.origin_provenance(f, recorded,
                                                   migrated=migrated)
    # Client Onboarding's own sentence is kept as the detail: it says WHICH
    # default or WHICH derivation, which the category alone cannot.
    sentence = str(case.provenance.get(indexed)
                   or case.provenance.get(path) or "")
    label = PROVENANCE_LABELS.get(provenance) or sentence or "not recorded"
    if sentence and PROVENANCE_LABELS.get(provenance) and sentence != label:
        label = f"{label} — {sentence}"
    return AnswerRow(
        section=section.key, section_label=section.label, field=f.key,
        label=f.label, value=value, item=item, index=index,
        provenance=provenance, provenance_label=label,
        writes_to=f.writes_to, sensitive=f.sensitive)


def access_actions(rows: List[Dict[str, Any]]) -> List[OperatorAction]:
    """Turn collected access requirements into administrator actions.

    Deliberately not a provisioning call. Trakt's operator list comes from
    environment configuration, so the honest output is a list of things a human
    has to do — each marked ``not_provisioned`` — rather than a claim that
    somebody now has access.
    """
    out: List[OperatorAction] = []
    for row in rows or []:
        who = str(row.get("user_name") or "").strip()
        email = str(row.get("user_email") or "").strip()
        role = str(row.get("user_role") or "").strip()
        subject = f"{who or 'Unnamed user'}" + (f" <{email}>" if email else "")
        scope = str(row.get("scope_note") or "").strip()
        if row.get("occ_access_required"):
            out.append(OperatorAction(
                kind="occ_access", subject=subject,
                detail=("Add to the OCC operator list for this environment "
                        f"as {role or 'a role that has not been stated'}"
                        + (f", scoped to {scope}" if scope else "")
                        + ". Trakt reads operators from environment "
                          "configuration; this is not done by activating the "
                          "client.")))
        if row.get("dashboard_access_required"):
            out.append(OperatorAction(
                kind="dashboard_access", subject=subject,
                detail="Grant dashboard access" + (f" for {scope}" if scope
                                                   else "")
                       + ". Not provisioned by Trakt."))
        if row.get("report_recipient"):
            out.append(OperatorAction(
                kind="report_distribution", subject=subject,
                detail="Add to the report distribution list. Trakt sends no "
                       "email in this environment; distribution is manual."))
        if role == "approver":
            out.append(OperatorAction(
                kind="approval_role", subject=subject,
                detail="Confirm this person is permitted to approve "
                       "configuration changes for this client."))
    return out


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return bool(str(value).strip())


def _render(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "—"
    return str(value)
