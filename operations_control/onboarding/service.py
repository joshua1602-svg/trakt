"""operations_control.onboarding.service — the governed onboarding workflow.

Draft -> answer -> review -> approve. Nothing is written to a governed artefact
until an operator has seen exactly what will change and approved it, and every
approval appends to the existing hash-chained audit trail with who, when, what
changed, before, after and why.

The service owns no configuration of its own: it reads the governed vocabularies
(:mod:`.model`), adopts existing clients from their legacy files
(:mod:`.migration`) and writes the existing artefacts (:mod:`.generation`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..contracts import new_id, now_iso
from ..engine import OpsError
from ..stores import OpsStore
from . import generation, migration
from .model import (
    PROFILE_DRAFT,
    ClientOnboardingProfile,
    derive_reporting,
    derive_sources,
    governed_vocabularies,
    standing_field_spec,
    validate,
)
from .store import OnboardingStore, ProfileVersion

STEPS = ("client", "portfolio", "reporting", "regime", "static_reporting",
         "sources", "review")

STEP_LABELS = {
    "client": "Client",
    "portfolio": "Portfolios",
    "reporting": "Reporting",
    "regime": "Regime configuration",
    "static_reporting": "Investor and static reporting",
    "sources": "Source registration",
    "review": "Review",
}


class OnboardingService:
    """Everything the Client Onboarding area does."""

    def __init__(self, store: OpsStore, *, registry_factory=None,
                 adopt_paths: Optional[Dict[str, Any]] = None):
        self.store = store
        self.profiles = OnboardingStore(store)
        self._registry_factory = registry_factory
        self._adopt_paths = adopt_paths or {}

    # ------------------------------------------------------------------ #
    def _registry(self):
        if self._registry_factory is not None:
            return self._registry_factory()
        try:
            from ..classification import open_source_registry
            return open_source_registry(storage=self.store.storage)
        except Exception:      # noqa: BLE001 — onboarding still works read-only
            return None

    def spec(self) -> Dict[str, Any]:
        return standing_field_spec()

    def vocabularies(self) -> Dict[str, Any]:
        return governed_vocabularies()

    # ------------------------------------------------------------------ #
    # The home screen
    # ------------------------------------------------------------------ #
    def list_clients(self, *, visible: Optional[List[str]] = None
                     ) -> List[Dict[str, Any]]:
        """Every client the operator can see, onboarded or not.

        A client that predates onboarding is listed as ``legacy`` — it works
        today, and adopting it is an offer rather than a demand.
        """
        known: List[str] = list(self.store.known_clients())
        registry = self._registry()
        if registry is not None:
            for record in registry.records():
                if record.client_id not in known:
                    known.append(record.client_id)
        onboarded = set(self.profiles.onboarded_clients())
        for client_id in onboarded:
            if client_id not in known:
                known.append(client_id)
        if visible is not None:
            known = [c for c in known if c in visible]

        out: List[Dict[str, Any]] = []
        for client_id in sorted(known):
            current = self.profiles.current(client_id)
            drafts = self.profiles.list_drafts(client_id)
            if current is None:
                out.append({
                    "client_id": client_id, "status": "legacy",
                    "display_name": client_id, "version": 0,
                    "portfolios": 0, "products": [],
                    "updated_at": "", "updated_by": "",
                    "open_drafts": len(drafts),
                    "summary": "Configured before Client Onboarding existed. "
                               "Its current values can be adopted without "
                               "re-entering them."})
                continue
            profile = current.as_profile()
            out.append({
                "client_id": client_id, "status": "onboarded",
                "display_name": profile.client.client_name or client_id,
                "version": current.version,
                "portfolios": len(profile.portfolios),
                "products": list(profile.reporting.products),
                "updated_at": current.approved_at,
                "updated_by": current.approved_by,
                "open_drafts": len(drafts),
                "summary": ""})
        return out

    # ------------------------------------------------------------------ #
    # Drafts
    # ------------------------------------------------------------------ #
    def start_draft(self, *, client_id: str = "", by: str,
                    adopt: bool = False) -> Dict[str, Any]:
        """Open a draft: blank for a new client, or populated from what exists.

        ``adopt`` reads the legacy configuration for ``client_id``; an already
        onboarded client is continued from its current version instead.
        """
        adoption = None
        base_documents: Dict[str, Any] = {}
        provenance: Dict[str, str] = {}
        gaps: List[Dict[str, str]] = []
        sources_read: List[str] = []

        current = self.profiles.current(client_id) if client_id else None
        if current is not None:
            profile = current.as_profile()
            based_on = current.version
            origin = "current"
        elif adopt and client_id:
            adoption = migration.adopt(client_id, registry=self._registry(),
                                       **self._adopt_paths)
            profile = adoption.profile
            base_documents = adoption.base_documents
            provenance = adoption.provenance
            gaps = adoption.gaps
            sources_read = adoption.sources_read
            based_on = None
            origin = "adopted"
        else:
            profile = ClientOnboardingProfile(client_id=client_id)
            profile.client.client_id = client_id
            based_on = None
            origin = "new"

        draft = {
            "draft_id": new_id("onb"),
            "client_id": client_id,
            "status": PROFILE_DRAFT,
            "origin": origin,
            "based_on_version": based_on,
            "profile": profile.to_dict(),
            "provenance": provenance,
            "gaps": gaps,
            "sources_read": sources_read,
            "base_documents": base_documents,
            "created_by": by,
            "created_at": now_iso(),
            "step": STEPS[0],
        }
        self.profiles.save_draft(client_id or "_new", draft)
        if client_id:
            self.store.append_audit(
                client_id, "onboarding_draft_opened", actor=by,
                detail={"draft_id": draft["draft_id"], "origin": origin})
        return self.present_draft(draft)

    def load_draft(self, client_id: str, draft_id: str) -> Dict[str, Any]:
        doc = self.profiles.load_draft(client_id or "_new", draft_id)
        if doc is None or doc.get("status") != PROFILE_DRAFT:
            raise OpsError("OPS_NOT_FOUND", "That could not be found.", 404)
        return doc

    def save_step(self, *, client_id: str, draft_id: str, step: str,
                  payload: Dict[str, Any], by: str) -> Dict[str, Any]:
        """Record one step's answers. Derivations are re-run on every save so
        the operator never sees a stale downstream step."""
        if step not in STEPS:
            raise OpsError("OPS_BAD_STEP", "That is not a step of this "
                                           "workflow.", 400)
        doc = self.load_draft(client_id, draft_id)
        profile = ClientOnboardingProfile.from_dict(doc["profile"])
        merged = {**profile.to_dict()}

        if step == "client":
            merged["client"] = {**merged["client"], **(payload or {})}
            new_client_id = str((payload or {}).get("client_id")
                                or merged["client"].get("client_id") or "")
            merged["client_id"] = new_client_id
        elif step == "portfolio":
            merged["portfolios"] = list((payload or {}).get("portfolios") or [])
        elif step == "reporting":
            merged["reporting"] = {
                "products": list((payload or {}).get("products") or []),
                "derivation": dict((payload or {}).get("derivation") or {})}
        elif step == "regime":
            regime = {**merged.get("regime", {})}
            for key, answers in ((payload or {}).get("regime") or {}).items():
                regime[key] = {**(regime.get(key) or {}), **(answers or {})}
            merged["regime"] = regime
        elif step == "static_reporting":
            merged["static_reporting"] = {**merged["static_reporting"],
                                          **(payload or {})}
        elif step == "sources":
            merged["sources"] = list((payload or {}).get("sources") or [])

        profile = ClientOnboardingProfile.from_dict(merged)

        # Source registrations follow from portfolios and reporting. They are
        # derived on every save and only kept where the operator has since
        # edited them, so an operator never maintains them by hand.
        if step in ("portfolio", "reporting"):
            existing = {(s.portfolio_id, s.dataset): s for s in profile.sources}
            rebuilt = []
            for derived in derive_sources(profile):
                previous = existing.get((derived.portfolio_id, derived.dataset))
                if previous is not None:
                    derived.source_system = (previous.source_system
                                             or derived.source_system)
                    derived.expected_files = list(previous.expected_files)
                    derived.status = previous.status
                rebuilt.append(derived)
            profile.sources = rebuilt

        # A draft opened without a client id becomes owned by one as soon as
        # step 1 names it; the placeholder draft is closed so it cannot be
        # resumed under the old key.
        new_client_id = profile.client_id
        old_key = client_id or "_new"
        new_key = new_client_id or "_new"
        doc["profile"] = profile.to_dict()
        doc["client_id"] = new_client_id
        doc["step"] = step
        doc["updated_by"] = by
        if new_key != old_key:
            self.profiles.discard_draft(old_key, draft_id, by=by,
                                        reason="client identified")
        self.profiles.save_draft(new_key, doc)
        return self.present_draft(doc)

    def discard_draft(self, *, client_id: str, draft_id: str, by: str,
                      reason: str = "") -> None:
        self.load_draft(client_id, draft_id)
        self.profiles.discard_draft(client_id or "_new", draft_id, by=by,
                                    reason=reason)
        if client_id:
            self.store.append_audit(client_id, "onboarding_draft_discarded",
                                    actor=by, detail={"draft_id": draft_id,
                                                      "reason": reason})

    # ------------------------------------------------------------------ #
    # Presentation
    # ------------------------------------------------------------------ #
    def present_draft(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        profile = ClientOnboardingProfile.from_dict(doc["profile"])
        spec = self.spec()
        problems = validate(profile, spec=spec)
        by_step: Dict[str, List[Dict[str, Any]]] = {s: [] for s in STEPS}
        for problem in problems:
            by_step.setdefault(problem.step, []).append(problem.to_dict())
        return {
            "draft_id": doc["draft_id"],
            "client_id": doc.get("client_id", ""),
            "origin": doc.get("origin", "new"),
            "based_on_version": doc.get("based_on_version"),
            "step": doc.get("step", STEPS[0]),
            "steps": [{"key": s, "label": STEP_LABELS[s],
                       "problems": len(by_step.get(s) or [])} for s in STEPS],
            "profile": profile.to_dict(),
            "derived_reporting": derive_reporting(profile, spec=spec),
            "derived_sources": [s.to_dict() for s in derive_sources(profile)],
            "provenance": doc.get("provenance") or {},
            "gaps": doc.get("gaps") or [],
            "sources_read": doc.get("sources_read") or [],
            "problems": [p.to_dict() for p in problems],
            "problems_by_step": by_step,
            "ready": not problems,
            "created_by": doc.get("created_by", ""),
            "created_at": doc.get("created_at", ""),
        }

    # ------------------------------------------------------------------ #
    # Review + approval
    # ------------------------------------------------------------------ #
    def review(self, *, client_id: str, draft_id: str) -> Dict[str, Any]:
        """Exactly what approving this draft would create or change."""
        doc = self.load_draft(client_id, draft_id)
        profile = ClientOnboardingProfile.from_dict(doc["profile"])
        problems = validate(profile, spec=self.spec())
        registry = self._registry()
        artefacts = generation.plan(
            profile, store=self.profiles, registry=registry, spec=self.spec(),
            base_documents=doc.get("base_documents") or {})
        current = (self.profiles.current(profile.client_id)
                   if profile.client_id else None)
        changes = diff_profiles(current.as_profile() if current else None,
                                profile)
        return {
            "draft_id": draft_id,
            "client_id": profile.client_id,
            "profile": profile.to_dict(),
            "artefacts": [a.to_dict() for a in artefacts],
            "changes": changes,
            "problems": [p.to_dict() for p in problems],
            "ready": not problems,
            "current_version": current.version if current else 0,
            "next_version": (self.profiles.next_version(profile.client_id)
                             if profile.client_id else 1),
            "unrepresented": self._unrepresented(profile),
        }

    def _unrepresented(self, profile: ClientOnboardingProfile
                       ) -> List[Dict[str, Any]]:
        """Standing information the current model has no artefact for. Shown at
        review so the operator knows what is recorded but not generated."""
        spec = self.spec()
        out: List[Dict[str, Any]] = []
        for key in profile.reporting.products:
            product = (spec.get("products") or {}).get(key) or {}
            for entry in product.get("unrepresented") or []:
                out.append({"product": product.get("label", key), **entry})
        return out

    def approve(self, *, client_id: str, draft_id: str, by: str,
                reason: str) -> Dict[str, Any]:
        """Write the artefacts and commit an immutable version."""
        if not str(reason or "").strip():
            raise OpsError("OPS_REASON_REQUIRED",
                           "Please say why this configuration is changing.",
                           400)
        doc = self.load_draft(client_id, draft_id)
        profile = ClientOnboardingProfile.from_dict(doc["profile"])
        if not profile.client_id:
            raise OpsError("OPS_CLIENT_REQUIRED",
                           "The client needs an identifier before its "
                           "configuration can be saved.", 400)
        problems = validate(profile, spec=self.spec())
        if problems:
            raise OpsError("OPS_ONBOARDING_INCOMPLETE",
                           problems[0].message, 409)

        registry = self._registry()
        base_documents = doc.get("base_documents") or {}
        artefacts = generation.plan(profile, store=self.profiles,
                                    registry=registry, spec=self.spec(),
                                    base_documents=base_documents)
        current = self.profiles.current(profile.client_id)
        changes = diff_profiles(current.as_profile() if current else None,
                                profile)
        version = self.profiles.next_version(profile.client_id)
        written = generation.apply(profile, store=self.profiles,
                                   version=version, registry=registry,
                                   artefacts=artefacts,
                                   base_documents=base_documents)
        record = self.profiles.commit(
            client_id=profile.client_id, profile=profile, approved_by=by,
            reason=reason, changes=changes, artefacts=written,
            based_on_version=(current.version if current else None))
        self.profiles.discard_draft(doc.get("client_id") or "_new", draft_id,
                                    by=by, reason="approved")
        self.store.append_audit(
            profile.client_id, "onboarding_configuration_approved", actor=by,
            detail={"version": record.version,
                    "based_on_version": record.based_on_version,
                    "reason": reason,
                    "changes": changes,
                    "artefacts": written,
                    "content_hash": record.content_hash})
        return {"version": record.version,
                "client_id": profile.client_id,
                "artefacts": written,
                "changes": changes}

    # ------------------------------------------------------------------ #
    # Existing client view
    # ------------------------------------------------------------------ #
    def client_view(self, client_id: str) -> Dict[str, Any]:
        """General / Portfolios / Reporting / Regimes / Source registrations /
        History for one client."""
        current = self.profiles.current(client_id)
        registry = self._registry()
        live_records = []
        if registry is not None:
            live_records = [r.to_dict() for r in registry.records()
                            if r.client_id == client_id]
        if current is None:
            return {"client_id": client_id, "status": "legacy",
                    "version": 0, "profile": None,
                    "live_source_registrations": live_records,
                    "history": [], "artefacts": [],
                    "summary": "This client has not been through Client "
                               "Onboarding. Adopt it to manage its "
                               "configuration here."}
        profile = current.as_profile()
        spec = self.spec()
        return {
            "client_id": client_id,
            "status": "onboarded",
            "version": current.version,
            "approved_by": current.approved_by,
            "approved_at": current.approved_at,
            "reason": current.reason,
            "content_hash": current.content_hash,
            "profile": profile.to_dict(),
            "derived_reporting": derive_reporting(profile, spec=spec),
            "live_source_registrations": live_records,
            "artefacts": current.artefacts,
            "unrepresented": self._unrepresented(profile),
            "history": [self._history_row(v)
                        for v in self.profiles.history(client_id)],
        }

    @staticmethod
    def _history_row(v: ProfileVersion) -> Dict[str, Any]:
        return {"version": v.version, "status": v.status,
                "approved_by": v.approved_by, "approved_at": v.approved_at,
                "reason": v.reason, "based_on_version": v.based_on_version,
                "change_count": len(v.changes), "changes": v.changes,
                "artefacts": v.artefacts, "content_hash": v.content_hash}

    def version_view(self, client_id: str, version: int) -> Dict[str, Any]:
        record = self.profiles.version(client_id, version)
        if record is None:
            raise OpsError("OPS_NOT_FOUND", "That could not be found.", 404)
        return {**self._history_row(record), "profile": record.profile}


# --------------------------------------------------------------------------- #
# Field-level change detection
# --------------------------------------------------------------------------- #

_FIELD_LABELS = {
    "client.client_name": "Client name",
    "client.legal_entity_name": "Legal entity name",
    "client.lei": "Legal entity identifier",
    "client.jurisdiction": "Jurisdiction",
    "client.reporting_currency": "Reporting currency",
    "client.time_zone": "Time zone",
    "client.primary_reporting_contact": "Primary reporting contact",
    "client.reporting_email": "Reporting email",
    "client.operational_contact": "Operational contact",
    "client.operational_email": "Operational email",
    "reporting.products": "Reporting products",
}


def _label(path: str) -> str:
    if path in _FIELD_LABELS:
        return _FIELD_LABELS[path]
    return path.replace("_", " ").replace(".", " → ")


def _flatten(node: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(node, list):
        if all(not isinstance(v, (dict, list)) for v in node):
            out[prefix] = ", ".join(str(v) for v in node)
        else:
            for i, v in enumerate(node):
                key = (v.get("portfolio_id") or v.get("key")
                       or f"{i}") if isinstance(v, dict) else str(i)
                out.update(_flatten(v, f"{prefix}[{key}]"))
    else:
        out[prefix] = node
    return out


#: Paths that restate another path and would show twice in a change list. The
#: profile carries the client id at the root as well as inside the client block
#: (the root copy is the storage key); an operator should see it once.
_MIRRORED_PATHS = frozenset({"client_id"})


def diff_profiles(before: Optional[ClientOnboardingProfile],
                  after: ClientOnboardingProfile) -> List[Dict[str, Any]]:
    """Field-level before/after between two profiles, in operator wording."""
    old = _flatten(before.to_dict()) if before else {}
    new = _flatten(after.to_dict())
    changes: List[Dict[str, Any]] = []
    for path in sorted((set(old) | set(new)) - _MIRRORED_PATHS):
        was, now = old.get(path), new.get(path)
        if was == now:
            continue
        if (was is None or str(was) == "") and (now is None or str(now) == ""):
            continue
        changes.append({
            "path": path, "label": _label(path),
            "before": "" if was is None else str(was),
            "after": "" if now is None else str(now),
            "action": ("added" if was in (None, "") else
                       "removed" if now in (None, "") else "changed")})
    return changes
