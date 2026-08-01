"""operations_control.onboarding.validation — readiness, in plain English.

Validation runs continuously, not only at the end: the wizard shows a step's
problems as they arise, the readiness screen shows all of them, and approval is
refused while any blocking problem remains.

Every rule is driven by the catalogue except the structural ones that no
per-field declaration can express — identifier collisions, entity references
that point nowhere, a regime asked of a book that cannot carry it. Those are
here, named, rather than scattered through the service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..contracts import REGIME_CAPABLE_DATASETS
from .case import OnboardingCase
from .catalogue import Catalogue, Field, Section, catalogue, validate_value

SEVERITY_BLOCKING = "blocking"
SEVERITY_ADVISORY = "advisory"


@dataclass
class Problem:
    section: str
    field: str
    message: str
    severity: str = SEVERITY_BLOCKING
    #: Index within a repeatable section, where applicable.
    index: Optional[int] = None
    #: Who can resolve it: client | operator.
    owner: str = "operator"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return bool(str(value).strip())


class Validator:
    """Validates a case against the catalogue plus the structural rules."""

    def __init__(self, cat: Optional[Catalogue] = None, *,
                 existing_clients: Sequence[str] = (),
                 existing_portfolios: Sequence[str] = ()):
        self.catalogue = cat or catalogue()
        #: Client identifiers already in use, so onboarding cannot mint a
        #: collision. Supplied by the service from the live platform.
        self.existing_clients = {str(c).lower() for c in existing_clients}
        #: ``client/portfolio`` pairs already registered.
        self.existing_portfolios = {str(p).lower() for p in existing_portfolios}

    # ------------------------------------------------------------------ #
    def validate(self, case: OnboardingCase) -> List[Problem]:
        problems: List[Problem] = []
        answers = case.answers
        for section in self.catalogue.sections:
            if section.repeatable:
                problems.extend(self._repeatable(section, case, answers))
            elif section.from_regime:
                problems.extend(self._regime(section, case, answers))
            else:
                problems.extend(self._block(section, case, answers))
        problems.extend(self._structural(case))
        return problems

    def blocking(self, case: OnboardingCase) -> List[Problem]:
        return [p for p in self.validate(case)
                if p.severity == SEVERITY_BLOCKING]

    def is_ready(self, case: OnboardingCase) -> bool:
        return not self.blocking(case) and not case.outstanding_requests

    # -- per section ---------------------------------------------------- #
    def _block(self, section: Section, case: OnboardingCase,
               answers: Dict[str, Any]) -> List[Problem]:
        out: List[Problem] = []
        block = answers.get(section.key) or {}
        for f in section.fields:
            if not f.collected:
                continue
            value = block.get(f.key)
            out.extend(self._field(section, f, value, answers, block))
        return out

    def _repeatable(self, section: Section, case: OnboardingCase,
                    answers: Dict[str, Any]) -> List[Problem]:
        out: List[Problem] = []
        items = answers.get(section.key) or []
        if section.min_items and len(items) < section.min_items:
            out.append(Problem(
                section.key, section.key,
                f"At least {section.min_items} "
                f"{section.label.lower().rstrip('s')}"
                f"{'' if section.min_items == 1 else 's'} "
                "must be added." if section.min_items == 1 else
                f"At least {section.min_items} entries are needed under "
                f"{section.label.lower()}."))
        for index, item in enumerate(items):
            for f in section.fields:
                if not f.collected:
                    continue
                out.extend(self._field(section, f, item.get(f.key), answers,
                                       item, index=index))
        return out

    def _regime(self, section: Section, case: OnboardingCase,
                answers: Dict[str, Any]) -> List[Problem]:
        """Only the selected products' fields are required — a regime the client
        does not receive must never block their onboarding."""
        out: List[Problem] = []
        chosen = set(case.products)
        held = answers.get(section.key) or {}
        for f in section.fields:
            if f.product and f.product not in chosen:
                continue
            value = (held.get(f.product) or {}).get(f.key)
            out.extend(self._field(section, f, value, answers, held,
                                   product=f.product))
        return out

    # -- one field ------------------------------------------------------- #
    def _field(self, section: Section, f: Field, value: Any,
               answers: Dict[str, Any], item: Dict[str, Any],
               index: Optional[int] = None,
               product: str = "") -> List[Problem]:
        out: List[Problem] = []
        key = f"{product}.{f.key}" if product else f.key
        owner = "client" if f.asked_of_client else "operator"
        where = ""
        if index is not None and section.item_label_field:
            name = str(item.get(section.item_label_field) or "").strip()
            where = f" for {name}" if name else f" (item {index + 1})"

        if not _present(value):
            if self.catalogue.is_required(f, answers, item):
                out.append(Problem(section.key, key,
                                   f"{f.label}{where} is needed.",
                                   index=index, owner=owner))
            return out

        problem = validate_value(f, value)
        if problem:
            out.append(Problem(section.key, key, f"{problem[:-1]}{where}."
                               if where and problem.endswith(".") else problem,
                               index=index, owner=owner))
        return out

    # -- structural rules ------------------------------------------------- #
    def _structural(self, case: OnboardingCase) -> List[Problem]:
        out: List[Problem] = []
        client = case.answers.get("client") or {}
        client_id = str(client.get("client_id") or "").strip()

        # The concentration-test request may not sit unresolved at approval: a
        # blank answer cannot masquerade as completion, and only an operator
        # decision (supplied / not_applicable / deferred_with_reason) closes
        # it. Applies only where the request is asked at all (MI clients).
        status_field = self.catalogue.field("risk_limits",
                                            "concentration_tests_status")
        if status_field is not None and \
                self.catalogue.is_asked(status_field, case.answers):
            risk = case.answers.get("risk_limits") or {}
            status = str(risk.get("concentration_tests_status") or "").strip()
            # A missing status is already reported by the catalogue's own
            # required_when; the structural rules cover what the catalogue
            # cannot say: "pending" is not a completion, and "supplied" with a
            # blank answer is not supplied.
            if status == "pending_client_response":
                out.append(Problem(
                    "risk_limits", "concentration_tests_status",
                    "The concentration-test request is still awaiting the "
                    "client's response. Record it as supplied, not "
                    "applicable, or deferred with a reason before approval.",
                    owner="client"))
            elif status == "supplied" and \
                    not _present(risk.get("concentration_tests")):
                out.append(Problem(
                    "risk_limits", "concentration_tests",
                    "The request is marked supplied but no response is "
                    "recorded — a blank answer cannot count as supplied."))

        # A new client may not take an identifier already in use. An amendment
        # or migration keeps the identifier it already has, so the collision
        # check applies only where the case is creating one.
        if client_id and case.based_on_version is None \
                and case.kind == "new_client" \
                and client_id.lower() in self.existing_clients:
            out.append(Problem(
                "client", "client_id",
                f"'{client_id}' is already in use by another client. "
                "Choose a different identifier."))

        seen_portfolios: set = set()
        for index, p in enumerate(case.items("portfolios")):
            pid = str(p.get("portfolio_id") or "").strip()
            if not pid:
                continue
            if pid.lower() in seen_portfolios:
                out.append(Problem("portfolios", "portfolio_id",
                                   f"'{pid}' is listed twice.", index=index))
            seen_portfolios.add(pid.lower())
            pair = f"{client_id}/{pid}".lower()
            if case.based_on_version is None and case.kind == "new_client" \
                    and pair in self.existing_portfolios:
                out.append(Problem(
                    "portfolios", "portfolio_id",
                    f"'{pid}' is already registered for this client.",
                    index=index))
            entity_ref = str(p.get("owning_entity") or "").strip()
            if entity_ref and case.entity(entity_ref) is None:
                out.append(Problem(
                    "portfolios", "owning_entity",
                    f"{p.get('display_name') or pid} points at an entity that "
                    "is no longer listed.", index=index))

        # Sources must belong to a portfolio, must not repeat, and must not ask
        # for regulatory reporting from a book that cannot carry it — the same
        # rule the intake enforces on every delivery.
        seen_sources: set = set()
        for index, s in enumerate(case.items("sources")):
            pid = str(s.get("portfolio_id") or "").strip()
            dataset = str(s.get("dataset") or "").strip()
            if pid and case.portfolio(pid) is None:
                out.append(Problem(
                    "sources", "portfolio_id",
                    f"A delivery is expected for '{pid}', which is not one of "
                    "this client's portfolios.", index=index))
            key = f"{pid}/{dataset}".lower()
            if key in seen_sources:
                out.append(Problem(
                    "sources", "dataset",
                    f"Two deliveries are expected for {pid} {dataset}. "
                    "Remove one.", index=index))
            seen_sources.add(key)
            if s.get("regime_required") and dataset \
                    and dataset not in REGIME_CAPABLE_DATASETS:
                out.append(Problem(
                    "sources", "regime_required",
                    "Regulatory reporting is prepared from the funded book. "
                    f"The {dataset} book cannot carry it.", index=index))

        # Entity roles: a product that names an entity needs that entity.
        products = set(case.products)
        if "esma_annex2" in products and not case.entities_with_role("originator"):
            out.append(Problem(
                "entities", "roles",
                "Regulatory reporting names an originator. Give one entity the "
                "originator role."))
        if "investor_reporting" in products \
                and not case.entities_with_role("reporting_entity"):
            out.append(Problem(
                "entities", "roles",
                "Investor reporting names a reporting entity. Give one entity "
                "the reporting entity role.", severity=SEVERITY_ADVISORY))

        # A product the portfolios cannot support.
        from .derivation import product_eligibility
        for key, state in product_eligibility(case, self.catalogue).items():
            if key in products and not state["eligible"]:
                out.append(Problem("reporting", "products", state["reason"]))

        # Unresolved client questions block approval, not editing.
        for question in case.unresolved_questions:
            out.append(Problem(
                "review", "open_questions",
                f"An open question is unanswered: "
                f"{question.get('question', 'unspecified')}",
                severity=SEVERITY_ADVISORY, owner="client"))
        return out
