"""operations_control.occ_agent.pack — the client onboarding pack.

Everything in the pack is derived from the configuration the platform already
uses. Nothing here is a hand-written questionnaire:

* which source files to ask for, and which are mandatory — the required and
  optional **semantic input roles** in
  ``config/system/workflow_input_requirements.yaml``, selected by the OCC
  delivery outcome the chosen products imply;
* asset-specific questions — the configured asset's supported regimes
  (``ASSET_MODEL``) and the resolved product profile's own characteristics
  (``config/asset/product_profiles.yaml``), so an asset whose profile says a
  field is not applicable is not asked about it;
* product-specific questions — the product catalogue in
  :mod:`.vocabulary`, itself keyed on the configured capability list;
* delivery and naming instructions — the governed raw path the production
  parser accepts (:mod:`operations_control.manual_intake`), so the instructions
  the client is given are the instructions the pipeline can actually receive;
* the expected onboarding stages — the lifecycle in :mod:`.states`.

The pack is a plain document. Generating it changes nothing outside the case,
and "issuing" it means recording that fact — :mod:`.policy` refuses to send an
email, and there is no code path here that tries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import BATCH_DATASET_DEFAULT, canonical_json, stable_hash
from ..manual_intake import ManualIntakeError, derive_raw_prefix, raw_container
from . import states as _states
from .case import ExtractedRequirements
from .vocabulary import (
    artefact_vocabulary,
    asset_vocabulary,
    product_vocabulary,
)

#: Placeholder used when a fact is not yet known, so the drafted pack shows the
#: human exactly where a gap is instead of quietly reading well.
UNKNOWN = "«to be confirmed»"


@dataclass
class PackSection:
    """One regenerable section of the pack.

    ``depends_on`` names the confirmed-requirement fields the section was built
    from, so a correction can regenerate only the sections it affects.
    """

    key: str
    title: str
    body: str = ""
    items: List[Dict[str, Any]] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OnboardingPack:
    """The generated pack. Hashed so an approval names an exact version."""

    sections: List[Dict[str, Any]] = field(default_factory=list)
    outcome: str = ""
    required_roles: List[str] = field(default_factory=list)
    optional_roles: List[str] = field(default_factory=list)
    content_hash: str = ""
    generated_from: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def section(self, key: str) -> Optional[Dict[str, Any]]:
        return next((s for s in self.sections if s["key"] == key), None)


# Section keys. Stable identifiers — the UI and the regeneration logic use them.
SEC_EMAIL = "client_email"
SEC_QUESTIONNAIRE = "questionnaire"
SEC_INITIAL_REQUEST = "initial_artefact_request"
SEC_RECURRING_REQUEST = "recurring_artefact_request"
SEC_DELIVERY = "delivery_instructions"
SEC_ASSET_QUESTIONS = "asset_questions"
SEC_PRODUCT_QUESTIONS = "product_questions"
SEC_STAGES = "expected_stages"
SEC_CHECKLIST = "occ_checklist"

SECTION_KEYS = (SEC_EMAIL, SEC_QUESTIONNAIRE, SEC_INITIAL_REQUEST,
                SEC_RECURRING_REQUEST, SEC_DELIVERY, SEC_ASSET_QUESTIONS,
                SEC_PRODUCT_QUESTIONS, SEC_STAGES, SEC_CHECKLIST)


def _or_unknown(value: str) -> str:
    return value if value else UNKNOWN


def build_pack(req: ExtractedRequirements, *,
               client_id: str = "", portfolio_id: str = "") -> OnboardingPack:
    """Build the pack from confirmed requirements and platform configuration."""
    assets = asset_vocabulary()
    products = product_vocabulary()
    roles = artefact_vocabulary()

    selected = [p for p in (products.by_id(pid) for pid in req.required_products)
                if p is not None]
    outcome = products.outcome_for(req.required_products)
    required_roles = roles.required_roles(outcome)
    optional_roles = roles.optional_roles(outcome)
    # Roles the client themselves named that configuration treats as optional
    # are still worth asking for explicitly — the client expects to send them.
    named = [r for r in req.expected_artefacts
             if r not in required_roles and r in
             {x.role for x in roles.roles}]
    for role in named:
        if role not in optional_roles:
            optional_roles.append(role)

    sections: List[PackSection] = [
        _email_section(req, assets, selected, roles, required_roles),
        _questionnaire_section(req, assets, roles, required_roles),
        _initial_request_section(req, roles, required_roles, optional_roles),
        _recurring_request_section(req, roles, required_roles, optional_roles),
        _delivery_section(req, client_id=client_id, portfolio_id=portfolio_id),
        _asset_questions_section(req, assets),
        _product_questions_section(selected),
        _stages_section(),
        _checklist_section(req, roles, required_roles, selected),
    ]

    pack = OnboardingPack(
        sections=[s.to_dict() for s in sections],
        outcome=outcome,
        required_roles=required_roles,
        optional_roles=optional_roles,
        generated_from={
            "input_requirements": "config/system/workflow_input_requirements.yaml",
            "product_profiles": "config/asset/product_profiles.yaml",
            "asset_model": "operations_control.configuration.packages.ASSET_MODEL",
            "delivery_paths": "operations_control.manual_intake",
            "lifecycle": "operations_control.occ_agent.states",
        })
    pack.content_hash = stable_hash(canonical_json(
        {"sections": pack.sections, "outcome": pack.outcome,
         "required_roles": pack.required_roles,
         "optional_roles": pack.optional_roles}))
    return pack


def regenerate_sections(pack: OnboardingPack, req: ExtractedRequirements,
                        changed_fields: List[str], *, client_id: str = "",
                        portfolio_id: str = "") -> OnboardingPack:
    """Rebuild only the sections that depend on a changed fact.

    A full rebuild would be simpler, but the operator asked to correct one fact
    — showing them which sections that touched is the point of the exercise.
    """
    fresh = build_pack(req, client_id=client_id, portfolio_id=portfolio_id)
    if not changed_fields:
        return fresh
    affected = {s["key"] for s in fresh.sections
                if set(s["depends_on"]) & set(changed_fields)}
    merged = []
    for old in pack.sections:
        if old["key"] in affected:
            merged.append(fresh.section(old["key"]) or old)
        else:
            merged.append(old)
    # A section present in the fresh pack but not the old one (configuration
    # changed) is added rather than dropped.
    for new in fresh.sections:
        if not any(s["key"] == new["key"] for s in merged):
            merged.append(new)
    out = OnboardingPack(sections=merged, outcome=fresh.outcome,
                         required_roles=fresh.required_roles,
                         optional_roles=fresh.optional_roles,
                         generated_from=fresh.generated_from)
    out.content_hash = stable_hash(canonical_json(
        {"sections": out.sections, "outcome": out.outcome,
         "required_roles": out.required_roles,
         "optional_roles": out.optional_roles}))
    return out


def affected_sections(changed_fields: List[str]) -> List[str]:
    """Which section keys a set of changed facts would regenerate."""
    probe = build_pack(ExtractedRequirements())
    return sorted({s["key"] for s in probe.sections
                   if set(s["depends_on"]) & set(changed_fields)})


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def _email_section(req, assets, selected, roles, required_roles) -> PackSection:
    product_lines = "\n".join(f"  - {p.label}" for p in selected) or \
        f"  - {UNKNOWN}"
    file_lines = "\n".join(f"  - {roles.label(r)}" for r in required_roles) or \
        f"  - {UNKNOWN}"
    body = (
        f"Dear {_or_unknown(req.client_name)} team,\n\n"
        f"We are setting up {_or_unknown(req.portfolio_name or req.client_name)} "
        f"on Trakt. Trakt will prepare:\n{product_lines}\n\n"
        f"To begin, we need the following files for the "
        f"{_or_unknown(req.reporting_date)} reporting date:\n{file_lines}\n\n"
        f"Thereafter we will need the same set "
        f"{_or_unknown(req.reporting_frequency)}.\n\n"
        f"The attached questionnaire covers the remaining set-up questions. "
        f"Please return it with the first delivery so we can complete "
        f"configuration ahead of your target go-live date of "
        f"{_or_unknown(req.target_go_live_date)}.\n\n"
        "Kind regards,\nTrakt Operations")
    return PackSection(
        key=SEC_EMAIL, title="Draft client email", body=body,
        depends_on=["client_name", "portfolio_name", "required_products",
                    "reporting_date", "reporting_frequency",
                    "target_go_live_date"])


def _questionnaire_section(req, assets, roles, required_roles) -> PackSection:
    """The base questions every onboarding needs, derived from what the
    configuration must know before a run can be configured."""
    items = [
        {"id": "q_client_legal_name",
         "question": "What is the client's full legal name?",
         "why": "It identifies the client on every report Trakt produces.",
         "answer": req.client_name},
        {"id": "q_portfolio_identifier",
         "question": "What identifier should this portfolio be known by?",
         "why": "It is part of the governed storage location for every "
                "delivery, so it cannot change later without a migration.",
         "answer": req.proposed_portfolio_id},
        {"id": "q_jurisdiction",
         "question": "In which jurisdiction do the loans sit?",
         "why": "It selects the applicable regulatory and geographic "
                "reference data.",
         "answer": req.jurisdiction},
        {"id": "q_reporting_date",
         "question": "What is the first reporting date?",
         "why": "Trakt files each delivery under its reporting period.",
         "answer": req.reporting_date},
        {"id": "q_frequency",
         "question": "How often will data be delivered?",
         "why": "It sets the expected delivery cadence and the governed "
                "storage path.",
         "answer": req.reporting_frequency},
        {"id": "q_delivery_convention",
         "question": "Which day of the cycle will the files be delivered on?",
         "why": "It tells Trakt when to expect a delivery and when one is "
                "late.",
         "answer": ""},
        {"id": "q_currency",
         "question": "What is the reporting currency?",
         "why": "Amounts are validated and presented in it.",
         "answer": ""},
    ]
    for role in required_roles:
        items.append({
            "id": f"q_source_{role}",
            "question": f"Which system produces the {roles.label(role)}, and "
                        "in what file format?",
            "why": "Trakt recognises each file by its contents; knowing the "
                   "producing system helps resolve column meanings the first "
                   "time.",
            "answer": ""})
    return PackSection(
        key=SEC_QUESTIONNAIRE, title="Onboarding questionnaire", items=items,
        depends_on=["client_name", "proposed_portfolio_id", "jurisdiction",
                    "reporting_date", "reporting_frequency",
                    "required_products"])


def _initial_request_section(req, roles, required_roles,
                             optional_roles) -> PackSection:
    items = [{"role": r, "label": roles.label(r), "required": True,
              "reporting_date": req.reporting_date or UNKNOWN}
             for r in required_roles]
    items += [{"role": r, "label": roles.label(r), "required": False,
               "reporting_date": req.reporting_date or UNKNOWN}
              for r in optional_roles]
    return PackSection(
        key=SEC_INITIAL_REQUEST, title="Initial data request", items=items,
        body="The first delivery establishes the mapping and configuration. "
             "Required files must be present before onboarding can run.",
        depends_on=["required_products", "reporting_date",
                    "expected_artefacts"])


def _recurring_request_section(req, roles, required_roles,
                               optional_roles) -> PackSection:
    frequency = req.reporting_frequency or UNKNOWN
    items = [{"role": r, "label": roles.label(r), "required": True,
              "frequency": frequency} for r in required_roles]
    items += [{"role": r, "label": roles.label(r), "required": False,
               "frequency": frequency} for r in optional_roles]
    return PackSection(
        key=SEC_RECURRING_REQUEST, title="Recurring data request", items=items,
        body=f"Once live, the same set is expected {frequency}. Trakt "
             "recognises each file by its contents, so file names may vary "
             "between periods.",
        depends_on=["required_products", "reporting_frequency",
                    "expected_artefacts"])


def _delivery_section(req, *, client_id: str, portfolio_id: str) -> PackSection:
    """File naming and delivery instructions, from the governed path rules.

    The prefix is produced by the SAME function a manual delivery uses, which
    re-parses its own output with the production path parser — so a location
    quoted here is one the automated intake route would accept. When the facts
    are not yet complete enough to derive a location, the section says so
    rather than inventing one.
    """
    client = client_id or req.proposed_client_id
    portfolio = portfolio_id or req.proposed_portfolio_id
    period = req.reporting_date
    frequency = req.reporting_frequency or "monthly"
    prefix = ""
    problem = ""
    if client and portfolio and period:
        try:
            prefix = derive_raw_prefix(
                client_id=client, portfolio_id=portfolio,
                reporting_period=period, dataset=BATCH_DATASET_DEFAULT,
                frequency=frequency)
        except ManualIntakeError as exc:
            problem = str(exc)
    else:
        problem = ("The delivery location cannot be quoted until the client "
                   "identifier, portfolio identifier and first reporting date "
                   "are confirmed.")
    items = [
        {"instruction": "Send spreadsheets (.xlsx/.xls/.xlsm) or "
                        "comma-separated (.csv) files only."},
        {"instruction": "One file per logical extract. Do not combine "
                        "extracts into a single sheet."},
        {"instruction": "Keep column headings stable between periods. Trakt "
                        "recognises a changed layout and will ask before "
                        "using it."},
        {"instruction": "Do not send a readiness marker file — Trakt decides "
                        "when a delivery is complete."},
    ]
    body = (f"Deliveries are filed under blob://{prefix}/ "
            f"(container {raw_container()})." if prefix
            else problem)
    return PackSection(
        key=SEC_DELIVERY, title="File naming and delivery instructions",
        body=body, items=items,
        depends_on=["proposed_client_id", "proposed_portfolio_id",
                    "reporting_date", "reporting_frequency"])


def _asset_questions_section(req, assets) -> PackSection:
    """Questions that follow from the configured asset type."""
    items: List[Dict[str, Any]] = []
    asset = req.asset_type
    if not asset:
        items.append({"id": "q_asset_type",
                      "question": "What asset type is this portfolio?",
                      "why": "The asset type selects the canonical field "
                             "requirements and the regimes Trakt can report "
                             "under."})
    else:
        label = assets.label(asset)
        supported = assets.supports_regimes(asset)
        items.append({
            "id": "q_asset_characteristics",
            "question": f"Do all loans in this {label.lower()} portfolio "
                        "share the same product characteristics?",
            "why": "Trakt applies a product profile to decide which fields are "
                   "meaningful for this asset; a mixed book may need more than "
                   "one."})
        if supported:
            items.append({
                "id": "q_regime_intent",
                "question": "Will this portfolio be reported under "
                            + ", ".join(supported) + "?",
                "why": "A regulatory regime adds mandatory fields to the first "
                       "delivery."})
        else:
            items.append({
                "id": "q_regime_none",
                "question": "Please confirm no regulatory submission is "
                            "required for this portfolio.",
                "why": "Trakt has no regulatory regime configured for this "
                       "asset type."})
    return PackSection(
        key=SEC_ASSET_QUESTIONS, title="Asset-specific questions", items=items,
        depends_on=["asset_type"])


def _product_questions_section(selected) -> PackSection:
    items: List[Dict[str, Any]] = []
    for product in selected:
        for i, question in enumerate(product.questions):
            items.append({
                "id": f"q_{product.product_id}_{i}",
                "product": product.product_id,
                "product_label": product.label,
                "question": question,
                "why": f"Required to configure {product.label}."})
    if not items:
        items.append({"id": "q_products",
                      "question": "Which Trakt products does the client need?",
                      "why": "Product selection determines the configuration "
                             "and the required source files."})
    return PackSection(
        key=SEC_PRODUCT_QUESTIONS, title="Product-specific questions",
        items=items, depends_on=["required_products"])


def _stages_section() -> PackSection:
    items = [{"state": s["state"], "label": s["label"],
              "approvals": s["required_approvals"]}
             for s in _states.lifecycle()
             if s["state"] not in (_states.BLOCKED, _states.CANCELLED)]
    return PackSection(
        key=SEC_STAGES, title="Expected onboarding stages", items=items,
        body="Trakt will move this onboarding through these stages. Stages "
             "that require an approval will wait for you.",
        depends_on=[])


def _checklist_section(req, roles, required_roles, selected) -> PackSection:
    """The internal OCC checklist — what the operator must satisfy."""
    items = [
        {"id": "chk_identity",
         "check": "Client and portfolio identifiers confirmed",
         "done": bool(req.proposed_client_id and req.proposed_portfolio_id)},
        {"id": "chk_products",
         "check": "Product scope confirmed",
         "done": bool(selected)},
        {"id": "chk_period",
         "check": "First reporting date confirmed",
         "done": bool(req.reporting_date)},
        {"id": "chk_frequency",
         "check": "Delivery frequency confirmed",
         "done": bool(req.reporting_frequency)},
    ]
    for role in required_roles:
        items.append({"id": f"chk_role_{role}",
                      "check": f"{roles.label(role)} requested",
                      "done": True})
    items += [
        {"id": "chk_config", "check": "Client configuration approved",
         "done": False},
        {"id": "chk_mappings", "check": "Field mappings resolved",
         "done": False},
        {"id": "chk_validation", "check": "Validation exceptions cleared",
         "done": False},
        {"id": "chk_readiness", "check": "Readiness approved", "done": False},
    ]
    return PackSection(
        key=SEC_CHECKLIST, title="Internal OCC checklist", items=items,
        depends_on=["proposed_client_id", "proposed_portfolio_id",
                    "required_products", "reporting_date",
                    "reporting_frequency"])
