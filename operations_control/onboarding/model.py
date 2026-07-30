"""operations_control.onboarding.model — the standing-configuration model.

One dataclass per wizard step. Every vocabulary offered to the operator is
resolved from configuration that ALREADY governs the platform — the asset /
regime support model, the intake dataset vocabulary, the path parser's frequency
vocabulary, the source registry's portfolio types — so onboarding can never
offer a value the rest of Trakt would refuse, and a new asset class or regime
becomes available here without a code change.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from apps.blob_trigger_app.path_parser import VALID_FREQUENCIES
from apps.blob_trigger_app.source_registry import (
    STATUS_ACTIVE,
    STATUS_PENDING_REVIEW,
)

from ..contracts import BATCH_DATASETS, REGIME_CAPABLE_DATASETS
from ..configuration.packages import ASSET_MODEL

REPO = Path(__file__).resolve().parents[2]

#: The governed declaration of which fields are standing, per reporting product.
STANDING_FIELDS_PATH = REPO / "config/regime/onboarding_standing_fields.yaml"

#: Where the Annex 12 enumerations live. Onboarding reads them; it never
#: restates them.
ANNEX12_TEMPLATE_PATH = REPO / "config/regime/annex12_template.yaml"

#: Portfolio types. The source registry records these on every registration
#: (``SourceRecord.source_portfolio_type``) and the governed default table in
#: ``trakt_core.portfolio`` keys origination behaviour off them.
PORTFOLIO_TYPES = ("direct", "acquired")

#: How a portfolio is held. Recorded as standing portfolio information and used
#: to label the book; it does not alter routing.
PORTFOLIO_STRUCTURES = ("on_balance_sheet", "warehouse", "spv", "managed")

#: Reporting products onboarding can switch on. ``mi`` is not listed because it
#: is not a choice — see ``config/regime/onboarding_standing_fields.yaml``.
SELECTABLE_PRODUCTS = ("esma_annex2", "investor_reporting", "static_pools")

LEI_RE = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")      # ISO 17442, as in annex2.preflight
CCY_RE = re.compile(r"^[A-Z]{3}$")
COUNTRY_RE = re.compile(r"^[A-Z]{2}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
IDENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

PROFILE_DRAFT = "draft"
PROFILE_ACTIVE = "active"
PROFILE_SUPERSEDED = "superseded"


# --------------------------------------------------------------------------- #
# Governed vocabulary resolution
# --------------------------------------------------------------------------- #

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


def standing_field_spec(path: Path = STANDING_FIELDS_PATH) -> Dict[str, Any]:
    """The governed per-product standing-field declaration.

    Enumerations declared as ``options_from: annex12_template:<path>`` are
    resolved here from the regime template, so the wizard renders the same
    values the projector accepts.
    """
    spec = _load_yaml(path)
    products = spec.get("products") or {}
    template = _load_yaml(ANNEX12_TEMPLATE_PATH)
    for product in products.values():
        for f in product.get("fields") or []:
            ref = f.get("options_from")
            if not ref or not str(ref).startswith("annex12_template:"):
                continue
            node: Any = template
            for part in str(ref).split(":", 1)[1].split("."):
                node = (node or {}).get(part) if isinstance(node, dict) else None
            if isinstance(node, dict):
                f["options"] = [{"value": k, "label": v}
                                for k, v in node.items()]
    return spec


def _asset_choices() -> List[Dict[str, Any]]:
    return [{"value": key, "label": entry.get("label", key),
             "supports_regimes": list(entry.get("supports_regimes") or [])}
            for key, entry in sorted(ASSET_MODEL.items())]


def governed_vocabularies() -> Dict[str, Any]:
    """Every option list the wizard may offer, with its provenance.

    The provenance strings are shown to administrators so it is obvious that
    onboarding restates nothing: each list has exactly one owner elsewhere.
    """
    spec = standing_field_spec()
    products = spec.get("products") or {}
    return {
        "asset_classes": _asset_choices(),
        "portfolio_types": list(PORTFOLIO_TYPES),
        "portfolio_structures": list(PORTFOLIO_STRUCTURES),
        "datasets": list(BATCH_DATASETS),
        "regime_capable_datasets": list(REGIME_CAPABLE_DATASETS),
        "cadences": list(dict.fromkeys(VALID_FREQUENCIES)),
        "registration_statuses": [STATUS_PENDING_REVIEW, STATUS_ACTIVE],
        "products": [
            {"key": key,
             "label": product.get("label", key),
             "selectable": key in SELECTABLE_PRODUCTS,
             "always_applies": bool(product.get("always_applies")),
             "derived_from": product.get("derived_from", ""),
             "requires_dataset": product.get("requires_dataset", ""),
             "supported_by_assets_declaring":
                 product.get("supported_by_assets_declaring", ""),
             "field_count": len(product.get("fields") or []),
             "unrepresented": product.get("unrepresented") or []}
            for key, product in products.items()
        ],
        "sources": {
            "asset_classes": "operations_control.configuration.packages.ASSET_MODEL",
            "portfolio_types": "apps.blob_trigger_app.source_registry.SourceRecord",
            "datasets": "operations_control.contracts.BATCH_DATASETS",
            "cadences": "apps.blob_trigger_app.path_parser.VALID_FREQUENCIES",
            "products": "config/regime/onboarding_standing_fields.yaml",
        },
    }


# --------------------------------------------------------------------------- #
# Step models
# --------------------------------------------------------------------------- #

@dataclass
class ClientStanding:
    """Step 1 — who the client is.

    ``client_id`` / ``client_name`` / ``environment`` map onto the existing
    ``client:`` block of the client configuration; ``jurisdiction`` and
    ``reporting_currency`` onto its ``portfolio:`` block; ``legal_entity_name``
    and ``lei`` onto its ``defaults:`` block (RREL82 / RREL83). Nothing here is
    a new field.
    """

    client_id: str = ""
    client_name: str = ""
    legal_entity_name: str = ""
    lei: str = ""
    jurisdiction: str = ""                 # ISO 3166-1 alpha-2
    reporting_currency: str = ""           # ISO 4217
    time_zone: str = "Europe/London"
    environment: str = "production"
    primary_reporting_contact: str = ""
    reporting_email: str = ""
    operational_contact: str = ""
    operational_email: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioStanding:
    """Step 2 — one book. Repeatable."""

    portfolio_id: str = ""
    display_name: str = ""
    portfolio_type: str = "direct"          # PORTFOLIO_TYPES
    asset_class: str = ""                   # ASSET_MODEL key
    structure: str = "on_balance_sheet"     # PORTFOLIO_STRUCTURES
    funded_cadence: str = "monthly"         # VALID_FREQUENCIES
    pipeline_cadence: str = ""              # blank = no pipeline book
    originates: Optional[bool] = None       # None = governed default by type
    warehouse_lender: str = ""
    source_system: str = ""

    @property
    def has_pipeline(self) -> bool:
        return bool(self.pipeline_cadence)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReportingStanding:
    """Step 3 — which reporting products apply.

    MI is not stored as a choice. ``derivation`` records, per product, whether
    the answer was derived by the platform or chosen by the operator.
    """

    products: List[str] = field(default_factory=list)
    derivation: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StaticReportingStanding:
    """Step 5 — standing investor / static reporting information.

    ``app_title`` and ``primary_color`` are the existing ``mi.branding`` block
    of the client configuration. The rest is standing information the current
    model has no artefact for; it is versioned and audited in the profile and
    reported as such.
    """

    app_title: str = ""
    primary_color: str = ""
    logo_uri: str = ""
    disclaimer: str = ""
    investor_contact: str = ""
    investor_email: str = ""
    trustee_name: str = ""
    warehouse_lender: str = ""
    payment_convention: str = ""            # e.g. QUARTERLY (loan_engine)
    day_count_convention: str = ""          # e.g. ACT365 (loan_engine)
    reporting_convention: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SourceRegistrationStanding:
    """Step 6 — one expected source. Generates one source registry record."""

    portfolio_id: str = ""
    dataset: str = "funded"                 # BATCH_DATASETS
    frequency: str = "monthly"              # VALID_FREQUENCIES
    source_system: str = ""
    expected_files: List[str] = field(default_factory=list)
    regime_required: bool = False
    status: str = STATUS_PENDING_REVIEW
    blob_prefix: str = ""                   # derived; shown, never typed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClientOnboardingProfile:
    """A client's complete standing configuration.

    This is the onboarding INPUT model — the operator's answers. The governed
    artefacts are generated from it (see :mod:`.generation`); the profile is
    never read by the delivery pipeline directly.
    """

    client_id: str = ""
    client: ClientStanding = field(default_factory=ClientStanding)
    portfolios: List[PortfolioStanding] = field(default_factory=list)
    reporting: ReportingStanding = field(default_factory=ReportingStanding)
    #: product key -> {field key: value}
    regime: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    static_reporting: StaticReportingStanding = field(
        default_factory=StaticReportingStanding)
    sources: List[SourceRegistrationStanding] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "client": self.client.to_dict(),
            "portfolios": [p.to_dict() for p in self.portfolios],
            "reporting": self.reporting.to_dict(),
            "regime": {k: dict(v) for k, v in self.regime.items()},
            "static_reporting": self.static_reporting.to_dict(),
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_dict(cls, doc: Optional[Dict[str, Any]]) -> "ClientOnboardingProfile":
        doc = doc or {}

        def build(kind, raw):
            known = {f for f in kind.__dataclass_fields__}   # type: ignore[attr-defined]
            return kind(**{k: v for k, v in (raw or {}).items() if k in known})

        return cls(
            client_id=str(doc.get("client_id") or ""),
            client=build(ClientStanding, doc.get("client")),
            portfolios=[build(PortfolioStanding, p)
                        for p in (doc.get("portfolios") or [])],
            reporting=build(ReportingStanding, doc.get("reporting")),
            regime={k: dict(v) for k, v in (doc.get("regime") or {}).items()},
            static_reporting=build(StaticReportingStanding,
                                   doc.get("static_reporting")),
            sources=[build(SourceRegistrationStanding, s)
                     for s in (doc.get("sources") or [])],
        )

    def portfolio(self, portfolio_id: str) -> Optional[PortfolioStanding]:
        return next((p for p in self.portfolios
                     if p.portfolio_id == portfolio_id), None)


# --------------------------------------------------------------------------- #
# Validation — operator wording only, no field paths, no traces
# --------------------------------------------------------------------------- #

@dataclass
class Problem:
    step: str
    field: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _blank(value: Any) -> bool:
    return value is None or not str(value).strip()


def validate(profile: ClientOnboardingProfile,
             *, spec: Optional[Dict[str, Any]] = None) -> List[Problem]:
    """Every problem that would stop this profile being approved."""
    spec = spec or standing_field_spec()
    products = spec.get("products") or {}
    out: List[Problem] = []
    c = profile.client

    if _blank(c.client_id):
        out.append(Problem("client", "client_id",
                           "The client needs an identifier."))
    elif not IDENT_RE.fullmatch(c.client_id.strip()):
        out.append(Problem("client", "client_id",
                           "A client identifier may use letters, numbers, "
                           "hyphens and underscores only."))
    if _blank(c.client_name):
        out.append(Problem("client", "client_name", "The client needs a name."))
    if _blank(c.legal_entity_name):
        out.append(Problem("client", "legal_entity_name",
                           "The legal entity name has not been given."))
    if _blank(c.lei):
        out.append(Problem("client", "lei",
                           "The legal entity identifier has not been given."))
    elif not LEI_RE.fullmatch(c.lei.strip().upper()):
        out.append(Problem("client", "lei",
                           "That is not a valid 20-character LEI."))
    if _blank(c.jurisdiction) or not COUNTRY_RE.fullmatch(
            c.jurisdiction.strip().upper()):
        out.append(Problem("client", "jurisdiction",
                           "Give the jurisdiction as a two-letter country code."))
    if _blank(c.reporting_currency) or not CCY_RE.fullmatch(
            c.reporting_currency.strip().upper()):
        out.append(Problem("client", "reporting_currency",
                           "Give the reporting currency as a three-letter code."))
    if _blank(c.time_zone):
        out.append(Problem("client", "time_zone",
                           "The reporting time zone has not been given."))
    for label, name, value in (("reporting", "reporting_email", c.reporting_email),
                               ("operational", "operational_email",
                                c.operational_email)):
        if _blank(value):
            out.append(Problem("client", name,
                               f"The {label} email address has not been given."))
        elif not EMAIL_RE.fullmatch(value.strip()):
            out.append(Problem("client", name,
                               f"That {label} email address does not look right."))
    if _blank(c.primary_reporting_contact):
        out.append(Problem("client", "primary_reporting_contact",
                           "The primary reporting contact has not been named."))
    if _blank(c.operational_contact):
        out.append(Problem("client", "operational_contact",
                           "The operational contact has not been named."))

    if not profile.portfolios:
        out.append(Problem("portfolio", "portfolios",
                           "A client needs at least one portfolio."))
    seen: set = set()
    for p in profile.portfolios:
        where = p.portfolio_id or "this portfolio"
        if _blank(p.portfolio_id):
            out.append(Problem("portfolio", "portfolio_id",
                               "Every portfolio needs an identifier."))
        elif not IDENT_RE.fullmatch(p.portfolio_id.strip()):
            out.append(Problem("portfolio", "portfolio_id",
                               f"'{p.portfolio_id}' is not a valid portfolio "
                               "identifier."))
        elif p.portfolio_id in seen:
            out.append(Problem("portfolio", "portfolio_id",
                               f"'{p.portfolio_id}' is listed twice."))
        else:
            seen.add(p.portfolio_id)
        if _blank(p.display_name):
            out.append(Problem("portfolio", "display_name",
                               f"{where} needs a display name."))
        if p.portfolio_type not in PORTFOLIO_TYPES:
            out.append(Problem("portfolio", "portfolio_type",
                               f"{where} needs to be direct or acquired."))
        if p.asset_class not in ASSET_MODEL:
            out.append(Problem("portfolio", "asset_class",
                               f"{where} has no recognised asset class."))
        if p.structure not in PORTFOLIO_STRUCTURES:
            out.append(Problem("portfolio", "structure",
                               f"{where} has no recognised structure."))
        if p.funded_cadence not in VALID_FREQUENCIES:
            out.append(Problem("portfolio", "funded_cadence",
                               f"{where} has no recognised funded reporting "
                               "cadence."))
        if p.pipeline_cadence and p.pipeline_cadence not in VALID_FREQUENCIES:
            out.append(Problem("portfolio", "pipeline_cadence",
                               f"{where} has no recognised pipeline reporting "
                               "cadence."))

    for product_key in profile.reporting.products:
        product = products.get(product_key)
        if product is None:
            out.append(Problem("reporting", "products",
                               "A reporting product was chosen that Trakt does "
                               "not offer."))
            continue
        needs = product.get("supported_by_assets_declaring")
        if needs:
            supported = [p for p in profile.portfolios
                         if needs in (ASSET_MODEL.get(p.asset_class, {})
                                      .get("supports_regimes") or [])]
            if not supported:
                out.append(Problem(
                    "reporting", "products",
                    f"No portfolio's asset class supports "
                    f"{product.get('label', product_key)}."))
        out.extend(_validate_regime(product_key, product,
                                    profile.regime.get(product_key) or {}))

    for s in profile.sources:
        if profile.portfolio(s.portfolio_id) is None:
            out.append(Problem("sources", "portfolio_id",
                               f"A source is registered against '{s.portfolio_id}', "
                               "which is not one of this client's portfolios."))
        if s.dataset not in BATCH_DATASETS:
            out.append(Problem("sources", "dataset",
                               "A source names a book Trakt does not process."))
        if s.frequency not in VALID_FREQUENCIES:
            out.append(Problem("sources", "frequency",
                               "A source has an unrecognised cadence."))
        if s.regime_required and s.dataset not in REGIME_CAPABLE_DATASETS:
            out.append(Problem(
                "sources", "regime_required",
                "Regulatory reporting is prepared from the funded book. "
                f"The {s.dataset} book cannot carry it."))
    return out


def _validate_regime(product_key: str, product: Dict[str, Any],
                     answers: Dict[str, Any]) -> List[Problem]:
    out: List[Problem] = []
    for f in product.get("fields") or []:
        key = f.get("key")
        value = answers.get(key)
        label = f.get("label", key)
        if _blank(value):
            if f.get("required"):
                out.append(Problem("regime", f"{product_key}.{key}",
                                   f"{label} is needed for "
                                   f"{product.get('label', product_key)}."))
            continue
        text = str(value).strip()
        fmt = f.get("format", "text")
        if fmt == "lei" and not LEI_RE.fullmatch(text.upper()):
            out.append(Problem("regime", f"{product_key}.{key}",
                               f"{label} is not a valid 20-character LEI."))
        elif fmt == "country" and not COUNTRY_RE.fullmatch(text.upper()):
            out.append(Problem("regime", f"{product_key}.{key}",
                               f"{label} must be a two-letter country code."))
        elif fmt == "email" and not EMAIL_RE.fullmatch(text):
            out.append(Problem("regime", f"{product_key}.{key}",
                               f"{label} does not look like an email address."))
        elif fmt == "yes_no" and text.upper() not in ("Y", "N"):
            out.append(Problem("regime", f"{product_key}.{key}",
                               f"{label} must be yes or no."))
        elif fmt == "date_or_nd" and not (DATE_RE.fullmatch(text)
                                          or text.upper().startswith("ND")):
            out.append(Problem("regime", f"{product_key}.{key}",
                               f"{label} must be a date, or ND where none "
                               "applies."))
        elif fmt == "enum":
            allowed = {str(o.get("value")) for o in (f.get("options") or [])
                       if isinstance(o, dict)}
            allowed |= {str(o) for o in (f.get("options") or [])
                        if not isinstance(o, dict)}
            if allowed and text not in allowed:
                out.append(Problem("regime", f"{product_key}.{key}",
                                   f"{label} is not one of the accepted values."))
        elif fmt == "text" and f.get("max_length") \
                and len(text) > int(f["max_length"]):
            out.append(Problem("regime", f"{product_key}.{key}",
                               f"{label} is longer than the regime allows."))
    return out


# --------------------------------------------------------------------------- #
# Derivation — what the platform decides, so the operator does not
# --------------------------------------------------------------------------- #

def derive_reporting(profile: ClientOnboardingProfile,
                     *, spec: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
    """Which reporting products apply, and why.

    MI and static pools follow from the books that exist. ESMA Annex 2
    eligibility follows from the asset support model — the SAME table the
    effective-configuration resolver uses to accept or refuse a regime outcome
    (``EffectiveConfigResolver.resolve``). Onboarding does not re-decide it; it
    reports it and lets the operator switch on what is eligible.
    """
    spec = spec or standing_field_spec()
    products = spec.get("products") or {}
    out: Dict[str, Any] = {}
    has_funded = any(p for p in profile.portfolios)
    for key, product in products.items():
        if product.get("always_applies"):
            out[key] = {"applies": True, "eligible": True, "derived": True,
                        "reason": product.get("derived_from", "")}
            continue
        needs = product.get("supported_by_assets_declaring")
        if needs:
            supporting = [p.portfolio_id for p in profile.portfolios
                          if needs in (ASSET_MODEL.get(p.asset_class, {})
                                       .get("supports_regimes") or [])]
            eligible = bool(supporting)
            reason = (f"Supported by {', '.join(supporting)}."
                      if eligible else
                      "No portfolio's asset class declares support for this "
                      "regime.")
        else:
            eligible = has_funded
            reason = ("Available for the funded book."
                      if eligible else "No portfolio has been added yet.")
        out[key] = {"applies": key in profile.reporting.products,
                    "eligible": eligible, "derived": False, "reason": reason}
    return out


def derive_sources(profile: ClientOnboardingProfile
                   ) -> List[SourceRegistrationStanding]:
    """The source registrations a set of portfolios implies.

    One funded registration per portfolio, plus one pipeline registration where
    a pipeline cadence was given. ``regime_required`` is set from the reporting
    answers, and only ever on a regime-capable dataset — the same rule the
    intake applies (``contracts.REGIME_CAPABLE_DATASETS``).
    """
    annex2 = "esma_annex2" in profile.reporting.products
    out: List[SourceRegistrationStanding] = []
    for p in profile.portfolios:
        supports = "ESMA_Annex2" in (ASSET_MODEL.get(p.asset_class, {})
                                     .get("supports_regimes") or [])
        out.append(SourceRegistrationStanding(
            portfolio_id=p.portfolio_id, dataset="funded",
            frequency=p.funded_cadence, source_system=p.source_system,
            regime_required=bool(annex2 and supports),
            status=STATUS_PENDING_REVIEW))
        if p.has_pipeline:
            out.append(SourceRegistrationStanding(
                portfolio_id=p.portfolio_id, dataset="pipeline",
                frequency=p.pipeline_cadence, source_system=p.source_system,
                regime_required=False, status=STATUS_PENDING_REVIEW))
    return out


def blob_prefix(client_id: str, portfolio_type: str, dataset: str,
                frequency: str, portfolio_id: str,
                container: str = "raw-v2") -> str:
    """Where deliveries for a registration are expected to land.

    The layout is the production one the path parser already enforces:
    ``{container}/{client}/{book_type}/{dataset}/{frequency}/{portfolio}/{period}/``.
    Shown to the operator as derived information; never typed in.
    """
    return (f"{container}/{client_id}/{portfolio_type}/{dataset}/"
            f"{frequency}/{portfolio_id}/{{reporting_period}}/")
