"""operations_control.onboarding.migration — adopt an existing client.

No operator recreates a client that already works. This module reads what is
already there and builds the onboarding profile from it, so the first screen for
an existing client shows current values ready for editing.

Sources read (all existing, none modified):

  ``config/client/config_client_ERM_UK.yaml``   client identity, jurisdiction,
                                                currency, regimes, Annex 2
                                                defaults, MI branding, loan
                                                engine conventions
  ``config/client/config_client_annex12.yaml``  standing investor-report fields
  ``config/client/portfolio_registry.yaml``     portfolio labels, origination,
                                                pipeline availability
  ``config/tenancy.yaml``                       display name, portfolio list
  the durable source registry                   portfolios, datasets, cadences,
                                                source systems, regime scope

Everything read is reported with its origin, and anything the legacy files do
not answer is left blank for the operator rather than guessed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from apps.blob_trigger_app.source_registry import SourceRecord, SourceRegistry

from ..configuration.packages import ASSET_MODEL
from .generation import (
    ARTEFACT_ANNEX12_CONFIG,
    ARTEFACT_CLIENT_CONFIG,
    ARTEFACT_PORTFOLIO_REGISTRY,
)
from .model import (
    ClientOnboardingProfile,
    ClientStanding,
    PortfolioStanding,
    ReportingStanding,
    SourceRegistrationStanding,
    StaticReportingStanding,
    standing_field_spec,
)

REPO = Path(__file__).resolve().parents[2]

DEFAULT_CLIENT_CONFIG = REPO / "config/client/config_client_ERM_UK.yaml"
DEFAULT_ANNEX12_CONFIG = REPO / "config/client/config_client_annex12.yaml"
DEFAULT_PORTFOLIO_REGISTRY = REPO / "config/client/portfolio_registry.yaml"
DEFAULT_TENANCY = REPO / "config/tenancy.yaml"


@dataclass
class Adoption:
    """A migrated profile plus the evidence behind it."""

    profile: ClientOnboardingProfile
    #: field path -> where the value came from, in operator wording
    provenance: Dict[str, str] = field(default_factory=dict)
    #: What the legacy configuration could not answer.
    gaps: List[Dict[str, str]] = field(default_factory=list)
    #: Parsed legacy documents, so generation can preserve unmanaged blocks.
    base_documents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sources_read: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"profile": self.profile.to_dict(),
                "provenance": dict(self.provenance),
                "gaps": list(self.gaps),
                "sources_read": list(self.sources_read)}


def _load(path: Path) -> Dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


def _find(node: Any, key: str) -> Any:
    """First occurrence of ``key`` anywhere in the tree — the same lookup the
    Annex 2 preflight uses, so migration sees exactly what preflight sees."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            hit = _find(v, key)
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for v in node:
            hit = _find(v, key)
            if hit is not None:
                return hit
    return None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def adopt(client_id: str, *,
          registry: Optional[SourceRegistry] = None,
          client_config_path: Optional[Path] = None,
          annex12_config_path: Optional[Path] = None,
          portfolio_registry_path: Optional[Path] = None,
          tenancy_path: Optional[Path] = None,
          spec: Optional[Dict[str, Any]] = None) -> Adoption:
    """Build an onboarding profile for an existing client."""
    spec = spec or standing_field_spec()
    client_config_path = Path(client_config_path or os.environ.get(
        "TRAKT_OPS_CLIENT_CONFIG", DEFAULT_CLIENT_CONFIG))
    cfg = _load(client_config_path)
    annex12 = _load(Path(annex12_config_path or DEFAULT_ANNEX12_CONFIG))
    portfolios_doc = _load(Path(portfolio_registry_path
                                or DEFAULT_PORTFOLIO_REGISTRY))
    tenancy = _load(Path(tenancy_path or DEFAULT_TENANCY))

    adoption = Adoption(profile=ClientOnboardingProfile(client_id=client_id))
    p = adoption.profile
    prov = adoption.provenance
    if cfg:
        adoption.sources_read.append(str(client_config_path))
        adoption.base_documents[ARTEFACT_CLIENT_CONFIG] = cfg
    if annex12:
        adoption.sources_read.append(str(annex12_config_path
                                         or DEFAULT_ANNEX12_CONFIG))
        adoption.base_documents[ARTEFACT_ANNEX12_CONFIG] = annex12
    if portfolios_doc:
        adoption.sources_read.append(str(portfolio_registry_path
                                         or DEFAULT_PORTFOLIO_REGISTRY))
        adoption.base_documents[ARTEFACT_PORTFOLIO_REGISTRY] = portfolios_doc
    if tenancy:
        adoption.sources_read.append(str(tenancy_path or DEFAULT_TENANCY))

    _adopt_client(p, cfg, tenancy, client_id, prov)
    records = _records(registry, client_id)
    if records:
        adoption.sources_read.append("source registry")
    _adopt_portfolios(p, records, portfolios_doc, cfg, client_id, prov)
    _adopt_reporting(p, cfg, annex12, records, spec, prov)
    _adopt_regime(p, cfg, annex12, spec, prov)
    _adopt_static_reporting(p, cfg, prov)
    _adopt_sources(p, records, prov)
    adoption.gaps = _gaps(p, spec)
    return adoption


# --------------------------------------------------------------------------- #

def _adopt_client(p: ClientOnboardingProfile, cfg: Dict[str, Any],
                  tenancy: Dict[str, Any], client_id: str,
                  prov: Dict[str, str]) -> None:
    block = cfg.get("client") or {}
    portfolio = cfg.get("portfolio") or {}
    defaults = cfg.get("defaults") or {}
    tenant = ((tenancy.get("tenants") or {}).get(client_id)
              or (tenancy.get("tenants") or {}).get(client_id.upper()) or {})

    c = ClientStanding(client_id=client_id)
    c.client_name = _text(block.get("display_name")) or _text(
        tenant.get("display_name"))
    if c.client_name:
        prov["client.client_name"] = (
            "client.display_name in the client configuration"
            if block.get("display_name") else "the tenant registry")
    c.legal_entity_name = (_text(block.get("legal_entity_name"))
                           or _text(defaults.get("originator_name")))
    if c.legal_entity_name:
        prov["client.legal_entity_name"] = (
            "defaults.originator_name (RREL82) in the client configuration")
    c.lei = _text(defaults.get("originator_legal_entity_identifier"))
    if c.lei:
        prov["client.lei"] = ("defaults.originator_legal_entity_identifier "
                              "(RREL83) in the client configuration")
    c.jurisdiction = _text(portfolio.get("country")).upper()
    if c.jurisdiction:
        prov["client.jurisdiction"] = "portfolio.country"
    c.reporting_currency = _text(portfolio.get("base_currency")).upper()
    if c.reporting_currency:
        prov["client.reporting_currency"] = "portfolio.base_currency"
    c.environment = _text(block.get("environment")) or "production"
    c.time_zone = _text(block.get("time_zone")) or "Europe/London"
    if block.get("time_zone"):
        prov["client.time_zone"] = "client.time_zone"

    # Contacts: the master configuration has never carried them. The Annex 12
    # overlay does (IVSS5 / IVSS7), so an existing investor-reporting client
    # arrives with its reporting contact already filled in.
    reporting = block.get("reporting_contact") or {}
    operational = block.get("operational_contact") or {}
    c.primary_reporting_contact = _text(reporting.get("name"))
    c.reporting_email = _text(reporting.get("email"))
    c.operational_contact = _text(operational.get("name"))
    c.operational_email = _text(operational.get("email"))
    p.client = c


def _records(registry: Optional[SourceRegistry],
             client_id: str) -> List[SourceRecord]:
    if registry is None:
        return []
    try:
        return [r for r in registry.records() if r.client_id == client_id]
    except Exception:      # noqa: BLE001 — adoption must never fail on registry
        return []


def _adopt_portfolios(p: ClientOnboardingProfile, records: List[SourceRecord],
                      portfolios_doc: Dict[str, Any], cfg: Dict[str, Any],
                      client_id: str, prov: Dict[str, str]) -> None:
    """Portfolios come from the source registry (what actually delivers) and
    are described by the portfolio registry (what the platform assumes)."""
    meta: Dict[str, Dict[str, Any]] = {}
    clients = portfolios_doc.get("clients")
    entries = []
    if isinstance(clients, dict):
        for cid, blk in clients.items():
            if str(cid).lower() == client_id.lower():
                entries = list((blk or {}).get("portfolios") or [])
    if not entries:
        entries = list(portfolios_doc.get("portfolios") or [])
    for entry in entries:
        if isinstance(entry, dict) and entry.get("source_portfolio_id"):
            meta[str(entry["source_portfolio_id"])] = entry

    asset_class = _text((cfg.get("portfolio") or {}).get("asset_class"))
    if asset_class not in ASSET_MODEL:
        asset_class = next(iter(ASSET_MODEL), "")

    by_id: Dict[str, PortfolioStanding] = {}
    for record in records:
        pid = record.source_portfolio_id
        standing = by_id.get(pid)
        if standing is None:
            entry = meta.get(pid, {})
            standing = PortfolioStanding(
                portfolio_id=pid,
                display_name=_text(entry.get("source_portfolio_label")) or pid,
                portfolio_type=(record.source_portfolio_type
                                or _text(entry.get("source_portfolio_type"))
                                or "direct"),
                asset_class=asset_class,
                source_system=_text(record.source_system),
            )
            if entry.get("originates") is not None:
                standing.originates = bool(entry["originates"])
            by_id[pid] = standing
            prov[f"portfolios.{pid}"] = (
                "the source registry"
                + (" and the portfolio registry" if entry else ""))
        if record.dataset == "funded":
            standing.funded_cadence = record.frequency
        elif record.dataset == "pipeline":
            standing.pipeline_cadence = record.frequency
    p.portfolios = list(by_id.values())


def _adopt_reporting(p: ClientOnboardingProfile, cfg: Dict[str, Any],
                     annex12: Dict[str, Any], records: List[SourceRecord],
                     spec: Dict[str, Any], prov: Dict[str, str]) -> None:
    products: List[str] = []
    derivation: Dict[str, str] = {}
    pipeline = cfg.get("pipeline") or {}
    supported = [str(r) for r in (cfg.get("supported_regimes") or [])]
    if pipeline.get("esma_enabled") or "ESMA_Annex2" in supported \
            or any(r.regime_required for r in records):
        products.append("esma_annex2")
        derivation["esma_annex2"] = (
            "pipeline.esma_enabled / supported_regimes in the client "
            "configuration, and regime_required source registrations")
    if annex12.get("annex12"):
        products.append("investor_reporting")
        derivation["investor_reporting"] = (
            "an Annex 12 client configuration already exists")
    p.reporting = ReportingStanding(products=products, derivation=derivation)
    for key, why in derivation.items():
        prov[f"reporting.{key}"] = why


def _adopt_regime(p: ClientOnboardingProfile, cfg: Dict[str, Any],
                  annex12: Dict[str, Any], spec: Dict[str, Any],
                  prov: Dict[str, str]) -> None:
    """Read each governed standing field back out of the artefact it writes to.

    The spec's ``writes_to`` is the single mapping, used in both directions —
    so a field can never be adopted from one place and written to another.
    """
    products = spec.get("products") or {}
    for product_key, product in products.items():
        answers: Dict[str, Any] = {}
        for f in product.get("fields") or []:
            target = str(f.get("writes_to") or "")
            if ":" not in target:
                continue
            kind, dotted = target.split(":", 1)
            source = (cfg if kind == ARTEFACT_CLIENT_CONFIG
                      else annex12 if kind == ARTEFACT_ANNEX12_CONFIG
                      else {})
            node: Any = source
            for part in dotted.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                # Fall back to a tree-wide lookup for the flat legacy shapes
                # (`defaults:` keys are also reachable by name alone).
                node = _find(source, dotted.rsplit(".", 1)[-1])
            if node is None or _text(node) == "":
                continue
            answers[f["key"]] = (bool(node) if f.get("format") == "boolean"
                                 else _text(node))
            prov[f"regime.{product_key}.{f['key']}"] = (
                f"{dotted} in the "
                + ("client configuration" if kind == ARTEFACT_CLIENT_CONFIG
                   else "Annex 12 configuration"))
        if answers:
            p.regime[product_key] = answers


def _adopt_static_reporting(p: ClientOnboardingProfile, cfg: Dict[str, Any],
                            prov: Dict[str, str]) -> None:
    branding = ((cfg.get("mi") or {}).get("branding") or {})
    engine = cfg.get("loan_engine") or {}
    parties = cfg.get("reporting_parties") or {}
    sr = StaticReportingStanding(
        app_title=_text(branding.get("app_title")),
        primary_color=_text((branding.get("theme") or {}).get("primary_color")),
        logo_uri=_text(branding.get("logo_uri")),
        disclaimer=_text(branding.get("disclaimer")),
        investor_contact=_text(parties.get("investor_contact")),
        investor_email=_text(parties.get("investor_email")),
        trustee_name=_text(parties.get("trustee_name")),
        warehouse_lender=_text(parties.get("warehouse_lender")),
        reporting_convention=_text(parties.get("reporting_convention")),
        payment_convention=_text(engine.get("interest_payment_frequency")),
        day_count_convention=_text(engine.get("day_count_convention")),
    )
    if sr.app_title or sr.primary_color:
        prov["static_reporting.branding"] = "mi.branding in the client configuration"
    if sr.day_count_convention or sr.payment_convention:
        prov["static_reporting.conventions"] = "loan_engine in the client configuration"
    p.static_reporting = sr


def _adopt_sources(p: ClientOnboardingProfile, records: List[SourceRecord],
                   prov: Dict[str, str]) -> None:
    p.sources = [
        SourceRegistrationStanding(
            portfolio_id=r.source_portfolio_id, dataset=r.dataset,
            frequency=r.frequency, source_system=_text(r.source_system),
            expected_files=list(r.expected_columns or []),
            regime_required=bool(r.regime_required), status=r.status)
        for r in sorted(records, key=lambda r: (r.source_portfolio_id,
                                                r.dataset, r.frequency))
    ]
    if p.sources:
        prov["sources"] = "the durable source registry"


def _gaps(p: ClientOnboardingProfile,
          spec: Dict[str, Any]) -> List[Dict[str, str]]:
    """What the legacy configuration could not answer. Shown on the first
    screen so an adopted client's missing standing information is visible
    rather than discovered at delivery time."""
    out: List[Dict[str, str]] = []

    def gap(step: str, field_name: str, label: str, why: str) -> None:
        out.append({"step": step, "field": field_name, "label": label,
                    "why": why})

    c = p.client
    if not c.primary_reporting_contact or not c.reporting_email:
        gap("client", "reporting_contact", "Primary reporting contact",
            "The client configuration has never carried reporting contacts.")
    if not c.operational_contact or not c.operational_email:
        gap("client", "operational_contact", "Operational contact",
            "The client configuration has never carried operational contacts.")
    if not c.time_zone:
        gap("client", "time_zone", "Reporting time zone",
            "No time zone is recorded anywhere in the legacy configuration.")
    for portfolio in p.portfolios:
        if not portfolio.structure:
            gap("portfolio", f"{portfolio.portfolio_id}.structure",
                f"{portfolio.display_name}: how the book is held",
                "Warehouse / SPV / managed is not recorded in the legacy model.")
    products = spec.get("products") or {}
    for product_key in p.reporting.products:
        answers = p.regime.get(product_key) or {}
        for f in (products.get(product_key) or {}).get("fields") or []:
            if f.get("required") and not _text(answers.get(f["key"])):
                gap("regime", f"{product_key}.{f['key']}", f.get("label", ""),
                    "Required by "
                    f"{(products.get(product_key) or {}).get('label', product_key)}"
                    " and not present in the legacy configuration.")
    return out
