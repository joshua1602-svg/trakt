"""operations_control.onboarding.migration — bringing an existing client in.

**Secondary to the product.** New-client onboarding does not import this module,
does not read these files, and works with none of them present. This exists so a
client Trakt already serves does not have to be re-keyed by hand.

It reads what is already there and produces answers in the SAME generic model a
blank onboarding produces, so a migrated client and a newly onboarded one are
indistinguishable afterwards. It writes nothing, and it changes no active
configuration: the case it produces goes through the same review, approval and
activation as any other.

Values that fail today's validation are returned as ``issues`` rather than
quietly carried across, so a legacy problem surfaces during migration instead of
at delivery time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from apps.blob_trigger_app.source_registry import SourceRecord, SourceRegistry

from ..configuration.packages import ASSET_MODEL
from .artefacts import (
    ARTEFACT_ANNEX12_CONFIG,
    ARTEFACT_CLIENT_CONFIG,
    ARTEFACT_PORTFOLIO_REGISTRY,
)
from .case import new_entity_id, source_key
from .catalogue import Catalogue, catalogue, validate_value

REPO = Path(__file__).resolve().parents[2]

#: Where a legacy client's documents are, when the caller does not say. There
#: is no repository default any more: a client whose documents are not supplied
#: adopts nothing, which is the correct outcome — the alternative was every
#: adoption silently reading one particular client's file. Deployments that
#: still hold legacy documents point at them explicitly, per adoption, or
#: through ``TRAKT_OPS_CLIENT_CONFIG``.
DEFAULT_CLIENT_CONFIG: Optional[Path] = None
DEFAULT_ANNEX12_CONFIG: Optional[Path] = None
DEFAULT_PORTFOLIO_REGISTRY = REPO / "config/client/portfolio_registry.yaml"
DEFAULT_TENANCY = REPO / "config/tenancy.yaml"


@dataclass
class Adoption:
    """Answers derived from a legacy client, plus the evidence behind them."""

    answers: Dict[str, Any] = field(default_factory=dict)
    #: answer path -> where the value came from, in operator wording
    provenance: Dict[str, str] = field(default_factory=dict)
    #: Legacy values that today's rules refuse.
    issues: List[Dict[str, str]] = field(default_factory=list)
    #: Parsed legacy documents, so blocks onboarding does not own survive.
    base_documents: Dict[str, Any] = field(default_factory=dict)
    sources_read: List[str] = field(default_factory=list)


def _load(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _find(node: Any, key: str) -> Any:
    """First occurrence anywhere — the same lookup the Annex 2 preflight uses,
    so migration sees exactly what preflight sees."""
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


def adopt(client_id: str, *,
          catalogue: Optional[Catalogue] = None,
          registry: Optional[SourceRegistry] = None,
          client_config_path: Optional[Path] = None,
          annex12_config_path: Optional[Path] = None,
          portfolio_registry_path: Optional[Path] = None,
          tenancy_path: Optional[Path] = None) -> Adoption:
    from .catalogue import catalogue as _default_catalogue
    cat = catalogue or _default_catalogue()

    configured = client_config_path or os.environ.get(
        "TRAKT_OPS_CLIENT_CONFIG") or DEFAULT_CLIENT_CONFIG
    client_config_path = Path(configured) if configured else None
    annex12_config_path = annex12_config_path or DEFAULT_ANNEX12_CONFIG
    cfg = _load(client_config_path)
    annex12 = _load(Path(annex12_config_path) if annex12_config_path else None)
    portfolios_doc = _load(Path(portfolio_registry_path
                                or DEFAULT_PORTFOLIO_REGISTRY))
    tenancy = _load(Path(tenancy_path or DEFAULT_TENANCY))

    adoption = Adoption()
    if cfg:
        adoption.sources_read.append(str(client_config_path))
        adoption.base_documents[ARTEFACT_CLIENT_CONFIG] = cfg
    if annex12:
        adoption.sources_read.append(str(annex12_config_path))
        adoption.base_documents[ARTEFACT_ANNEX12_CONFIG] = annex12
    if portfolios_doc:
        adoption.sources_read.append(str(portfolio_registry_path
                                         or DEFAULT_PORTFOLIO_REGISTRY))
        adoption.base_documents[ARTEFACT_PORTFOLIO_REGISTRY] = portfolios_doc
    if tenancy:
        adoption.sources_read.append(str(tenancy_path or DEFAULT_TENANCY))

    records = _records(registry, client_id)
    if records:
        adoption.sources_read.append("source registrations")

    _client(adoption, cfg, tenancy, client_id)
    _entities(adoption, cfg, annex12)
    _contacts(adoption, cfg, annex12)
    _portfolios(adoption, cfg, portfolios_doc, records, client_id)
    _sources(adoption, records, client_id)
    _reporting(adoption, cfg, annex12, records)
    _regime(adoption, cfg, annex12, cat)
    _presentation(adoption, cfg)
    _check(adoption, cat)
    return adoption


# --------------------------------------------------------------------------- #

def _client(a: Adoption, cfg: Dict[str, Any], tenancy: Dict[str, Any],
            client_id: str) -> None:
    block = cfg.get("client") or {}
    portfolio = cfg.get("portfolio") or {}
    tenant = ((tenancy.get("tenants") or {}).get(client_id)
              or (tenancy.get("tenants") or {}).get(client_id.upper()) or {})
    client: Dict[str, Any] = {"client_id": client_id}

    name = _text(block.get("display_name")) or _text(tenant.get("display_name"))
    if name:
        client["client_name"] = name
        a.provenance["client.client_name"] = (
            "the client configuration" if block.get("display_name")
            else "the tenant registry")
    for key, source, label in (
            ("jurisdiction", portfolio.get("country"), "portfolio country"),
            ("reporting_currency", portfolio.get("base_currency"),
             "portfolio base currency")):
        if _text(source):
            client[key] = _text(source).upper()
            a.provenance[f"client.{key}"] = f"the {label}"
    if _text(block.get("environment")):
        client["environment"] = _text(block.get("environment"))
    if _text(block.get("time_zone")):
        client["time_zone"] = _text(block.get("time_zone"))
        a.provenance["client.time_zone"] = "the client configuration"
    a.answers["client"] = client


def _entities(a: Adoption, cfg: Dict[str, Any],
              annex12: Dict[str, Any]) -> None:
    """The legacy model has no entities — it has an originator's three fields
    and, where investor reporting exists, a reporting entity name. Those become
    entities here, which is the whole point of the corrected model."""
    defaults = cfg.get("defaults") or {}
    entities: List[Dict[str, Any]] = []

    originator_name = _text(defaults.get("originator_name"))
    originator_lei = _text(defaults.get("originator_legal_entity_identifier"))
    if originator_name or originator_lei:
        entity = {"entity_id": new_entity_id(), "legal_name": originator_name,
                  "roles": ["originator"]}
        if originator_lei:
            entity["lei"] = originator_lei.upper()
        country = _text(defaults.get("originator_establishment_country"))
        if country:
            entity["country_of_establishment"] = country.upper()
        entities.append(entity)
        a.provenance["entities.originator"] = (
            "the originator defaults in the client configuration "
            "(RREL82/83/84)")

    deal = (annex12.get("annex12") or {}).get("deal") or {}
    reporting_name = _text(deal.get("IVSS4")) or _text(deal.get("IVSS3"))
    if reporting_name:
        match = next((e for e in entities
                      if e["legal_name"].lower() == reporting_name.lower()),
                     None)
        if match is not None:
            # The same entity in both roles — captured once, exactly as the
            # corrected model requires.
            if "reporting_entity" not in match["roles"]:
                match["roles"].append("reporting_entity")
        else:
            entities.append({"entity_id": new_entity_id(),
                             "legal_name": reporting_name,
                             "roles": ["reporting_entity"]})
        a.provenance["entities.reporting_entity"] = (
            "the investor report configuration (IVSS3/IVSS4)")

    original_lender_lei = _text(defaults.get(
        "original_lender_legal_entity_identifier"))
    if original_lender_lei:
        entities.append({"entity_id": new_entity_id(), "legal_name": "",
                         "roles": ["original_lender"],
                         "lei": original_lender_lei.upper()})
    a.answers["entities"] = entities


def _contacts(a: Adoption, cfg: Dict[str, Any],
              annex12: Dict[str, Any]) -> None:
    block = cfg.get("client") or {}
    reporting = block.get("reporting_contact") or {}
    operational = block.get("operational_contact") or {}
    deal = (annex12.get("annex12") or {}).get("deal") or {}
    contacts: Dict[str, Any] = {}
    for key, value, origin in (
            ("reporting_contact_name", reporting.get("name"),
             "the client configuration"),
            ("reporting_contact_email", reporting.get("email"),
             "the client configuration"),
            ("operational_contact_name", operational.get("name"),
             "the client configuration"),
            ("operational_contact_email", operational.get("email"),
             "the client configuration"),
            ("reporting_contact_phone", deal.get("IVSS6"),
             "the investor report configuration (IVSS6)")):
        if _text(value):
            contacts[key] = _text(value)
            a.provenance[f"contacts.{key}"] = origin
    # Annex 12 carries a contact name and email the master config never had.
    if not contacts.get("reporting_contact_name") and _text(deal.get("IVSS5")):
        contacts["reporting_contact_name"] = _text(deal["IVSS5"])
        a.provenance["contacts.reporting_contact_name"] = \
            "the investor report configuration (IVSS5)"
    if not contacts.get("reporting_contact_email") and _text(deal.get("IVSS7")):
        contacts["reporting_contact_email"] = _text(deal["IVSS7"])
        a.provenance["contacts.reporting_contact_email"] = \
            "the investor report configuration (IVSS7)"
    a.answers["contacts"] = contacts


def _records(registry: Optional[SourceRegistry],
             client_id: str) -> List[SourceRecord]:
    if registry is None:
        return []
    try:
        return [r for r in registry.records() if r.client_id == client_id]
    except Exception:      # noqa: BLE001 — migration must not fail on registry
        return []


def _portfolios(a: Adoption, cfg: Dict[str, Any], doc: Dict[str, Any],
                records: List[SourceRecord], client_id: str) -> None:
    meta: Dict[str, Dict[str, Any]] = {}
    entries = []
    clients = doc.get("clients")
    if isinstance(clients, dict):
        for cid, blk in clients.items():
            if str(cid).lower() == client_id.lower():
                entries = list((blk or {}).get("portfolios") or [])
    if not entries:
        entries = list(doc.get("portfolios") or [])
    for entry in entries:
        if isinstance(entry, dict) and entry.get("source_portfolio_id"):
            meta[str(entry["source_portfolio_id"])] = entry

    asset_class = _text((cfg.get("portfolio") or {}).get("asset_class"))
    if asset_class not in ASSET_MODEL:
        asset_class = ""
    owning = next((e["entity_id"] for e in a.answers.get("entities") or []
                   if "originator" in (e.get("roles") or [])), "")

    out: Dict[str, Dict[str, Any]] = {}
    for record in records:
        pid = record.source_portfolio_id
        if pid in out:
            continue
        entry = meta.get(pid, {})
        portfolio = {
            "portfolio_id": pid,
            "display_name": _text(entry.get("source_portfolio_label")) or pid,
            "portfolio_type": (record.source_portfolio_type
                               or _text(entry.get("source_portfolio_type"))
                               or "direct"),
            "asset_class": asset_class,
            "owning_entity": owning,
        }
        if entry.get("originates") is not None:
            portfolio["originates"] = bool(entry["originates"])
        out[pid] = portfolio
        a.provenance[f"portfolios.{pid}"] = (
            "the source registrations"
            + (" and the portfolio metadata" if entry else ""))
    a.answers["portfolios"] = list(out.values())


def _sources(a: Adoption, records: List[SourceRecord], client_id: str) -> None:
    a.answers["sources"] = [
        {"source_key": source_key(r.source_portfolio_id, r.dataset),
         "portfolio_id": r.source_portfolio_id, "dataset": r.dataset,
         "cadence": r.frequency, "source_party": _text(r.source_system),
         "expected_files": list(r.expected_columns or []),
         "regime_required": bool(r.regime_required),
         "mapping_complete": bool(r.approved_mapping_id)}
        for r in sorted(records, key=lambda r: (r.source_portfolio_id,
                                                r.dataset))]
    if a.answers["sources"]:
        a.provenance["sources"] = "the durable source registrations"


def _reporting(a: Adoption, cfg: Dict[str, Any], annex12: Dict[str, Any],
               records: List[SourceRecord]) -> None:
    products: List[str] = []
    pipeline = cfg.get("pipeline") or {}
    supported = [str(r) for r in (cfg.get("supported_regimes") or [])]
    if pipeline.get("esma_enabled") or "ESMA_Annex2" in supported \
            or any(r.regime_required for r in records):
        products.append("esma_annex2")
        a.provenance["reporting.esma_annex2"] = (
            "the client configuration and the regulatory source registrations")
    if annex12.get("annex12"):
        products.append("investor_reporting")
        a.provenance["reporting.investor_reporting"] = (
            "an investor report configuration already exists")
    a.answers["reporting"] = {"products": products}


def _regime(a: Adoption, cfg: Dict[str, Any], annex12: Dict[str, Any],
            cat: Catalogue) -> None:
    """Read each standing field back out of the artefact it writes to, using the
    catalogue's own routing so a value can never be adopted from one place and
    written to another."""
    held: Dict[str, Dict[str, Any]] = {}
    section = cat.section("regime")
    for f in (section.fields if section else []):
        target = str(f.writes_to)
        if ":" not in target:
            continue
        kind, dotted = target.split(":", 1)
        source = (cfg if kind == ARTEFACT_CLIENT_CONFIG
                  else annex12 if kind == ARTEFACT_ANNEX12_CONFIG else {})
        node: Any = source
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            node = _find(source, dotted.rsplit(".", 1)[-1])
        if node is None or _text(node) == "":
            continue
        held.setdefault(f.product, {})[f.key] = (
            bool(node) if f.type == "boolean" else _text(node))
        a.provenance[f"regime.{f.product}.{f.key}"] = (
            f"{dotted} in the "
            + ("client configuration" if kind == ARTEFACT_CLIENT_CONFIG
               else "investor report configuration"))
    a.answers["regime"] = held


def _presentation(a: Adoption, cfg: Dict[str, Any]) -> None:
    branding = ((cfg.get("mi") or {}).get("branding") or {})
    engine = cfg.get("loan_engine") or {}
    presentation: Dict[str, Any] = {}
    for key, value, origin in (
            ("report_title", branding.get("app_title"), "the branding block"),
            ("brand_colour", (branding.get("theme") or {}).get("primary_color"),
             "the branding block"),
            ("logo_uri", branding.get("logo_uri"), "the branding block"),
            ("disclaimer", branding.get("disclaimer"), "the branding block"),
            ("day_count_convention", engine.get("day_count_convention"),
             "the calculation settings"),
            ("payment_frequency", engine.get("interest_payment_frequency"),
             "the calculation settings")):
        if _text(value):
            presentation[key] = _text(value)
            a.provenance[f"presentation.{key}"] = origin
    a.answers["presentation"] = presentation


def _check(a: Adoption, cat: Catalogue) -> None:
    """Legacy values today's rules refuse. Surfaced as migration issues so a
    long-standing problem is fixed during migration rather than at delivery."""
    for section in cat.sections:
        if section.repeatable:
            for index, item in enumerate(a.answers.get(section.key) or []):
                for f in section.fields:
                    problem = validate_value(f, item.get(f.key))
                    if problem:
                        a.issues.append({
                            "section": section.key, "field": f.key,
                            "index": str(index),
                            "message": f"{problem} (carried over from the "
                                       "existing configuration)"})
            continue
        if section.from_regime:
            held = a.answers.get(section.key) or {}
            for f in section.fields:
                problem = validate_value(f, (held.get(f.product) or {})
                                         .get(f.key))
                if problem:
                    a.issues.append({
                        "section": section.key,
                        "field": f"{f.product}.{f.key}",
                        "message": f"{problem} (carried over from the existing "
                                   "configuration)"})
            continue
        block = a.answers.get(section.key) or {}
        for f in section.fields:
            problem = validate_value(f, block.get(f.key))
            if problem:
                a.issues.append({
                    "section": section.key, "field": f.key,
                    "message": f"{problem} (carried over from the existing "
                               "configuration)"})
