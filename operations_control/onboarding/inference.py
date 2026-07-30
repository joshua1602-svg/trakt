"""operations_control.onboarding.inference — what Trakt answers for itself.

Every rule here removes a question. Three mechanisms, deliberately distinct
because they differ in how much the operator should trust them:

``derive``    A fixed relationship to another answer. The originator entity's
              name IS the Annex 2 originator name. Never asked, never
              separately overridable — two copies of one fact is the
              duplication onboarding exists to remove.

``infer``     A lookup or a signal: jurisdiction to currency, asset class to
              exposure type, a sample file's extension to its format. Filled
              only when blank, shown with its origin, always overridable.

``propose``   Trakt suggests a value the operator owns — identifiers. Offered
              with a collision check; the operator confirms or edits.

Nothing here is silent. Every value this module supplies is recorded in the
case's provenance, so the wizard can say where it came from and the review
screen can list it.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .case import OnboardingCase
from .catalogue import Catalogue, catalogue

REPO = Path(__file__).resolve().parents[2]
RULES_PATH = REPO / "config/onboarding/inference_rules.yaml"

_RULES: Optional[Dict[str, Any]] = None


def rules(path: Path = RULES_PATH) -> Dict[str, Any]:
    global _RULES
    if _RULES is None:
        try:
            _RULES = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            _RULES = {}
    return _RULES


def reset_cache() -> None:
    global _RULES
    _RULES = None


def _blank(value: Any) -> bool:
    if value is None:
        return False if isinstance(value, bool) else True
    if isinstance(value, bool):
        return False
    if isinstance(value, (list, tuple, dict)):
        return not value
    return not str(value).strip()


#: Fields the operator edited on the save being processed, per section. Cleared
#: and repopulated by :func:`apply`.
_TOUCHED: Dict[str, set] = {}


def _touched(path: str) -> bool:
    section = path.split(".", 1)[0].split("[", 1)[0]
    key = path.rsplit(".", 1)[-1]
    return key in _TOUCHED.get(section, set())


def _set(block: Dict[str, Any], key: str, value: Any,
         provenance: Dict[str, str], path: str, origin: str,
         *, force: bool = False) -> None:
    """Fill a value and record where it came from."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return
    if _touched(path):
        return
    if not force and not _blank(block.get(key)):
        return
    if block.get(key) == value:
        provenance.setdefault(path, origin)
        return
    block[key] = value
    provenance[path] = origin


# --------------------------------------------------------------------------- #
# Identifier proposals
# --------------------------------------------------------------------------- #

def upper_slug(text: str, max_length: int = 12) -> str:
    """A stable, legible identifier from a name.

    ``Nordic Lending AB`` -> ``NORDIC``. Whole words only: an identifier is
    permanent and appears in every storage location this client will ever have,
    so truncating one mid-word (``NORDICLENDIN``) is worse than using fewer
    words. Company-form suffixes are dropped because they carry no meaning in
    an identifier.
    """
    stop = {"the", "and", "of", "ltd", "limited", "plc", "ab", "as", "asa",
            "gmbh", "sa", "nv", "bv", "inc", "llc", "lp", "llp", "group",
            "holdings", "holding", "company", "co"}
    words = [w for w in re.split(r"[^A-Za-z0-9]+", text or "") if w]
    meaningful = [w for w in words if w.lower() not in stop] or words
    slug = ""
    for word in meaningful:
        if slug and len(slug) + len(word) > max_length:
            break
        slug += word
    # A single word longer than the limit is the only case where cutting is
    # unavoidable.
    slug = (slug or "".join(meaningful))[:max_length].upper()
    return slug if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,63}", slug) else ""


def propose_client_id(case: OnboardingCase, taken: Optional[set] = None) -> str:
    """A client identifier the operator can accept or edit.

    Collision-checked against identifiers already in use, because an identifier
    is permanent and appears in every storage location this client will ever
    have.
    """
    name = str((case.answers.get("client") or {}).get("client_name") or "")
    spec = (rules().get("proposals") or {}).get("client_id") or {}
    base = upper_slug(name, int(spec.get("max_length") or 12))
    if not base:
        return ""
    taken = {str(t).lower() for t in (taken or set())}
    if base.lower() not in taken:
        return base
    for n in range(2, 100):
        candidate = f"{base}{n}"
        if candidate.lower() not in taken:
            return candidate
    return ""


def propose_portfolio_id(portfolio_type: str, sequence: int) -> str:
    """``direct_001`` — the convention already in the source registry and the
    storage layout, so an operator never has to know it."""
    kind = str(portfolio_type or "direct").strip() or "direct"
    return f"{kind}_{sequence:03d}"


# --------------------------------------------------------------------------- #
# Sample packs
# --------------------------------------------------------------------------- #

def infer_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """What a sample pack tells us: file format, expected file names, and the
    asset class its columns imply.

    ``sample`` is ``{"files": [{"name", "headers": [...]}, ...]}`` — the same
    shape the existing intake produces when it reads a pack.
    """
    spec = (rules().get("from_sample") or {})
    out: Dict[str, Any] = {}
    files = [f for f in (sample.get("files") or []) if f.get("name")]
    if not files:
        return out

    by_ext = spec.get("file_format", {}).get("by_extension") or {}
    formats = {by_ext.get(str(f["name"]).rsplit(".", 1)[-1].lower())
               for f in files}
    formats.discard(None)
    if len(formats) == 1:
        out["file_format"] = formats.pop()

    out["expected_files"] = [str(f["name"]) for f in files]

    headers = [str(h).strip().lower().replace(" ", "_")
               for f in files for h in (f.get("headers") or [])]
    asset_class = _asset_class_from_headers(headers, spec)
    if asset_class:
        out["asset_class"] = asset_class
    return out


def _asset_class_from_headers(headers: List[str],
                              spec: Dict[str, Any]) -> str:
    """Score the tape's columns against the signal sets the existing bootstrap
    agent uses, so onboarding and that agent cannot disagree about what a tape
    looks like."""
    ref = str(spec.get("asset_class_signals_from") or "")
    if not ref:
        return ""
    module_path, _, attr = ref.rpartition(".")
    try:
        signals = getattr(importlib.import_module(module_path), attr)
    except (ImportError, AttributeError):
        return ""
    minimum = int(spec.get("asset_class_min_hits") or 2)
    scores: List[Tuple[int, str]] = []
    for asset_class, words in (signals or {}).items():
        hits = sum(1 for w in words
                   if any(str(w).replace(" ", "_") in h for h in headers))
        if hits >= minimum:
            scores.append((hits, asset_class))
    if not scores:
        return ""
    scores.sort(reverse=True)
    # An ambiguous tape is not an answer. Two asset classes scoring equally is
    # exactly when a human should look.
    if len(scores) > 1 and scores[0][0] == scores[1][0]:
        return ""
    return scores[0][1]


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #

def apply(case: OnboardingCase, cat: Optional[Catalogue] = None, *,
          existing_clients: Optional[set] = None,
          touched: Optional[Dict[str, set]] = None) -> Dict[str, str]:
    """Fill everything Trakt can answer for itself. Returns new provenance.

    Order matters: jurisdiction feeds currency, currency feeds portfolios,
    entities feed the regime blocks. Each pass reads what the previous one
    established.

    ``touched`` names the fields the operator has just edited, per section. Those
    are left alone for this pass: someone clearing a currency to retype it must
    not have the inferred value put back underneath their cursor.
    """
    cat = cat or catalogue()
    r = rules()
    provenance: Dict[str, str] = {}
    _TOUCHED.clear()
    _TOUCHED.update({k: set(v) for k, v in (touched or {}).items()})

    _client(case, r, provenance, existing_clients or set())
    _portfolios(case, r, provenance)
    _sources(case, r, provenance)
    _presentation(case, r, provenance)
    _regime(case, r, provenance)
    _defaults(case, cat, provenance)
    case.provenance.update(provenance)
    return provenance


def _defaults(case: OnboardingCase, cat: Catalogue,
              prov: Dict[str, str]) -> None:
    """Apply every declared default, wherever it lives.

    Done generally rather than per field so a default added to the catalogue —
    or to a future regime's declaration — takes effect without a change here.

    Two rules make this safe. It runs **last**, so anything better informed —
    a jurisdiction lookup, a sample pack, a fact already given — wins over a
    bare default; a default is what is used when nothing is known, not a value
    that displaces what is. And it only fills a section the operator has
    reached: an item exists because someone created it, a regime block exists
    because someone bought the product, and a plain section counts as reached
    once it holds an answer. A case started and abandoned holds nothing.
    """
    for section in cat.sections:
        if section.repeatable:
            for index, item in enumerate(case.items(section.key)):
                for f in section.fields:
                    if f.default is None:
                        continue
                    _set(item, f.key, f.default, prov,
                         f"{section.key}[{index}].{f.key}",
                         "the standard value where nothing else is given")
            continue
        block = case.block(section.key)
        for f in section.fields:
            if f.default is None:
                continue
            if section.from_regime:
                if f.product and f.product not in case.products:
                    continue
                target = block.setdefault(f.product, {})
                path = f"{section.key}.{f.product}.{f.key}"
            elif block:
                target, path = block, f"{section.key}.{f.key}"
            else:
                continue
            _set(target, f.key, f.default, prov, path,
                 "the standard value where nothing else is given")


# -- client ----------------------------------------------------------------- #

def _client(case: OnboardingCase, r: Dict[str, Any],
            prov: Dict[str, str], taken: set) -> None:
    block = case.block("client")
    jurisdiction = str(block.get("jurisdiction") or "").strip().upper()
    if jurisdiction:
        block["jurisdiction"] = jurisdiction
        table = (r.get("jurisdiction") or {}).get(jurisdiction) or {}
        fallback = r.get("jurisdiction_fallback") or {}
        _set(block, "reporting_currency", table.get("reporting_currency"),
             prov, "client.reporting_currency",
             f"the currency used in {jurisdiction}")
        _set(block, "time_zone",
             table.get("time_zone") or fallback.get("time_zone"),
             prov, "client.time_zone",
             f"the time zone for {jurisdiction}"
             if table.get("time_zone") else
             "the default where no rule matched the jurisdiction")

    # The identifier is the operator's, but Trakt offers one.
    if _blank(block.get("client_id")):
        proposed = propose_client_id(case, taken)
        _set(block, "client_id", proposed, prov, "client.client_id",
             "the client's name, checked against identifiers already in use")


# -- portfolios ------------------------------------------------------------- #

def _portfolios(case: OnboardingCase, r: Dict[str, Any],
                prov: Dict[str, str]) -> None:
    client = case.block("client")
    entities = case.items("entities")
    sample = case.answers.get("sample") or {}
    inferred_from_sample = infer_from_sample(sample) if sample else {}
    counts: Dict[str, int] = {}

    for index, p in enumerate(case.items("portfolios")):
        kind = str(p.get("portfolio_type") or "direct")
        counts[kind] = counts.get(kind, 0) + 1
        path = f"portfolios[{index}]"

        _set(p, "portfolio_id", propose_portfolio_id(kind, counts[kind]),
             prov, f"{path}.portfolio_id",
             "Trakt's naming convention for books of this type")

        if inferred_from_sample.get("asset_class"):
            _set(p, "asset_class", inferred_from_sample["asset_class"],
                 prov, f"{path}.asset_class",
                 "the columns in the sample the client supplied")

        # One entity means no question: it owns everything.
        if len(entities) == 1:
            _set(p, "owning_entity", entities[0].get("entity_id"),
                 prov, f"{path}.owning_entity",
                 "the only entity on this client")

        _set(p, "reporting_currency", client.get("reporting_currency"),
             prov, f"{path}.reporting_currency",
             "the client's reporting currency")

        if p.get("originates") is None:
            p["originates"] = kind == "direct"
            prov[f"{path}.originates"] = (
                "books this client originated are still originating unless "
                "told otherwise")

        cadence = _cadence_for(case, str(p.get("portfolio_id") or ""))
        convention = (r.get("cadence") or {}).get(cadence)
        _set(p, "period_convention", convention, prov,
             f"{path}.period_convention",
             f"a {cadence} book reporting to period end")


def _cadence_for(case: OnboardingCase, portfolio_id: str) -> str:
    for s in case.items("sources"):
        if s.get("portfolio_id") == portfolio_id and s.get("dataset") == "funded":
            return str(s.get("cadence") or "monthly")
    return "monthly"


# -- sources ---------------------------------------------------------------- #

def _sources(case: OnboardingCase, r: Dict[str, Any],
             prov: Dict[str, str]) -> None:
    sample = case.answers.get("sample") or {}
    inferred = infer_from_sample(sample) if sample else {}
    for index, s in enumerate(case.items("sources")):
        path = f"sources[{index}]"
        _set(s, "cadence", "monthly", prov, f"{path}.cadence",
             "the cadence most books report at")
        cadence = str(s.get("cadence") or "monthly")
        _set(s, "cut_off_convention", (r.get("cadence") or {}).get(cadence),
             prov, f"{path}.cut_off_convention",
             "the reporting cadence")
        if inferred.get("file_format"):
            _set(s, "file_format", inferred["file_format"], prov,
                 f"{path}.file_format",
                 "the sample the client supplied")
        if inferred.get("expected_files") and s.get("dataset") == "funded":
            _set(s, "expected_files", inferred["expected_files"], prov,
                 f"{path}.expected_files",
                 "the files in the sample the client supplied")


# -- presentation ----------------------------------------------------------- #

def _presentation(case: OnboardingCase, r: Dict[str, Any],
                  prov: Dict[str, str]) -> None:
    block = case.block("presentation")
    client = case.block("client")
    name = str(client.get("client_name") or "").strip()
    if name:
        _set(block, "report_title", f"{name} Portfolio MI", prov,
             "presentation.report_title", "the client's name")

    portfolios = case.items("portfolios")
    asset_class = str(portfolios[0].get("asset_class") or "") if portfolios else ""
    table = (r.get("asset_class") or {}).get(asset_class) or {}
    _set(block, "day_count_convention", table.get("day_count_convention"),
         prov, "presentation.day_count_convention",
         f"the standard convention for {asset_class.replace('_', ' ')}"
         if asset_class else "")
    _set(block, "payment_frequency", table.get("payment_frequency"), prov,
         "presentation.payment_frequency",
         f"the standard convention for {asset_class.replace('_', ' ')}"
         if asset_class else "")


# -- regime ------------------------------------------------------------------ #

def _regime(case: OnboardingCase, r: Dict[str, Any],
            prov: Dict[str, str]) -> None:
    """Everything the regime blocks can take from answers already given.

    These are DERIVED, not inferred: they are forced to match their source on
    every save, because an Annex 2 originator name that has drifted from the
    entity it names is a defect, not a preference.
    """
    products = set(case.products)
    held = case.block("regime")
    client = case.block("client")
    contacts = case.block("contacts")

    if "esma_annex2" in products:
        annex2 = held.setdefault("esma_annex2", {})
        originator = next(iter(case.entities_with_role("originator")), None)
        if originator:
            _set(annex2, "originator_name", originator.get("legal_name"),
                 prov, "regime.esma_annex2.originator_name",
                 "the originator entity", force=True)
            _set(annex2, "originator_legal_entity_identifier",
                 originator.get("lei"), prov,
                 "regime.esma_annex2.originator_legal_entity_identifier",
                 "the originator entity", force=True)
            _set(annex2, "originator_establishment_country",
                 originator.get("country_of_establishment"), prov,
                 "regime.esma_annex2.originator_establishment_country",
                 "the originator entity", force=True)
        lender = next(iter(case.entities_with_role("original_lender")), None)
        if lender:
            _set(annex2, "original_lender_legal_entity_identifier",
                 lender.get("lei"), prov,
                 "regime.esma_annex2.original_lender_legal_entity_identifier",
                 "the original lender entity", force=True)
            _set(annex2, "original_lender_establishment_country",
                 lender.get("country_of_establishment"), prov,
                 "regime.esma_annex2.original_lender_establishment_country",
                 "the original lender entity", force=True)
        jurisdiction = str(client.get("jurisdiction") or "").upper()
        table = (r.get("jurisdiction") or {}).get(jurisdiction) or {}
        if table.get("uk_geography") and annex2.get("uk_geography_override") is None:
            annex2["uk_geography_override"] = True
            prov["regime.esma_annex2.uk_geography_override"] = (
                "UK books report region as GBZZZ by convention")

    if "investor_reporting" in products:
        annex12 = held.setdefault("investor_reporting", {})
        reporting = next(iter(case.entities_with_role("reporting_entity")), None)
        if reporting:
            _set(annex12, "IVSS3", reporting.get("legal_name"), prov,
                 "regime.investor_reporting.IVSS3", "the reporting entity",
                 force=True)
            _set(annex12, "IVSS4", reporting.get("legal_name"), prov,
                 "regime.investor_reporting.IVSS4", "the reporting entity",
                 force=True)
            _set(annex12, "IVSS1", reporting.get("lei"), prov,
                 "regime.investor_reporting.IVSS1", "the reporting entity",
                 force=True)
        _set(annex12, "IVSS5", contacts.get("reporting_contact_name"), prov,
             "regime.investor_reporting.IVSS5", "the reporting contact",
             force=True)
        _set(annex12, "IVSS6", contacts.get("reporting_contact_phone"), prov,
             "regime.investor_reporting.IVSS6", "the reporting contact",
             force=True)
        _set(annex12, "IVSS7", contacts.get("reporting_contact_email"), prov,
             "regime.investor_reporting.IVSS7", "the reporting contact",
             force=True)

        holder = next(iter(case.entities_with_role("risk_retention_holder")),
                      None)
        if holder:
            code = _retention_code(holder, r)
            _set(annex12, "IVSS9", code, prov,
                 "regime.investor_reporting.IVSS9",
                 "the role of the entity holding the retention", force=True)

        portfolios = case.items("portfolios")
        asset_class = str(portfolios[0].get("asset_class") or "") if portfolios else ""
        table = (r.get("asset_class") or {}).get(asset_class) or {}
        _set(annex12, "IVSS10", table.get("underlying_exposure_type"), prov,
             "regime.investor_reporting.IVSS10",
             f"the portfolio's asset class" if asset_class else "", force=True)


def _retention_code(holder: Dict[str, Any], r: Dict[str, Any]) -> str:
    """The Annex 12 holder code implied by the entity's other roles.

    An entity that holds the retention is normally also the originator, sponsor,
    original lender or seller — and that is precisely what IVSS9 reports. Where
    it is none of those, the code is not guessable and the field stays a
    question.
    """
    table = r.get("risk_retention_holder_by_role") or {}
    for role in (holder.get("roles") or []):
        if role in table:
            return str(table[role])
    return ""
