"""Read-only governed metadata, and the tool surface that exposes it to Opus.

WHY THIS REPLACED A HANDCRAFTED VOCABULARY
------------------------------------------
The first build handed the interpreter one flat, business-name-only vocabulary
derived from a single registry. It was an intentionally impoverished view, and
measurement showed exactly what that costs: the interpreter clarified on
sixteen of twenty questions because it could not verify that "drawdown" is a
product or that "Offer" is a pipeline stage.

Both were governed all along, in sources the vocabulary never read:

    config/asset/product_profiles.yaml     drawdown, lump_sum, rio, erm, ...
    question_interpretation.lexical        KFI, APPLICATION, OFFER, COMPLETED,
                                           WITHDRAWN (22 spellings)
    config/business_semantics_registry     analytical concept, temporality,
                                           asset applicability, categories
    config/risk/concentration_test_library 42 governed limit metrics
    config/mi/buckets.yaml                 the governed band labels

So this module is a read-only adapter over the authoritative estate, not a
second copy of it. Nothing here is curated, nothing is redesigned, and nothing
is written back.

THE BOUNDARY THAT DID NOT MOVE
------------------------------
Opus may SEE canonical identifiers and refer to them. That does not make it
authoritative: every concept it proposes is re-derived by the compiler against
these same sources before it can enter a plan.

What it may never see is DATA. That distinction is enforced field by field, not
by good intentions — ``config/risk/funding_facilities.yaml`` carries a
£250m commitment and an advance rate beside the facility's identity, and
:func:`_facility_context` takes the identity and leaves the figures. The
assertion is ``tests/interpretation_v2/test_model_sees_no_data.py``, which
walks every tool's output looking for a number that could be a portfolio value.

    loan rows sent to Opus             = 0
    borrower-level data sent to Opus   = 0
    calculated MI answers sent to Opus = 0
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

METADATA_VERSION = "governed_metadata/1.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every authoritative source this adapter reads. Listed in one place so the
#: estate it depends on is inspectable, and so a test can assert that each one
#: is read rather than quietly skipped.
SOURCES: Mapping[str, Path] = {
    "canonical_fields": _REPO_ROOT / "config" / "system" / "fields_registry.yaml",
    "mi_semantics": _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml",
    "business_semantics": _REPO_ROOT / "config" / "business_semantics_registry.yaml",
    "enum_synonyms": _REPO_ROOT / "config" / "system" / "enum_synonyms.yaml",
    "capabilities": _REPO_ROOT / "config" / "system" / "mi_capability_registry.yaml",
    "asset_profiles": _REPO_ROOT / "config" / "asset" / "product_profiles.yaml",
    "asset_geography": _REPO_ROOT / "config" / "asset" / "mi_geography.yaml",
    "buckets": _REPO_ROOT / "config" / "mi" / "buckets.yaml",
    "region_taxonomy": _REPO_ROOT / "config" / "mi" / "region_taxonomy.yaml",
    "concentration_tests": _REPO_ROOT / "config" / "risk"
                           / "concentration_test_library.yaml",
    "funding_facilities": _REPO_ROOT / "config" / "risk" / "funding_facilities.yaml",
    "client": _REPO_ROOT / "config" / "client" / "config_client_ERE.yaml",
}


@lru_cache(maxsize=32)
def _load(name: str) -> Mapping[str, Any]:
    """One authoritative source, read once. Never written."""
    path = SOURCES[name]
    if not path.exists():  # pragma: no cover - the estate is committed
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _slug(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# Governed value vocabularies
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _enum_domain_values() -> Mapping[str, Tuple[str, ...]]:
    """domain -> the BUSINESS spellings its values may take.

    The ESMA codes on the right of each synonym map are a physical encoding and
    are deliberately not exposed; the model speaks the business words.
    """
    out: Dict[str, Tuple[str, ...]] = {}
    for domain, block in (_load("enum_synonyms") or {}).items():
        words: List[str] = []
        for mapping in (block or {}).values():
            if not isinstance(mapping, Mapping):
                continue
            for spelling, code in mapping.items():
                text = str(spelling).strip()
                if text and text != code and text not in words:
                    words.append(text)
        if words:
            out[domain] = tuple(sorted(words))
    return out


@lru_cache(maxsize=1)
def _pipeline_stage_values() -> Tuple[str, ...]:
    """The governed funnel stages.

    ``question_interpretation.lexical`` is the estate's ONE question-side stage
    vocabulary — the same one the shipped parser reads — so the canonical stage
    names come from there rather than from a list retyped here.
    """
    try:
        from question_interpretation.lexical import pipeline_stage_vocabulary
        return tuple(sorted(set(pipeline_stage_vocabulary().values())))
    except Exception:  # noqa: BLE001 - metadata degrades, it never fails a parse
        return ()


@lru_cache(maxsize=1)
def _product_type_values() -> Tuple[str, ...]:
    """The governed product values, from the asset profile that matches them.

    ``config/asset/product_profiles.yaml`` declares which product types belong
    to each asset profile. This is where "drawdown" and "lump_sum" live — the
    first measurement clarified on both because nothing read this file.
    """
    profiles = (_load("asset_profiles") or {}).get("profiles") or {}
    values: List[str] = []
    for profile in profiles.values():
        for value in ((profile or {}).get("match") or {}).get("product_type") or ():
            text = str(value).strip()
            if text and text not in values:
                values.append(text)
    return tuple(sorted(values))


@lru_cache(maxsize=1)
def _bucket_labels() -> Mapping[str, Tuple[str, ...]]:
    """Governed band labels per bucket dimension ("<20%", "20-30%", ...).

    Labels only. The bucket EDGES are a calculation input and are not metadata
    a reader of a question needs.
    """
    out: Dict[str, Tuple[str, ...]] = {}
    for name, block in ((_load("buckets") or {}).get("buckets") or {}).items():
        labels = tuple(str(v) for v in ((block or {}).get("labels") or ()))
        if labels:
            out[str(name)] = labels
            semantic = (block or {}).get("semantic_field")
            if semantic and str(semantic) != str(name):
                out[str(semantic)] = labels
    return out


@lru_cache(maxsize=1)
def _region_values() -> Tuple[str, ...]:
    """The governed reporting-region values for the configured taxonomy."""
    taxonomy = _load("region_taxonomy") or {}
    default = taxonomy.get("default_taxonomy")
    block = (taxonomy.get("taxonomies") or {}).get(default) or {}
    values: List[str] = []
    for key in ("regions", "values", "members"):
        for value in (block.get(key) or ()):
            text = str(value).strip() if not isinstance(value, Mapping) \
                else str(value.get("name") or value.get("id") or "").strip()
            if text and text not in values:
                values.append(text)
    if not values and isinstance(block, Mapping):
        values = sorted(str(k) for k in block if isinstance(k, str))
    return tuple(values)


@lru_cache(maxsize=1)
def governed_values_for_field() -> Mapping[str, Tuple[str, ...]]:
    """canonical field -> its governed values, from wherever they are governed.

    Four sources, joined. A field with no governed values is ABSENT from this
    map, and that absence is reported to the model as "no governed value list"
    rather than as silence — a value that cannot be checked is a guess, and the
    interpreter can only decline to guess if it is told.
    """
    fields = (_load("canonical_fields") or {}).get("fields") or {}
    domains = _enum_domain_values()
    out: Dict[str, Tuple[str, ...]] = {}

    for canonical_field, entry in fields.items():
        domain = (entry or {}).get("allowed_values")
        if domain and domain in domains:
            out[canonical_field] = domains[domain]

    stages = _pipeline_stage_values()
    if stages:
        out["pipeline_stage"] = stages
    products = _product_type_values()
    if products:
        out["erm_product_type"] = products
    regions = _region_values()
    for region_field in ("canonical_region_reporting", "canonical_region_detail",
                         "collateral_geography"):
        if regions:
            out[region_field] = regions
    for name, labels in _bucket_labels().items():
        out.setdefault(name, labels)
    return out


# --------------------------------------------------------------------------- #
# Analytical metadata (relationships, applicability)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def business_semantics() -> Mapping[str, Mapping[str, Any]]:
    """canonical field -> its governed ANALYTICAL metadata.

    From the Business Semantics Registry v2: the analytical concept a field
    belongs to, its role and temporality, which asset classes it applies to,
    and the workflows it participates in. This is the "relationships to other
    governed concepts" and "applicability metadata" an interpreter needs to tell
    a measure from a dimension and a stock from a flow.
    """
    entries = (_load("business_semantics") or {}).get("fields") or {}
    if isinstance(entries, list):
        return {str(e.get("source_field")): e for e in entries if e.get("source_field")}
    return {str(k): v for k, v in entries.items()}


@lru_cache(maxsize=1)
def canonical_fields() -> Mapping[str, Mapping[str, Any]]:
    return (_load("canonical_fields") or {}).get("fields") or {}


@lru_cache(maxsize=1)
def capability_catalogue() -> Tuple[Mapping[str, Any], ...]:
    """The 28 registered analytical capabilities and their semantic purpose."""
    return tuple((_load("capabilities") or {}).get("capabilities") or ())


@lru_cache(maxsize=1)
def concentration_tests() -> Tuple[Mapping[str, Any], ...]:
    """The governed limit/concentration metrics, as semantic descriptions."""
    return tuple((_load("concentration_tests") or {}).get("metrics") or ())


# --------------------------------------------------------------------------- #
# Asset and portfolio semantic context
# --------------------------------------------------------------------------- #

#: Keys on a facility record that are CONFIGURATION (what exists, in what
#: currency, of what type) rather than POSITION (how much). Everything else on
#: the record — commitment, advance rate, drawn amount, floors, dates — is a
#: portfolio value and never leaves this module.
_FACILITY_SAFE_KEYS = ("facility_id", "facility_label", "facility_type",
                       "currency", "client_id")


def _facility_context() -> List[Dict[str, Any]]:
    """Which funding facilities EXIST. Never how much is drawn against them.

    The addendum permits "presence/absence of a funding facility" as governed
    context. The same file carries a £250m commitment and an advance rate two
    keys away, so the allowlist is positive: named keys are copied, everything
    else is dropped, and a key added to the config upstream cannot leak by
    default.
    """
    facilities = (_load("funding_facilities") or {}).get("facilities") or ()
    return [{k: v for k, v in (facility or {}).items() if k in _FACILITY_SAFE_KEYS}
            for facility in facilities]


@lru_cache(maxsize=1)
def portfolio_semantic_context() -> Mapping[str, Any]:
    """The governed facts about THIS environment that change what a question means.

    Asset class, currency, country, which lenses and geography bases are
    configured, which capabilities are registered, whether a funding facility
    exists at all. No dates: a reporting date is a snapshot handle, and the
    interpreter states time semantically so that the governed period contract
    can resolve it against a book this package never sees.
    """
    client = _load("client") or {}
    portfolio = client.get("portfolio") or {}
    asset_class = str(portfolio.get("asset_class") or "").strip()
    basis_by_asset = (_load("asset_geography") or {}).get(
        "primary_basis_by_asset_class") or {}
    facilities = _facility_context()
    return {
        "client_id": str((client.get("client") or {}).get("client_id") or ""),
        "asset_class": asset_class,
        "country": str(portfolio.get("country") or ""),
        "base_currency": str(portfolio.get("base_currency") or ""),
        "primary_geography_basis": basis_by_asset.get(asset_class, "collateral"),
        "available_portfolio_lenses": ["direct", "acquired", "all"],
        "configured_geography_levels": ["reporting", "nuts3", "itl3", "postcode"],
        "funding_facility_configured": bool(facilities),
        "funding_facilities": facilities,
        "concentration_tests_configured": len(concentration_tests()),
        "note": "Configuration only. No loan rows, no balances, no calculated "
                "figures, and no reporting date — time is stated semantically "
                "and resolved downstream.",
    }


@lru_cache(maxsize=1)
def asset_metadata() -> Mapping[str, Any]:
    """The asset-class semantics: which profile applies and what it implies."""
    profiles = _load("asset_profiles") or {}
    context = portfolio_semantic_context()
    asset_class = context.get("asset_class")
    matched: Optional[Dict[str, Any]] = None
    for profile_id, profile in (profiles.get("profiles") or {}).items():
        match = (profile or {}).get("match") or {}
        if asset_class in [str(a) for a in (match.get("asset_class") or ())]:
            matched = {
                "profile_id": profile_id,
                "label": (profile or {}).get("label"),
                "description": (profile or {}).get("description"),
                "product_types": sorted(str(p) for p in
                                        (match.get("product_type") or ())),
            }
            break
    return {
        "asset_class": asset_class,
        "profile": matched,
        "asset_capabilities": sorted(str(c) for c in
                                     (profiles.get("capabilities") or ())),
        "primary_geography_basis": context.get("primary_geography_basis"),
        "governed_pipeline_stages": list(_pipeline_stage_values()),
        "governed_product_types": list(_product_type_values()),
    }


def applies_to_asset_class(applicability: Sequence[str], asset_class: str) -> bool:
    """Whether a concept's declared applicability covers this asset class.

    ``cross_asset`` covers everything; an empty declaration is treated as
    cross-asset, because the registry leaves applicability unstated for fields
    that were never asset-specific and reading silence as "applies to nothing"
    would empty the vocabulary.
    """
    if not applicability:
        return True
    declared = {str(a).strip().lower() for a in applicability}
    if "cross_asset" in declared or "all" in declared:
        return True
    return str(asset_class or "").strip().lower() in declared


# --------------------------------------------------------------------------- #
# The tool surface
# --------------------------------------------------------------------------- #

#: Metadata tools only. Nothing here reads a tape, runs an engine, or writes.
TOOL_NAMES = ("search_concepts", "get_concept_metadata", "get_allowed_values",
              "search_capabilities", "get_capability_metadata",
              "get_asset_metadata", "get_portfolio_semantic_context")


def metadata_tool_schemas() -> List[Dict[str, Any]]:
    """The read-only metadata tools offered alongside the intent tool."""
    return [
        {
            "name": "search_concepts",
            "description": (
                "Search Trakt's governed concept registry. Returns concept "
                "identifiers with their definition, role, aliases and units. "
                "Use it to find the governed identifier for a business word "
                "before you name it in the intent."),
            "input_schema": {
                "type": "object", "additionalProperties": False,
                "required": ["query"],
                "properties": {
                    "query": {"type": "string",
                              "description": "A business word or phrase, e.g. "
                                             "'balance', 'loan to value', "
                                             "'region', 'borrower age'."},
                    "role": {"type": "string",
                             "enum": ["measure", "dimension", "date", "flag"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 40},
                },
            },
        },
        {
            "name": "get_concept_metadata",
            "description": (
                "Full governed metadata for one concept identifier: definition, "
                "data type, role, temporality, permitted statistics, default "
                "statistic and weight, related bucket dimension, analytical "
                "concept, and which asset classes it applies to."),
            "input_schema": {
                "type": "object", "additionalProperties": False,
                "required": ["concept_id"],
                "properties": {"concept_id": {"type": "string"}},
            },
        },
        {
            "name": "get_allowed_values",
            "description": (
                "The governed values a dimension may take. Returns "
                "has_governed_values=false when Trakt governs no value list for "
                "it — in that case do NOT assert a filter value against it."),
            "input_schema": {
                "type": "object", "additionalProperties": False,
                "required": ["concept_id"],
                "properties": {"concept_id": {"type": "string"}},
            },
        },
        {
            "name": "search_capabilities",
            "description": (
                "Discover registered analytical capabilities and their semantic "
                "purpose — what Trakt can be asked to do, and by which owner."),
            "input_schema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 40},
                },
            },
        },
        {
            "name": "get_capability_metadata",
            "description": (
                "What one capability computes, its unit, time basis, category, "
                "and the governed operations it offers."),
            "input_schema": {
                "type": "object", "additionalProperties": False,
                "required": ["capability_id"],
                "properties": {"capability_id": {"type": "string"}},
            },
        },
        {
            "name": "get_asset_metadata",
            "description": (
                "The asset-class semantics for this environment: the matched "
                "product profile, governed product types, governed pipeline "
                "stages, and the primary geography basis for the asset class."),
            "input_schema": {"type": "object", "additionalProperties": False,
                             "properties": {}},
        },
        {
            "name": "get_portfolio_semantic_context",
            "description": (
                "Governed configuration for this environment: asset class, "
                "currency, country, available portfolio lenses, configured "
                "geography levels, whether a funding facility exists, and how "
                "many concentration tests are configured. Configuration only — "
                "never loan data or calculated figures."),
            "input_schema": {"type": "object", "additionalProperties": False,
                             "properties": {}},
        },
    ]


@dataclass(frozen=True)
class ToolCallRecord:
    """One metadata retrieval, for the run's provenance."""

    name: str
    arguments: Mapping[str, Any]
    result_summary: str


class GovernedMetadataService:
    """Read-only metadata retrieval. The only thing Opus can call.

    Holds a reference to the compiler's authoritative concept index so that what
    the model is shown and what the compiler binds against are the SAME index —
    a metadata service that drifted from the registry the compiler validates on
    would be worse than none, because it would advertise concepts that then
    refuse.
    """

    version = METADATA_VERSION

    def __init__(self, vocabulary) -> None:
        self.vocabulary = vocabulary
        self.calls: List[ToolCallRecord] = []

    # -- dispatch ----------------------------------------------------------- #

    def call(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        """Run one metadata tool. Unknown names are an error, never a guess."""
        handler = {
            "search_concepts": self.search_concepts,
            "get_concept_metadata": self.get_concept_metadata,
            "get_allowed_values": self.get_allowed_values,
            "search_capabilities": self.search_capabilities,
            "get_capability_metadata": self.get_capability_metadata,
            "get_asset_metadata": self.get_asset_metadata,
            "get_portfolio_semantic_context": self.get_portfolio_semantic_context,
        }.get(name)
        if handler is None:
            return {"error": f"unknown metadata tool {name!r}",
                    "available": list(TOOL_NAMES)}
        try:
            result = handler(**dict(arguments or {}))
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}
        self.calls.append(ToolCallRecord(
            name=name, arguments=dict(arguments or {}),
            result_summary=json.dumps(result, default=str)[:200]))
        return result

    # -- concepts ----------------------------------------------------------- #

    def search_concepts(self, query: str, role: Optional[str] = None,
                        limit: int = 20) -> Dict[str, Any]:
        needle = str(query or "").strip().lower()
        limit = max(1, min(int(limit or 20), 40))
        scored: List[Tuple[int, Any]] = []
        for concept in self.vocabulary.concepts.values():
            if role and concept.role != role:
                continue
            score = _match_score(needle, concept)
            if score:
                scored.append((score, concept))
        scored.sort(key=lambda pair: (-pair[0], pair[1].concept_id))
        return {
            "query": query,
            "match_count": len(scored),
            "concepts": [c.search_view() for _, c in scored[:limit]],
            "note": ("A concept identifier here is a PROPOSAL you may name in "
                     "the intent. The compiler validates every one against this "
                     "same registry before anything runs."),
        }

    def get_concept_metadata(self, concept_id: str) -> Dict[str, Any]:
        concept = self.vocabulary.resolve(concept_id)
        if concept is None:
            candidates = self.vocabulary.candidates(concept_id)
            if len(candidates) > 1:
                return {"concept_id": concept_id, "found": False,
                        "reason": "AMBIGUOUS — this word names more than one "
                                  "governed concept; choose one identifier",
                        "candidates": [c.concept_id for c in candidates]}
            return {"concept_id": concept_id, "found": False,
                    "reason": "not a governed concept",
                    "did_you_mean": [c["concept_id"] for c in
                                     self.search_concepts(concept_id,
                                                          limit=5)["concepts"]]}
        return {"found": True, **concept.metadata_view()}

    def get_allowed_values(self, concept_id: str) -> Dict[str, Any]:
        concept = self.vocabulary.resolve(concept_id)
        if concept is None:
            return {"concept_id": concept_id, "found": False,
                    "reason": "not a governed concept"}
        if concept.values:
            return {"concept_id": concept.concept_id, "found": True,
                    "has_governed_values": True,
                    "values": list(concept.values),
                    "source": concept.values_source}
        return {
            "concept_id": concept.concept_id, "found": True,
            "has_governed_values": False,
            "guidance": ("Trakt governs no value list for this concept. Do NOT "
                         "assert a filter value against it — record it in "
                         "`ambiguity` with blocking=true instead."),
        }

    # -- capabilities ------------------------------------------------------- #

    def search_capabilities(self, query: Optional[str] = None,
                            limit: int = 20) -> Dict[str, Any]:
        from .vocabulary import CAPABILITY_OPERATIONS

        needle = str(query or "").strip().lower()
        rows: List[Dict[str, Any]] = []
        for capability in sorted(self.vocabulary.capabilities):
            blob = f"{capability} {' '.join(CAPABILITY_OPERATIONS.get(capability, ()))}"
            if needle and needle not in blob.lower():
                continue
            rows.append(self._capability_row(capability))
        registry_rows = [
            {"registered_capability": c.get("id"), "name": c.get("name"),
             "description": c.get("description"), "category": c.get("category"),
             "unit": c.get("unit"), "time_basis": c.get("time_basis")}
            for c in capability_catalogue()
            if not needle or needle in json.dumps(c, default=str).lower()]
        return {
            "query": query,
            "intent_capabilities": rows[:limit],
            "registered_analytical_capabilities": registry_rows[:limit],
            "note": ("`intent_capabilities` are the values the intent's "
                     "`capability` slot accepts. The registered list is the "
                     "deterministic catalogue behind them."),
        }

    def get_capability_metadata(self, capability_id: str) -> Dict[str, Any]:
        wanted = str(capability_id or "").strip().lower()
        if wanted in self.vocabulary.capabilities:
            return {"found": True, **self._capability_row(wanted)}
        for entry in capability_catalogue():
            if str(entry.get("id") or "").lower() == wanted:
                return {"found": True, "registered_capability": entry.get("id"),
                        "name": entry.get("name"),
                        "description": entry.get("description"),
                        "category": entry.get("category"),
                        "unit": entry.get("unit"),
                        "basis": entry.get("basis"),
                        "time_basis": entry.get("time_basis"),
                        "aggregation": entry.get("aggregation")}
        return {"capability_id": capability_id, "found": False,
                "available": sorted(self.vocabulary.capabilities)}

    def _capability_row(self, capability: str) -> Dict[str, Any]:
        from .vocabulary import CAPABILITY_OPERATIONS

        measures = sorted(c.concept_id for c in self.vocabulary.concepts.values()
                          if c.owning_capability == capability
                          and c.role == "measure")
        dimensions = sorted(c.concept_id for c in self.vocabulary.concepts.values()
                            if c.owning_capability == capability
                            and c.role == "dimension")
        row: Dict[str, Any] = {
            "capability": capability,
            "operations": sorted(CAPABILITY_OPERATIONS.get(capability, ())),
        }
        if measures:
            row["owned_measures"] = measures
            row["note"] = ("This capability owns how these are calculated. Name "
                           "them; do not decompose them, do not impose a "
                           "statistic or weight, and do not state a period for "
                           "a movement it defines itself.")
        if dimensions:
            row["owned_dimensions"] = dimensions
        if capability in ("concentration", "limit_assessment"):
            row["governed_tests"] = [
                {"metric_id": t.get("metric_id"),
                 "display_name": t.get("display_name"),
                 "category": t.get("category")}
                for t in concentration_tests()][:42]
        return row

    # -- environment -------------------------------------------------------- #

    def get_asset_metadata(self) -> Dict[str, Any]:
        return dict(asset_metadata())

    def get_portfolio_semantic_context(self) -> Dict[str, Any]:
        context = dict(portfolio_semantic_context())
        context["registered_intent_capabilities"] = sorted(
            self.vocabulary.capabilities)
        return context


def _match_score(needle: str, concept) -> int:
    """How well a search term matches a concept. Deterministic, no fuzziness.

    Ranked so an exact identifier or business name beats a substring, and a
    substring of an alias beats a substring of the description. Approximate
    matching is deliberately absent: a near-miss that scores is how a wrong
    concept gets proposed confidently.
    """
    if not needle:
        return 1
    if needle == concept.concept_id or needle == _slug(concept.label):
        return 100
    if needle in {a for a in concept.aliases}:
        return 90
    if needle in concept.concept_id or needle in _slug(concept.label):
        return 60
    if any(needle in alias for alias in concept.aliases):
        return 40
    if needle in (concept.description or "").lower():
        return 15
    if any(word and word in concept.concept_id
           for word in needle.split() if len(word) > 3):
        return 10
    return 0
