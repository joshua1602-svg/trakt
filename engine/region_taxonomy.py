"""engine.region_taxonomy — governed cross-portfolio region harmonisation.

Direct and acquired books name geography however their source system did.
Consolidating them without harmonisation produces a Total geography chart with
"South West" and "south-west" as two separate bars; harmonising them by
flattening to a lowest common denominator throws away granularity the source
actually supplied. This module does neither.

    raw source region
        ↓  deterministic cleaning        (trim · case · punctuation · separators)
        ↓  exact canonical match
        ↓  approved synonym              (config/mi/region_taxonomy.yaml)
        ↓  LLM-assisted PROPOSAL         ← onboarding only, reviewable, never live
        ↓  governed approval, persisted back into the synonym table
        ↓  canonical_region_detail + canonical_region_reporting  (+ provenance)
        ↓  runtime MI execution          ← reads the persisted columns only

Two levels, because one is not enough:

``canonical_region_detail``
    the most granular governed value the source supports. London stays London.

``canonical_region_reporting``
    the client-level consolidated value used when books are combined. Equal to
    the detail value unless the client's approved reporting taxonomy declares a
    consolidation. There is deliberately NO unconditional London → South East
    rule: that mapping exists only in a taxonomy that names it, is selected per
    client, is visible in provenance, and is reversible because the detail
    column is retained.

The runtime MI path — queries, charts, the geography view, exports — reads the
persisted canonical columns. :func:`propose_unresolved` is an ONBOARDING helper
and is never called while answering a question.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("engine.region_taxonomy")

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Governed taxonomy configuration. Overridable for tests / per deployment.
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "mi" / "region_taxonomy.yaml"
ENV_CONFIG_PATH = "TRAKT_REGION_TAXONOMY"

#: Canonical output columns.
FIELD_DETAIL = "canonical_region_detail"
FIELD_REPORTING = "canonical_region_reporting"
FIELD_SOURCE_VALUE = "region_source_value"
FIELD_METHOD = "region_mapping_method"
FIELD_RULE = "region_mapping_rule"

#: How a raw value was resolved. Recorded per row so a consolidated answer can
#: always explain where each category came from.
METHOD_EXACT = "exact"                # already a governed canonical value
METHOD_SYNONYM = "synonym"            # matched an approved synonym
METHOD_UNRESOLVED = "unresolved"      # no governed mapping — NOT guessed
METHOD_ABSENT = "absent"              # no raw value supplied

#: Applied on top of the detail value to reach the reporting level.
RULE_IDENTITY = "identity"
RULE_CONSOLIDATED = "consolidated"

# Punctuation and separators collapse to single spaces so "South-West",
# "SOUTH_WEST" and "south  west" all clean to "south west".
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"[\s_]+")

_NULL_TOKENS = {"", "nan", "none", "nat", "<na>", "null", "n/a", "na", "unknown"}


def clean(value: Any) -> str:
    """Deterministic cleaning: the key both the taxonomy and its synonyms use.

    Lower-cases, strips punctuation and collapses separators, so every casing and
    punctuation variant of one region converges on one key before any mapping is
    attempted. Returns ``""`` for an absent value.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in _NULL_TOKENS:
        return ""
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip().lower()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegionTaxonomy:
    """One client's resolved region taxonomy: detail vocabulary + reporting view."""

    name: str
    label: str
    #: Governed canonical values, keyed by their cleaned form → display value.
    values_by_key: Dict[str, str] = field(default_factory=dict)
    #: Approved synonyms, keyed by cleaned raw value → canonical display value.
    synonyms: Dict[str, str] = field(default_factory=dict)
    #: The reporting taxonomy name and its approved consolidations.
    reporting_name: str = ""
    reporting_label: str = ""
    consolidations: Dict[str, str] = field(default_factory=dict)

    @property
    def values(self) -> List[str]:
        return sorted(set(self.values_by_key.values()))

    @property
    def consolidates(self) -> bool:
        return bool(self.consolidations)

    def resolve_detail(self, raw: Any) -> Tuple[Optional[str], str]:
        """``(canonical detail value | None, method)`` for one raw value.

        An unresolved value is returned as ``None`` with ``METHOD_UNRESOLVED`` —
        never bucketed into a nearby region and never dropped silently.
        """
        key = clean(raw)
        if not key:
            return None, METHOD_ABSENT
        if key in self.values_by_key:
            return self.values_by_key[key], METHOD_EXACT
        if key in self.synonyms:
            return self.synonyms[key], METHOD_SYNONYM
        return None, METHOD_UNRESOLVED

    def to_reporting(self, detail: Optional[str]) -> Tuple[Optional[str], str]:
        """``(reporting value, rule)`` for a resolved detail value."""
        if detail is None:
            return None, RULE_IDENTITY
        target = self.consolidations.get(detail)
        if target and target != detail:
            return target, RULE_CONSOLIDATED
        return detail, RULE_IDENTITY

    def describe(self) -> Dict[str, Any]:
        """The taxonomy facts the MI Agent discloses in a consolidated answer."""
        return {
            "taxonomy": self.name,
            "taxonomy_label": self.label,
            "reporting_taxonomy": self.reporting_name,
            "reporting_taxonomy_label": self.reporting_label,
            "consolidations": dict(self.consolidations),
            "values": self.values,
        }


def config_path() -> Optional[Path]:
    configured = os.environ.get(ENV_CONFIG_PATH)
    if configured:
        path = Path(configured)
        return path if path.exists() else None
    return DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """The raw taxonomy document, or ``{}`` when none is configured."""
    target = path or config_path()
    if not target:
        return {}
    try:
        import yaml
        return yaml.safe_load(Path(target).read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - harmonisation must never break a run
        logger.warning("region taxonomy config unreadable (%s): %s", target, exc)
        return {}


def resolve_taxonomy(client_id: Optional[str] = None, *,
                     config: Optional[Mapping[str, Any]] = None,
                     path: Optional[Path] = None) -> Optional[RegionTaxonomy]:
    """The governed taxonomy for a client, or ``None`` when none is configured.

    ``None`` means "no harmonisation configured", and every caller treats that as
    a no-op — a deployment without this config behaves exactly as before.
    """
    doc = dict(config) if config is not None else load_config(path)
    if not doc:
        return None
    taxonomies = doc.get("taxonomies") or {}
    if not taxonomies:
        return None

    client_cfg = ((doc.get("clients") or {}).get(client_id) or {}) if client_id else {}
    name = client_cfg.get("taxonomy") or doc.get("default_taxonomy")
    spec = taxonomies.get(name)
    if not spec:
        name, spec = next(iter(taxonomies.items()))

    values_by_key: Dict[str, str] = {}
    for value in spec.get("values") or ():
        values_by_key[clean(value)] = str(value)
    synonyms: Dict[str, str] = {}
    for raw, target in (spec.get("synonyms") or {}).items():
        canonical = values_by_key.get(clean(target))
        if canonical:  # a synonym may only point AT a governed value
            synonyms[clean(raw)] = canonical

    reporting_name = (client_cfg.get("reporting_taxonomy")
                      or doc.get("default_reporting_taxonomy") or name)
    reporting_spec = (doc.get("reporting_taxonomies") or {}).get(reporting_name) or {}
    consolidations: Dict[str, str] = {}
    for source, target in (reporting_spec.get("consolidations") or {}).items():
        src = values_by_key.get(clean(source))
        tgt = values_by_key.get(clean(target))
        if src and tgt:
            consolidations[src] = tgt

    return RegionTaxonomy(
        name=str(name), label=str(spec.get("label") or name),
        values_by_key=values_by_key, synonyms=synonyms,
        reporting_name=str(reporting_name),
        reporting_label=str(reporting_spec.get("label") or reporting_name),
        consolidations=consolidations)


# --------------------------------------------------------------------------- #
# Applying the taxonomy to a tape
# --------------------------------------------------------------------------- #
#: Canonical source columns a raw region may arrive under, most specific first.
SOURCE_FIELDS: Tuple[str, ...] = (
    "geographic_region_obligor", "geographic_region_collateral",
    "collateral_geography", "property_region", "region",
)


def apply(df, taxonomy: Optional[RegionTaxonomy], *,
          source_fields: Sequence[str] = SOURCE_FIELDS) -> Dict[str, Any]:
    """Stamp the governed region columns onto ``df`` in place. Returns a report.

    Adds ``canonical_region_detail`` / ``canonical_region_reporting`` plus the
    provenance the disclosure needs: the raw value, how it was mapped, and which
    rule took it to the reporting level. Rows whose raw value has no governed
    mapping keep NULL canonical values and are counted as unresolved — they are
    disclosed, never assigned to an arbitrary region.

    A no-op (empty report) when no taxonomy is configured or no source column is
    present, so this is safe to call unconditionally.
    """
    import pandas as pd

    if df is None or taxonomy is None or not len(getattr(df, "index", [])):
        return {}
    present = [c for c in source_fields if c in df.columns]
    if not present:
        return {}

    # First populated source column per row, in preference order.
    raw = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in present:
        candidate = df[col]
        blank = raw.isna() | raw.astype(str).str.strip().str.lower().isin(_NULL_TOKENS)
        take = blank & candidate.notna()
        if take.any():
            raw = raw.mask(take, candidate)

    # Resolve once per DISTINCT value, then map — a tape has thousands of rows
    # and a handful of regions.
    distinct = {v for v in raw.dropna().astype(str) if clean(v)}
    detail_by_raw: Dict[str, Optional[str]] = {}
    method_by_raw: Dict[str, str] = {}
    reporting_by_raw: Dict[str, Optional[str]] = {}
    rule_by_raw: Dict[str, str] = {}
    unresolved: Dict[str, int] = {}
    for value in distinct:
        detail, method = taxonomy.resolve_detail(value)
        reporting, rule = taxonomy.to_reporting(detail)
        detail_by_raw[value] = detail
        method_by_raw[value] = method
        reporting_by_raw[value] = reporting
        rule_by_raw[value] = rule

    as_text = raw.astype("object").where(raw.notna())
    df[FIELD_SOURCE_VALUE] = as_text
    df[FIELD_DETAIL] = as_text.map(lambda v: detail_by_raw.get(str(v)) if v is not None else None)
    df[FIELD_REPORTING] = as_text.map(lambda v: reporting_by_raw.get(str(v)) if v is not None else None)
    df[FIELD_METHOD] = as_text.map(
        lambda v: method_by_raw.get(str(v), METHOD_UNRESOLVED) if v is not None else METHOD_ABSENT)
    df[FIELD_RULE] = as_text.map(
        lambda v: rule_by_raw.get(str(v), RULE_IDENTITY) if v is not None else RULE_IDENTITY)

    for value in distinct:
        if method_by_raw[value] == METHOD_UNRESOLVED:
            unresolved[value] = int((as_text.astype(str) == value).sum())

    resolved_rows = int(df[FIELD_DETAIL].notna().sum())
    return {
        "applied": True,
        "source_fields_present": present,
        "taxonomy": taxonomy.name,
        "reporting_taxonomy": taxonomy.reporting_name,
        "consolidations": dict(taxonomy.consolidations),
        "distinct_source_values": len(distinct),
        "rows_resolved": resolved_rows,
        "rows_unresolved": int(len(df) - resolved_rows),
        "unresolved_values": dict(sorted(unresolved.items(), key=lambda kv: -kv[1])),
        "methods": {m: int((df[FIELD_METHOD] == m).sum())
                    for m in (METHOD_EXACT, METHOD_SYNONYM, METHOD_UNRESOLVED, METHOD_ABSENT)},
    }


# --------------------------------------------------------------------------- #
# Onboarding: LLM-assisted proposals for values deterministic mapping missed
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegionMappingProposal:
    """A reviewable proposal. Never applied to runtime data without approval."""

    raw_value: str
    proposed_canonical_value: Optional[str]
    confidence: float
    reason: str
    source_portfolio_id: Optional[str] = None
    row_count: int = 0
    method: str = "llm"
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_value": self.raw_value,
            "proposed_canonical_value": self.proposed_canonical_value,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "source_portfolio_id": self.source_portfolio_id,
            "row_count": self.row_count,
            "method": self.method,
            "status": self.status,
        }


def unresolved_values(report: Mapping[str, Any]) -> Dict[str, int]:
    """The raw values :func:`apply` could not map, with their row counts."""
    return dict((report or {}).get("unresolved_values") or {})


def propose_unresolved(
    values: Mapping[str, int],
    taxonomy: RegionTaxonomy,
    *,
    source_portfolio_id: Optional[str] = None,
    llm_callable=None,
) -> List[RegionMappingProposal]:
    """Propose a canonical mapping for each unresolved raw value. ONBOARDING ONLY.

    This is the single point at which an LLM may be involved in region mapping,
    and it produces a *proposal* — structured, scored, reasoned and attributed to
    a source portfolio — that a human or a governed approval step accepts before
    anything is persisted. Nothing here writes to runtime data.

    ``llm_callable`` takes a prompt and returns JSON; without one, no proposal is
    made (rather than a guess). A proposal naming a value outside the governed
    taxonomy is discarded.
    """
    if not values or taxonomy is None:
        return []
    if llm_callable is None:
        return [RegionMappingProposal(
            raw_value=value, proposed_canonical_value=None, confidence=0.0,
            reason="No LLM configured for onboarding region mapping; this value "
                   "needs a governed decision before it can be consolidated.",
            source_portfolio_id=source_portfolio_id, row_count=count,
            method="none")
            for value, count in values.items()]

    prompt = _proposal_prompt(values, taxonomy, source_portfolio_id)
    try:
        raw = llm_callable(prompt)
        parsed = _parse_proposals(raw)
    except Exception as exc:  # noqa: BLE001 - a proposal step must never break onboarding
        logger.warning("region mapping proposal failed: %s", exc)
        return []

    allowed = set(taxonomy.values)
    out: List[RegionMappingProposal] = []
    for item in parsed:
        value = str(item.get("raw_value") or "").strip()
        if value not in values:
            continue
        proposed = item.get("proposed_canonical_value")
        proposed = str(proposed).strip() if proposed else None
        if proposed not in allowed:
            proposed = None  # never propose outside the governed taxonomy
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        out.append(RegionMappingProposal(
            raw_value=value, proposed_canonical_value=proposed,
            confidence=max(0.0, min(1.0, confidence)),
            reason=str(item.get("reason") or ""),
            source_portfolio_id=source_portfolio_id,
            row_count=int(values.get(value, 0))))
    return out


def _proposal_prompt(values: Mapping[str, int], taxonomy: RegionTaxonomy,
                     source_portfolio_id: Optional[str]) -> str:
    import json
    return (
        "Map each raw geographic region value onto exactly one value from the "
        "governed taxonomy. Return ONLY a JSON array; one object per raw value, "
        "with keys raw_value, proposed_canonical_value, confidence (0-1) and "
        "reason. Use null for proposed_canonical_value when no governed value is "
        "a faithful match — do not guess.\n\n"
        f"Governed taxonomy ({taxonomy.label}):\n"
        f"{json.dumps(taxonomy.values, indent=2)}\n\n"
        f"Source portfolio: {source_portfolio_id or 'unknown'}\n"
        f"Raw values:\n{json.dumps(sorted(values), indent=2)}\n"
    )


def _parse_proposals(raw: Any) -> List[Dict[str, Any]]:
    import json
    if isinstance(raw, (list, tuple)):
        return [dict(x) for x in raw if isinstance(x, Mapping)]
    text = str(raw or "").strip()
    if not text:
        return []
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    parsed = json.loads(text[start:end + 1])
    return [dict(x) for x in parsed if isinstance(x, Mapping)]


def approved_synonyms(proposals: Iterable[RegionMappingProposal]) -> Dict[str, str]:
    """The synonym entries an APPROVED proposal set contributes.

    Keyed by the cleaned raw value, so persisting these into the taxonomy's
    ``synonyms`` block makes the mapping deterministic from the next run onward —
    the LLM is consulted once per new value, never again, and never at runtime.
    """
    out: Dict[str, str] = {}
    for proposal in proposals or ():
        if proposal.status != "approved" or not proposal.proposed_canonical_value:
            continue
        key = clean(proposal.raw_value)
        if key:
            out[key] = proposal.proposed_canonical_value
    return out
