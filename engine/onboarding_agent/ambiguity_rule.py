"""
ambiguity_rule.py
=================

PART 1 — the regulatory-preference ambiguity rule.

When a source column has multiple plausible canonical targets whose top
confidence scores are *close* (within a configurable delta) and one target is
regulatory while another is non-regulatory, the system is uncertain. This
module decides which target to prefer, per onboarding mode, and always flags
the decision for review:

  * regulatory_mi             - prefer the regulatory target.
  * mna_dd                    - prefer the regulatory target (diligence
                                visibility); non-blocking.
  * warehouse_securitisation  - prefer the regulatory target only if regulatory
                                reporting is enabled OR the regulatory target is
                                core_canonical; otherwise prefer the
                                operational/warehouse (non-regulatory) target.
  * mi_only                   - prefer the regulatory target only if it is
                                core_canonical; never select a regulatory
                                non-core target (it is diverted out of scope).

The rule is intentionally pure and registry-driven (it reads `category` and
`core_canonical` via the resolved :class:`field_scope.FieldScopeResult`) so it
can be unit-tested in isolation with constructed candidate lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Reasons (stable identifiers used in artefacts / tests).
REASON_REGULATORY_MI = "regulatory_preference_ambiguity_rule"
REASON_MNA_DD = "regulatory_preference_ambiguity_rule_diligence_visibility"
REASON_WAREHOUSE = "regulatory_preference_ambiguity_rule_warehouse"
REASON_MI_ONLY_CORE = "regulatory_preference_ambiguity_rule_mi_only_core"
REASON_MI_ONLY_DIVERTED = "regulatory_preference_ambiguity_rule_mi_only_noncore_diverted"

DEFAULT_AMBIGUITY_DELTA = 0.10
# Floor below which two competing candidates are treated as noise rather than a
# genuine regulatory/non-regulatory ambiguity (keeps weak token-overlap matches
# from manufacturing spurious regulatory mappings).
DEFAULT_MIN_CANDIDATE_CONFIDENCE = 0.60

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "system" / "onboarding_agent.yaml"


def _mapping_cfg(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else _CONFIG_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw.get("mapping", {}) or {}


def load_ambiguity_delta(config_path: Path | None = None) -> float:
    """Read ``mapping.ambiguity_delta_threshold`` from onboarding_agent.yaml."""
    try:
        return float(_mapping_cfg(config_path).get(
            "ambiguity_delta_threshold", DEFAULT_AMBIGUITY_DELTA))
    except (TypeError, ValueError):
        return DEFAULT_AMBIGUITY_DELTA


def load_min_candidate_confidence(config_path: Path | None = None) -> float:
    """Read ``mapping.min_ambiguity_candidate_confidence`` (plausibility floor)."""
    try:
        return float(_mapping_cfg(config_path).get(
            "min_ambiguity_candidate_confidence", DEFAULT_MIN_CANDIDATE_CONFIDENCE))
    except (TypeError, ValueError):
        return DEFAULT_MIN_CANDIDATE_CONFIDENCE


@dataclass
class Candidate:
    """One scored canonical candidate for a source column."""

    field: str
    confidence: float
    category: str = ""          # regulatory | analytics | ""
    core_canonical: bool = False
    method: str = ""

    @property
    def is_regulatory(self) -> bool:
        return self.category == "regulatory"


@dataclass
class AmbiguityResolution:
    """Outcome of applying the regulatory-preference rule to a column."""

    selected: Optional[Candidate]
    alternative: Optional[Candidate]
    review_required: bool
    reason: str
    rule_applied: str
    # When True the originally top regulatory non-core candidate must be diverted
    # to 05a_out_of_scope_fields.csv (mi_only only).
    divert_regulatory_to_out_of_scope: bool = False
    diverted_field: Optional[Candidate] = None

    @property
    def confidence_delta(self) -> float:
        if self.selected is None or self.alternative is None:
            return 0.0
        return round(abs(self.selected.confidence - self.alternative.confidence), 4)


def classify_candidate(
    field_name: str,
    confidence: float,
    field_scope,
    method: str = "",
) -> Candidate:
    """Build a :class:`Candidate` using the resolved field scope for metadata."""
    category = ""
    core = False
    if field_scope is not None:
        category = field_scope.category_of(field_name) or ""
        core = field_name in getattr(field_scope, "core_canonical_fields", set())
    return Candidate(
        field=field_name,
        confidence=float(confidence),
        category=category,
        core_canonical=core,
        method=method,
    )


def _best_by_category(candidates: List[Candidate], regulatory: bool) -> Optional[Candidate]:
    pool = [c for c in candidates if c.is_regulatory == regulatory and c.field]
    if not pool:
        return None
    return max(pool, key=lambda c: c.confidence)


def detect_regulatory_ambiguity(
    candidates: List[Candidate],
    delta_threshold: float,
    min_candidate_confidence: float = 0.0,
) -> Optional[Tuple[Candidate, Candidate]]:
    """Return (regulatory_candidate, non_regulatory_candidate) when the two best
    competing candidates are within ``delta_threshold`` and straddle the
    regulatory / non-regulatory boundary; otherwise None.

    Both candidates must score at or above ``min_candidate_confidence`` so weak
    token-overlap matches do not manufacture a spurious ambiguity.
    """
    reg = _best_by_category(candidates, regulatory=True)
    non_reg = _best_by_category(candidates, regulatory=False)
    if reg is None or non_reg is None:
        return None
    if reg.confidence < min_candidate_confidence or non_reg.confidence < min_candidate_confidence:
        return None
    if abs(reg.confidence - non_reg.confidence) <= delta_threshold:
        return reg, non_reg
    return None


def resolve_regulatory_preference(
    candidates: List[Candidate],
    *,
    mode: str,
    delta_threshold: float = DEFAULT_AMBIGUITY_DELTA,
    regulatory_reporting_enabled: bool = False,
    min_candidate_confidence: float = 0.0,
) -> Optional[AmbiguityResolution]:
    """Apply the regulatory-preference ambiguity rule.

    Returns an :class:`AmbiguityResolution` when the rule fires, or ``None`` when
    there is no regulatory/non-regulatory ambiguity (caller keeps its
    deterministic selection unchanged).
    """
    pair = detect_regulatory_ambiguity(
        candidates, delta_threshold, min_candidate_confidence)
    if pair is None:
        return None
    reg, non_reg = pair

    if mode == "regulatory_mi":
        return AmbiguityResolution(
            selected=reg, alternative=non_reg, review_required=True,
            reason=REASON_REGULATORY_MI, rule_applied=REASON_REGULATORY_MI,
        )

    if mode == "mna_dd":
        # Prefer regulatory for diligence visibility; non-blocking review.
        return AmbiguityResolution(
            selected=reg, alternative=non_reg, review_required=True,
            reason=REASON_MNA_DD, rule_applied=REASON_MNA_DD,
        )

    if mode == "warehouse_securitisation":
        if regulatory_reporting_enabled or reg.core_canonical:
            return AmbiguityResolution(
                selected=reg, alternative=non_reg, review_required=True,
                reason=REASON_WAREHOUSE, rule_applied=REASON_WAREHOUSE,
            )
        # Otherwise prefer the operational / warehouse (non-regulatory) target.
        return AmbiguityResolution(
            selected=non_reg, alternative=reg, review_required=True,
            reason=REASON_WAREHOUSE, rule_applied=REASON_WAREHOUSE,
        )

    if mode == "mi_only":
        if reg.core_canonical:
            # Regulatory core fields may be selected even in MI-only.
            return AmbiguityResolution(
                selected=reg, alternative=non_reg, review_required=True,
                reason=REASON_MI_ONLY_CORE, rule_applied=REASON_MI_ONLY_CORE,
            )
        # Regulatory non-core must NEVER be selected in MI-only. Prefer the
        # in-scope analytics target; divert the regulatory candidate to
        # 05a_out_of_scope_fields.csv.
        return AmbiguityResolution(
            selected=non_reg, alternative=reg, review_required=True,
            reason=REASON_MI_ONLY_DIVERTED, rule_applied=REASON_MI_ONLY_DIVERTED,
            divert_regulatory_to_out_of_scope=True, diverted_field=reg,
        )

    # Unknown mode: be conservative, prefer regulatory and flag for review.
    return AmbiguityResolution(
        selected=reg, alternative=non_reg, review_required=True,
        reason=REASON_REGULATORY_MI, rule_applied=REASON_REGULATORY_MI,
    )


def ambiguity_record(
    resolution: AmbiguityResolution,
    *,
    source_file: str,
    source_column: str,
    mode: str,
) -> Dict[str, Any]:
    """Flatten an :class:`AmbiguityResolution` into a 05b artefact row dict."""
    sel = resolution.selected
    alt = resolution.alternative
    return {
        "source_file": source_file,
        "source_column": source_column,
        "selected_canonical_field": sel.field if sel else "",
        "selected_category": sel.category if sel else "",
        "selected_core_canonical": bool(sel.core_canonical) if sel else False,
        "selected_confidence": round(sel.confidence, 4) if sel else 0.0,
        "alternative_canonical_field": alt.field if alt else "",
        "alternative_category": alt.category if alt else "",
        "alternative_core_canonical": bool(alt.core_canonical) if alt else False,
        "alternative_confidence": round(alt.confidence, 4) if alt else 0.0,
        "confidence_delta": resolution.confidence_delta,
        "ambiguity_rule_applied": resolution.rule_applied,
        "review_required": resolution.review_required,
        "reason": resolution.reason,
        "mode": mode,
    }
