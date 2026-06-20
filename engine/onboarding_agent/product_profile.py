"""
product_profile.py
==================

Asset-agnostic, config-driven **product profile** layer for onboarding hardening.

The onboarding engine stays generic. A *product profile* (loaded from
``config/asset/product_profiles.yaml``) captures the structural characteristics
of a product family — e.g. an equity-release / lifetime mortgage capitalises
interest and has no contractual maturity — so that a handful of generic
"required" fields become *not applicable*, *derivable*, or *defaultable* for base
MI, **without** weakening other asset classes, regulatory requirements, or the
field registry.

Design guarantees
-----------------
* Config-driven: profiles are defined in YAML, keyed by ``profile_id``. No client
  names, no source-file names, no tape-specific keys appear in this module.
* Explicit > inferred: an operator/config selection is trusted outright; a
  detected profile is only auto-applied at/above ``apply_confidence``; between
  ``confirm_confidence`` and ``apply_confidence`` it is *proposed for
  confirmation*, never silently applied.
* Auditable: every default / derivation produced here is returned as a
  :class:`DerivationRecord` carrying ``field``, ``method``, ``value``,
  ``source``, ``confidence`` and ``rationale`` so callers can write it to the
  review / handoff artefacts.
* Never fabricates loan-level values silently and never derives
  ``number_of_borrowers`` from a unique loan-id count.

This module deliberately does NOT mutate coverage / readiness state itself; it
exposes pure helpers that the orchestrator, coverage layer
(``target_coverage``) and capability-readiness layer consume.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field as dc_field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROFILES_PATH = _REPO_ROOT / "config" / "asset" / "product_profiles.yaml"

# Resolution decisions (controlled vocabulary, recorded in artefacts).
DECISION_EXPLICIT = "explicit_config_or_operator"
DECISION_DETECTED = "detected_high_confidence"
DECISION_NEEDS_CONFIRMATION = "detected_needs_confirmation"
DECISION_NONE = "no_profile_generic_behaviour"

# Base-MI field policy vocabulary.
POLICY_NOT_APPLICABLE = "not_applicable"
POLICY_DERIVED = "derived"
POLICY_DEFAULTED = "defaulted"
POLICY_OPTIONAL = "optional"
POLICY_REQUIRED = "required"

# Policies that are NON-BLOCKING for base MI (still surfaced as visible gaps).
_NON_BLOCKING_BASE_MI = {
    POLICY_NOT_APPLICABLE, POLICY_DERIVED, POLICY_DEFAULTED, POLICY_OPTIONAL,
}


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s or "").strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass
class DerivationRecord:
    """A single profile-driven default or derivation, fully auditable."""
    field: str
    method: str                 # e.g. from_field / from_dates / from_artefact_role / default
    value: Any = None
    source: str = ""            # what the value/derivation came from
    confidence: float = 0.0
    rationale: str = ""
    profile_id: str = ""
    blocking: bool = False      # profile-driven derivations are never base-MI blocking

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "method": self.method,
            "value": self.value,
            "source": self.source,
            "confidence": round(float(self.confidence), 4),
            "rationale": self.rationale,
            "profile_id": self.profile_id,
            "blocking": self.blocking,
        }


@dataclass
class ResolvedProfile:
    """Outcome of resolving which product profile (if any) applies to a run."""
    profile_id: str = ""
    decision: str = DECISION_NONE
    confidence: float = 0.0
    applied: bool = False               # True only when the profile is active
    needs_confirmation: bool = False
    rationale: str = ""
    evidence: List[str] = dc_field(default_factory=list)
    profile: Optional["ProductProfile"] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "decision": self.decision,
            "confidence": round(float(self.confidence), 4),
            "applied": self.applied,
            "needs_confirmation": self.needs_confirmation,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
        }


# --------------------------------------------------------------------------- #
# Profile object
# --------------------------------------------------------------------------- #

class ProductProfile:
    """A single config-driven product profile."""

    def __init__(self, profile_id: str, spec: Dict[str, Any], defaults: Dict[str, Any]):
        self.profile_id = profile_id
        self.spec = spec or {}
        self.label = self.spec.get("label", profile_id)
        self.description = self.spec.get("description", "")
        cp = self.spec.get("confidence_policy", {}) or {}
        self.apply_confidence = float(cp.get("apply_confidence",
                                              defaults.get("apply_confidence", 0.80)))
        self.confirm_confidence = float(cp.get("confirm_confidence",
                                               defaults.get("confirm_confidence", 0.55)))
        self.field_policies: Dict[str, Dict[str, Any]] = {
            _norm(k): v for k, v in (self.spec.get("field_policies", {}) or {}).items()
        }
        self.capability_fields: Dict[str, Dict[str, Any]] = (
            self.spec.get("capability_fields", {}) or {}
        )

    # -- field policy access ------------------------------------------------ #

    def field_policy(self, field: str) -> Dict[str, Any]:
        return self.field_policies.get(_norm(field), {})

    def base_mi_policy(self, field: str) -> str:
        return str(self.field_policy(field).get("base_mi", POLICY_REQUIRED))

    def is_non_blocking_for_base_mi(self, field: str) -> bool:
        """A field is non-blocking for base MI when the profile marks it
        not_applicable / derived / defaulted / optional."""
        return self.base_mi_policy(field) in _NON_BLOCKING_BASE_MI

    def required_for_capabilities(self, field: str) -> List[str]:
        return list(self.field_policy(field).get("required_for_capabilities", []) or [])

    def reporting_date_period_inference_allowed(self) -> bool:
        return bool((self.spec.get("reporting_date", {}) or {}).get(
            "allow_period_inference", True))

    # -- defaults & derivations (all return auditable records) -------------- #

    def default_record(self, field: str) -> Optional[DerivationRecord]:
        pol = self.field_policy(field)
        if str(pol.get("base_mi")) != POLICY_DEFAULTED:
            return None
        value = pol.get("default_value", pol.get("default_rule", ""))
        return DerivationRecord(
            field=_norm(field), method="default", value=value,
            source="product_profile_default", confidence=0.90,
            rationale=pol.get("rationale", ""), profile_id=self.profile_id,
        )

    def not_applicable_record(self, field: str) -> Optional[DerivationRecord]:
        pol = self.field_policy(field)
        if str(pol.get("base_mi")) != POLICY_NOT_APPLICABLE:
            return None
        return DerivationRecord(
            field=_norm(field), method="not_applicable", value=None,
            source="product_profile_applicability", confidence=0.95,
            rationale=pol.get("rationale", ""), profile_id=self.profile_id,
        )

    def derive_current_outstanding_balance(
        self, row: Dict[str, Any]
    ) -> Optional[DerivationRecord]:
        """Equity-release proxy: current outstanding balance == current loan balance
        because interest capitalises. Returns a record only when a current-balance
        source value is present (never fabricated)."""
        pol = self.field_policy("current_outstanding_balance")
        if str(pol.get("base_mi")) != POLICY_DERIVED:
            return None
        sources = [pol.get("derivation_source", "current_principal_balance")]
        sources += list(pol.get("derivation_source_alternatives", []) or [])
        norm_row = {_norm(k): v for k, v in row.items()}
        for src in sources:
            val = norm_row.get(_norm(src))
            if val not in (None, ""):
                return DerivationRecord(
                    field="current_outstanding_balance", method="from_field",
                    value=val, source=_norm(src), confidence=0.92,
                    rationale=pol.get("rationale", ""), profile_id=self.profile_id,
                )
        return None

    def derive_funded_status(
        self, artefact_role: str, source_status: Any = None
    ) -> Optional[DerivationRecord]:
        """Funded status from the funded-loan-extract artefact role unless a source
        status value clearly contradicts it. Documents the derivation source."""
        pol = self.field_policy("funded_status")
        if str(pol.get("base_mi")) != POLICY_DERIVED:
            return None
        funded_roles = {_norm(r) for r in
                        (pol.get("funded_when_artefact_role_in", []) or [])}
        if _norm(artefact_role) not in funded_roles:
            return None
        contradicting = [t for t in (pol.get("contradicting_status_tokens", []) or [])]
        status_norm = str(source_status or "").strip().lower()
        if status_norm and any(t in status_norm for t in contradicting):
            return DerivationRecord(
                field="funded_status", method="from_source_status",
                value=status_norm, source="source_status_column", confidence=0.9,
                rationale=("source status contradicts the funded-extract role; "
                           "deferring to the source status value"),
                profile_id=self.profile_id,
            )
        return DerivationRecord(
            field="funded_status", method="from_artefact_role", value="funded",
            source=f"artefact_role:{_norm(artefact_role)}", confidence=0.88,
            rationale=pol.get("rationale", ""), profile_id=self.profile_id,
        )

    def derive_months_on_book(
        self, origination_date: Any, reporting_date: Any
    ) -> Optional[DerivationRecord]:
        """Whole months between origination_date and reporting_date."""
        pol = self.field_policy("months_on_book")
        if str(pol.get("base_mi")) != POLICY_DERIVED:
            return None
        od, rd = _parse_iso(origination_date), _parse_iso(reporting_date)
        if not od or not rd:
            return None
        months = (rd.year - od.year) * 12 + (rd.month - od.month)
        if rd.day < od.day:
            months -= 1
        months = max(months, 0)
        return DerivationRecord(
            field="months_on_book", method="from_dates", value=months,
            source="origination_date,reporting_date", confidence=0.93,
            rationale=pol.get("rationale",
                              "months between origination_date and reporting_date"),
            profile_id=self.profile_id,
        )

    def derive_number_of_borrowers(
        self, row: Dict[str, Any]
    ) -> Optional[DerivationRecord]:
        """Count distinct borrower/person/applicant fields present in the row.

        NEVER derives from a unique loan-id count — forbidden source tokens are
        excluded explicitly. Returns ``None`` (non-blocking) when no borrower
        evidence exists rather than fabricating a value.
        """
        pol = self.field_policy("number_of_borrowers")
        if str(pol.get("base_mi")) not in (POLICY_OPTIONAL, POLICY_DERIVED):
            return None
        want = [t for t in (pol.get("derivation_source_tokens", []) or [])]
        forbid = {_norm(t) for t in (pol.get("forbid_derivation_from", []) or [])}
        present = 0
        used: List[str] = []
        for k, v in row.items():
            nk = _norm(k)
            if nk in forbid:
                continue                      # never count from loan-id-like fields
            if any(t in nk for t in want) and v not in (None, ""):
                present += 1
                used.append(nk)
        if present == 0:
            return None
        return DerivationRecord(
            field="number_of_borrowers", method="from_borrower_fields",
            value=present, source=",".join(sorted(used)), confidence=0.7,
            rationale=pol.get("rationale", ""), profile_id=self.profile_id,
        )


# Mapping from profile base-MI policy -> coverage overlay status consumed by
# target_coverage._classify (all are non-blocking for base MI).
_POLICY_TO_OVERLAY_STATUS = {
    POLICY_NOT_APPLICABLE: "not_applicable",
    POLICY_DERIVED: "derived",
    POLICY_DEFAULTED: "defaulted",
    POLICY_OPTIONAL: "optional_for_mi",
}


def profile_overlay_rules(profile: ProductProfile) -> Dict[str, Dict[str, Any]]:
    """Express a profile's field policies as coverage-overlay rules.

    The returned ``{field: rule}`` map is shape-compatible with the existing MI
    applicability overlay consumed by ``target_coverage``: each rule carries
    ``coverage_status_if_no_source`` (non-blocking) plus rationale and any default
    rule, so a no-source field the profile marks not_applicable / derived /
    defaulted / optional is not reported as ``missing_required``. ``required``
    fields are intentionally omitted (they keep generic blocking behaviour).
    """
    rules: Dict[str, Dict[str, Any]] = {}
    for field, pol in profile.field_policies.items():
        base = str(pol.get("base_mi", POLICY_REQUIRED))
        status = _POLICY_TO_OVERLAY_STATUS.get(base)
        if not status:
            continue
        rule: Dict[str, Any] = {
            "field": field,
            "coverage_status_if_no_source": status,
            "blocking": False,
            "reason": pol.get("rationale", ""),
            "applicability_status": ("not_applicable"
                                     if base == POLICY_NOT_APPLICABLE else "applicable"),
            "product_profile_id": profile.profile_id,
        }
        default_rule = pol.get("default_rule") or pol.get("default_value")
        if default_rule:
            rule["default_rule"] = default_rule
        rules[field] = rule
    return rules


def _parse_iso(value: Any) -> Optional[date]:
    s = str(value or "").strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        if not (1 <= mo <= 12 and 1 <= d <= calendar.monthrange(y, mo)[1]):
            return None
        return date(y, mo, d)
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------- #
# Loading & resolution
# --------------------------------------------------------------------------- #

def load_product_profiles(
    path: Optional[str | Path] = None,
) -> Dict[str, ProductProfile]:
    """Load all configured product profiles, keyed by profile_id."""
    p = Path(path) if path else _PROFILES_PATH
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults", {}) or {}
    profiles: Dict[str, ProductProfile] = {}
    for pid, spec in (data.get("profiles", {}) or {}).items():
        profiles[pid] = ProductProfile(pid, spec or {}, defaults)
    return profiles


def configured_capabilities(path: Optional[str | Path] = None) -> List[str]:
    p = Path(path) if path else _PROFILES_PATH
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return list(data.get("capabilities", []) or [])


def _match_score(profile: ProductProfile, context: Dict[str, Any]) -> tuple[float, List[str]]:
    """Deterministic evidence score for a profile against the run context."""
    match = profile.spec.get("match", {}) or {}
    evidence: List[str] = []
    score = 0.0

    asset = _norm(context.get("asset_class"))
    ptype = _norm(context.get("product_type"))
    asset_opts = {_norm(a) for a in (match.get("asset_class", []) or [])}
    ptype_opts = {_norm(a) for a in (match.get("product_type", []) or [])}

    if asset and asset in asset_opts:
        score += 0.6
        evidence.append(f"asset_class={asset}")
    if ptype and ptype in ptype_opts:
        score += 0.25
        evidence.append(f"product_type={ptype}")

    # Soft signal tokens found in the context rationale / supporting evidence.
    blob_parts: List[str] = [str(context.get("rationale", ""))]
    blob_parts += [str(x) for x in (context.get("supporting_evidence", []) or [])]
    blob = " ".join(blob_parts).lower()
    tokens = [t.lower() for t in (match.get("signal_tokens", []) or [])]
    hits = [t for t in tokens if t and t in blob]
    if hits:
        score += min(0.3, 0.1 * len(hits))
        evidence.append("signals=" + ",".join(hits[:5]))

    # Detector may already carry its own confidence; blend gently (never above 1).
    det_conf = context.get("confidence")
    if asset in asset_opts and isinstance(det_conf, (int, float)):
        score = max(score, min(1.0, 0.5 * score + 0.5 * float(det_conf) + 0.1))

    return round(min(score, 1.0), 4), evidence


def resolve_product_profile(
    context: Optional[Dict[str, Any]] = None,
    *,
    explicit_profile_id: str = "",
    profiles: Optional[Dict[str, ProductProfile]] = None,
    profiles_path: Optional[str | Path] = None,
) -> ResolvedProfile:
    """Resolve which product profile applies.

    Priority:
      1. ``explicit_profile_id`` (config/operator) -> trusted, applied outright.
      2. best deterministic evidence match:
           * >= apply_confidence       -> applied, evidence recorded;
           * >= confirm_confidence      -> proposed for confirmation (NOT applied);
           * otherwise                  -> no profile (generic stricter behaviour).
    """
    context = context or {}
    profiles = profiles if profiles is not None else load_product_profiles(profiles_path)

    if explicit_profile_id:
        prof = profiles.get(explicit_profile_id)
        if prof is not None:
            return ResolvedProfile(
                profile_id=explicit_profile_id, decision=DECISION_EXPLICIT,
                confidence=1.0, applied=True, needs_confirmation=False,
                rationale="Profile set explicitly by config/operator; trusted.",
                evidence=["explicit_selection"], profile=prof,
            )
        # Explicit id that does not exist -> surface, do not guess.
        return ResolvedProfile(
            profile_id=explicit_profile_id, decision=DECISION_NONE, confidence=0.0,
            applied=False, needs_confirmation=True,
            rationale=f"Explicit profile '{explicit_profile_id}' is not configured.",
            evidence=[], profile=None,
        )

    best: Optional[ProductProfile] = None
    best_score = 0.0
    best_evidence: List[str] = []
    for prof in profiles.values():
        sc, ev = _match_score(prof, context)
        if sc > best_score:
            best, best_score, best_evidence = prof, sc, ev

    if best is None or best_score <= 0.0:
        return ResolvedProfile(decision=DECISION_NONE, rationale="No profile matched the context.")

    if best_score >= best.apply_confidence:
        return ResolvedProfile(
            profile_id=best.profile_id, decision=DECISION_DETECTED,
            confidence=best_score, applied=True, needs_confirmation=False,
            rationale=(f"Detected '{best.profile_id}' with confidence {best_score} "
                       f">= apply threshold {best.apply_confidence}."),
            evidence=best_evidence, profile=best,
        )
    if best_score >= best.confirm_confidence:
        return ResolvedProfile(
            profile_id=best.profile_id, decision=DECISION_NEEDS_CONFIRMATION,
            confidence=best_score, applied=False, needs_confirmation=True,
            rationale=(f"Detected '{best.profile_id}' with confidence {best_score} "
                       f"in the confirm band [{best.confirm_confidence}, "
                       f"{best.apply_confidence}); proposing for confirmation."),
            evidence=best_evidence, profile=best,
        )
    return ResolvedProfile(
        profile_id=best.profile_id, decision=DECISION_NONE, confidence=best_score,
        applied=False, needs_confirmation=False,
        rationale=(f"Best match '{best.profile_id}' confidence {best_score} below "
                   f"confirm threshold {best.confirm_confidence}; generic behaviour."),
        evidence=best_evidence, profile=None,
    )
