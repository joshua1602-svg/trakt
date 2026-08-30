"""
canonical_derivations.py
========================

Deterministic, configuration-driven canonical field derivations for the
Transformation Agent.

A *derivation* computes one canonical field from another canonical field using a
rule from a fixed library. It is the correct home for relationships that are
properties of the canonical model itself — "the protected-equity flag is a view
of the protected-equity percentage" — rather than properties of any one client,
portfolio or source file.

Design rules this module holds to:

  * **Deterministic only.** Every rule is a pure function of the parsed canonical
    value. No inference, no LLM, no locale guessing.
  * **Null in, null out.** A derivation never manufactures a value from an absent
    source. Preserving nulls is what keeps "we don't know" distinguishable from
    "we know it is N".
  * **Never destructive.** The source field is left exactly as parsed; the
    derived field is written to its own column.
  * **Always attributed.** Each application returns lineage naming the rule and
    the field it was derived from, so a derived value can never be mistaken for a
    sourced one.

Rules are declared in ``config/system/canonical_derivations.yaml``; this module
owns the rule library and the application mechanics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "system" / "canonical_derivations.yaml"

#: Canonical boolean representation used across the canonical model.
FLAG_TRUE = "Y"
FLAG_FALSE = "N"


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    if value is pd.NA:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in ("", "nan", "nat", "none", "null", "<na>")


class DerivationError(ValueError):
    """A source value that the rule cannot deterministically interpret."""


# --------------------------------------------------------------------------- #
# Rule library
# --------------------------------------------------------------------------- #

def _positive_number_to_flag(value: Any) -> Optional[str]:
    """``> 0 -> "Y"``, ``<= 0 -> "N"``, blank -> ``None``.

    The magnitude is irrelevant to the flag, so this works on either percentage
    scale without needing to know which one is in play — but the value must be a
    number. A non-numeric value is an error, never a silent ``N``.
    """
    if _is_blank(value):
        return None
    try:
        number = float(str(value).strip().replace("%", "").replace(",", ""))
    except (TypeError, ValueError) as exc:
        raise DerivationError(f"not a number: {value!r}") from exc
    if math.isnan(number):
        return None
    return FLAG_TRUE if number > 0 else FLAG_FALSE


def _presence_to_flag(value: Any) -> Optional[str]:
    """``any value -> "Y"``, blank -> ``"N"``."""
    return FLAG_FALSE if _is_blank(value) else FLAG_TRUE


def _ratio_percentage(numerator: Any, denominator: Any) -> Optional[str]:
    """``numerator / denominator`` as a percentage, to four decimal places.

    Null in, null out: a missing balance or a missing valuation yields nothing.
    A denominator of zero or less yields nothing either — a ratio against no
    valuation is not a small number, it is an unknown one.
    """
    if _is_blank(numerator) or _is_blank(denominator):
        return None
    try:
        num = float(str(numerator).replace(",", "").strip())
        den = float(str(denominator).replace(",", "").strip())
    except (TypeError, ValueError):
        raise DerivationError(f"not numeric: {numerator!r} / {denominator!r}")
    if den <= 0 or math.isnan(num) or math.isnan(den):
        return None
    return f"{(num / den) * 100.0:.4f}"


RULES: Dict[str, Callable[[Any], Optional[str]]] = {
    "positive_number_to_flag": _positive_number_to_flag,
    "presence_to_flag": _presence_to_flag,
}

#: Rules that read TWO canonical columns. Declared separately because the
#: single-source mechanics above are the common case and stay unchanged.
RULES_MULTI: Dict[str, Callable[..., Optional[str]]] = {
    "ratio_percentage": _ratio_percentage,
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_derivations(config_path: str | Path = "") -> Dict[str, Dict[str, Any]]:
    """Load ``{target_canonical_field: {from, rule, description}}``.

    A missing / unreadable config yields an empty map: derivations are additive,
    so their absence must never break a run.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — additive layer, never fatal
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for target, spec in (data.get("derivations", {}) or {}).items():
        spec = spec or {}
        raw_from = spec.get("from", "")
        sources = ([str(x).strip() for x in raw_from if str(x).strip()]
                   if isinstance(raw_from, (list, tuple))
                   else [str(raw_from or "").strip()])
        sources = [x for x in sources if x]
        rule = str(spec.get("rule", "") or "").strip()
        multi = rule in RULES_MULTI
        if not target or not sources or (rule not in RULES and not multi):
            continue
        if multi and len(sources) != 2:
            continue
        out[str(target).strip()] = {
            "from": sources[0] if not multi else list(sources),
            "sources": list(sources),
            "rule": rule,
            "multi": multi,
            "description": str(spec.get("description", "") or "").strip(),
            "preserve_source": bool(spec.get("preserve_source", True)),
            # A derivation calculates what the lender did not supply. Where the
            # lender DID supply it, the real value wins: the derivation fills
            # blanks only. Existing single-source entries keep their behaviour
            # of rewriting the whole column, because there the derived field is
            # by definition a view of its source.
            "fill_blank_only": bool(spec.get("fill_blank_only", False)),
        }
    return out


def derived_field_parents(config_path: str | Path = "") -> Dict[str, str]:
    """``{derived_canonical_field: source_canonical_field}`` for guard checks.

    A multi-source derivation reports its first source; the guard only asks
    whether a field is derived and from where, not how many inputs it took.
    """
    return {t: (s["sources"][0] if s.get("multi") else s["from"])
            for t, s in load_derivations(config_path).items()}


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

def apply_derivations(
    df: pd.DataFrame,
    derivations: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    config_path: str | Path = "",
) -> Dict[str, Dict[str, Any]]:
    """Materialise every configured derivation whose source column is present.

    Returns ``{target_field: result}`` where ``result`` carries ``applied``,
    ``rule``, ``derived_from``, the value counts, and any ``failures`` (source
    values the rule could not interpret — surfaced, never guessed).

    The derived column is written in full: it is a function of the source column,
    so any pre-existing content is a stale copy of the same relationship. The
    SOURCE column is never touched.
    """
    derivations = (load_derivations(config_path) if derivations is None
                   else derivations)
    results: Dict[str, Dict[str, Any]] = {}

    for target, spec in derivations.items():
        rule_name = spec["rule"]
        if spec.get("multi"):
            results[target] = _apply_multi(df, target, spec)
            continue
        source = spec["from"]
        rule = RULES[rule_name]
        if source not in df.columns:
            results[target] = {
                "applied": False, "rule": rule_name, "derived_from": source,
                "reason": "source_field_absent",
            }
            continue

        values: List[Optional[str]] = []
        failures: List[str] = []
        counts = {FLAG_TRUE: 0, FLAG_FALSE: 0, "null": 0}
        for raw in df[source].tolist():
            try:
                flag = rule(raw)
            except DerivationError:
                if len(failures) < 5:
                    failures.append(str(raw))
                flag = None
            values.append(flag)
            counts["null" if flag is None else flag] += 1

        df[target] = pd.Series(values, index=df.index, dtype="object")
        results[target] = {
            "applied": True,
            "rule": rule_name,
            "derived_from": source,
            "description": spec.get("description", ""),
            "value_counts": counts,
            "failure_count": sum(1 for v, raw in zip(values, df[source].tolist())
                                 if v is None and not _is_blank(raw)),
            "sample_failures": failures,
        }
    return results


def _apply_multi(df: pd.DataFrame, target: str,
                 spec: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a two-source rule (currently only ``ratio_percentage``)."""
    rule_name = spec["rule"]
    rule = RULES_MULTI[rule_name]
    sources: List[str] = list(spec["sources"])
    missing = [c for c in sources if c not in df.columns]
    if missing:
        return {"applied": False, "rule": rule_name,
                "derived_from": ", ".join(sources),
                "reason": "source_field_absent",
                "missing_sources": missing}

    fill_blank_only = bool(spec.get("fill_blank_only"))
    existing = (df[target] if target in df.columns else None)
    values: List[Optional[str]] = []
    failures: List[str] = []
    derived_count = kept_count = null_count = 0
    for idx in range(len(df)):
        supplied = None if existing is None else existing.iloc[idx]
        if fill_blank_only and not _is_blank(supplied):
            values.append(str(supplied))
            kept_count += 1
            continue
        try:
            value = rule(df[sources[0]].iloc[idx], df[sources[1]].iloc[idx])
        except DerivationError as exc:
            if len(failures) < 5:
                failures.append(str(exc))
            value = None
        values.append(value)
        if value is None:
            null_count += 1
        else:
            derived_count += 1

    df[target] = pd.Series(values, index=df.index, dtype="object")
    return {
        "applied": True,
        "rule": rule_name,
        "derived_from": ", ".join(sources),
        "description": spec.get("description", ""),
        "value_counts": {"derived": derived_count,
                         "kept_supplied_value": kept_count,
                         "null": null_count},
        "failure_count": len(failures),
        "sample_failures": failures,
    }
