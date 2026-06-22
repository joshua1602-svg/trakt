"""The MI dataset metadata contract exposed to React (via /health) and reused by
the static review generator.

There is ONE prepared MI dataset (``funded_prep.prepare_funded_mi_dataset``) and
ONE metadata contract built here from the single per-field profile in
``mi_agent.mi_dataset_profile``. For every prepared field it carries:

  field, semantic type, storage scale (percent_fraction / percent_points),
  display format, non-null count, numeric-parse count, dimension availability,
  metric availability, source field / derivation basis, and bucket source.

The frontend reads this contract instead of guessing field meaning, numeric
parsing, bucket availability, or display scale.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from mi_agent.mi_dataset_profile import profile_dataset


def _bucket_source_map() -> Dict[str, str]:
    """``{bucket_dimension: source_field}`` from the single bucket config."""
    try:
        from analytics_lib.buckets import load_bucket_config
        out: Dict[str, str] = {}
        for spec in (load_bucket_config().get("buckets") or {}).values():
            tgt = spec.get("semantic_field") or spec.get("target")
            src = spec.get("source_field")
            if tgt and src:
                out[str(tgt)] = str(src)
        return out
    except Exception:
        return {}


def _derivation_index(prep_report: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{field: derivation_basis}`` from a funded_prep report (LTV/age/etc.)."""
    if not prep_report:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for basis in prep_report.get("ltv_derivation_basis", []) or []:
        tgt = basis.get("target")
        if tgt:
            out[tgt] = {"method": basis.get("method"),
                        "source_fields": basis.get("source_fields"),
                        "confidence": basis.get("confidence")}
    for f in prep_report.get("derived_fields", []) or []:
        out.setdefault(f, {"method": "derived", "source_fields": None})
    return out


def build_dataset_contract(
    df: pd.DataFrame,
    semantics: dict,
    prep_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the per-field MI dataset contract from the single dataset profile."""
    profile = profile_dataset(df, semantics)
    buckets = _bucket_source_map()
    derivations = _derivation_index(prep_report)

    fields: List[Dict[str, Any]] = []
    dims_available: List[str] = []
    dims_missing: List[Dict[str, str]] = []
    metrics_available: List[str] = []

    for name, p in profile["fields"].items():
        derivation = derivations.get(name) or derivations.get(p["canonical_field"])
        entry = {
            "field": name,
            "semantic_type": p["semantic_type"],
            "storage_scale": p["storage_scale"],
            "display_format": p["display_format"],
            "non_null": p["non_null"],
            "numeric_parse": p["numeric_parse"],
            "dimension_available": p["dimension_available"],
            "metric_available": p["metric_available"],
            "source_field": p["canonical_field"],
            "derivation_basis": derivation,
            "bucket_source": buckets.get(name),
        }
        fields.append(entry)
        if p["dimension_available"]:
            dims_available.append(name)
        elif p["semantic_type"] == "bucket" or p["role"] in ("dimension", "date", "flag"):
            dims_missing.append({"dimension": name, "reason": "no_values_after_preparation",
                                 "detail": f"{name!r} present but has no non-blank values"})
        if p["metric_available"]:
            metrics_available.append(name)

    return {
        "fields": fields,
        "display_hints": profile["display_hints"],
        "dimensions_available": sorted(dims_available),
        "dimensions_missing": dims_missing,
        "metrics_available": sorted(metrics_available),
    }


def summary_numbers(df: pd.DataFrame, semantics: dict) -> Dict[str, Any]:
    """Shared headline numbers for the static review generator — uses the same
    numeric coercion as the dataset (never a separate parser)."""
    from analytics_lib.numeric import coerce_numeric
    out: Dict[str, Any] = {"loan_count": int(len(df))}
    for col in ("current_outstanding_balance", "current_principal_balance"):
        if col in df.columns:
            out["balance_field"] = col
            out["balance_sum"] = float(coerce_numeric(df[col]).sum())
            break
    return out
