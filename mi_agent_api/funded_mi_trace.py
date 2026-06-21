"""Trace funded-MI target fields end-to-end across the promoted run artefacts.

For each target canonical field, walk: raw source discovery (05_mapping_candidates)
-> source period eligibility (04c) -> registry scope -> central tape promotion
(18_central_lender_tape.csv) -> MI prep dimension (funded_prep) -> React (/health).
Emits a reason-coded status per field so "missing" is never silent.

Reason codes: ``promoted`` | ``dimension_available`` | ``derivation_inputs_missing``
| ``raw_not_found`` | ``mapped_but_out_of_scope`` | ``source_period_ineligible``
| ``join_failed`` | ``not_in_central_tape`` | ``not_consumed_by_mi_prep``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .funded_prep import _DIM_SPEC, _LTV_INPUTS, prepare_funded_mi_dataset

# Target canonical field -> the MI dimension it backs (or "" for a raw value).
TARGETS: List[Dict[str, str]] = [
    {"canonical_field": "youngest_borrower_age", "dimension": "age_bucket"},
    {"canonical_field": "geographic_region_obligor", "dimension": "geographic_region_obligor"},
    {"canonical_field": "collateral_geography", "dimension": "geographic_region_obligor"},
    {"canonical_field": "current_valuation_amount", "dimension": "ltv_bucket"},
    {"canonical_field": "current_loan_to_value", "dimension": "ltv_bucket"},
    {"canonical_field": "original_valuation_amount", "dimension": "original_ltv_bucket"},
    {"canonical_field": "original_principal_balance", "dimension": "original_ltv_bucket"},
    {"canonical_field": "original_loan_to_value", "dimension": "original_ltv_bucket"},
    {"canonical_field": "origination_channel", "dimension": "origination_channel"},
    {"canonical_field": "broker_channel", "dimension": "origination_channel"},
]


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _enrichment_fields() -> set:
    try:
        import yaml
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "system"
             / "onboarding_agent.yaml").read_text(encoding="utf-8")) or {}
        return set((cfg.get("central_lender_tape", {}) or {}).get("mi_enrichment_fields", []) or [])
    except Exception:
        return set()


def trace(project_dir: str | Path, central_tape_path: str | Path) -> List[Dict[str, Any]]:
    project_dir = Path(project_dir)
    central = pd.read_csv(central_tape_path) if Path(central_tape_path).exists() else pd.DataFrame()
    cols = set(central.columns)

    mapping = _load_json(project_dir / "05_mapping_candidates.json") or []
    map_by_canon: Dict[str, Dict[str, Any]] = {}
    for m in mapping:
        canon = m.get("candidate_canonical_field", "")
        if canon and canon not in map_by_canon:
            map_by_canon[canon] = m

    elig_doc = _load_json(project_dir / "04c_source_period_eligibility.json") or {}
    elig_rows = elig_doc.get("rows", []) if isinstance(elig_doc, dict) else []
    elig_by_file = {}
    for r in elig_rows:
        if r.get("output_domain") == "central_lender_tape":
            elig_by_file[r.get("source_file", "")] = r

    enrichment = _enrichment_fields()
    prep_df, report = (prepare_funded_mi_dataset(central) if not central.empty else (central, {}))
    available = set(report.get("dimensions_available", []))
    miss_by_dim = {m["dimension"]: m for m in report.get("missing_dimensions", [])}
    basis_by_target = {b["target"]: b for b in report.get("ltv_derivation_basis", [])}

    rows: List[Dict[str, Any]] = []
    for t in TARGETS:
        canon = t["canonical_field"]
        dim = t["dimension"]
        m = map_by_canon.get(canon, {})
        src_file = m.get("source_file", "")
        elig = elig_by_file.get(src_file, {})
        in_tape = canon in cols
        non_null = int(central[canon].notna().sum()) if in_tape else 0
        period_eligible = bool(elig.get("is_period_eligible")) if elig else None

        status, reason = _classify(
            canon, dim, mapped=bool(m), src_file=src_file, period_eligible=period_eligible,
            in_tape=in_tape, non_null=non_null, in_enrichment=(canon in enrichment),
            available=available, miss_by_dim=miss_by_dim, basis=basis_by_target.get(canon))

        rows.append({
            "canonical_field": canon,
            "dimension": dim,
            "raw_source_file": src_file,
            "raw_source_column": m.get("source_column", ""),
            "mapping_method": m.get("method", ""),
            "source_period_eligible": period_eligible,
            "in_mi_enrichment_config": canon in enrichment,
            "in_central_tape": in_tape,
            "non_null_count": non_null,
            "dimension_available": dim in available,
            "status": status,
            "reason": reason,
        })
    return rows


def _classify(canon, dim, *, mapped, src_file, period_eligible, in_tape, non_null,
              in_enrichment, available, miss_by_dim, basis) -> tuple:
    if in_tape and non_null > 0:
        if dim in available:
            return "available", "promoted_and_dimension_available"
        return "promoted", "in_central_tape"
    # LTV dims can still be available via derivation from balance/valuation.
    if canon in _LTV_INPUTS and dim in available:
        return "available", "dimension_available_via_derivation"
    if dim in available:
        return "available", "dimension_available"
    if canon in _LTV_INPUTS:
        mm = miss_by_dim.get(dim, {})
        return "unavailable", mm.get("reason", "derivation_inputs_missing")
    if not mapped:
        return "unavailable", "raw_not_found"
    if period_eligible is False:
        return "unavailable", "source_period_ineligible"
    if not in_tape:
        # mapped + eligible but absent: out-of-scope (not enriched) vs join failure.
        return ("unavailable", "join_failed") if in_enrichment else \
               ("unavailable", "mapped_but_out_of_scope")
    return "unavailable", "not_in_central_tape"


def render_markdown(rows: List[Dict[str, Any]], *, run_id: str = "", client_id: str = "") -> str:
    head = (f"# Funded MI — missing-dimension trace ({client_id}/{run_id})\n\n"
            "**Framing.** MI availability is decided by the active MI target contract + "
            "MI enrichment configuration (`central_lender_tape.mi_enrichment_fields`) and the "
            "source fields actually present — NOT by the registry category/layer. A field that "
            "is regulatory/collateral in the registry can still be an MI dimension (this is MI "
            "contract enrichment using source fields that may also be relevant to regulatory "
            "reporting — not contract leakage).\n\n"
            "raw source → mapping → MI contract/scope → period eligibility → entity-key join → "
            "central tape → MI prep → React health.\n\n"
            "| canonical_field | dimension | source file:col | period_eligible | "
            "in_enrich_cfg | in_tape (non-null) | dim_available | status | reason |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        body += (f"| `{r['canonical_field']}` | `{r['dimension']}` | "
                 f"{r['raw_source_file']}:{r['raw_source_column'] or '—'} | "
                 f"{r['source_period_eligible']} | {r['in_mi_enrichment_config']} | "
                 f"{r['in_central_tape']} ({r['non_null_count']}) | "
                 f"{r['dimension_available']} | **{r['status']}** | `{r['reason']}` |\n")
    legend = ("\n## Reason codes\n"
              "- `raw_not_found` — no source column maps to the canonical field.\n"
              "- `mapped_but_out_of_scope` — mapped but excluded by registry scope and not in "
              "`central_lender_tape.mi_enrichment_fields`.\n"
              "- `source_period_ineligible` — the mapping source is not period-eligible for the run.\n"
              "- `join_failed` — eligible + in enrichment config but did not join the funded universe.\n"
              "- `not_in_central_tape` — absent from the promoted tape for another reason.\n"
              "- `derivation_inputs_missing` — an LTV bucket whose balance/valuation inputs are absent.\n"
              "- `promoted` / `available` — field reached the tape / its dimension is prepared.\n")
    return head + body + legend


def trace_to_file(project_dir: str | Path, central_tape_path: str | Path, out_path: str | Path,
                  *, run_id: str = "", client_id: str = "") -> str:
    rows = trace(project_dir, central_tape_path)
    md = render_markdown(rows, run_id=run_id, client_id=client_id)
    Path(out_path).write_text(md, encoding="utf-8")
    return str(out_path)


def _main(argv: List[str]) -> int:
    """CLI: trace one promoted run. Prints the reason-coded markdown table.

      python -m mi_agent_api.funded_mi_trace <project_dir> <central_tape.csv>

    ``project_dir`` is the onboarding output dir holding 05_mapping_candidates.json
    and 04c_source_period_eligibility.json; ``central_tape`` is the promoted
    18_central_lender_tape.csv. Also surfaces the central tape's own enrichment
    diagnostics (18f) when present, so collateral join breaks are explicit.
    """
    if len(argv) < 3:
        print(_main.__doc__)
        return 2
    project_dir, tape = argv[1], argv[2]
    print(render_markdown(trace(project_dir, tape)))
    dbg = _load_json(Path(project_dir) / "18f_central_universe_debug.json") or \
        _load_json(Path(tape).resolve().parent / "18f_central_universe_debug.json")
    if isinstance(dbg, dict) and dbg.get("enrichment_field_diagnostics"):
        print("\n## Central-tape enrichment diagnostics (18f)\n")
        for d in dbg["enrichment_field_diagnostics"]:
            print(f"- `{d.get('canonical_field')}`: **{d.get('status')}** "
                  f"reason=`{d.get('reason', d.get('status'))}` "
                  f"populated={d.get('populated_rows')}/{d.get('universe_rows')}; "
                  f"candidates={d.get('candidate_source_columns')}")
            for c in d.get("candidate_join_detail", []):
                print(f"    - {c}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv))
