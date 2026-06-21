"""
central_tape_builder.py
=======================

PART 7 / PART 8 — build consolidated raw input tapes from APPROVED onboarding
decisions.

This is a controlled, domain-based consolidator — *not* a generic "merge
anything" engine. The files are containers; the domains are what matter. Loan,
borrower and collateral fields can all be sourced from the same combined master
tape, so the builder never requires a separate collateral file.

Outputs (loan-level central lender tape)::
    18_central_lender_tape.csv      one row per funded / live loan
    18b_central_tape_lineage.csv    one row per populated field
    18c_central_tape_gaps.csv       unmapped / conflicting / missing fields
    18d_central_tape_summary.json

Outputs (pipeline / origination — applications need no funded loan id)::
    18a_central_pipeline_tape.csv
    18a_central_pipeline_lineage.csv
    18a_central_pipeline_summary.json

Selection logic per canonical field (PART 7):
  1. approved mapping override
  2. approved source precedence rule (13_source_precedence_rules.yaml)
  3. unambiguous in-scope mapping candidate
  4. regulatory-preference ambiguity-selected candidate
  5. otherwise leave blank and raise a gap

Material conflicts are never silently resolved: differing values across sources
with no approved precedence become conflict gaps.

Audit note (deterministic-first): this builder does NOT invent source-to-canonical
mappings. Every canonical field source comes from the existing mapping artefacts
(12_approved_mapping_overrides + 05_mapping_candidates, which already encode the
alias/registry/context/ambiguity decisions and are traced in 05c_mapping_trace).
The only header-name logic here is loan/application KEY detection and pipeline
field extraction (pipeline data lives outside the loan registry); no canonical
field is ever guessed by header name.

Input order: approved mapping overrides -> approved source precedence rules ->
approved enum decisions -> 05_mapping_candidates (which reflect 05b ambiguity
resolution) -> domain coverage.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from engine.gate_1_alignment.semantic_alignment import load_field_registry
from . import domain_coverage as dc
from . import source_period_eligibility as spe
from .field_scope import resolve_field_scope
from .mode_policy import load_mode_policy

# Loan-level domains that belong in the central lender tape. Cashflow is NOT a
# lender-tape domain (per-period schedule rows belong to the pipeline/cashflow
# extracts); however a *loan-domain* field such as current_principal_balance is
# still consolidated even when its authoritative source is the cashflow extract,
# because domain membership follows the canonical field, not the file.
_LENDER_DOMAINS = {dc.LOAN, dc.BORROWER, dc.COLLATERAL}

# Candidate loan-key column names (PART 7 primary key).
_LOAN_KEY_NAMES = [
    "loan_identifier", "loan_id", "loanid", "account_number", "account_no",
    "facility_id", "account_id",
]

_LENDER_LINEAGE_COLUMNS = [
    "loan_identifier", "canonical_field", "value", "source_file", "source_sheet",
    "source_column", "source_value", "mapping_method", "confidence", "domain",
    "source_precedence_applied", "validation_sources", "alternate_values",
    "conflict_status", "enum_decision_applied", "review_required", "notes",
]

_GAP_COLUMNS = [
    "gap_id", "severity", "mode", "domain", "canonical_field", "source_file",
    "source_column", "issue_type", "description", "candidate_actions", "blocking",
]

_PIPELINE_COLUMNS = [
    "application_id", "linked_loan_identifier", "broker_name", "pipeline_stage",
    "application_date", "expected_completion_date", "expected_funded_amount",
    "expected_ltv", "property_region", "property_post_code",
]

_PIPELINE_FIELD_PATTERNS: Dict[str, List[str]] = {
    "application_id": ["application_id", "pipeline_id", "case_id", "app_id", "application_no"],
    "linked_loan_identifier": ["linked_loan_id", "linked_loan_identifier",
                               "funded_loan_id", "loan_identifier", "loan_id"],
    "broker_name": ["broker_name", "broker", "intermediary"],
    "pipeline_stage": ["pipeline_stage", "application_stage", "stage", "status"],
    "application_date": ["application_date", "app_date", "offer_date"],
    "expected_completion_date": ["expected_completion_date", "completion_date",
                                 "expected_completion"],
    "expected_funded_amount": ["expected_funding_amount", "expected_funded_amount",
                               "requested_loan_amount", "expected_amount"],
    "expected_ltv": ["expected_ltv", "anticipated_ltv"],
    "property_region": ["property_region", "collateral_region", "region"],
    "property_post_code": ["property_post_code", "post_code", "postcode"],
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_df(file_path: str, sheet: str = "") -> Optional[pd.DataFrame]:
    p = Path(file_path)
    try:
        if p.suffix.lower() in (".xlsx", ".xls"):
            if sheet:
                xl = pd.ExcelFile(p)
                target = str(sheet).strip()
                match = next((sh for sh in xl.sheet_names if str(sh).strip() == target), None)
                if match is None:
                    match = next((sh for sh in xl.sheet_names
                                  if _norm(sh) == _norm(sheet)), None)
                return xl.parse(match if match is not None else xl.sheet_names[0])
            return pd.read_excel(p)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p, low_memory=False)
    except Exception:
        return None
    return None


def _norm_key(v: Any) -> str:
    """Normalise a loan-id key so it joins across sources regardless of how each
    file typed it (e.g. ``76034101.0`` from a float column == ``76034101``)."""
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "nat", "<na>"):
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


def _norm(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_")


def _alias_norm(text: str) -> str:
    try:
        from engine.gate_1_alignment.semantic_alignment import normalise_name
        return normalise_name(str(text))
    except Exception:
        return _norm(text)


def _alias_loan_columns() -> set:
    """Normalised headers (alias keys) that resolve to ``loan_identifier``."""
    try:
        from engine.gate_1_alignment.semantic_alignment import load_aliases_from_dir
        amap = load_aliases_from_dir(_CONFIG_DIR)
        return {k for k, v in amap.items() if v == "loan_identifier"}
    except Exception:
        return set()


def _loan_key_hints(project_dir: Path) -> Dict[Tuple[str, str], str]:
    """``{(file_name, sheet_name): column}`` of onboarding-detected loan-id keys.

    Uses 02_column_profiles (``likely_identifier``) so promotion joins on the same
    key column onboarding profiled, even when it is a non-standard header.
    """
    hints: Dict[Tuple[str, str], str] = {}
    rows = _load_json(Path(project_dir) / "02_column_profiles.json") or []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if str(r.get("likely_identifier", "")).strip().lower() in ("true", "1", "yes"):
            key = (r.get("file_name", ""), r.get("sheet_name", ""))
            col = r.get("source_column", "")
            if col and key not in hints:
                hints[key] = col
    return hints


def _resolve_key_column(
    df: pd.DataFrame, file_sheet: Tuple[str, str],
    loan_key_hints: Dict[Tuple[str, str], str], alias_loan_cols: set,
) -> Optional[str]:
    """Resolve the loan-id key column: onboarding hint -> alias -> name list."""
    hint = loan_key_hints.get(file_sheet) if loan_key_hints else None
    if hint:
        for c in df.columns:
            if _norm(c) == _norm(hint):
                return c
    if alias_loan_cols:
        for c in df.columns:
            if _alias_norm(c) in alias_loan_cols:
                return c
    return _find_key_column(df, _LOAN_KEY_NAMES)


def _find_key_column(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cols = {_norm(c): c for c in df.columns}
    for n in names:
        if n in cols:
            return cols[n]
    # substring fallback
    for n in names:
        for nc, orig in cols.items():
            if n in nc:
                return orig
    return None


def _values_match(a: Any, b: Any, tol: float = 0.01) -> bool:
    if a is None or b is None:
        return False
    sa, sb = str(a).strip(), str(b).strip()
    if sa == "" or sb == "":
        return False
    try:
        fa, fb = float(sa), float(sb)
        denom = max(abs(fa), abs(fb), 1.0)
        return abs(fa - fb) / denom <= tol
    except (ValueError, TypeError):
        return sa == sb


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return str(v).strip() == ""


# ---------------------------------------------------------------------------
# Source assembly
# ---------------------------------------------------------------------------


class _Source:
    """A (file, column) source for a canonical field, with its frame."""

    def __init__(self, file_name: str, file_path: str, column: str, sheet: str,
                 method: str, confidence: float, classification: str):
        self.file_name = file_name
        self.file_path = file_path
        self.column = column
        self.sheet = sheet
        self.method = method
        self.confidence = confidence
        self.classification = classification


def _collect_field_sources(
    mapping_candidates: List[Dict[str, Any]],
    overrides: Dict[str, Any],
    inventory_by_name: Dict[str, Dict[str, Any]],
    included_fields: set,
) -> Dict[str, List[_Source]]:
    """canonical_field -> ordered list of candidate sources (in-scope only)."""
    sources: Dict[str, List[_Source]] = {}
    seen: set = set()

    def add(file_name: str, column: str, canon: str, method: str, conf: float):
        if not canon or not file_name or not column:
            return
        if included_fields and canon not in included_fields:
            return
        key = (canon, file_name, column)
        if key in seen:
            return
        seen.add(key)
        inv = inventory_by_name.get(file_name, {})
        sources.setdefault(canon, []).append(
            _Source(
                file_name=file_name,
                file_path=inv.get("file_path", ""),
                column=column,
                sheet=inv.get("sheet_name", ""),
                method=method,
                confidence=float(conf or 0.0),
                classification=inv.get("classification", ""),
            )
        )

    # 1. Approved user overrides (highest priority — listed first).
    for o in (overrides or {}).get("user_overrides", []) or []:
        add(o.get("source_file", ""), o.get("source_column", ""),
            o.get("canonical_field", ""), o.get("method", "approved_override"),
            o.get("confidence", 1.0))
    # 2. Approved high-confidence mappings.
    for o in (overrides or {}).get("approved_high_confidence_mappings", []) or []:
        add(o.get("source_file", ""), o.get("source_column", ""),
            o.get("canonical_field", ""), o.get("method", "approved"),
            o.get("confidence", 0.92))
    # 3. Full deterministic mapping candidates (context hints etc.).
    for m in mapping_candidates or []:
        add(m.get("source_file", ""), m.get("source_column", ""),
            m.get("candidate_canonical_field", ""), m.get("method", ""),
            m.get("confidence", 0.0))
    return sources


_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "system"
_CONFIG_PATH = _CONFIG_DIR / "onboarding_agent.yaml"


# Coverage statuses (28a) the central tape consumes as RESOLVED selections.
_COV_SOURCE_MAPPED = {"source_mapped", "source_mapped_with_alternatives"}
_COV_CONSTANT = {"defaulted_value", "configured_static", "defaulted_ND"}
_COV_DERIVED = "derived"


def _coverage_selections(
    coverage_rows: List[Dict[str, Any]],
    inventory_by_name: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, "_Source"], Dict[str, Any], Dict[str, str]]:
    """Turn the resolved 28a coverage matrix into authoritative selections.

    Gate 4 already applied the artefact-role guardrail (funded over pipeline) and
    the product-profile proxy / reporting-date inference, so promotion consumes
    those decisions instead of re-discovering mappings. Returns:

    * ``forced``      — ``{canonical_field: _Source}`` (the single selected source);
    * ``constants``   — ``{canonical_field: value}`` (inferred/defaulted constants,
                        e.g. reporting_date / data_cut_off_date from run-id);
    * ``derive_from`` — ``{canonical_field: source_canonical_field}`` (profile
                        proxy, e.g. current_principal_balance <- current_outstanding_balance).
    """
    forced: Dict[str, "_Source"] = {}
    constants: Dict[str, Any] = {}
    derive_from: Dict[str, str] = {}
    for r in coverage_rows or []:
        canon = r.get("target_field", "")
        status = r.get("coverage_status", "")
        if not canon:
            continue
        if status in _COV_SOURCE_MAPPED:
            f = r.get("selected_source_file", "")
            c = r.get("selected_source_column", "")
            if f and c:
                inv = inventory_by_name.get(f, {})
                forced[canon] = _Source(
                    file_name=f, file_path=inv.get("file_path", ""), column=c,
                    sheet=r.get("selected_source_sheet", "") or inv.get("sheet_name", ""),
                    method="target_first_resolved",
                    confidence=float(r.get("selected_source_confidence") or 1.0),
                    classification=inv.get("classification", ""))
        elif status in _COV_CONSTANT and str(r.get("selected_value", "")).strip():
            constants[canon] = r.get("selected_value")
        elif status == _COV_DERIVED:
            dr = str(r.get("derivation_rule", "") or "")
            m = re.match(r"=\s*([a-z0-9_]+)", dr.strip())
            if m:
                derive_from[canon] = m.group(1)
    return forced, constants, derive_from



def _load_source_precedence_defaults() -> Dict[str, List[str]]:
    """PART 9 — domain -> ordered classification precedence (config-driven)."""
    try:
        cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return dict((cfg.get("source_precedence", {}) or {}).get("defaults_by_domain", {}) or {})
    except Exception:
        return {}


def _order_sources(
    canon: str,
    srcs: List[_Source],
    precedence: Dict[str, Any],
    registry_fields: Dict[str, Any],
    defaults_by_domain: Dict[str, List[str]],
) -> List[_Source]:
    """Order sources: approved precedence primary first, else domain default.

    The domain default (config/system/onboarding_agent.yaml) only chooses a
    sensible primary; it never *resolves* a material conflict (that still raises
    a gap unless an approved precedence rule exists).
    """
    rule = (precedence or {}).get(canon)
    if rule and rule.get("primary_source_file"):
        primary_file = rule["primary_source_file"]
        return sorted(srcs, key=lambda s: 0 if s.file_name == primary_file else 1)

    # Default ordering by the field's domain -> classification precedence.
    domains = dc.field_domains(canon, registry_fields.get(canon, {}))
    ranking: List[str] = []
    for d in (dc.LOAN, dc.COLLATERAL, dc.BORROWER, dc.CASHFLOW):
        if d in domains and d in defaults_by_domain:
            ranking = defaults_by_domain[d]
            break

    def rank(s: _Source) -> int:
        try:
            return ranking.index(s.classification)
        except ValueError:
            return len(ranking) + 1

    return sorted(srcs, key=rank)


# ---------------------------------------------------------------------------
# Central lender tape
# ---------------------------------------------------------------------------


def _build_lender_tape(
    mapping_candidates: List[Dict[str, Any]],
    overrides: Dict[str, Any],
    precedence: Dict[str, Any],
    enum_decisions: Dict[str, Any],
    inventory: List[Dict[str, Any]],
    field_scope: Any,
    registry_fields: Dict[str, Any],
    mode: str,
    coverage_rows: Optional[List[Dict[str, Any]]] = None,
    loan_key_hints: Optional[Dict[Tuple[str, str], str]] = None,
    alias_loan_cols: Optional[set] = None,
    entity_keys: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    debug_dir: Optional[Path] = None,
    period_gate: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    inventory_by_name = {i.get("file_name", ""): i for i in inventory}
    # Case/basename-tolerant file lookup (28a selected_source_file should match the
    # inventory file_name, but normalise to be safe).
    inv_by_basename = {Path(k).name.lower(): v for k, v in inventory_by_name.items()}

    def _resolve_inv(file_name: str) -> Dict[str, Any]:
        return (inventory_by_name.get(file_name)
                or inv_by_basename.get(Path(str(file_name)).name.lower())
                or {})

    loan_key_hints = loan_key_hints or {}
    alias_loan_cols = alias_loan_cols or set()
    entity_keys = entity_keys or {}
    from . import entity_key_resolver as _ekr

    def _entity_for(pk: Tuple[str, str]) -> Dict[str, Any]:
        return entity_keys.get(pk) or {}

    def _join_key(value: Any, rule: str) -> str:
        # Resolved entity-key normalisation when available; else the generic
        # numeric/decimal id normalisation (76034101.0 -> 76034101).
        return _ekr.normalise_key(value, rule) if rule else _norm_key(value)

    included = getattr(field_scope, "included_fields", set()) or set()

    field_sources = _collect_field_sources(
        mapping_candidates, overrides, inventory_by_name, included
    )

    # Restrict to loan-level domains for the lender tape.
    def in_lender_scope(canon: str) -> bool:
        return bool(dc.field_domains(canon, registry_fields.get(canon, {})) & _LENDER_DOMAINS)

    field_sources = {f: s for f, s in field_sources.items() if in_lender_scope(f)}

    # --- Consume the resolved target-first coverage (28a) ------------------
    # Gate 4 already selected the authoritative source per field (artefact-role
    # guardrail: funded over pipeline) and recorded profile proxy / inferred
    # constants. Promotion consumes those decisions and does NOT re-resolve them,
    # so a single authoritative source replaces the rediscovered candidates
    # (eliminating funded/pipeline conflicts and pipeline leakage into funded
    # fields). Fields without a resolved selection keep the legacy behaviour.
    # Period eligibility gate (MI modes): the run only consumes sources that
    # belong to its reporting period, and the lender-tape universe is built from
    # the eligible funded / current-book source(s) — not every loan-like key. A
    # cumulative current-book file is row-filtered by its Month Run / cut-off
    # column; future-period pipeline/KFI files contribute no rows or values.
    period_gate = period_gate or {}
    gate_active = bool(period_gate.get("active"))
    eligibility: Dict[Tuple[str, str], Dict[str, Any]] = period_gate.get("eligibility", {}) or {}
    run_period = period_gate.get("run_period", "")
    run_year = period_gate.get("run_year")
    pipeline_roles: set = period_gate.get("pipeline_roles", set()) or set()
    role_by_file: Dict[str, str] = period_gate.get("role_by_file", {}) or {}

    def _is_pipeline_role(classification: str) -> bool:
        return spe._norm_col(classification) in pipeline_roles

    forced, tape_constants, tape_derivations = _coverage_selections(
        coverage_rows or [], inventory_by_name)
    for canon, src in forced.items():
        if not in_lender_scope(canon):
            continue
        if gate_active:
            # Keep the authoritative funded source first, but retain other
            # funded/current-book candidates (e.g. the correct reporting period's
            # file) as fallbacks. Never fall back to a pipeline source (the
            # artefact-role guardrail still holds). The period gate then routes
            # each run to the source eligible for its period.
            fallbacks = [s for s in field_sources.get(canon, [])
                         if (s.file_name, s.sheet or "") != (src.file_name, src.sheet or "")
                         and not _is_pipeline_role(s.classification)]
            field_sources[canon] = [src] + fallbacks
        else:
            field_sources[canon] = [src]   # single authoritative source -> no conflict

    # Load frames + loan-key columns per (file, SHEET) — a source selected from a
    # specific sheet of a multi-sheet workbook must read THAT sheet, not the first.
    def _play_key(s: "_Source") -> Tuple[str, str]:
        return (s.file_name, s.sheet or "")

    universe_roles: set = period_gate.get("universe_roles", set()) or set()

    def _role_for(fname: str) -> str:
        return role_by_file.get(fname) or _resolve_inv(fname).get("classification", "")

    def _elig_for(fname: str, sheet: str, df) -> Optional[Dict[str, Any]]:
        e = eligibility.get((fname, sheet))
        if e is not None or not gate_active:
            return e
        inv = _resolve_inv(fname)
        recs = [{"file_name": fname, "file_path": inv.get("file_path", ""),
                 "sheet_name": sheet, "artefact_role": _role_for(fname),
                 "detected_reporting_date": inv.get("detected_reporting_date", ""),
                 "df": df}]
        rows = spe.compute_eligibility(
            recs, period_gate.get("run_id", ""), config=period_gate.get("config"),
            input_dir=period_gate.get("input_dir", ""))
        return rows[0].as_dict() if rows else None

    frames: Dict[Tuple[str, str], pd.DataFrame] = {}
    key_cols: Dict[Tuple[str, str], str] = {}
    load_debug: Dict[Tuple[str, str], Dict[str, Any]] = {}
    excluded_sources: List[Dict[str, Any]] = []
    plays = {_play_key(s) for srcs in field_sources.values() for s in srcs}
    for (fname, sheet) in plays:
        inv = _resolve_inv(fname)
        path = inv.get("file_path", "")
        df = _read_df(path, sheet)
        dbg: Dict[str, Any] = {
            "file_in_inventory": bool(inv), "resolved_file_path": path,
            "frame_loaded": df is not None,
            "df_shape": list(df.shape) if df is not None else None,
            "df_columns": [str(c) for c in df.columns] if df is not None else [],
            "artefact_role": _role_for(fname),
        }
        if df is None:
            load_debug[(fname, sheet)] = {**dbg, "key_column": "", "key_count": 0}
            continue
        # Reporting-period eligibility (MI modes). An ineligible source contributes
        # no rows and no values: it is recorded in 18f / 04c but not loaded.
        elig = _elig_for(fname, sheet, df)
        dbg["inferred_reporting_period"] = (elig or {}).get("inferred_reporting_period", "")
        dbg["period_column"] = (elig or {}).get("period_column", "")
        dbg["is_universe_source"] = bool((elig or {}).get("is_universe_source"))
        dbg["period_eligible"] = bool((elig or {}).get("is_period_eligible", True))
        if gate_active and elig is not None and not elig.get("is_period_eligible", True):
            excluded_sources.append({
                "source_file": fname, "source_sheet": sheet, "artefact_role": _role_for(fname),
                "inferred_reporting_period": elig.get("inferred_reporting_period", ""),
                "reason": elig.get("reason_excluded", "period_mismatch"),
                "row_count": int(df.shape[0])})
            load_debug[(fname, sheet)] = {**dbg, "key_column": "", "key_count": 0,
                                          "excluded": True}
            continue
        ent = _entity_for((fname, sheet))
        key = None
        if ent.get("key_column"):
            for c in df.columns:
                if _norm(c) == _norm(ent["key_column"]):
                    key = c
                    break
        if key is None:
            key = _resolve_key_column(df, (fname, sheet), loan_key_hints, alias_loan_cols)
        dbg["key_column"] = key or ""
        dbg["key_resolution_basis"] = (ent.get("basis", "entity_key_resolution")
                                       if ent.get("key_column") else "profile/alias/name")
        dbg["normalisation_rule"] = ent.get("normalisation_rule", "")
        if key is None:
            load_debug[(fname, sheet)] = {**dbg, "key_count": 0}
            continue
        frames[(fname, sheet)] = df
        key_cols[(fname, sheet)] = key
        load_debug[(fname, sheet)] = dbg

    # Normalised-header -> actual-column map per frame, so a selected source column
    # joins even if its header differs only by case / whitespace.
    col_maps: Dict[Tuple[str, str], Dict[str, str]] = {
        pk: {_norm(c): c for c in df.columns} for pk, df in frames.items()
    }

    def _actual_col(pk: Tuple[str, str], col: str) -> str:
        m = col_maps.get(pk, {})
        return m.get(_norm(col), col)

    # Per-(file,sheet) lookup index: {(file,sheet): {loan_id: {col: value}}}.
    # Loan ids are normalised (76034101.0 -> 76034101) so they join across files.
    indexes: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    all_ids: set = set()
    universe_ids: set = set()
    has_universe_source = False
    universe_source_debug: List[Dict[str, Any]] = []
    for pk, df in frames.items():
        key = key_cols[pk]
        rule = _entity_for(pk).get("normalisation_rule", "")
        ld = load_debug.setdefault(pk, {})
        period_col = ld.get("period_column", "")
        actual_period_col = _actual_col(pk, period_col) if period_col else ""
        is_universe = bool(ld.get("is_universe_source"))
        has_universe_source = has_universe_source or is_universe
        idx: Dict[str, Dict[str, Any]] = {}
        collisions = 0
        rows_raw = 0
        for _, row in df.iterrows():
            rows_raw += 1
            # Row-level period filter for a cumulative current-book file: keep only
            # rows whose Month Run / cut-off period equals the run period.
            if (gate_active and actual_period_col and run_period
                    and spe.period_of_value(row.get(actual_period_col), run_year) != run_period):
                continue
            lid = _join_key(row[key], rule)
            if not lid:
                continue
            if lid in idx:
                collisions += 1
            idx[lid] = row.to_dict()
            all_ids.add(lid)
            if is_universe:
                universe_ids.add(lid)
        indexes[pk] = idx
        ld["key_count"] = len(idx)
        ld["key_collision_count"] = collisions
        ld["rows_raw"] = rows_raw
        ld["rows_after_period_filter"] = len(idx)
        if is_universe:
            universe_source_debug.append({
                "source_file": pk[0], "source_sheet": pk[1],
                "artefact_role": ld.get("artefact_role", ""),
                "inferred_reporting_period": ld.get("inferred_reporting_period", ""),
                "period_column": period_col, "rows_raw": rows_raw,
                "rows_after_period_filter": len(idx)})

    # Loan universe: when the period gate is active and an eligible funded /
    # current-book source defines the universe, the tape rows come ONLY from that
    # source (period-filtered). Otherwise fall back to every discovered id (legacy
    # / regulatory behaviour).
    if gate_active and has_universe_source:
        loan_ids = sorted(universe_ids)
    else:
        loan_ids = sorted(all_ids)
    universe_keys = set(loan_ids)
    assigned_by_field: Dict[str, int] = {}

    # The loan identifier is the primary key column, never also a data column.
    # Include resolved profile-proxy and inferred-constant fields as columns even
    # when they have no per-row source (they materialise below).
    extra_cols = {c for c in tape_constants if c != "loan_identifier"}
    extra_cols |= {c for c in tape_derivations if c != "loan_identifier"}
    canonical_order = sorted(
        (set(field_sources.keys()) | extra_cols) - {"loan_identifier"})
    enum_decisions = enum_decisions or {}
    precedence_defaults = _load_source_precedence_defaults()

    tape: List[Dict[str, Any]] = []
    lineage: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []
    gap_seq = 0

    def new_gap(severity, domain, canon, file, col, issue, desc, actions, blocking):
        nonlocal gap_seq
        gap_seq += 1
        gaps.append({
            "gap_id": f"CTG{gap_seq:03d}",
            "severity": severity, "mode": mode, "domain": domain,
            "canonical_field": canon, "source_file": file, "source_column": col,
            "issue_type": issue, "description": desc,
            "candidate_actions": "; ".join(actions), "blocking": blocking,
        })

    conflict_count = 0
    populated_cells = 0

    for lid in loan_ids:
        row: Dict[str, Any] = {"loan_identifier": lid}
        for canon in canonical_order:
            if canon not in field_sources:
                continue  # constant / derived field — materialised below
            srcs = _order_sources(
                canon, field_sources[canon], precedence, registry_fields, precedence_defaults
            )
            srcs = [s for s in srcs
                    if _play_key(s) in indexes and lid in indexes[_play_key(s)]]
            if not srcs:
                continue
            domain = next(iter(dc.field_domains(canon, registry_fields.get(canon, {})) & _LENDER_DOMAINS), dc.LOAN)
            primary = srcs[0]
            pk_primary = _play_key(primary)
            primary_value = indexes[pk_primary][lid].get(_actual_col(pk_primary, primary.column))

            validation_sources: List[str] = []
            alternate_values: List[str] = []
            conflict_status = "single_source"
            precedence_applied = bool((precedence or {}).get(canon, {}).get("primary_source_file"))

            for other in srcs[1:]:
                pk_other = _play_key(other)
                ov = indexes[pk_other][lid].get(_actual_col(pk_other, other.column))
                if _is_blank(ov):
                    continue
                if _values_match(primary_value, ov):
                    validation_sources.append(f"{other.file_name}:{other.column}")
                    conflict_status = "validated"
                else:
                    alternate_values.append(f"{other.file_name}:{other.column}={ov}")
                    if precedence_applied:
                        conflict_status = "resolved_by_precedence"
                    else:
                        conflict_status = "conflict"

            if conflict_status == "conflict":
                conflict_count += 1
                new_gap(
                    "blocking" if mode in ("regulatory_mi", "warehouse_securitisation") else "high",
                    domain, canon, primary.file_name, primary.column, "value_conflict",
                    f"Loan {lid}: '{canon}' differs across sources "
                    f"({primary.file_name}:{primary.column}={primary_value} vs "
                    f"{'; '.join(alternate_values)}). No approved precedence.",
                    ["approve_source_precedence", "confirm_authoritative_value"],
                    True,
                )

            if _is_blank(primary_value):
                continue

            # Enum decision applied (e.g. employment_status = manual).
            enum_applied = ""
            if canon in enum_decisions:
                raw = str(primary_value).strip()
                dec = enum_decisions[canon].get(raw)
                if dec:
                    enum_applied = dec.get("decision", "")

            row[canon] = primary_value
            populated_cells += 1
            assigned_by_field[canon] = assigned_by_field.get(canon, 0) + 1
            lineage.append({
                "loan_identifier": lid,
                "canonical_field": canon,
                "value": primary_value,
                "source_file": primary.file_name,
                "source_sheet": primary.sheet,
                "source_column": primary.column,
                "source_value": primary_value,
                "mapping_method": primary.method,
                "confidence": round(primary.confidence, 3),
                "domain": domain,
                "source_precedence_applied": precedence_applied,
                "validation_sources": "; ".join(validation_sources),
                "alternate_values": "; ".join(alternate_values),
                "conflict_status": conflict_status,
                "enum_decision_applied": enum_applied,
                "review_required": conflict_status == "conflict",
                "notes": "",
            })

        # Materialise resolved profile-proxy derivations (e.g. equity-release
        # current_principal_balance <- current_outstanding_balance) and inferred
        # constants (reporting_date / data_cut_off_date from the run-id period).
        for canon, src_field in tape_derivations.items():
            if not _is_blank(row.get(canon)):
                continue
            v = row.get(src_field, "")
            if not _is_blank(v):
                row[canon] = v
                populated_cells += 1
                lineage.append({
                    "loan_identifier": lid, "canonical_field": canon, "value": v,
                    "source_file": "", "source_sheet": "", "source_column": src_field,
                    "source_value": v, "mapping_method": "product_profile_proxy",
                    "confidence": 0.9, "domain": dc.LOAN,
                    "source_precedence_applied": False, "validation_sources": "",
                    "alternate_values": "", "conflict_status": "derived",
                    "enum_decision_applied": "", "review_required": False,
                    "notes": f"derived from {src_field} (target-first resolved)"})
        for canon, val in tape_constants.items():
            if _is_blank(row.get(canon)) and not _is_blank(val):
                row[canon] = val
                populated_cells += 1
                lineage.append({
                    "loan_identifier": lid, "canonical_field": canon, "value": val,
                    "source_file": "", "source_sheet": "", "source_column": "",
                    "source_value": val, "mapping_method": "run_context_inference",
                    "confidence": 0.9, "domain": dc.LOAN,
                    "source_precedence_applied": False, "validation_sources": "",
                    "alternate_values": "", "conflict_status": "inferred",
                    "enum_decision_applied": "", "review_required": False,
                    "notes": "inferred/defaulted (target-first resolved)"})
        tape.append(row)

    # Required-domain coverage gaps: representative fields that never mapped.
    for domain in (dc.LOAN, dc.BORROWER, dc.COLLATERAL):
        for canon in dc._DOMAIN_REQUIRED_FIELDS.get(domain, []):
            if included and canon not in included:
                continue
            if canon not in field_sources:
                new_gap(
                    "high" if domain in (dc.LOAN,) else "medium",
                    domain, canon, "", "", "unmapped_required_field",
                    f"Required {domain} field '{canon}' has no approved/in-scope source.",
                    ["map_source_column", "confirm_field_absent"],
                    domain == dc.LOAN and mode in ("regulatory_mi", "warehouse_securitisation"),
                )

    # Per-field materialisation diagnostics for the RESOLVED (28a-selected)
    # source-mapped fields — so a real-pack run that still produces nulls reveals
    # exactly where (file/sheet/column/key/join) the assignment failed.
    materialisation_debug: List[Dict[str, Any]] = []
    for canon, src in sorted(forced.items()):
        pk = _play_key(src)
        ld = load_debug.get(pk, {})
        src_keys = set(indexes.get(pk, {}).keys())
        materialisation_debug.append({
            "canonical_field": canon,
            "selected_source_file": src.file_name,
            "selected_source_sheet": src.sheet,
            "selected_source_column": src.column,
            "resolved_file_path": ld.get("resolved_file_path", ""),
            "file_in_inventory": ld.get("file_in_inventory", False),
            "frame_loaded": ld.get("frame_loaded", False),
            "df_shape": ld.get("df_shape"),
            "selected_column_present": (
                _norm(src.column) in {_norm(c) for c in ld.get("df_columns", [])}),
            "loan_key_column_used": ld.get("key_column", ""),
            "key_resolution_basis": ld.get("key_resolution_basis", ""),
            "normalisation_rule": ld.get("normalisation_rule", ""),
            "source_key_count": ld.get("key_count", len(src_keys)),
            "central_universe_count": len(universe_keys),
            "key_intersection": len(src_keys & universe_keys),
            "key_collision_count": ld.get("key_collision_count", 0),
            "assigned_non_null": assigned_by_field.get(canon, 0),
            "in_lender_scope": in_lender_scope(canon),
        })
    if debug_dir is not None:
        try:
            (Path(debug_dir) / "18e_central_tape_materialisation_debug.json").write_text(
                json.dumps(materialisation_debug, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    # Period-eligible universe diagnostics (18f): which source(s) defined the
    # lender-tape rows for this run, and which were excluded and why.
    universe_debug = {
        "run_id": period_gate.get("run_id", ""),
        "run_reporting_period": run_period,
        "period_gate_active": gate_active,
        "universe_basis": ("period_eligible_funded_current_book"
                           if (gate_active and has_universe_source) else "all_discovered_ids"),
        "selected_universe_sources": universe_source_debug,
        "raw_universe_rows": sum(s["rows_raw"] for s in universe_source_debug),
        "canonical_universe_rows": len(loan_ids),
        "excluded_sources": excluded_sources,
        "excluded_row_counts": sum(e.get("row_count", 0) for e in excluded_sources),
    }

    summary = {
        "loan_count": len(tape),
        "canonical_fields_populated": len(canonical_order),
        "populated_cells": populated_cells,
        "conflict_count": conflict_count,
        "gap_count": len(gaps),
        "lineage_rows": len(lineage),
        "mode": mode,
        "materialisation_debug": materialisation_debug,
        "universe_debug": universe_debug,
        "columns": ["loan_identifier"] + canonical_order,
    }
    if debug_dir is not None:
        try:
            (Path(debug_dir) / "18f_central_universe_debug.json").write_text(
                json.dumps(universe_debug, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
    # Ensure every tape row has all columns (blank where unpopulated).
    for r in tape:
        for c in ["loan_identifier"] + canonical_order:
            r.setdefault(c, "")
    return tape, lineage, gaps, summary


# ---------------------------------------------------------------------------
# Central pipeline tape (PART 8)
# ---------------------------------------------------------------------------


def _build_pipeline_tape(
    inventory: List[Dict[str, Any]],
    lender_loan_ids: set,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    lineage: List[Dict[str, Any]] = []

    pipeline_files = [
        i for i in inventory
        if i.get("classification") == "pipeline_report"
        or dc.PIPELINE in (i.get("domains_detected") or [])
    ]

    linked_count = 0
    application_only = 0
    for inv in pipeline_files:
        df = _read_df(inv.get("file_path", ""))
        if df is None:
            continue
        norm_cols = {_norm(c): c for c in df.columns}
        field_cols: Dict[str, str] = {}
        for target, pats in _PIPELINE_FIELD_PATTERNS.items():
            for p in pats:
                if p in norm_cols:
                    field_cols[target] = norm_cols[p]
                    break
        if "application_id" not in field_cols:
            continue  # not a pipeline tape we can key
        for _, r in df.iterrows():
            out = {c: "" for c in _PIPELINE_COLUMNS}
            for target, col in field_cols.items():
                val = r.get(col)
                out[target] = "" if _is_blank(val) else val
                lineage.append({
                    "application_id": out.get("application_id", ""),
                    "pipeline_field": target,
                    "value": out[target],
                    "source_file": inv.get("file_name", ""),
                    "source_column": col,
                })
            if not str(out.get("application_id", "")).strip():
                continue
            linked = str(out.get("linked_loan_identifier", "")).strip()
            if linked:
                out["linked_to_central_lender_tape"] = linked in lender_loan_ids
                linked_count += 1
            else:
                out["linked_to_central_lender_tape"] = False
                application_only += 1
            rows.append(out)

    summary = {
        "pipeline_count": len(rows),
        "linked_to_funded_loans": linked_count,
        "application_only_rows": application_only,
        "columns": _PIPELINE_COLUMNS + ["linked_to_central_lender_tape"],
    }
    return rows, lineage, summary


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_rows(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_central_tapes(
    project_dir: str | Path,
    run_paths: Any,
    registry_path: str | Path,
    mode: str = "",
    regulatory_reporting_enabled: bool = False,
) -> Dict[str, Any]:
    """Build the central lender + pipeline tapes from approved artefacts."""
    project_dir = Path(project_dir)
    run_summary = _load_json(project_dir / "09_onboarding_run_summary.json") or {}
    mode = mode or run_summary.get("onboarding_mode", "regulatory_mi")

    mapping_candidates = _load_json(project_dir / "05_mapping_candidates.json") or []
    overrides = _load_yaml(project_dir / "12_approved_mapping_overrides.yaml") or {}
    precedence = _load_yaml(project_dir / "13_source_precedence_rules.yaml") or {}
    enum_decisions = _load_yaml(project_dir / "14_enum_review_decisions.yaml") or {}
    inventory = _load_json(project_dir / "01_file_inventory.json") or []
    # Resolved target-first coverage (28a) — the authoritative source selection
    # from Gate 4 (role guardrail + profile proxy/inference). Consumed so
    # promotion does not re-resolve already-decided fields. Only for MI modes,
    # whose 28a target fields are canonical field names; the regulatory (Annex 2)
    # contract uses ESMA codes and keeps the legacy generic tape unchanged.
    coverage_rows: List[Dict[str, Any]] = []
    if mode in ("mi_only", "mna_dd"):
        coverage_doc = _load_json(project_dir / "28a_target_coverage_matrix.json") or {}
        coverage_rows = (coverage_doc.get("rows", [])
                         if isinstance(coverage_doc, dict) else [])

    policy = load_mode_policy(mode)
    field_scope = resolve_field_scope(
        str(registry_path), policy,
        regulatory_reporting_enabled=regulatory_reporting_enabled,
    )
    registry_fields = load_field_registry(Path(registry_path)).get("fields", {}) or {}

    # Onboarding-detected loan-key hints per (file, sheet) — from 02 column
    # profiles (likely_identifier) and the loan_identifier alias set — so the
    # central tape uses the SAME key column onboarding detected rather than
    # guessing from a fixed name list (handles non-standard key headers).
    loan_key_hints = _loan_key_hints(project_dir)
    alias_loan_cols = _alias_loan_columns()
    # Generic entity-key resolution (04b): the resolved join key + normalisation
    # rule per (file, sheet) that links the same loan entity across sheets/files.
    # Consumed for the MI central-tape flow only; regulatory (Annex 2) is untouched.
    entity_keys: Dict[Tuple[str, str], Dict[str, Any]] = {}
    period_gate: Dict[str, Any] = {}
    if mode in ("mi_only", "mna_dd"):
        from . import entity_key_resolver as _ekr
        entity_keys = _ekr.load_resolution(project_dir)
        # Reporting-period eligibility (04c): the lender-tape universe is built from
        # the run-period funded / current-book source(s) only; future-period or
        # other-period files contribute no rows or values. Gated to MI modes;
        # regulatory (Annex 2) keeps the legacy generic universe.
        spe_cfg = spe.load_config()
        if spe_cfg.get("enabled", True):
            run_id = getattr(run_paths, "run_id", "") or run_summary.get("run_id", "")
            input_dir = getattr(run_paths, "input_dir", "") or run_summary.get("input_dir", "")
            run_p, _run_cut = spe.run_period(run_id, input_dir)
            period_gate = {
                "active": True,
                "run_id": run_id,
                "input_dir": input_dir,
                "run_period": run_p,
                "run_year": int(run_p[:4]) if re.fullmatch(r"\d{4}-\d{2}", run_p or "") else None,
                "config": spe_cfg,
                "eligibility": spe.load_eligibility(project_dir),
                "pipeline_roles": {spe._norm_col(r) for r in (spe_cfg.get("pipeline_roles") or [])},
                "universe_roles": {spe._norm_col(r) for r in (spe_cfg.get("funded_current_book_roles") or [])},
                "role_by_file": {i.get("file_name", ""): i.get("classification", "")
                                 for i in inventory},
            }

    tape, lineage, gaps, summary = _build_lender_tape(
        mapping_candidates, overrides, precedence, enum_decisions, inventory,
        field_scope, registry_fields, mode, coverage_rows=coverage_rows,
        loan_key_hints=loan_key_hints, alias_loan_cols=alias_loan_cols,
        entity_keys=entity_keys, debug_dir=project_dir, period_gate=period_gate,
    )

    central_dir = Path(run_paths.central_dir)
    lineage_dir = Path(run_paths.lineage_dir)
    gaps_dir = Path(run_paths.gaps_dir)
    run_paths.guard(central_dir)
    run_paths.guard(lineage_dir)
    run_paths.guard(gaps_dir)

    cols = summary["columns"]
    tape_path = central_dir / "18_central_lender_tape.csv"
    lineage_path = lineage_dir / "18b_central_tape_lineage.csv"
    gaps_path = gaps_dir / "18c_central_tape_gaps.csv"
    lender_summary_path = central_dir / "18d_central_tape_summary.json"

    _write_rows(tape_path, cols, tape)
    _write_rows(lineage_path, _LENDER_LINEAGE_COLUMNS, lineage)
    _write_rows(gaps_path, _GAP_COLUMNS, gaps)
    summary["central_lender_tape_path"] = str(tape_path)

    # Aggregate the funded balance over the resolved universe (for the universe
    # debug + an optional configured expected-balance check). Parses currency
    # strings ("112,619.77") tolerantly; never fabricates a value.
    def _to_float(v: Any) -> Optional[float]:
        s = re.sub(r"[^0-9.\-]", "", str(v or "").strip())
        try:
            return float(s) if s not in ("", "-", ".") else None
        except ValueError:
            return None

    udbg = summary.get("universe_debug") or {}
    if "current_outstanding_balance" in cols:
        vals = [_to_float(r.get("current_outstanding_balance")) for r in tape]
        nums = [v for v in vals if v is not None]
        udbg["aggregate_current_outstanding_balance"] = round(sum(nums), 2) if nums else 0.0
        udbg["current_outstanding_balance_populated"] = len(nums)
    summary["universe_debug"] = udbg
    lender_summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Mirror the period-universe + materialisation debug into the central dir
    # (alongside 18_/18d_), so they are discoverable next to the tape they explain.
    try:
        (central_dir / "18f_central_universe_debug.json").write_text(
            json.dumps(udbg, indent=2, default=str), encoding="utf-8")
        (central_dir / "18e_central_tape_materialisation_debug.json").write_text(
            json.dumps(summary.get("materialisation_debug", []), indent=2, default=str),
            encoding="utf-8")
    except Exception:
        pass

    # Pipeline tape.
    lender_loan_ids = {str(r.get("loan_identifier", "")).strip() for r in tape}
    p_rows, p_lineage, p_summary = _build_pipeline_tape(inventory, lender_loan_ids)
    pipeline_created = bool(p_rows)
    pipeline_tape_path = central_dir / "18a_central_pipeline_tape.csv"
    pipeline_lineage_path = lineage_dir / "18a_central_pipeline_lineage.csv"
    pipeline_summary_path = central_dir / "18a_central_pipeline_summary.json"
    if pipeline_created:
        _write_rows(pipeline_tape_path, p_summary["columns"], p_rows)
        _write_rows(pipeline_lineage_path,
                    ["application_id", "pipeline_field", "value", "source_file", "source_column"],
                    p_lineage)
        p_summary["central_pipeline_tape_path"] = str(pipeline_tape_path)
        pipeline_summary_path.write_text(
            json.dumps(p_summary, indent=2, default=str), encoding="utf-8"
        )

    return {
        "mode": mode,
        "central_lender_tape_created": bool(tape),
        "central_lender_tape_path": str(tape_path),
        "central_tape_lineage_path": str(lineage_path),
        "central_tape_gaps_path": str(gaps_path),
        "central_tape_summary_path": str(lender_summary_path),
        "lender_summary": summary,
        "central_pipeline_tape_created": pipeline_created,
        "central_pipeline_tape_path": str(pipeline_tape_path) if pipeline_created else "",
        "pipeline_summary": p_summary,
        "loan_count": summary["loan_count"],
        "pipeline_count": p_summary["pipeline_count"],
        "conflict_count": summary["conflict_count"],
        "gap_count": summary["gap_count"],
        "mapped_field_count": summary["canonical_fields_populated"],
    }
