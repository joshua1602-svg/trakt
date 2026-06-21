"""
entity_key_resolver.py
======================

Generic, reusable **entity-key resolution** for cross-sheet / cross-file row
linkage — the join key that connects the same loan/policy entity across a
lender's funded extracts, KFI/pipeline sheets and cashflow ledgers.

It is deliberately NOT lender-specific: it identifies candidate key columns by
alias + data profiling, scores cross-sheet overlap under a small set of
controlled normalisation rules (exact / numeric / decimal-suffix / repeated
trailing-suffix), guards against collisions, and records the resolved outcome as
an auditable artefact (``04b_entity_key_resolution.csv`` / ``.json``). Promotion
consumes that artefact rather than guessing a key from a fixed name list.

Normalisation never mutates source data: it produces a *comparison* key only;
original values are preserved for lineage.

Real-pack examples handled:
* ``Base Policy Number`` (one sheet) == ``Loan ID`` (another) — same values,
  different column names -> direct link.
* ``Loan Policy Number`` = ``760341`` links to ``Account Number`` = ``76034101``
  — a stable trailing ``01`` suffix -> ``strip_trailing_01`` (only when unique +
  high overlap + no collisions, else flagged for operator review).
* ``76034101`` vs ``76034101.0`` — decimal-suffix -> both join.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config" / "system"

# Alias headers that suggest a loan/policy entity key (config-extensible).
KEY_ALIASES = [
    "loan id", "loan identifier", "loan reference", "loan ref", "account number",
    "account no", "mortgage account reference", "policy number", "base policy number",
    "loan policy number", "agreement number", "contract id", "facility id",
    "unique identifier", "loan_identifier",
]

# Document / non-loan-level roles whose sheets never carry a loan entity key.
# Everything else is considered (a sheet only participates if it actually has a
# candidate identifier column), so mis-classified loan-level files (e.g. a funder
# cashflow tagged "warehouse_agreement") are still linked.
_NON_ENTITY_ROLES = {
    "data_dictionary", "securitisation_document", "investor_report", "document",
}

# Scoring / safety thresholds.
_MIN_NON_NULL = 0.80
_MIN_UNIQUENESS = 0.80
_OVERLAP_LINK_THRESHOLD = 0.50      # normalised overlap to call a link
_SUFFIX_DOMINANCE = 0.80            # fraction sharing one trailing suffix
_COLLISION_TOLERANCE = 0.99         # normalised distinct / raw distinct must stay >= this
_CONFIDENCE_REVIEW = 0.70

_RULE_EXACT = "exact"
_RULE_NUMERIC = "numeric_string"
_RULE_DECIMAL = "strip_decimal_suffix"
# strip_trailing_<suffix> is parametric, e.g. strip_trailing_01


def _norm_col(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s or "").strip().lower()).strip("_")


def _alias_norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").strip().lower()).strip()


_ALIAS_SET = {_alias_norm(a) for a in KEY_ALIASES}


# --------------------------------------------------------------------------- #
# Normalisation (comparison key only — never mutates source values)
# --------------------------------------------------------------------------- #

def _base_canonical(v: Any) -> str:
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "nat", "<na>"):
        return ""
    m = re.fullmatch(r"(-?\d+)\.0+", s)
    if m:
        return m.group(1)
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


def normalise_key(value: Any, rule: str) -> str:
    """Map a raw identifier to its comparison key under ``rule``. Empty -> ''."""
    s = _base_canonical(value)
    if not s:
        return ""
    rule = str(rule or _RULE_EXACT)
    if rule == _RULE_EXACT:
        return s.upper()
    if rule in (_RULE_NUMERIC, _RULE_DECIMAL):
        digits = re.sub(r"\D", "", s)
        return digits or s.upper()
    if rule.startswith("strip_trailing_"):
        suffix = rule[len("strip_trailing_"):]
        digits = re.sub(r"\D", "", s)
        if suffix and digits.endswith(suffix) and len(digits) > len(suffix):
            return digits[: -len(suffix)]
        return digits or s.upper()
    if rule == "remove_separators":
        return re.sub(r"[\s\-_/]", "", s).upper()
    return s.upper()


# --------------------------------------------------------------------------- #
# Candidate key detection
# --------------------------------------------------------------------------- #

@dataclass
class _Candidate:
    column: str
    non_null_pct: float
    uniqueness_pct: float
    alias_hit: bool
    identifier_like: bool
    values: List[str]          # base-canonical, non-empty
    score: float = 0.0


def _candidate_keys(df, profile_by_col: Dict[str, Dict[str, Any]]) -> List[_Candidate]:
    cands: List[_Candidate] = []
    n = len(df)
    if n == 0:
        return cands
    for col in df.columns:
        prof = profile_by_col.get(_norm_col(col), {})
        dtype = str(prof.get("inferred_type", "")).lower()
        if dtype in ("decimal", "date", "boolean", "rate"):
            continue  # monetary / date / rate / boolean are not entity keys
        series = df[col]
        non_null = series.dropna()
        nn_pct = len(non_null) / n if n else 0.0
        vals = [c for c in (_base_canonical(v) for v in non_null.tolist()) if c]
        if not vals:
            continue
        uniq_pct = len(set(vals)) / len(vals) if vals else 0.0
        alias_hit = _alias_norm(col) in _ALIAS_SET
        id_like = (alias_hit or bool(prof.get("likely_identifier"))
                   or dtype in ("identifier", "integer"))
        if not (alias_hit or id_like or (uniq_pct >= 0.95 and nn_pct >= _MIN_NON_NULL)):
            continue
        if nn_pct < _MIN_NON_NULL or uniq_pct < _MIN_UNIQUENESS:
            # keep only as weak candidate if alias hit
            if not alias_hit:
                continue
        score = ((0.5 if alias_hit else 0.0) + (0.25 if id_like else 0.0)
                 + 0.15 * uniq_pct + 0.10 * nn_pct)
        cands.append(_Candidate(column=str(col), non_null_pct=round(nn_pct, 4),
                                uniqueness_pct=round(uniq_pct, 4), alias_hit=alias_hit,
                                identifier_like=id_like, values=vals, score=round(score, 4)))
    cands.sort(key=lambda c: -c.score)
    return cands


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

@dataclass
class EntityKeyResolution:
    source_file: str
    source_sheet: str
    artefact_role: str
    inferred_entity: str
    selected_key_column: str
    selected_key_basis: str
    normalisation_rule: str
    non_null_pct: float
    uniqueness_pct: float
    overlap_partner_file: str = ""
    overlap_partner_sheet: str = ""
    overlap_partner_column: str = ""
    overlap_count_raw: int = 0
    overlap_count_normalised: int = 0
    overlap_pct_normalised: float = 0.0
    key_collision_count: int = 0
    confidence: float = 0.0
    needs_operator_review: bool = False
    rationale: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in _COLUMNS}


_COLUMNS = [
    "source_file", "source_sheet", "artefact_role", "inferred_entity",
    "selected_key_column", "selected_key_basis", "normalisation_rule",
    "non_null_pct", "uniqueness_pct", "overlap_partner_file", "overlap_partner_sheet",
    "overlap_partner_column", "overlap_count_raw", "overlap_count_normalised",
    "overlap_pct_normalised", "key_collision_count", "confidence",
    "needs_operator_review", "rationale",
]


def _numeric_set(values: List[str]) -> set:
    out = set()
    for v in values:
        d = re.sub(r"\D", "", _base_canonical(v))
        if d:
            out.add(d)
    return out


def _detect_global_suffix(value_sets: List[set]) -> Optional[str]:
    """Find a dominant trailing digit-suffix S that explains length differences
    between sheets (long-form key == short-form key + S). Controlled: only
    returned when one suffix dominates the observed prefix relationships."""
    counts: Counter = Counter()
    total = 0
    n = len(value_sets)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            short, longset = value_sets[i], value_sets[j]
            if not short or not longset:
                continue
            for b in list(longset)[:2000]:
                for k in (1, 2, 3):
                    if len(b) > k and b[:-k] in short:
                        counts[b[-k:]] += 1
                        total += 1
                        break
    if total and counts:
        suffix, cnt = counts.most_common(1)[0]
        if cnt / total >= _SUFFIX_DOMINANCE:
            return suffix
    return None


def _canonical_rule_for(numeric_set: set, suffix: Optional[str]) -> Tuple[str, set, int]:
    """Pick numeric vs strip_trailing_<suffix> for a sheet; return (rule, canon_set,
    collisions). Strip only when most keys carry the suffix and it doesn't collide."""
    if not suffix:
        return _RULE_NUMERIC, set(numeric_set), 0
    with_suffix = [d for d in numeric_set if d.endswith(suffix) and len(d) > len(suffix)]
    if len(with_suffix) / max(len(numeric_set), 1) < _SUFFIX_DOMINANCE:
        return _RULE_NUMERIC, set(numeric_set), 0
    rule = f"strip_trailing_{suffix}"
    canon = {normalise_key(d, rule) for d in numeric_set}
    canon.discard("")
    collisions = len(numeric_set) - len(canon)
    if len(canon) / max(len(numeric_set), 1) < _COLLISION_TOLERANCE:
        # Stripping collapses distinct ids -> unsafe; keep numeric, flag later.
        return _RULE_NUMERIC, set(numeric_set), collisions
    return rule, canon, collisions


def resolve_entity_keys(sheets: List[Dict[str, Any]]) -> List[EntityKeyResolution]:
    """Resolve the loan entity key per sheet.

    ``sheets``: list of ``{source_file, source_sheet, artefact_role, df,
    profiles: {norm_col: {inferred_type, likely_identifier, ...}}}``.
    """
    sheet_cands: List[Tuple[Dict[str, Any], List[_Candidate]]] = []
    _blocked = {_norm_col(r) for r in _NON_ENTITY_ROLES}
    for s in sheets:
        if _norm_col(str(s.get("artefact_role", "") or "")) in _blocked:
            continue
        cands = _candidate_keys(s["df"], s.get("profiles", {}) or {})
        if cands:
            sheet_cands.append((s, cands))
    if not sheet_cands:
        return []

    # Best candidate per sheet + its numeric key set.
    chosen: List[Tuple[Dict[str, Any], _Candidate, set]] = []
    for s, cands in sheet_cands:
        cand = cands[0]
        chosen.append((s, cand, _numeric_set(cand.values)))

    # Global trailing-suffix detection (bidirectional: canonicalise to short form).
    suffix = _detect_global_suffix([c[2] for c in chosen])

    # Seed: the most "central" candidate (alias + uniqueness) provides the
    # consensus key space (in the canonical/short form).
    seed_idx = max(range(len(chosen)),
                   key=lambda i: (chosen[i][1].alias_hit, chosen[i][1].uniqueness_pct))
    seed_s, seed_c, seed_num = chosen[seed_idx]
    _, seed_canon, _ = _canonical_rule_for(seed_num, suffix)
    consensus = set(seed_canon)

    resolutions: List[EntityKeyResolution] = []
    for idx, (s, cand, numeric) in enumerate(chosen):
        rule, canon, collisions = _canonical_rule_for(numeric, suffix)
        inter = len(canon & consensus)
        overlap = round(inter / max(len(canon), 1), 4)
        raw_distinct = len(set(cand.values))
        is_seed = (idx == seed_idx)

        basis = ("alias" if cand.alias_hit
                 else "profile" if cand.identifier_like else "overlap")
        suffix_rule = rule.startswith("strip_trailing_")
        if suffix_rule:
            basis = basis + "+suffix"
        elif overlap >= _OVERLAP_LINK_THRESHOLD and not is_seed:
            basis = basis + "+overlap"

        needs_review = False
        bits = [f"key='{cand.column}' basis={basis}",
                f"non_null={cand.non_null_pct} uniq={cand.uniqueness_pct}"]
        if is_seed:
            confidence = round(min(1.0, 0.6 + 0.4 * cand.uniqueness_pct), 4)
            bits.append("seed consensus key")
        else:
            confidence = round(min(1.0, 0.3 + 0.5 * overlap + 0.2 * cand.uniqueness_pct), 4)
            bits.append(f"normalised overlap={overlap} via {rule}")

        # Collision guard for suffix transforms.
        if suffix and collisions > 0 and not suffix_rule:
            needs_review = True
            confidence = round(min(confidence, 0.5), 4)
            bits.append(f"suffix '{suffix}' stripping would collide "
                        f"({collisions}); kept numeric, operator review")
        if not is_seed and overlap < _OVERLAP_LINK_THRESHOLD:
            needs_review = True
            bits.append("low cross-sheet overlap; operator review")
        if confidence < _CONFIDENCE_REVIEW:
            needs_review = True

        resolutions.append(EntityKeyResolution(
            source_file=s.get("source_file", ""), source_sheet=s.get("source_sheet", ""),
            artefact_role=str(s.get("artefact_role", "")), inferred_entity="loan",
            selected_key_column=cand.column, selected_key_basis=basis,
            normalisation_rule=rule, non_null_pct=cand.non_null_pct,
            uniqueness_pct=cand.uniqueness_pct,
            overlap_partner_file=("" if is_seed else seed_s.get("source_file", "")),
            overlap_partner_sheet=("" if is_seed else seed_s.get("source_sheet", "")),
            overlap_partner_column=("" if is_seed else seed_c.column),
            overlap_count_raw=len(numeric & seed_num),
            overlap_count_normalised=inter, overlap_pct_normalised=overlap,
            key_collision_count=max(0, raw_distinct - len(canon)),
            confidence=confidence, needs_operator_review=needs_review,
            rationale="; ".join(bits)))
    return resolutions


# --------------------------------------------------------------------------- #
# Artefacts + loading
# --------------------------------------------------------------------------- #

_ARTEFACT_CSV = "04b_entity_key_resolution.csv"
_ARTEFACT_JSON = "04b_entity_key_resolution.json"


def write_artifacts(resolutions: List[EntityKeyResolution], out_dir: str | Path) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r.as_dict() for r in resolutions]
    csv_path = out / _ARTEFACT_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json_path = out / _ARTEFACT_JSON
    json_path.write_text(json.dumps({"rows": rows}, indent=2, default=str), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}


def load_resolution(project_dir: str | Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Load 04b as ``{(file_name, sheet_name): {key_column, normalisation_rule,
    basis, needs_operator_review}}``. Empty when absent."""
    p = Path(project_dir) / _ARTEFACT_JSON
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in (data.get("rows", []) if isinstance(data, dict) else []):
        key = (r.get("source_file", ""), r.get("source_sheet", ""))
        out[key] = {
            "key_column": r.get("selected_key_column", ""),
            "normalisation_rule": r.get("normalisation_rule", _RULE_EXACT),
            "basis": r.get("selected_key_basis", ""),
            "needs_operator_review": bool(r.get("needs_operator_review", False)),
            "overlap_pct_normalised": float(r.get("overlap_pct_normalised", 0.0) or 0.0),
            "confidence": float(r.get("confidence", 0.0) or 0.0),
            "uniqueness_pct": float(r.get("uniqueness_pct", 0.0) or 0.0),
        }
    return out


def resolve_and_write(
    inventory: List[Dict[str, Any]],
    profiles: List[Dict[str, Any]],
    out_dir: str | Path,
    enable_conversion: bool = False,
) -> Dict[str, Any]:
    """Load all (file, sheet) tables, resolve entity keys, write 04b."""
    from . import source_table_loader as stl

    role_by_file = {i.get("file_name", ""): i.get("classification", "") for i in inventory}
    prof_by_sheet: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for p in profiles:
        k = (p.get("file_name", ""), p.get("sheet_name", ""))
        prof_by_sheet.setdefault(k, {})[_norm_col(p.get("source_column", ""))] = p

    tables, _, _ = stl.load_source_tables(inventory, enable_conversion=enable_conversion)
    sheets: List[Dict[str, Any]] = []
    for t in tables:
        sheets.append({
            "source_file": t.file_name, "source_sheet": t.sheet_name,
            "artefact_role": role_by_file.get(t.file_name, ""),
            "df": t.df,
            "profiles": prof_by_sheet.get((t.file_name, t.sheet_name), {}),
        })
    resolutions = resolve_entity_keys(sheets)
    paths = write_artifacts(resolutions, out_dir)
    return {"resolutions": [r.as_dict() for r in resolutions], "paths": paths}
