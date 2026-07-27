"""
source_value_rules.py
=====================

Operator-approved **source-value normalisation** for onboarding.

Real source files carry placeholder text where a value is not yet known — a date
column containing ``TBC``, a rate column containing ``n/k``. Deterministic
parsing must reject those, because a parser that quietly swallows unknown text is
a parser that will one day swallow a genuine data error. But once an operator has
looked at a specific value, in a specific column, for a specific source, and
decided what it means, that decision should be recorded once and reapplied
forever.

This module owns that decision:

    "TBC", in AcquiredLoanTape.csv's `Borrower 1 DOB`, feeding `borrower_1_DOB`,
    for ERE/acquired_001, under the MI semantics contract, means NULL.

What it deliberately does NOT do:

  * add ``TBC`` (or anything else) to the GLOBAL blank tokens in
    ``canonical_transform``. A global null marker would apply to every client,
    every column and every contract at once — the opposite of a scoped decision;
  * normalise anything that has not been explicitly approved. An unapproved bad
    value still fails deterministically and still surfaces as a parse error;
  * decide anything itself. Detection only *proposes* a decision; a human
    approves it.

Scope
-----
Every rule records all five scope dimensions — ``client_id``,
``source_portfolio_id``, ``source_column``, ``canonical_field`` and
``target_contract_id`` — plus ``source_file`` and ``source_schema_fingerprint``
as evidence. They are enforced at two different moments, because they answer two
different questions:

  * **Is this rule for this run?** ``client_id``, ``source_portfolio_id`` and
    ``target_contract_id`` (:meth:`SourceValueRule.in_scope_for_run`). The target
    contract is never a wildcard, so an MI-approved normalisation has NO effect on
    a regulatory run — a regulatory contract approves its own treatment, or none.
  * **What does it apply to?** ``canonical_field`` selects the column, and
    ``source_column`` is checked against the contract's CURRENT mapping: if the
    field is now fed by a different source column, the rule is skipped and the
    decision must be re-taken, because it was made about something else.

``source_schema_fingerprint`` is recorded but never matched on, so a non-material
schema change (an additive optional column, which the approval policy
auto-approves) does not silently drop an approved rule and re-block a recurring
pack.

Note that a rule's ``canonical_field`` need NOT be a field of the target
contract. ``borrower_1_DOB`` is parsed into the canonical tape but is not in the
MI semantics registry, so rules are carried as their own artefact
(``28d_source_value_rules.json``) rather than as an annotation on the coverage
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: The operator action this module implements. Generic by design: it names the
#: TREATMENT ("treat this source value as null"), not any field or client.
ACTION_TREAT_AS_NULL = "treat_source_value_as_null"

#: Controlled treatments. Only null-ing is supported today; the rule model has
#: room for more (e.g. map-to-value) without changing the decision plumbing.
TREATMENT_NULL = "null"

#: Decision type used in the 28c queue / 34 template.
DECISION_TYPE = "source_value_normalisation"

#: Canonical formats whose values are parsed deterministically, and are therefore
#: capable of producing a "this value cannot be parsed" operator decision.
_PARSEABLE_FORMATS = {
    "date", "percentage", "percent", "pct", "decimal", "number", "float",
    "integer", "int", "boolean", "bool", "y/n",
}

#: Cap on distinct unparseable values surfaced per column, so a genuinely broken
#: column raises one reviewable decision set rather than thousands.
MAX_CANDIDATES_PER_COLUMN = 5


def _norm_value(value: Any) -> str:
    """Comparison key for a source value: trimmed, case-insensitive.

    Case and padding are rendering, not meaning — an operator approving ``TBC``
    is approving ``tbc`` and `` TBC `` too. The ORIGINAL spelling is what gets
    recorded in lineage.
    """
    return str(value if value is not None else "").strip().lower()


@dataclass(frozen=True)
class SourceValueRule:
    """One approved normalisation, with the scope it is approved for."""
    source_value: str
    canonical_field: str
    source_column: str
    target_contract_id: str
    client_id: str = ""
    source_portfolio_id: str = ""
    source_file: str = ""
    treatment: str = TREATMENT_NULL
    decision_id: str = ""
    approved_by: str = ""
    approved_at: str = ""
    source_schema_fingerprint: str = ""

    @property
    def value_key(self) -> str:
        return _norm_value(self.source_value)

    def in_scope_for_run(self, context: "NormalisationContext") -> bool:
        """True when this rule belongs to the run being processed.

        Checks the RUN dimensions only — tenant, source portfolio and target
        contract. The field/column dimensions identify WHAT the rule applies to
        and are used when selecting the column, not when deciding whether the run
        is the right one.
        """
        def _eq(rule_value: str, ctx_value: str) -> bool:
            rule_value = (rule_value or "").strip()
            if not rule_value:
                return True     # unscoped on this dimension
            return rule_value.lower() == (ctx_value or "").strip().lower()

        return (
            # Never a wildcard: an MI-approved treatment must not leak into a
            # regulatory run.
            (self.target_contract_id or "").strip().lower()
            == (context.target_contract_id or "").strip().lower()
            and _eq(self.client_id, context.client_id)
            and _eq(self.source_portfolio_id, context.source_portfolio_id)
        )

    def as_row(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "action": ACTION_TREAT_AS_NULL,
            "treatment": self.treatment,
            "source_value": self.source_value,
            "canonical_field": self.canonical_field,
            "source_column": self.source_column,
            "source_file": self.source_file,
            "target_contract_id": self.target_contract_id,
            "client_id": self.client_id,
            "source_portfolio_id": self.source_portfolio_id,
            "source_schema_fingerprint": self.source_schema_fingerprint,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
        }


@dataclass
class NormalisationContext:
    """The run's identity, used to scope rule matching."""
    client_id: str = ""
    source_portfolio_id: str = ""
    target_contract_id: str = ""
    source_schema_fingerprint: str = ""


# --------------------------------------------------------------------------- #
# Rules from approved decisions
# --------------------------------------------------------------------------- #

def rule_from_decision(decision: Dict[str, Any],
                       context: Optional[NormalisationContext] = None
                       ) -> Optional[SourceValueRule]:
    """Build a rule from ONE approved 34_ decision entry, or None if unusable.

    The decision must name the value it is about; a ``treat_source_value_as_null``
    with no ``source_value`` is not actionable and is rejected rather than guessed
    at (nulling an unspecified value would be catastrophic).
    """
    if str(decision.get("selected_action") or "").strip() != ACTION_TREAT_AS_NULL:
        return None
    source_value = decision.get("source_value")
    if source_value is None or str(source_value).strip() == "":
        return None
    canonical = str(decision.get("canonical_field") or "").strip()
    column = str(decision.get("source_column") or "").strip()
    contract = str(decision.get("target_contract_id") or "").strip()
    if context:
        contract = contract or context.target_contract_id
    if not (canonical and column and contract):
        return None
    return SourceValueRule(
        source_value=str(source_value),
        canonical_field=canonical,
        source_column=column,
        target_contract_id=contract,
        client_id=str(decision.get("client_id") or
                      (context.client_id if context else "") or ""),
        source_portfolio_id=str(decision.get("source_portfolio_id") or
                                (context.source_portfolio_id if context else "") or ""),
        source_file=str(decision.get("source_file") or ""),
        treatment=TREATMENT_NULL,
        decision_id=str(decision.get("decision_id") or ""),
        approved_by=str(decision.get("approved_by") or ""),
        approved_at=str(decision.get("approved_at") or ""),
        source_schema_fingerprint=str(
            decision.get("source_schema_fingerprint") or
            (context.source_schema_fingerprint if context else "") or ""),
    )


def rules_from_decisions(approved: Iterable[Dict[str, Any]],
                         context: Optional[NormalisationContext] = None
                         ) -> List[SourceValueRule]:
    """Every usable normalisation rule in a list of APPROVED decisions."""
    out: List[SourceValueRule] = []
    for d in approved or []:
        if not isinstance(d, dict):
            continue
        rule = rule_from_decision(d, context)
        if rule is not None:
            out.append(rule)
    return out


def rules_from_rows(rows: Iterable[Dict[str, Any]]) -> List[SourceValueRule]:
    """Rehydrate rules from their serialised ``as_row`` form (handoff contract)."""
    out: List[SourceValueRule] = []
    for r in rows or []:
        if not isinstance(r, dict) or not r.get("source_value"):
            continue
        out.append(SourceValueRule(
            source_value=str(r.get("source_value", "")),
            canonical_field=str(r.get("canonical_field", "")),
            source_column=str(r.get("source_column", "")),
            target_contract_id=str(r.get("target_contract_id", "")),
            client_id=str(r.get("client_id", "")),
            source_portfolio_id=str(r.get("source_portfolio_id", "")),
            source_file=str(r.get("source_file", "")),
            treatment=str(r.get("treatment", TREATMENT_NULL)),
            decision_id=str(r.get("decision_id", "")),
            approved_by=str(r.get("approved_by", "")),
            approved_at=str(r.get("approved_at", "")),
            source_schema_fingerprint=str(r.get("source_schema_fingerprint", "")),
        ))
    return out


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

def apply_rules_to_column(
    df: Any, canonical_field: str, rules: Sequence[SourceValueRule],
) -> Dict[str, Any]:
    """Null every approved value in ``df[canonical_field]``. Returns a report.

    Runs BEFORE type normalisation, so the parser never sees the placeholder at
    all. The report carries the ORIGINAL source value and the approved treatment
    for each rule, plus the row count affected, for lineage and the non-blocking
    warning.
    """
    import pandas as pd

    if canonical_field not in getattr(df, "columns", []):
        return {"canonical_field": canonical_field, "applied": False,
                "reason": "field_absent", "normalised_row_count": 0, "rules": []}

    keys = {r.value_key: r for r in rules if r.treatment == TREATMENT_NULL}
    if not keys:
        return {"canonical_field": canonical_field, "applied": False,
                "reason": "no_rules", "normalised_row_count": 0, "rules": []}

    column = df[canonical_field]
    normalised = column.map(_norm_value)
    applied_rows: List[Dict[str, Any]] = []
    total = 0
    for key, rule in keys.items():
        mask = normalised == key
        count = int(mask.sum())
        if count:
            df.loc[mask, canonical_field] = pd.NA
            total += count
        applied_rows.append({**rule.as_row(), "normalised_row_count": count})

    return {
        "canonical_field": canonical_field,
        "applied": total > 0,
        "normalised_row_count": total,
        "rules": applied_rows,
    }


def apply_rules(
    df: Any, rules: Sequence[SourceValueRule],
    contract_rows: Sequence[Dict[str, Any]], context: NormalisationContext,
) -> Dict[str, Dict[str, Any]]:
    """Apply every in-scope rule to the tape. ``{canonical_field: report}``.

    Driven by the RULES, not by the target-contract rows: a rule can legitimately
    apply to a mapped column that the target contract does not list
    (``borrower_1_DOB`` is parsed into the canonical tape but is not a field of
    the MI semantics registry).

    ``contract_rows`` are consulted only as a SAFETY CHECK: if the contract maps
    this canonical field from a different source column than the one the decision
    was approved against, the rule is skipped. The mapping has changed, so the
    operator's decision is about something else and must be re-taken.
    """
    column_by_canonical: Dict[str, str] = {}
    for row in contract_rows or []:
        canonical = str(row.get("canonical_field") or "").strip()
        column = str(row.get("selected_source_column") or "").strip()
        if canonical and column:
            column_by_canonical.setdefault(canonical, column)

    by_field: Dict[str, List[SourceValueRule]] = {}
    for rule in rules:
        if not rule.in_scope_for_run(context):
            continue
        if not rule.canonical_field:
            continue
        mapped_column = column_by_canonical.get(rule.canonical_field)
        if mapped_column and mapped_column.strip().lower() != rule.source_column.strip().lower():
            continue
        by_field.setdefault(rule.canonical_field, []).append(rule)

    reports: Dict[str, Dict[str, Any]] = {}
    for canonical, field_rules in by_field.items():
        report = apply_rules_to_column(df, canonical, field_rules)
        if report.get("rules"):
            reports[canonical] = report
    return reports


# --------------------------------------------------------------------------- #
# Detection — propose decisions, never take them
# --------------------------------------------------------------------------- #

def _parser_for_format(fmt: str):
    """The deterministic parser for a canonical format, or None."""
    from engine.gate_2_transform import canonical_transform as ct

    fmt = (fmt or "").strip().lower()
    if fmt == "date":
        return ct.to_iso_date
    if fmt in {"percentage", "percent", "pct"}:
        return ct.to_percentage
    if fmt in {"decimal", "number", "float"}:
        return ct.to_decimal
    if fmt in {"integer", "int"}:
        return ct.to_integer
    if fmt in {"boolean", "bool", "y/n"}:
        return ct.to_bool_yn
    return None


def unparseable_values(values: Iterable[Any], fmt: str) -> List[Tuple[str, int]]:
    """Distinct values of a column that the deterministic parser rejects.

    Returns ``[(original_value, row_count), …]`` most-frequent first. Blank
    renderings and ND codes are excluded — they are already handled as absent,
    not as values needing a decision.
    """
    import pandas as pd
    from engine.gate_2_transform import canonical_transform as ct

    parser = _parser_for_format(fmt)
    if parser is None:
        return []
    series = pd.Series(list(values), dtype="object")
    if series.empty:
        return []

    counts: Dict[str, int] = {}
    originals: Dict[str, str] = {}
    for raw in series.tolist():
        # Blank renderings are ABSENT, not values needing a decision — using the
        # same definition typing uses, so the two never disagree about which
        # cells hold a real value.
        if ct.is_blank_token(raw):
            continue
        key = _norm_value(raw)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        originals.setdefault(key, str(raw).strip())

    if not counts:
        return []
    distinct = pd.Series(list(originals.values()), dtype="object")
    parsed = parser(distinct)
    nd_mask = distinct.astype(str).str.match(ct.ND_PATTERN, na=False)

    failures: List[Tuple[str, int]] = []
    for i, original in enumerate(originals.values()):
        if bool(nd_mask.iloc[i]):
            continue
        if pd.isna(parsed.iloc[i]):
            failures.append((original, counts[_norm_value(original)]))
    failures.sort(key=lambda t: (-t[1], t[0]))
    return failures


def mapping_rows_from(
    resolved_rows: Optional[Sequence[Dict[str, Any]]] = None,
    coverage_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Every source column → canonical field mapping this run established.

    Drawn from the RESOLVER output rather than the target-contract coverage
    matrix, because a column can be mapped into the canonical tape without being
    a field of the target contract — ``borrower_1_DOB`` is an analytics field
    that the MI semantics registry does not list, but it is still parsed, and a
    placeholder in it is still an operator decision.

    Coverage rows supply the friendly ``target_field`` label where the mapping is
    also a target-contract field.
    """
    target_by_canonical: Dict[str, str] = {}
    for row in coverage_rows or []:
        canonical = str(row.get("canonical_field") or row.get("match_field") or "").strip()
        if canonical:
            target_by_canonical.setdefault(canonical, str(row.get("target_field") or ""))

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for row in resolved_rows or []:
        canonical = str(row.get("resolved_target_field")
                        or row.get("canonical_field")
                        or row.get("candidate_canonical_field") or "").strip()
        column = str(row.get("source_column") or "").strip()
        src_file = str(row.get("source_file") or "").strip()
        sheet = str(row.get("source_sheet") or "").strip()
        if not (canonical and column and src_file):
            continue
        key = (src_file, sheet, column, canonical)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "canonical_field": canonical,
            "source_file": src_file,
            "source_sheet": sheet,
            "source_column": column,
            "target_field": target_by_canonical.get(canonical, canonical),
        })
    return out


class CandidateMatchError(ValueError):
    """No candidate matched, or the request was ambiguous / out of scope."""


#: Scope dimensions an operator may pin on the command line. Each is optional;
#: supplying one means "and it must be exactly this", so a request can never
#: approve a candidate whose scope differs from what the operator believed.
_REQUESTABLE_SCOPE = (
    "target_contract_id", "client_id", "source_portfolio_id",
    "source_schema_fingerprint", "source_file", "source_sheet",
)


def match_candidate(
    candidates: Sequence[Dict[str, Any]],
    *,
    canonical_field: str,
    source_column: str,
    source_value: str,
    action: str = ACTION_TREAT_AS_NULL,
    scope: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Find the ONE pending candidate an approval request identifies.

    Fails closed, and says which of the four ways it failed:

      * the action is not one this module implements;
      * nothing matches — approving a value that was never detected would create
        a rule for a value the pipeline has no evidence of;
      * more than one matches — approving "whichever" is never right;
      * a scope dimension the operator pinned differs from the candidate's.

    Matching is exact on canonical field, source column and source value (the
    value case-insensitively, since case is rendering), plus every scope
    dimension the request pins.
    """
    if str(action or "").strip() != ACTION_TREAT_AS_NULL:
        raise CandidateMatchError(
            f"unsupported action {action!r}; supported: {ACTION_TREAT_AS_NULL}")

    scope = {k: str(v).strip() for k, v in (scope or {}).items()
             if v not in (None, "")}
    unknown = sorted(set(scope) - set(_REQUESTABLE_SCOPE))
    if unknown:
        raise CandidateMatchError(
            f"unknown scope dimension(s): {unknown}; "
            f"supported: {sorted(_REQUESTABLE_SCOPE)}")

    want_field = str(canonical_field or "").strip().lower()
    want_column = str(source_column or "").strip().lower()
    want_value = _norm_value(source_value)
    if not (want_field and want_column and want_value):
        raise CandidateMatchError(
            "canonical_field, source_column and source_value are all required")

    identified = [
        c for c in (candidates or [])
        if str(c.get("canonical_field", "")).strip().lower() == want_field
        and str(c.get("source_column", "")).strip().lower() == want_column
        and _norm_value(c.get("source_value")) == want_value
    ]
    if not identified:
        raise CandidateMatchError(
            f"no pending candidate for {source_value!r} in "
            f"{source_column!r} -> {canonical_field!r}")

    # Scope is checked AFTER identification so a mismatch reports what actually
    # differs, rather than a bare "no match".
    in_scope: List[Dict[str, Any]] = []
    scope_conflicts: List[str] = []
    for c in identified:
        differences = [
            f"{k}: candidate={c.get(k, '') or '(none)'!r} requested={v!r}"
            for k, v in scope.items()
            if str(c.get(k, "") or "").strip().lower() != v.lower()
        ]
        if differences:
            scope_conflicts.extend(differences)
        else:
            in_scope.append(c)

    if not in_scope:
        raise CandidateMatchError(
            "candidate scope differs from the requested scope — "
            + "; ".join(sorted(set(scope_conflicts))))
    if len(in_scope) > 1:
        where = "; ".join(
            f"{c.get('source_file', '')}::{c.get('source_sheet', '') or '-'}"
            for c in in_scope)
        raise CandidateMatchError(
            f"{len(in_scope)} candidates match — narrow the request with "
            f"--source-file / --source-sheet (matched: {where})")
    return in_scope[0]


def decision_from_candidate(
    candidate: Dict[str, Any], *, decision_id: str = "",
    approved_by: str = "", approved_at: str = "",
) -> Dict[str, Any]:
    """The APPROVED 34_ decision entry for a matched candidate.

    Carries the candidate's scope VERBATIM — it is not re-derived here and must
    not be re-derived downstream, so the rule that runs is provably the rule the
    operator approved.
    """
    return {
        "decision_id": (decision_id or candidate.get("decision_id", "")
                        or f"SV-{candidate.get('canonical_field', '')}-"
                           f"{_norm_value(candidate.get('source_value'))}"),
        "decision_type": DECISION_TYPE,
        "status": "approved",
        "selected_action": ACTION_TREAT_AS_NULL,
        "target_field": candidate.get("target_field", "")
        or candidate.get("canonical_field", ""),
        "canonical_field": candidate.get("canonical_field", ""),
        "source_column": candidate.get("source_column", ""),
        "source_file": candidate.get("source_file", ""),
        "source_sheet": candidate.get("source_sheet", ""),
        "source_value": candidate.get("source_value", ""),
        "affected_row_count": candidate.get("affected_row_count", 0),
        "target_contract_id": candidate.get("target_contract_id", ""),
        "client_id": candidate.get("client_id", ""),
        "source_portfolio_id": candidate.get("source_portfolio_id", ""),
        "source_schema_fingerprint": candidate.get("source_schema_fingerprint", ""),
        "approved_by": approved_by or "operator",
        "approved_at": approved_at,
    }


def scope_of_rules(rules: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """The scope shared by a set of approved rule rows, for the handoff block.

    Taken from the rules themselves so the scope Transformation matches against
    is the one that was approved, never one re-derived from the current run.
    Dimensions the rules disagree on are left blank (a wildcard) rather than
    picking one arbitrarily.
    """
    dims = ("client_id", "source_portfolio_id", "target_contract_id",
            "source_schema_fingerprint")
    out: Dict[str, str] = {}
    for dim in dims:
        values = {str(r.get(dim, "") or "").strip() for r in (rules or [])}
        values.discard("")
        out[dim] = values.pop() if len(values) == 1 else ""
    return out


def detect_candidates(
    tables: Sequence[Tuple[str, str, Any]],
    mapping_rows: Sequence[Dict[str, Any]],
    registry_fields: Dict[str, Any],
    context: NormalisationContext,
    *,
    existing_rules: Optional[Sequence[SourceValueRule]] = None,
    max_per_column: int = MAX_CANDIDATES_PER_COLUMN,
) -> List[Dict[str, Any]]:
    """Propose source-value normalisation decisions for operator review.

    For every mapped source column whose canonical field is parsed
    deterministically, finds the distinct source values the parser rejects and
    returns one candidate per (column, value). Values an approved rule already
    covers are skipped — a settled decision is not re-asked.

    ``mapping_rows`` are ``{canonical_field, source_file, source_sheet,
    source_column, target_field}`` — see :func:`mapping_rows_from`.

    Purely advisory: this proposes, it never normalises anything.
    """
    frames = {(str(f), str(s or "")): df for f, s, df in (tables or [])}
    by_file = {str(f): df for f, _s, df in (tables or [])}
    settled = {
        (r.canonical_field.lower(), r.source_column.lower(), r.value_key)
        for r in (existing_rules or [])
    }

    candidates: List[Dict[str, Any]] = []
    seen_columns: set = set()
    for row in mapping_rows or []:
        canonical = str(row.get("canonical_field") or "").strip()
        column = str(row.get("source_column") or "").strip()
        src_file = str(row.get("source_file") or "").strip()
        sheet = str(row.get("source_sheet") or "").strip()
        if not (canonical and column and src_file):
            continue
        fmt = str((registry_fields.get(canonical, {}) or {}).get("format", "")).lower()
        if fmt not in _PARSEABLE_FORMATS:
            continue
        key = (src_file, sheet, column, canonical)
        if key in seen_columns:
            continue
        seen_columns.add(key)

        # Explicit None check: a DataFrame has no usable truth value.
        df = frames.get((src_file, sheet))
        if df is None:
            df = by_file.get(src_file)
        if df is None or column not in getattr(df, "columns", []):
            continue

        for original, count in unparseable_values(df[column].tolist(), fmt)[:max_per_column]:
            if (canonical.lower(), column.lower(), _norm_value(original)) in settled:
                continue
            candidates.append({
                "target_field": row.get("target_field", "") or canonical,
                "canonical_field": canonical,
                "source_file": src_file,
                "source_sheet": sheet,
                "source_column": column,
                "source_value": original,
                "affected_row_count": count,
                "canonical_format": fmt,
                "target_contract_id": context.target_contract_id,
                "client_id": context.client_id,
                "source_portfolio_id": context.source_portfolio_id,
                "source_schema_fingerprint": context.source_schema_fingerprint,
            })
    return candidates
