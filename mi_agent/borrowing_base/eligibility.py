"""mi_agent.borrowing_base.eligibility — governed loan-level eligibility.

This is the ONE place a loan is decided to be an Eligible Mortgage Loan for a
funding facility. It runs in the canonical preparation layer, so Dashboard, MI,
Teams, PPTX, forecasting and any future regulatory or funding component all
read the same four columns off the same frame rather than each deciding for
themselves.

The three rules that keep it honest:

* **Nothing is inferred from the concentration limits.** Schedule 8 states what
  a PORTFOLIO of Eligible Mortgage Loans must look like; it does not define an
  Eligible Mortgage Loan. Passing every concentration test says nothing about
  whether an individual loan qualifies.
* **Fail closed.** With no approved criteria the governed status is
  UNDETERMINED, for every loan, with the reason named. A production facility
  never becomes eligible by omission.
* **A missing input is not a failure.** A rule whose input a loan does not
  carry makes that loan UNDETERMINED, not INELIGIBLE.

Field resolution reuses the concentration-test library's ``field_roles``, so a
rule written against ``balance_current`` keeps working across the column names
different books arrive with — one governed field vocabulary, not two.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics_lib.numeric import coerce_numeric

from .models import (
    ELIGIBLE,
    FIELD_ELIGIBILITY_REASON,
    FIELD_ELIGIBILITY_STATUS,
    FIELD_ELIGIBLE,
    FIELD_FACILITY_ID,
    INELIGIBLE,
    REASON_NO_APPROVED_RULES,
    REASON_OUTSIDE_FINANCING_PORTFOLIO,
    REASON_PROTOTYPE_ASSUMPTION,
    REASON_RULES_SATISFIED,
    REASON_RULE_INPUT_MISSING,
    UNDETERMINED,
    EligibilityRule,
    FacilityConfiguration,
)

logger = logging.getLogger("mi_agent.borrowing_base.eligibility")

#: Columns that may carry the source-portfolio identity used to scope the
#: Financing Portfolio. Same names the platform provenance stamps.
_PORTFOLIO_ID_COLUMNS = ("source_portfolio_id", "portfolio_id")
_PORTFOLIO_TYPE_COLUMNS = ("source_portfolio_type", "portfolio_type")

_BLANK_TOKENS = ("", "nan", "none", "nat", "<na>", "null")


def _blank(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return series.isna() | text.isin(_BLANK_TOKENS)


def _library():
    """The governed field-role vocabulary, or None when it cannot be loaded."""
    try:
        from mi_agent.concentration_tests.library import load_library
        return load_library()
    except Exception as exc:  # noqa: BLE001 — eligibility must never break prep
        logger.info("concentration library unavailable for role resolution: %s",
                    exc)
        return None


def resolve_rule_column(df: pd.DataFrame, rule: EligibilityRule,
                        lib: Any = None) -> Optional[str]:
    """The frame column one rule reads, or None when the input is absent."""
    if rule.field:
        return rule.field if rule.field in df.columns else None
    if not rule.field_role:
        return None
    if lib is not None:
        try:
            from mi_agent.concentration_tests.metrics import resolve_role_column
            col = resolve_role_column(df, lib, rule.field_role)
            if col:
                return col
        except Exception:  # noqa: BLE001
            pass
    return rule.field_role if rule.field_role in df.columns else None


def financing_portfolio_mask(df: pd.DataFrame,
                             facility: FacilityConfiguration) -> pd.Series:
    """Which rows are in this facility's Financing Portfolio.

    An empty selector means the whole governed canonical population — the ERE
    shape, and the safe default: a facility that does not narrow its portfolio
    is not silently narrowed by us. A selector naming values the frame cannot
    express (no source-portfolio column at all) selects NOTHING, so an
    unenforceable narrowing fails closed instead of quietly selecting the whole
    book it was meant to restrict.
    """
    scope = facility.financing_portfolio or {}
    ids = [str(v).strip().lower() for v in (scope.get("source_portfolio_ids") or [])
           if str(v).strip()]
    types = [str(v).strip().lower()
             for v in (scope.get("source_portfolio_types") or []) if str(v).strip()]
    if not ids and not types:
        return pd.Series(True, index=df.index)

    mask = pd.Series(False, index=df.index)
    matched_any_column = False
    if ids:
        for col in _PORTFOLIO_ID_COLUMNS:
            if col in df.columns:
                matched_any_column = True
                mask = mask | df[col].astype(str).str.strip().str.lower().isin(ids)
                break
    if types:
        for col in _PORTFOLIO_TYPE_COLUMNS:
            if col in df.columns:
                matched_any_column = True
                mask = mask | df[col].astype(str).str.strip().str.lower().isin(types)
                break
    if not matched_any_column:
        return pd.Series(False, index=df.index)
    return mask


def _apply_rule(df: pd.DataFrame, rule: EligibilityRule, column: str
                ) -> Tuple[pd.Series, pd.Series]:
    """``(fails, unknown)`` boolean masks for one rule over the frame."""
    series = df[column]
    unknown = _blank(series)

    if rule.operator == "present":
        return unknown.copy(), pd.Series(False, index=df.index)

    if rule.operator in ("max", "min"):
        numeric = coerce_numeric(series)
        unknown = unknown | numeric.isna()
        threshold = float(rule.value)
        if rule.operator == "max":
            fails = (numeric > threshold).fillna(False)
        else:
            fails = (numeric < threshold).fillna(False)
        return fails & ~unknown, unknown

    text = series.astype(str).str.strip().str.lower()
    if rule.operator == "equals":
        fails = text != str(rule.value).strip().lower()
    elif rule.operator == "not_equals":
        fails = text == str(rule.value).strip().lower()
    elif rule.operator == "in":
        allowed = {str(v).strip().lower() for v in (rule.value or [])}
        fails = ~text.isin(allowed)
    elif rule.operator == "not_in":
        blocked = {str(v).strip().lower() for v in (rule.value or [])}
        fails = text.isin(blocked)
    else:  # unreachable for a validated rule
        return pd.Series(False, index=df.index), pd.Series(True, index=df.index)
    return fails & ~unknown, unknown


def derive_eligibility(df: pd.DataFrame, facility: FacilityConfiguration
                       ) -> Dict[str, Any]:
    """Materialise the four governed eligibility columns onto ``df`` IN PLACE.

    Returns the derivation receipt: the rules applied, what each of them could
    and could not read, the status counts, and every prototype assumption used.
    The receipt is what the transformation/derivation provenance carries, so a
    derived status can never be mistaken for a sourced one.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return {"applied": False, "reason": "no frame"}

    total = len(df)
    index = df.index
    in_scope = financing_portfolio_mask(df, facility)

    status = pd.Series(pd.NA, index=index, dtype="object")
    reason = pd.Series(pd.NA, index=index, dtype="object")

    lib = _library() if facility.eligibility_rules else None
    rules_applied: List[Dict[str, Any]] = []
    assumptions: List[str] = []

    if not facility.eligibility_rules:
        # No approved contractual criteria exist for this facility.
        if facility.prototype_assumption_active:
            status = status.mask(in_scope, ELIGIBLE)
            reason = reason.mask(in_scope, REASON_PROTOTYPE_ASSUMPTION)
            assumptions.append(
                "prototype_assume_financing_portfolio_eligible: every loan in "
                "the configured Financing Portfolio is treated as an Eligible "
                "Mortgage Loan. This is a PROTOTYPE ASSUMPTION, not a "
                "contractual eligibility determination.")
        else:
            status = status.mask(in_scope, UNDETERMINED)
            reason = reason.mask(in_scope, REASON_NO_APPROVED_RULES)
    else:
        fails_any = pd.Series(False, index=index)
        unknown_any = pd.Series(False, index=index)
        fail_reason = pd.Series(pd.NA, index=index, dtype="object")
        unknown_reason = pd.Series(pd.NA, index=index, dtype="object")

        for rule in facility.eligibility_rules:
            column = resolve_rule_column(df, rule, lib)
            if column is None:
                # The whole input is absent from the book: every in-scope loan
                # is undetermined against this rule, never ineligible.
                missing = in_scope & ~unknown_any
                unknown_reason = unknown_reason.mask(
                    missing, f"{REASON_RULE_INPUT_MISSING}:{rule.rule_id}")
                unknown_any = unknown_any | in_scope
                rules_applied.append({
                    "rule_id": rule.rule_id, "resolved_column": None,
                    "input_available": False,
                    "description": rule.description,
                    "failed_loans": 0, "undetermined_loans": int(in_scope.sum()),
                })
                continue
            fails, unknown = _apply_rule(df, rule, column)
            fails = fails & in_scope
            unknown = unknown & in_scope
            code = rule.reason_code or rule.rule_id
            fail_reason = fail_reason.mask(fails & fail_reason.isna(), code)
            unknown_reason = unknown_reason.mask(
                unknown & unknown_reason.isna(),
                f"{REASON_RULE_INPUT_MISSING}:{rule.rule_id}")
            fails_any = fails_any | fails
            unknown_any = unknown_any | unknown
            rules_applied.append({
                "rule_id": rule.rule_id, "resolved_column": column,
                "input_available": True,
                "description": rule.description,
                "operator": rule.operator, "value": rule.value,
                "failed_loans": int(fails.sum()),
                "undetermined_loans": int(unknown.sum()),
            })

        # A definite failure beats a missing input: a loan that breaches an
        # approved criterion is ineligible whatever else could not be read.
        ineligible = in_scope & fails_any
        undetermined = in_scope & ~fails_any & unknown_any
        eligible = in_scope & ~fails_any & ~unknown_any
        status = status.mask(ineligible, INELIGIBLE)
        status = status.mask(undetermined, UNDETERMINED)
        status = status.mask(eligible, ELIGIBLE)
        reason = reason.mask(ineligible, fail_reason[ineligible])
        reason = reason.mask(undetermined, unknown_reason[undetermined])
        reason = reason.mask(eligible, REASON_RULES_SATISFIED)

    # Out of the Financing Portfolio: NOT this facility's population at all.
    # Left with no status so the facility's own reconciliation partitions
    # exactly the loans it governs, and marked in the reason so the exclusion
    # is visible rather than looking like a gap.
    reason = reason.mask(~in_scope, REASON_OUTSIDE_FINANCING_PORTFOLIO)

    eligible_flag = pd.Series(pd.NA, index=index, dtype="object")
    eligible_flag = eligible_flag.mask(status == ELIGIBLE, True)
    eligible_flag = eligible_flag.mask(status == INELIGIBLE, False)
    # UNDETERMINED and out-of-scope stay NA: a tri-state cannot be flattened
    # into a boolean without turning "we cannot tell" into "no".

    df[FIELD_ELIGIBILITY_STATUS] = status
    df[FIELD_ELIGIBILITY_REASON] = reason
    df[FIELD_ELIGIBLE] = eligible_flag.astype("boolean")
    df[FIELD_FACILITY_ID] = pd.Series(pd.NA, index=index, dtype="object").mask(
        in_scope, facility.facility_id)

    counts = {
        ELIGIBLE: int((status == ELIGIBLE).sum()),
        INELIGIBLE: int((status == INELIGIBLE).sum()),
        UNDETERMINED: int((status == UNDETERMINED).sum()),
    }
    return {
        "applied": True,
        "facility_id": facility.facility_id,
        "client_id": facility.client_id,
        "environment": facility.environment,
        "eligibility_rule_version": facility.eligibility_rule_version,
        "eligibility_governed": facility.eligibility_governed,
        "rules_applied": rules_applied,
        "derived_fields": list((FIELD_ELIGIBLE, FIELD_ELIGIBILITY_STATUS,
                                FIELD_ELIGIBILITY_REASON, FIELD_FACILITY_ID)),
        "derivation": "mi_agent.borrowing_base.eligibility.derive_eligibility",
        "config_source": facility.config_source,
        "config_version": facility.config_version,
        "config_hash": facility.content_hash(),
        "total_rows": int(total),
        "financing_portfolio_rows": int(in_scope.sum()),
        "out_of_scope_rows": int(total - in_scope.sum()),
        "status_counts": counts,
        "prototype_assumptions_used": assumptions,
    }


# --------------------------------------------------------------------------- #
# Population selection — the ONE way every consumer selects a population
# --------------------------------------------------------------------------- #
def status_series(df: pd.DataFrame) -> pd.Series:
    """The governed status column, or an all-NA series when it is absent."""
    if df is None or FIELD_ELIGIBILITY_STATUS not in getattr(df, "columns", []):
        idx = getattr(df, "index", pd.Index([]))
        return pd.Series(pd.NA, index=idx, dtype="object")
    return df[FIELD_ELIGIBILITY_STATUS]


def eligible_mask(df: pd.DataFrame,
                  facility: Optional[FacilityConfiguration] = None) -> pd.Series:
    """Boolean mask of Eligible Mortgage Loans.

    Reads the governed STATUS, never the boolean flag, so UNDETERMINED can
    never be silently counted as eligible. Scoped to the facility when one is
    given, so a frame carrying several facilities' loans cannot leak between
    them.
    """
    status = status_series(df)
    mask = (status == ELIGIBLE).fillna(False)
    if facility is not None and FIELD_FACILITY_ID in getattr(df, "columns", []):
        mask = mask & (df[FIELD_FACILITY_ID] == facility.facility_id).fillna(False)
    return mask.astype(bool)


def status_mask(df: pd.DataFrame, status: str,
                facility: Optional[FacilityConfiguration] = None) -> pd.Series:
    mask = (status_series(df) == status).fillna(False)
    if facility is not None and FIELD_FACILITY_ID in getattr(df, "columns", []):
        mask = mask & (df[FIELD_FACILITY_ID] == facility.facility_id).fillna(False)
    return mask.astype(bool)


def in_financing_portfolio_mask(df: pd.DataFrame,
                                facility: Optional[FacilityConfiguration] = None
                                ) -> pd.Series:
    """Every loan this facility governs, whatever its status.

    Membership is read from the FACILITY ATTRIBUTION, not from the status.
    That is deliberate and it is what makes the reconciliation invariant able
    to fail: a loan attributed to the facility whose status is missing or
    unrecognised stays in the Financing Portfolio and lands in none of the
    three buckets, so the partition breaks loudly. Deriving membership from
    the status instead would quietly shrink the portfolio to fit whatever
    statuses happened to be present, and no loan could ever be seen to
    disappear.

    A frame with no attribution column at all (nothing has been derived on it)
    falls back to the statuses, so an empty frame is empty rather than wrong.
    """
    if FIELD_FACILITY_ID in getattr(df, "columns", []):
        attributed = df[FIELD_FACILITY_ID].notna()
        if facility is not None:
            attributed = attributed & (
                df[FIELD_FACILITY_ID] == facility.facility_id)
        return attributed.fillna(False).astype(bool)
    mask = status_series(df).isin(list((ELIGIBLE, INELIGIBLE, UNDETERMINED)))
    return mask.fillna(False).astype(bool)


def eligibility_available(df: pd.DataFrame) -> bool:
    """Whether the governed derivation has run on this frame."""
    return (df is not None
            and FIELD_ELIGIBILITY_STATUS in getattr(df, "columns", [])
            and bool(status_series(df).notna().any()))


__all__ = [
    "derive_eligibility", "eligible_mask", "status_mask", "status_series",
    "in_financing_portfolio_mask", "eligibility_available",
    "financing_portfolio_mask", "resolve_rule_column",
]
