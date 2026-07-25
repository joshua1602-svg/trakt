"""demo_platform.assertions — the reconciliation gate.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

The demo run FAILS unless every claim the film makes is true of the data the
pipeline actually produced. Values are never adjusted to match the script: this
module is the check that the engineered source data really does produce the
narrative.

Checked here:

  * current and prior consolidated totals reconcile to the assembled platform
    canonicals;
  * the consolidated movement is the target ~£18.2m (within tolerance);
  * each portfolio's movement reconciles, and the two sum exactly to the
    consolidated movement;
  * the South East is the primary regional contributor, by both share and lead
    over the runner-up;
  * the completions attribution the answer asserts is evidenced;
  * weighted-average current LTV and average borrower age hit their targets;
  * LTV is broadly stable and borrower age rises, as the narrative states;
  * the consolidated loan count sits in the credible band;
  * loan identifiers are unique within each portfolio and the composite platform
    key is unique across the platform;
  * every derived economic field reconciles at row level (LTV to balance and
    valuation; redeemed accounts carry nil balance);
  * the two source schemas are genuinely different;
  * the React MI Agent answer and the Copilot answer carry the SAME figures;
  * no prohibited client string appears in any generated file.

:func:`run` returns a machine-readable assertion report. :func:`enforce` raises
on any failure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import config as cfg
from . import schemas


class AssertionFailure(RuntimeError):
    """Raised when the demonstration does not reconcile."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    observed: Any = None
    expected: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.name, "passed": self.passed, "detail": self.detail,
                "observed": self.observed, "expected": self.expected}


def _platform_csv(prd: cfg.PeriodSpec) -> Path:
    return (cfg.local_blob_root() / "processed" / "platform" / cfg.CLIENT_ID
            / prd.reporting_date / "platform_canonical_typed.csv")


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _money(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"£{value / 1e6:,.3f}m"


# --------------------------------------------------------------------------- #
# Row-level economic coherence
# --------------------------------------------------------------------------- #
def _row_level_checks(df: pd.DataFrame, label: str) -> List[Check]:
    checks: List[Check] = []

    bal = _num(df["current_outstanding_balance"])
    val = _num(df["current_valuation_amount"])
    ltv = _num(df["current_loan_to_value"])
    testable = bal.notna() & val.notna() & (val > 0) & ltv.notna() & (bal > 0)
    if testable.any():
        implied = bal[testable] / val[testable] * 100.0
        worst = float((implied - ltv[testable]).abs().max())
        checks.append(Check(
            f"{label}: current LTV reconciles to balance and valuation",
            worst <= 0.01,
            f"largest row-level difference between the reported LTV and "
            f"balance ÷ valuation is {worst:.6f} percentage points",
            observed=round(worst, 8), expected="<= 0.01pp"))

    obal = _num(df["original_outstanding_balance"]) if "original_outstanding_balance" in df.columns else None
    oprin = _num(df["original_principal_balance"])
    oval = _num(df["original_valuation_amount"])
    oltv = _num(df["original_loan_to_value"])
    ot = oprin.notna() & oval.notna() & (oval > 0) & oltv.notna()
    if ot.any():
        implied = oprin[ot] / oval[ot] * 100.0
        worst = float((implied - oltv[ot]).abs().max())
        checks.append(Check(
            f"{label}: original LTV reconciles to advance and original valuation",
            worst <= 0.01,
            f"largest row-level difference is {worst:.6f} percentage points",
            observed=round(worst, 8), expected="<= 0.01pp"))

    # Outstanding = principal + accrued interest (nil on this product).
    prin = _num(df["current_principal_balance"])
    both = bal.notna() & prin.notna()
    if both.any():
        worst = float((bal[both] - prin[both]).abs().max())
        checks.append(Check(
            f"{label}: outstanding balance reconciles to principal",
            worst <= 0.02,
            f"largest row-level difference is £{worst:.4f} "
            f"(interest is capitalised monthly, so accrued interest is nil)",
            observed=round(worst, 6), expected="<= £0.02"))

    # Redeemed accounts must carry nil balance.
    if "account_status" in df.columns:
        status = df["account_status"].astype(str).str.strip().str.lower()
        redeemed = status.isin(("redeemed", "repaid", "closed", "matured"))
        if redeemed.any():
            leaked = float(bal[redeemed].fillna(0.0).abs().sum())
            checks.append(Check(
                f"{label}: redeemed accounts carry nil balance",
                leaked <= 0.01,
                f"{int(redeemed.sum())} redeemed account(s) carry "
                f"£{leaked:,.2f} of balance in total",
                observed=round(leaked, 2), expected="£0.00"))

    # Borrower age within the product's own bounds, except the deliberately
    # injected exception rows.
    age = _num(df["youngest_borrower_age"]).dropna()
    if not age.empty:
        below = int((age < 55).sum())
        above = int((age > 110).sum())
        expected_injected = cfg.INJECTED_EXCEPTIONS_PER_PERIOD * len(cfg.PORTFOLIOS)
        checks.append(Check(
            f"{label}: borrower ages are within product bounds",
            below <= expected_injected and above == 0,
            f"{below} row(s) below the product minimum of 55 (deliberately "
            f"injected exceptions, budget {expected_injected}), {above} above 110",
            observed={"below_55": below, "above_110": above},
            expected=f"below_55 <= {expected_injected}, above_110 == 0"))

    return checks


# --------------------------------------------------------------------------- #
# The assertion run
# --------------------------------------------------------------------------- #
def run(export_payload: Optional[Dict[str, Any]] = None, *,
        verbose: bool = True) -> Dict[str, Any]:
    """Run every reconciliation check and return the assertion report."""
    checks: List[Check] = []

    if export_payload is None:
        path = cfg.export_dir() / "demo_metrics.json"
        if not path.exists():
            raise AssertionFailure(
                f"{path} not found — export the metrics before asserting "
                f"(python -m demo_platform.run_demo --metrics).")
        export_payload = json.loads(path.read_text(encoding="utf-8"))

    lenses = export_payload["metrics"]["lenses"]
    total = lenses["total"]
    movement = total["movement"]
    summary = total["summary"]
    a_id, b_id = (p.source_portfolio_id for p in cfg.PORTFOLIOS)

    # -- 1. Period totals reconcile to the assembled canonicals ------------ #
    cur_csv, pri_csv = _platform_csv(cfg.CURRENT_PERIOD), _platform_csv(cfg.PRIOR_PERIOD)
    for path in (cur_csv, pri_csv):
        if not path.exists():
            raise AssertionFailure(f"assembled platform canonical missing: {path}")
    cur_df = pd.read_csv(cur_csv, low_memory=False)
    pri_df = pd.read_csv(pri_csv, low_memory=False)
    cur_total = round(float(_num(cur_df["current_outstanding_balance"]).fillna(0.0).sum()), 2)
    pri_total = round(float(_num(pri_df["current_outstanding_balance"]).fillna(0.0).sum()), 2)

    for label, tape_total, metric_total in (
            ("current", cur_total, movement["current"]["funded_balance"]),
            ("prior", pri_total, movement["prior"]["funded_balance"])):
        diff = abs(tape_total - float(metric_total or 0.0))
        checks.append(Check(
            f"{label} funded balance reconciles to the assembled canonical",
            diff <= 0.05,
            f"tape {_money(tape_total)} vs governed metric {_money(metric_total)} "
            f"(difference £{diff:.4f})",
            observed=tape_total, expected=metric_total))

    # -- 2. Consolidated movement hits the target -------------------------- #
    consolidated = movement["delta"]["funded_balance"]
    t = cfg.TARGET_MOVEMENT_TOTAL
    checks.append(Check(
        "consolidated funded balance movement",
        t.ok(consolidated),
        f"{_money(consolidated)} against a target of {_money(t.target)} "
        f"± {_money(t.tolerance)}",
        observed=consolidated, expected=f"{t.target} ± {t.tolerance}"))
    checks.append(Check(
        "movement equals current minus prior",
        abs((cur_total - pri_total) - float(consolidated or 0.0)) <= 0.05,
        f"{_money(cur_total - pri_total)} from the tapes vs "
        f"{_money(consolidated)} from the governed metric",
        observed=round(cur_total - pri_total, 2), expected=consolidated))

    # -- 3. Per-portfolio movements ---------------------------------------- #
    cohort_moves = {c["id"]: c["delta"] for c in movement["cohortMovements"]}
    for pid, target in ((a_id, cfg.TARGET_MOVEMENT_A), (b_id, cfg.TARGET_MOVEMENT_B)):
        value = cohort_moves.get(pid)
        checks.append(Check(
            f"{cfg.portfolio(pid).display_id} movement",
            target.ok(value),
            f"{_money(value)} against a target of {_money(target.target)} "
            f"± {_money(target.tolerance)}",
            observed=value, expected=f"{target.target} ± {target.tolerance}"))
    cohort_sum = round(sum(cohort_moves.values()), 2)
    checks.append(Check(
        "portfolio movements sum to the consolidated movement",
        abs(cohort_sum - float(consolidated or 0.0)) <= 0.05,
        f"{_money(cohort_sum)} vs {_money(consolidated)}",
        observed=cohort_sum, expected=consolidated))

    # -- 4. Regional attribution ------------------------------------------- #
    primary = movement.get("primaryRegion") or {}
    runner_up = movement.get("runnerUpRegion") or {}
    checks.append(Check(
        "the South East is the primary regional contributor",
        primary.get("region") == cfg.PRIMARY_MOVEMENT_REGION,
        f"largest positive regional contribution is "
        f"{primary.get('region')!r} at {_money(primary.get('delta'))}",
        observed=primary.get("region"), expected=cfg.PRIMARY_MOVEMENT_REGION))
    share = movement.get("primaryRegionShare")
    checks.append(Check(
        "the primary region carries a material share of the movement",
        share is not None and share >= cfg.PRIMARY_REGION_MIN_SHARE,
        f"{(share or 0) * 100:.1f}% of the consolidated movement "
        f"(minimum {cfg.PRIMARY_REGION_MIN_SHARE * 100:.0f}%)",
        observed=share, expected=f">= {cfg.PRIMARY_REGION_MIN_SHARE}"))
    lead = movement.get("primaryRegionLead")
    checks.append(Check(
        "the primary region leads the runner-up decisively",
        lead is not None and lead >= cfg.PRIMARY_REGION_MIN_LEAD,
        f"{lead}x the next largest contributor "
        f"({runner_up.get('region')} at {_money(runner_up.get('delta'))}); "
        f"minimum {cfg.PRIMARY_REGION_MIN_LEAD}x",
        observed=lead, expected=f">= {cfg.PRIMARY_REGION_MIN_LEAD}"))
    region_residual = (movement.get("reconciliation") or {}).get("region_residual")
    checks.append(Check(
        "regional contributions reconcile to the consolidated movement",
        region_residual is not None and abs(region_residual) <= 5_000.0,
        f"residual {_money(region_residual)} (the unattributed-geography bucket "
        f"carries the remainder)",
        observed=region_residual, expected="<= £5,000"))
    checks.append(Check(
        "the completions attribution in the answer is evidenced",
        bool(movement.get("completionsEvidenced")),
        f"loans completed in the month carry "
        f"{_money(movement.get('completionsBalanceInPrimaryRegion'))} in the "
        f"{primary.get('region')}, i.e. "
        f"{(movement.get('completionsShareOfPrimaryRegion') or 0) * 100:.1f}% of "
        f"that region's increase",
        observed=movement.get("completionsShareOfPrimaryRegion"),
        expected=">= 0.5 before the answer may say 'driven by completions'"))

    # -- 5. LTV, age, loan count ------------------------------------------- #
    for target, value in (
            (cfg.TARGET_WA_LTV_CURRENT, movement["current"]["wa_ltv_points"]),
            (cfg.TARGET_BORROWER_AGE_CURRENT, movement["current"]["avg_borrower_age"])):
        checks.append(Check(
            target.name, target.ok(value),
            f"{value:.3f} against a target of {target.target} ± {target.tolerance}",
            observed=value, expected=f"{target.target} ± {target.tolerance}"))

    ltv_delta = movement["delta"]["wa_ltv_points"]
    checks.append(Check(
        "weighted-average LTV is broadly stable, as the answer states",
        ltv_delta is not None and abs(ltv_delta) < cfg.WA_LTV_STABILITY_MAX_CHANGE_POINTS,
        f"moved {ltv_delta:+.3f} percentage points "
        f"(materiality threshold {cfg.WA_LTV_STABILITY_MAX_CHANGE_POINTS})",
        observed=ltv_delta,
        expected=f"|Δ| < {cfg.WA_LTV_STABILITY_MAX_CHANGE_POINTS}"))
    age_delta = movement["delta"]["avg_borrower_age"]
    checks.append(Check(
        "average borrower age increases, as the answer states",
        age_delta is not None and (age_delta > 0) == cfg.BORROWER_AGE_MUST_INCREASE,
        f"moved {age_delta:+.4f} years",
        observed=age_delta, expected="> 0"))

    loan_count = movement["current"]["loan_count"]
    lo, hi = cfg.LOAN_COUNT_BAND
    checks.append(Check(
        "consolidated loan count is within the intended band",
        loan_count is not None and lo <= loan_count <= hi,
        f"{loan_count:,} loans (band {lo:,}–{hi:,})",
        observed=loan_count, expected=f"{lo}–{hi}"))

    # -- 6. Identity + assembly integrity ---------------------------------- #
    if "platform_loan_key" in cur_df.columns:
        dupes = int(cur_df["platform_loan_key"].duplicated().sum())
        checks.append(Check(
            "the composite platform key is unique",
            dupes == 0,
            f"{dupes} duplicate source_portfolio_id + loan_identifier key(s)",
            observed=dupes, expected=0))
    ids = cur_df["source_portfolio_id"].astype(str).str.strip()
    checks.append(Check(
        "both portfolios retain their identity in the platform view",
        set(ids.unique()) == {a_id, b_id},
        f"source portfolios present: {sorted(set(ids.unique()))}",
        observed=sorted(set(ids.unique())), expected=[a_id, b_id]))
    for pid in (a_id, b_id):
        sub = cur_df[ids == pid]
        dup = int(sub["loan_identifier"].duplicated().sum())
        checks.append(Check(
            f"{cfg.portfolio(pid).display_id}: loan identifiers are unique",
            dup == 0, f"{dup} duplicate loan identifier(s) in {len(sub):,} rows",
            observed=dup, expected=0))

    # -- 7. Reporting periods ---------------------------------------------- #
    dates = sorted(pd.to_datetime(cur_df["data_cut_off_date"], errors="coerce")
                   .dropna().dt.date.unique())
    checks.append(Check(
        "the current period carries exactly one reporting date",
        len(dates) == 1 and dates[0].isoformat() == cfg.CURRENT_PERIOD.reporting_date,
        f"reporting date(s) in the current platform canonical: "
        f"{[d.isoformat() for d in dates]}",
        observed=[d.isoformat() for d in dates],
        expected=[cfg.CURRENT_PERIOD.reporting_date]))
    checks.append(Check(
        "at least two reporting periods are available",
        len(total["series"]) >= 2,
        f"{len(total['series'])} governed reporting period(s): "
        f"{[p['period'] for p in total['series']]}",
        observed=len(total["series"]), expected=">= 2"))
    checks.append(Check(
        "all reporting dates are historical",
        all(p["reportingDate"] < "2026-07-25" for p in total["series"]),
        f"periods {[p['reportingDate'] for p in total['series']]}",
        observed=[p["reportingDate"] for p in total["series"]],
        expected="before the demonstration build date"))

    # -- 8. Row-level economic coherence ---------------------------------- #
    checks.extend(_row_level_checks(cur_df, "current period"))

    # -- 9. Source schemas are genuinely different ------------------------- #
    shared = [h for h in schemas.shared_headers() if h != "Synthetic Data Notice"]
    checks.append(Check(
        "the two source schemas share no business column names",
        not shared,
        f"shared headers (excluding the synthetic marker): {shared or 'none'}",
        observed=shared, expected=[]))

    # -- 10. Both surfaces carry the same figures -------------------------- #
    checks.extend(_surface_agreement_checks(export_payload))

    # -- 11. No prohibited content ----------------------------------------- #
    from . import safety
    safety_report = safety.scan()
    checks.append(Check(
        "no prohibited client string or credential in generated material",
        safety_report["ok"],
        f"{safety_report['filesScanned']} files scanned, "
        f"{safety_report['findingCount']} finding(s)",
        observed=safety_report["findingCount"], expected=0))

    passed = [c for c in checks if c.passed]
    failed = [c for c in checks if not c.passed]
    report = {
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
        "ok": not failed,
        "checksRun": len(checks),
        "checksPassed": len(passed),
        "checksFailed": len(failed),
        "checks": [c.to_dict() for c in checks],
        "metrics": exact_metrics(export_payload),
    }
    out = cfg.export_dir() / "assertion_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if verbose:
        for c in checks:
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name} — {c.detail}")
        print(f"  [assert] {len(passed)}/{len(checks)} checks passed")
    return report


# --------------------------------------------------------------------------- #
# Cross-surface agreement
# --------------------------------------------------------------------------- #
_MONEY_RE = re.compile(r"£([\d,]+(?:\.\d+)?)(bn|m|k)?")
_PCT_RE = re.compile(r"([\d.]+)%")


def _monies(text: str) -> List[float]:
    out: List[float] = []
    for amount, unit in _MONEY_RE.findall(text or ""):
        value = float(amount.replace(",", ""))
        value *= {"bn": 1e9, "m": 1e6, "k": 1e3}.get(unit or "", 1.0)
        out.append(round(value, 2))
    return out


def _surface_agreement_checks(export_payload: Dict[str, Any]) -> List[Check]:
    """The React MI Agent and Copilot answers must carry the same figures.

    Both come from the same ``/mi/query`` handler, so this is a regression guard
    on that contract rather than a reconciliation of two computations: if the two
    surfaces ever diverge, the demonstration must not render.
    """
    checks: List[Check] = []
    mi_turns = export_payload["miAgent"]["turns"]
    copilot = export_payload["copilot"]["answers"]
    if len(mi_turns) != len(copilot):
        checks.append(Check(
            "both surfaces answer the same number of questions", False,
            f"MI Agent {len(mi_turns)} turns vs Copilot {len(copilot)}",
            observed=len(copilot), expected=len(mi_turns)))
        return checks

    for idx, (turn, cop) in enumerate(zip(mi_turns, copilot), start=1):
        question = turn.get("question")
        mi_answer = turn.get("answer") or ""
        cop_answer = cop.get("answer") or ""
        mi_values, cop_values = _monies(mi_answer), _monies(cop_answer)
        same_money = mi_values == cop_values
        mi_pcts = _PCT_RE.findall(mi_answer)
        cop_pcts = _PCT_RE.findall(cop_answer)
        checks.append(Check(
            f"question {idx}: MI Agent and Copilot report identical figures",
            same_money and mi_pcts == cop_pcts,
            f"{question!r}: monetary values {mi_values} vs {cop_values}; "
            f"percentages {mi_pcts} vs {cop_pcts}",
            observed={"copilotMoney": cop_values, "copilotPercent": cop_pcts},
            expected={"miAgentMoney": mi_values, "miAgentPercent": mi_pcts}))
    return checks


# --------------------------------------------------------------------------- #
# The exact figures the film may quote
# --------------------------------------------------------------------------- #
def exact_metrics(export_payload: Dict[str, Any]) -> Dict[str, Any]:
    """The exact calculated values used in the film, for the final report."""
    lenses = export_payload["metrics"]["lenses"]
    mv = lenses["total"]["movement"]
    a_id, b_id = (p.source_portfolio_id for p in cfg.PORTFOLIOS)
    cohorts = {c["id"]: c for c in mv["cohortMovements"]}
    primary = mv.get("primaryRegion") or {}
    return {
        "priorPeriod": mv["priorPeriod"],
        "currentPeriod": mv["currentPeriod"],
        "priorFundedBalance": mv["prior"]["funded_balance"],
        "currentFundedBalance": mv["current"]["funded_balance"],
        "movement": mv["delta"]["funded_balance"],
        "portfolioAMovement": cohorts.get(a_id, {}).get("delta"),
        "portfolioBMovement": cohorts.get(b_id, {}).get("delta"),
        "primaryRegion": primary.get("region"),
        "primaryRegionMovement": primary.get("delta"),
        "primaryRegionShare": mv.get("primaryRegionShare"),
        "currentWaCurrentLtvPoints": mv["current"]["wa_ltv_points"],
        "priorWaCurrentLtvPoints": mv["prior"]["wa_ltv_points"],
        "currentAvgBorrowerAge": mv["current"]["avg_borrower_age"],
        "priorAvgBorrowerAge": mv["prior"]["avg_borrower_age"],
        "currentWaInterestRate": mv["current"]["wa_interest_rate"],
        "consolidatedLoanCount": mv["current"]["loan_count"],
        "priorLoanCount": mv["prior"]["loan_count"],
        "portfolioABalance": cohorts.get(a_id, {}).get("current"),
        "portfolioBBalance": cohorts.get(b_id, {}).get("current"),
        "topRegions": lenses["total"]["summary"].get("topRegions"),
    }


def enforce(export_payload: Optional[Dict[str, Any]] = None, *,
            verbose: bool = True) -> Dict[str, Any]:
    """Run the checks and RAISE on any failure so the demo run fails closed."""
    report = run(export_payload, verbose=verbose)
    if not report["ok"]:
        failed = [c for c in report["checks"] if not c["passed"]]
        lines = [f"    {c['check']}: {c['detail']}" for c in failed]
        raise AssertionFailure(
            f"{len(failed)} reconciliation check(s) failed — the demonstration "
            f"does not support its own narrative:\n" + "\n".join(lines))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    ap = argparse.ArgumentParser(description="Run the demo reconciliation gate.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        enforce(verbose=not args.quiet)
    except AssertionFailure as exc:
        print(f"ASSERTION GATE FAILED\n{exc}")
        return 2
    print("Assertion gate passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
