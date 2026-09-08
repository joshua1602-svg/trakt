"""Independent truth for the MI Query Agent V1 live acceptance.

WHERE TRUTH COMES FROM, AND WHY NOT FROM /mi/query.

The live book does not exist in this repository. It cannot: the certification
snapshot beside this file records what happened the last time figures were
written into a fixture and called production truth — the gate failed the
deployed service for not matching a book it does not hold.

So truth is taken at run time from the DASHBOARD's own GET endpoints: a
different URL, a different handler, a different response contract, and the
surface a human actually reads. Asking /mi/query twice and calling the
agreement independent is not done here and would not be evidence if it were.

Where even that is unavailable, the harness RECOMPUTES the value arithmetically
from independently fetched components — a share from its part and its whole, a
delta from two levels, a total from its cells. A service cannot satisfy those
by repeating itself.

WHAT THIS DOES NOT CLAIM. Below the handlers, the GET endpoints and /mi/query
read the same governed engines. This is an independent SURFACE, not an
independent IMPLEMENTATION: a defect inside a shared engine moves both sides
together and is invisible to a cross-surface check. The report says, per case,
which method carried the check, so nobody reads more into a pass than it holds.

TRANSPORT. `MI_BEARER` from the environment, the same convention as
`certify_mi_api._live_asker` and `migration_phase0/replay_probe.py`. It is
never placed on a command line. Only the verb differs — the certification
client POSTs questions; truth is READ — so this file adds a GET, not a second
credential.

STDLIB ONLY. The certification workflows install no dependencies, and a gate
that dies on `ModuleNotFoundError` after authenticating has certified nothing.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

#: Returned when a truth value could not be established. It is NOT a value and
#: never compares equal to one: a case whose truth is UNAVAILABLE is reported as
#: unverified, never as passed.
UNAVAILABLE = "__UNAVAILABLE__"

#: Returned by an availability rule that could not be resolved either way.
UNRESOLVED = "__UNRESOLVED__"

SCOTLAND_TOKENS = ("scotland", "scottish")


def reader(base_url: str, portfolio_id: Optional[str]
           ) -> Callable[[str, Optional[Dict[str, str]]], Dict[str, Any]]:
    """A GET reader for the dashboard surface, bearing the same credential."""
    bearer = os.environ.get("MI_BEARER", "").strip()
    root = base_url.rstrip("/")

    def get(path: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        query = dict(params or {})
        if portfolio_id and "portfolioId" not in query:
            query["portfolioId"] = portfolio_id
        url = root + path
        if query:
            from urllib.parse import urlencode
            url += "?" + urlencode(query)
        request = urllib.request.Request(url, method="GET")
        if bearer:
            request.add_header(
                "Authorization", "Bearer " + bearer.removeprefix("Bearer ").strip())
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read().decode("utf-8") or "{}"
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            return {"__error__": f"HTTP {exc.code}", "__status__": exc.code}
        except Exception as exc:  # noqa: BLE001
            return {"__error__": str(exc), "__status__": None}

    return get


def _ok(doc: Any) -> bool:
    """Usable, not merely delivered.

    These endpoints never 500: they answer HTTP 200 carrying ``ok: false`` and
    a reason. Treating that as a successful read is how a truth collector ends
    up reporting that the book has no regions, no LTV and no prior period —
    and how a service that answered all three correctly gets failed for it. A
    document that says it failed is not truth."""
    if not isinstance(doc, dict) or "__error__" in doc:
        return False
    return doc.get("ok") is not False


def _status(doc: Any) -> str:
    if not isinstance(doc, dict):
        return "unreadable"
    if "__error__" in doc:
        return str(doc["__error__"])
    if doc.get("ok") is False:
        return f"answered ok=false: {doc.get('error') or 'no reason given'}"
    return "ok"


def _kpi(snapshot: Dict[str, Any], kpi_id: str) -> Any:
    for k in (snapshot.get("kpis") or []):
        if isinstance(k, dict) and k.get("id") == kpi_id:
            if not k.get("available", True):
                return UNAVAILABLE
            raw = k.get("raw")
            return UNAVAILABLE if raw is None else float(raw)
    return UNAVAILABLE


def _strat_state(snapshot: Dict[str, Any], key: str) -> Any:
    """Whether this book supports one stratification: True, False, or
    UNRESOLVED when the snapshot itself could not be read.

    The three are kept apart deliberately. "The tape does not carry product"
    is a fact about the book and makes a refusal CORRECT; "I could not read
    the snapshot" is a fact about this harness and must never be scored as
    either."""
    entries = snapshot.get("stratifications")
    if not isinstance(entries, list) or not entries:
        return UNRESOLVED
    for entry in entries:
        if isinstance(entry, dict) and entry.get("key") == key:
            return bool(entry.get("bars"))
    return False


def _strat(snapshot: Dict[str, Any], key: str) -> Any:
    """``[{label, balance, count, sharePct}]`` for one stratification, or
    UNAVAILABLE when the book does not support it. A dimension the service
    reported as unavailable is UNAVAILABLE here too — that is a fact about the
    book, and the acceptance must not read it as a defect."""
    for entry in (snapshot.get("stratifications") or []):
        if isinstance(entry, dict) and entry.get("key") == key:
            bars = entry.get("bars") or []
            if not bars:
                return UNAVAILABLE
            return [{"label": str(b.get("label")),
                     "balance": b.get("balance"),
                     "count": b.get("count"),
                     "sharePct": b.get("sharePct")} for b in bars]
    return UNAVAILABLE


def _scotland_row(rows: Any) -> Any:
    if not isinstance(rows, list):
        return UNAVAILABLE
    for row in rows:
        label = str(row.get("label") or "").strip().lower()
        if any(token in label for token in SCOTLAND_TOKENS):
            return row
    return UNAVAILABLE


def _series(evolution: Dict[str, Any], metric: str, scale: float = 1.0) -> Any:
    periods = evolution.get("periods") or []
    if not periods:
        return UNAVAILABLE
    out: List[Dict[str, Any]] = []
    for p in periods:
        metrics = (p or {}).get("metrics") or {}
        value = metrics.get(metric)
        out.append({"period": p.get("period"),
                    "reporting_date": p.get("reporting_date"),
                    "value": None if value is None else float(value) * scale})
    if all(row["value"] is None for row in out):
        return UNAVAILABLE
    return out


def _stage_rows(pipeline: Dict[str, Any]) -> Any:
    rows = pipeline.get("stageBreakdown") or []
    if not rows:
        return UNAVAILABLE
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        out.append({"label": str(r.get("stage")),
                    "count": r.get("caseCount"),
                    "balance": r.get("pipelineAmount")})
    return out or UNAVAILABLE


def collect(base_url: str, portfolio_id: Optional[str]) -> Dict[str, Any]:
    """Read every independent surface once, and derive the truth keys.

    One read per endpoint, before any question is asked, so the book cannot
    move underneath the comparison and every case is scored against the same
    observation.
    """
    get = reader(base_url, portfolio_id)

    endpoints: Dict[str, Dict[str, Any]] = {
        "snapshot": get("/mi/snapshot"),
        "geo": get("/mi/geo/exposure"),
        "pipeline": get("/mi/pipeline/snapshot"),
        "evolution": get("/mi/evolution/funded"),
        "risk_limits": get("/mi/risk-limits"),
        "borrowing_base": get("/mi/borrowing-base"),
        "forecast": get("/mi/forecast/extrapolation"),
        "cohort_progression": get("/mi/cohorts/progression"),
        "source_portfolios": get("/mi/source-portfolios"),
        "pipeline_snapshots": get("/mi/pipeline/snapshots"),
        "pipeline_evolution": get("/mi/evolution/pipeline"),
        "portfolio_context": get("/mi/portfolio-context"),
    }

    snap = endpoints["snapshot"] if _ok(endpoints["snapshot"]) else {}
    geo = endpoints["geo"] if _ok(endpoints["geo"]) else {}
    pipe = endpoints["pipeline"] if _ok(endpoints["pipeline"]) else {}
    evo = endpoints["evolution"] if _ok(endpoints["evolution"]) else {}
    risk = endpoints["risk_limits"] if _ok(endpoints["risk_limits"]) else {}
    bbase = endpoints["borrowing_base"] if _ok(endpoints["borrowing_base"]) else {}
    fcast = endpoints["forecast"] if _ok(endpoints["forecast"]) else {}
    cohort = endpoints["cohort_progression"] if _ok(endpoints["cohort_progression"]) else {}
    sources = endpoints["source_portfolios"] if _ok(endpoints["source_portfolios"]) else {}
    pipe_snaps = endpoints["pipeline_snapshots"] if _ok(endpoints["pipeline_snapshots"]) else {}
    pipe_evo = endpoints["pipeline_evolution"] if _ok(endpoints["pipeline_evolution"]) else {}
    context = endpoints["portfolio_context"] if _ok(endpoints["portfolio_context"]) else {}

    total_balance = snap.get("current_outstanding_balance")
    loan_count = snap.get("loan_count")
    total_balance = float(total_balance) if isinstance(total_balance, (int, float)) else UNAVAILABLE
    loan_count = int(loan_count) if isinstance(loan_count, (int, float)) else UNAVAILABLE

    avg_balance = UNAVAILABLE
    if total_balance is not UNAVAILABLE and loan_count is not UNAVAILABLE and loan_count:
        # RECOMPUTED HERE, not read from the tile: the identity is the check.
        avg_balance = total_balance / loan_count

    strat_region = _strat(snap, "region")
    strat_ltv = _strat(snap, "ltv")
    strat_product = _strat(snap, "product")

    region_top = UNAVAILABLE
    region_top_share = UNAVAILABLE
    if isinstance(strat_region, list):
        ranked = sorted((r for r in strat_region if isinstance(r.get("balance"), (int, float))),
                        key=lambda r: r["balance"], reverse=True)
        if ranked:
            region_top = ranked[0]["label"]
            if total_balance is not UNAVAILABLE and total_balance:
                region_top_share = ranked[0]["balance"] / total_balance * 100.0

    scotland = _scotland_row(strat_region)
    scotland_balance = (scotland["balance"] if isinstance(scotland, dict)
                        and isinstance(scotland.get("balance"), (int, float))
                        else UNAVAILABLE)

    monthly = snap.get("monthly_change") or {}
    mom_balance_change = (float(monthly["balance_change"])
                          if isinstance(monthly.get("balance_change"), (int, float))
                          else UNAVAILABLE)
    mom_new = monthly.get("new_loans")
    mom_exited = monthly.get("exited_loans")
    mom_new_exited = ({"new_loans": mom_new, "exited_loans": mom_exited}
                      if isinstance(mom_new, int) and isinstance(mom_exited, int)
                      else UNAVAILABLE)

    balance_series = _series(evo, "funded_balance")
    count_series = _series(evo, "loan_count")
    # The evolution surface stores weighted-average LTV as a FRACTION while the
    # snapshot tile states percentage POINTS. Normalised here, in the open,
    # rather than by widening a tolerance until both conventions pass.
    wa_ltv_series = _series(evo, "wa_ltv", scale=100.0)

    prior_period_balance = UNAVAILABLE
    last_two = UNAVAILABLE
    if isinstance(balance_series, list) and len(balance_series) >= 2:
        prior_period_balance = balance_series[-2]["value"]
        last_two = {"prior": balance_series[-2], "current": balance_series[-1]}

    region_breakdown = evo.get("breakdowns", {}).get("region") if isinstance(
        evo.get("breakdowns"), dict) else None
    region_breakdown = region_breakdown if region_breakdown else UNAVAILABLE

    scotland_series = UNAVAILABLE
    if isinstance(region_breakdown, list):
        rows = [r for r in region_breakdown
                if any(t in str(r.get("key") or r.get("label") or "").lower()
                       for t in SCOTLAND_TOKENS)]
        if rows:
            by_period: Dict[str, float] = {}
            for r in rows:
                value = r.get("balance")
                if isinstance(value, (int, float)):
                    by_period[str(r.get("period"))] = by_period.get(
                        str(r.get("period")), 0.0) + float(value)
            if by_period:
                scotland_series = [{"period": k, "value": v}
                                   for k, v in sorted(by_period.items())]

    stage_rows = _stage_rows(pipe)
    kfi_stock = UNAVAILABLE
    if isinstance(stage_rows, list):
        for row in stage_rows:
            if "kfi" in str(row.get("label") or "").lower():
                kfi_stock = row.get("count")
                break

    pipeline_count = pipe.get("pipelineRowCount")
    pipeline_count = (int(pipeline_count)
                      if isinstance(pipeline_count, (int, float)) and pipe.get("ok") is not False
                      else UNAVAILABLE)

    risk_summary = risk.get("summary") if isinstance(risk.get("summary"), dict) else {}
    risk_summary_truth = UNAVAILABLE
    if risk.get("available") and risk_summary:
        risk_summary_truth = {"testsPassed": risk_summary.get("testsPassed"),
                              "breaches": risk_summary.get("breaches"),
                              "total": risk_summary.get("total")}
    closest = risk_summary.get("closestHeadroom")
    closest_name = UNAVAILABLE
    if isinstance(closest, dict):
        closest_name = closest.get("name") or closest.get("test") or closest.get("label")
    closest_name = closest_name or UNAVAILABLE

    cohort_counts = UNAVAILABLE
    if cohort.get("available") and isinstance(cohort.get("periods"), list):
        cohort_counts = [p.get("survivingLoanCount") for p in cohort["periods"]]

    # `/mi/source-portfolios` returns {available, lenses, source} — it names its
    # scopes `lenses`, not `portfolios`. Reading for the wrong key made a
    # readable, authoritative answer look like an absent one, and two whole
    # cases were reported UNSCOREABLE for it.
    source_count = UNAVAILABLE
    if isinstance(sources.get("lenses"), list):
        source_count = len(sources["lenses"])
    else:
        for key in ("portfolios", "sourcePortfolios", "items"):
            if isinstance(sources.get(key), list):
                source_count = len(sources[key])
                break

    geo_supported = UNAVAILABLE
    for key in ("supportedBases", "availableBases", "bases"):
        if isinstance(geo.get(key), list):
            geo_supported = [str(b) for b in geo[key]]
            break

    truths: Dict[str, Any] = {
        "funded_total_balance": total_balance,
        "funded_loan_count": loan_count,
        "funded_avg_balance": avg_balance,
        "funded_wa_current_ltv": _kpi(snap, "wa_current_ltv"),
        "funded_wa_rate": _kpi(snap, "wa_rate"),
        "strat_region": strat_region,
        "strat_ltv": strat_ltv,
        "strat_product": strat_product,
        "strat_region_top": region_top,
        "region_scotland_balance": scotland_balance,
        "region_top_share": region_top_share,
        "geo_total": geo.get("total") if geo.get("available") else UNAVAILABLE,
        "geo_areas": geo.get("areas") if geo.get("available") else UNAVAILABLE,
        "geo_basis": geo.get("basis") if geo.get("available") else UNAVAILABLE,
        "geo_supported_bases": geo_supported,
        "pipeline_case_count": pipeline_count,
        "pipeline_stage_breakdown": stage_rows,
        "pipeline_kfi_stock": kfi_stock if kfi_stock is not None else UNAVAILABLE,
        "source_portfolio_count": source_count,
        "evolution_balance_series": balance_series,
        "evolution_count_series": count_series,
        "evolution_wa_ltv_series": wa_ltv_series,
        "evolution_region_breakdown": region_breakdown,
        "evolution_scotland_series": scotland_series,
        "prior_period_balance": (prior_period_balance if prior_period_balance is not None
                                 else UNAVAILABLE),
        "last_two_period_balances": last_two,
        "mom_balance_change": mom_balance_change,
        "mom_new_and_exited": mom_new_exited,
        "forecast_current_balance": (float(fcast["currentFundedBalance"])
                                     if isinstance(fcast.get("currentFundedBalance"), (int, float))
                                     and fcast.get("currentFundedBalance")
                                     else UNAVAILABLE),
        "cohort_progression_counts": cohort_counts,
        "risk_limits_summary": risk_summary_truth,
        "risk_limits_closest": closest_name,
        "borrowing_base_envelope": {
            "available": bbase.get("available"),
            "reason": bbase.get("reason") or bbase.get("unavailableReason"),
            "measures": bbase.get("measures"),
        } if bbase else UNAVAILABLE,
    }

    truths["funded_total_balance+funded_loan_count"] = [
        truths["funded_total_balance"], truths["funded_loan_count"]]
    truths["funded_total_balance+funded_loan_count+funded_avg_balance"] = [
        truths["funded_total_balance"], truths["funded_loan_count"],
        truths["funded_avg_balance"]]
    truths["funded_wa_current_ltv+funded_wa_rate"] = [
        truths["funded_wa_current_ltv"], truths["funded_wa_rate"]]

    availability = _availability(truths, snap, geo, pipe, evo, risk, bbase,
                                 fcast, cohort, pipe_snaps, pipe_evo, context)

    return {
        "truths": truths,
        "availability": availability,
        "endpoint_status": {name: _status(doc) for name, doc in endpoints.items()},
    }


def _stated_availability(doc: Dict[str, Any]) -> Any:
    """``available`` as the surface stated it — unless it declined for a MISSING
    PARAMETER, which is a fact about the request and not about the book."""
    if not doc:
        return UNRESOLVED
    if doc.get("available"):
        return True
    reason = str(doc.get("reason") or doc.get("error") or "").lower()
    if "required" in reason or "portfolioid" in reason:
        return UNRESOLVED
    return False


def _region_evidence(snap: Dict[str, Any], evo: Dict[str, Any],
                     geo: Dict[str, Any], t: Dict[str, Any]) -> Any:
    """CAN THIS BOOK BE MEASURED BY REGION — asked of every surface that knows.

    This rule used to read one thing: whether the Portfolio tab drew a region
    chart. It does not, on this book, and twenty questions were therefore
    expected to refuse a regional breakdown that the service produced correctly.
    The dashboard tile and the query route resolve region through DIFFERENT
    owners; a tile that is not drawn says nothing about whether the analytic is
    supported.

    So the rule is now the union of the surfaces that would each independently
    establish it, and it is False only when a surface that could have shown
    region was read and none did."""
    signals = []
    strat = _strat_state(snap, "region")
    if strat is not UNRESOLVED:
        signals.append(bool(strat))
    breakdown = (evo.get("breakdowns") or {}).get("region") if isinstance(
        evo.get("breakdowns"), dict) else None
    if evo:
        signals.append(bool(breakdown))
    if geo:
        signals.append(bool(geo.get("available")))
    if not signals:
        return UNRESOLVED
    return any(signals)


def _obligor_basis_state(geo: Dict[str, Any]) -> Any:
    """Whether an OBLIGOR geography basis is established for this book.

    When the surfaces disagree this returns UNRESOLVED rather than picking one.
    The geography endpoint reported `postcode_derived` on the live book while
    the query route labelled its own answers `Obligor Region (NUTS3)`; two
    production surfaces naming different bases is a finding to be audited, not
    a rule to be resolved by whichever this harness happened to read."""
    if not geo:
        return UNRESOLVED
    for key in ("supportedBases", "availableBases", "bases"):
        if isinstance(geo.get(key), list):
            return any("obligor" in str(b).lower() for b in geo[key])
    basis = str(geo.get("basis") or "").lower()
    if not basis:
        return UNRESOLVED
    if "obligor" in basis:
        return True
    if "collateral" in basis:
        return False
    # `postcode_derived` names neither side: the basis was DERIVED, so this
    # surface cannot say which of the two the book reports on.
    return UNRESOLVED


def _pipeline_history_state(pipe_evo: Dict[str, Any],
                            pipe_snaps: Dict[str, Any]) -> Any:
    """More than one governed weekly extract retained, from the surface that
    owns the weekly series rather than the one that lists sources."""
    for key in ("availableExtractDates", "extractDates", "reportingDates"):
        dates = pipe_evo.get(key)
        if isinstance(dates, list):
            return len(dates) >= 2
    used = pipe_evo.get("uniqueWeeklyExtractsUsed")
    if isinstance(used, int):
        return used >= 2
    for key in ("extractDates", "availableExtractDates", "dates", "reportingDates"):
        dates = pipe_snaps.get(key)
        if isinstance(dates, list):
            return len(dates) >= 2
    return UNRESOLVED


def _scope_state(sources_doc: Any, count: Any, context: Dict[str, Any]) -> Any:
    """At least two governed portfolio scopes to compare."""
    if isinstance(count, int):
        return count >= 2
    contexts = context.get("contexts") if isinstance(context, dict) else None
    if isinstance(contexts, list) and contexts:
        # `Total` is a scope but not a comparison partner: two scopes means two
        # things to put side by side.
        return len(contexts) >= 3
    portfolios = context.get("portfolios") if isinstance(context, dict) else None
    if isinstance(portfolios, list):
        return len(portfolios) >= 2
    return UNRESOLVED


def _geo_state(geo: Dict[str, Any]) -> Any:
    """A geography surface that refused for a MISSING PARAMETER is telling us
    about the request, not about the book."""
    if not geo:
        return UNRESOLVED
    if geo.get("available"):
        return True
    reason = str(geo.get("reason") or "").lower()
    if "required" in reason or "portfolioid" in reason:
        return UNRESOLVED
    return False


def _availability(t: Dict[str, Any], snap: Dict[str, Any], geo: Dict[str, Any],
                  pipe: Dict[str, Any], evo: Dict[str, Any], risk: Dict[str, Any],
                  bbase: Dict[str, Any], fcast: Dict[str, Any],
                  cohort: Dict[str, Any], pipe_snaps: Dict[str, Any],
                  pipe_evo: Dict[str, Any], context: Dict[str, Any]
                  ) -> Dict[str, Any]:
    """Resolve every frozen answerability rule from the independent surfaces.

    A rule that cannot be resolved returns UNRESOLVED, and its cases are
    reported as unscoreable rather than defaulted to a pass.
    """
    def present(value: Any) -> Any:
        return UNRESOLVED if value is UNAVAILABLE else value is not UNAVAILABLE

    periods = t["evolution_balance_series"]
    period_count = len(periods) if isinstance(periods, list) else UNRESOLVED
    multi = (period_count >= 2) if isinstance(period_count, int) else UNRESOLVED

    ltv_series = t["evolution_wa_ltv_series"]
    region_bd = t["evolution_region_breakdown"]

    pipeline_dates = None
    for key in ("extractDates", "availableExtractDates", "dates", "reportingDates"):
        if isinstance(pipe_snaps.get(key), list):
            pipeline_dates = pipe_snaps[key]
            break

    region_evidence = _region_evidence(snap, evo, geo, t)
    basis_state = _obligor_basis_state(geo)

    return {
        "kpi_wa_current_ltv_available": (
            UNRESOLVED if not snap else t["funded_wa_current_ltv"] is not UNAVAILABLE),
        "kpi_wa_ltv_and_rate_available": (
            UNRESOLVED if not snap
            else (t["funded_wa_current_ltv"] is not UNAVAILABLE
                  and t["funded_wa_rate"] is not UNAVAILABLE)),
        "strat_region_available": region_evidence,
        "strat_ltv_available": (
            True if _strat_state(snap, "ltv") is True
            # A governed LTV BAND is configuration applied to the LTV column.
            # If the book carries a weighted-average current LTV, the column
            # has values and the bands are computable, whatever the dashboard
            # happened to draw.
            else (True if t["funded_wa_current_ltv"] is not UNAVAILABLE
                  else _strat_state(snap, "ltv"))),
        "strat_product_available": _strat_state(snap, "product"),
        "region_scotland_present": (
            UNRESOLVED if region_evidence is UNRESOLVED
            else (t["region_scotland_balance"] is not UNAVAILABLE
                  or isinstance(t["evolution_scotland_series"], list))),
        "geo_available": _geo_state(geo),
        "geo_obligor_basis_supported": basis_state,
        "pipeline_available": (
            UNRESOLVED if not pipe else bool(pipe.get("ok") is not False
                                             and t["pipeline_case_count"] is not UNAVAILABLE)),
        "pipeline_history_available": _pipeline_history_state(pipe_evo, pipe_snaps),
        "two_governed_scopes": _scope_state(sources_doc=None,
                                            count=t["source_portfolio_count"],
                                            context=context),
        "multi_period": multi,
        "multi_period_ltv": (
            UNRESOLVED if multi is UNRESOLVED
            else bool(multi and isinstance(ltv_series, list))),
        "multi_period_region": (
            UNRESOLVED if multi is UNRESOLVED
            else bool(multi and isinstance(region_bd, list))),
        "multi_period_scotland": (
            UNRESOLVED if multi is UNRESOLVED
            else bool(multi and isinstance(t["evolution_scotland_series"], list))),
        "prior_period_available": (
            UNRESOLVED if not snap
            else bool(snap.get("prior")) or (multi is True)),
        "loan_movement_identifiable": (
            bool((snap.get("monthly_change") or {}).get("loans_identifiable"))
            if snap.get("monthly_change") else UNRESOLVED),
        "forecast_available": (
            UNRESOLVED if not fcast else t["forecast_current_balance"] is not UNAVAILABLE),
        "cohort_progression_available": _stated_availability(cohort),
        "risk_limits_available": _stated_availability(risk),
        "borrowing_base_available": (
            bool(bbase.get("available")) if bbase else UNRESOLVED),
    }


#: WHAT EACH TRUTH VALUE IS, IN WORDS. A number compared against another number
#: is not evidence until both sides say what they are measuring, over which
#: population, in which units. Two correct figures on different populations
#: disagree; so do two figures of the same population in different units. This
#: table is what lets a disagreement be classified instead of merely counted.
TRUTH_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "funded_total_balance": {
        "unit": "GBP", "population": "every funded loan at the run's reporting date",
        "source": "GET /mi/snapshot .current_outstanding_balance"},
    "funded_loan_count": {
        "unit": "count of loans", "population": "every funded loan at the reporting date",
        "source": "GET /mi/snapshot .loan_count"},
    "funded_avg_balance": {
        "unit": "GBP per loan", "population": "every funded loan at the reporting date",
        "source": "recomputed here: .current_outstanding_balance / .loan_count"},
    "funded_wa_current_ltv": {
        "unit": "percentage points", "population": "balance-weighted over funded loans",
        "source": "GET /mi/snapshot kpi wa_current_ltv .raw"},
    "funded_wa_rate": {
        "unit": "percentage points", "population": "balance-weighted over funded loans",
        "source": "GET /mi/snapshot kpi wa_rate .raw"},
    "strat_region": {
        "unit": "GBP per region", "population": "funded loans grouped by the snapshot's region dimension",
        "source": "GET /mi/snapshot .stratifications[key=region].bars"},
    "strat_ltv": {
        "unit": "GBP per band", "population": "funded loans grouped into governed LTV bands",
        "source": "GET /mi/snapshot .stratifications[key=ltv].bars"},
    "strat_product": {
        "unit": "count per product", "population": "funded loans grouped by product",
        "source": "GET /mi/snapshot .stratifications[key=product].bars"},
    "strat_region_top": {
        "unit": "region name", "population": "the largest region by balance",
        "source": "GET /mi/snapshot .stratifications[key=region] ordered"},
    "region_scotland_balance": {
        "unit": "GBP", "population": "funded loans whose snapshot region is Scotland",
        "source": "GET /mi/snapshot .stratifications[key=region] Scotland row"},
    "region_top_share": {
        "unit": "per cent of funded balance", "population": "largest region over the whole book",
        "source": "recomputed here: largest region balance / total balance x 100"},
    "geo_total": {
        "unit": "GBP", "population": "ONLY funded loans whose ITL3 area resolved",
        "source": "GET /mi/geo/exposure .total"},
    "geo_basis": {
        "unit": "basis name", "population": "the geography basis the configuration resolved",
        "source": "GET /mi/geo/exposure .basis"},
    "pipeline_case_count": {
        "unit": "count of cases", "population": "the latest governed weekly pipeline extract",
        "source": "GET /mi/pipeline/snapshot .pipelineRowCount"},
    "pipeline_stage_breakdown": {
        "unit": "count per stage", "population": "the latest weekly extract grouped by stage",
        "source": "GET /mi/pipeline/snapshot .stageBreakdown"},
    "pipeline_kfi_stock": {
        "unit": "count of cases", "population": "cases standing at KFI in the latest extract",
        "source": "GET /mi/pipeline/snapshot .stageBreakdown KFI row"},
    "evolution_balance_series": {
        "unit": "GBP per period", "population": "the funded book at each governed reporting period",
        "source": "GET /mi/evolution/funded .periods[].metrics.funded_balance"},
    "evolution_count_series": {
        "unit": "count per period", "population": "the funded book at each governed reporting period",
        "source": "GET /mi/evolution/funded .periods[].metrics.loan_count"},
    "evolution_wa_ltv_series": {
        "unit": "percentage points (surface stores a fraction; scaled x100 here)",
        "population": "balance-weighted over the funded book each period",
        "source": "GET /mi/evolution/funded .periods[].metrics.wa_ltv"},
    "evolution_region_breakdown": {
        "unit": "GBP per region per period", "population": "funded book per region per period",
        "source": "GET /mi/evolution/funded .breakdowns.region"},
    "evolution_scotland_series": {
        "unit": "GBP per period", "population": "Scottish rows of the region breakdown",
        "source": "GET /mi/evolution/funded .breakdowns.region Scotland rows"},
    "prior_period_balance": {
        "unit": "GBP", "population": "the funded book at the SECOND-TO-LAST governed period",
        "source": "GET /mi/evolution/funded .periods[-2]"},
    "last_two_period_balances": {
        "unit": "GBP", "population": "the funded book at the last two governed periods",
        "source": "GET /mi/evolution/funded .periods[-2:]"},
    "mom_balance_change": {
        "unit": "GBP, signed", "population": "current run minus the snapshot's own prior run",
        "source": "GET /mi/snapshot .monthly_change.balance_change"},
    "mom_new_and_exited": {
        "unit": "counts of loans", "population": "loan ids present in one run and not the other",
        "source": "GET /mi/snapshot .monthly_change.new_loans / .exited_loans"},
    "forecast_current_balance": {
        "unit": "GBP", "population": "the funded balance the forecast extrapolates from",
        "source": "GET /mi/forecast/extrapolation .currentFundedBalance"},
    "cohort_progression_counts": {
        "unit": "count per period", "population": "a static pool fixed at formation",
        "source": "GET /mi/cohorts/progression .periods[].survivingLoanCount"},
    "risk_limits_summary": {
        "unit": "counts of tests", "population": "the approved concentration tests",
        "source": "GET /mi/risk-limits .summary"},
    "risk_limits_closest": {
        "unit": "test name", "population": "the approved test with least headroom",
        "source": "GET /mi/risk-limits .summary.closestHeadroom"},
    "source_portfolio_count": {
        "unit": "count of scopes", "population": "governed portfolio scopes for this client",
        "source": "GET /mi/source-portfolios"},
    "borrowing_base_envelope": {
        "unit": "GBP / state", "population": "the configured funding facility, if any",
        "source": "GET /mi/borrowing-base"},
}


def describe(key: Optional[str]) -> Dict[str, str]:
    if not key:
        return {"unit": "n/a", "population": "n/a", "source": "n/a"}
    if key in TRUTH_DESCRIPTIONS:
        return TRUTH_DESCRIPTIONS[key]
    parts = [TRUTH_DESCRIPTIONS[k] for k in key.split("+") if k in TRUTH_DESCRIPTIONS]
    if parts:
        return {"unit": " + ".join(p["unit"] for p in parts),
                "population": " + ".join(p["population"] for p in parts),
                "source": " + ".join(p["source"] for p in parts)}
    return {"unit": "not described", "population": "not described",
            "source": "not described"}
