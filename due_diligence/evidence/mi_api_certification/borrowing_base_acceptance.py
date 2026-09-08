#!/usr/bin/env python3
"""BORROWING BASE — live acceptance against the deployed service.

WHY THIS IS NOT A SECOND CERTIFICATION CLIENT. It asks its questions through
``certify_mi_api._live_asker`` and establishes the build through
``certify_mi_api.preflight``, so the repository keeps ONE live transport, ONE
``MI_BEARER`` convention, ONE reachability/auth classifier and ONE deployed-SHA
provenance check. What it adds is the borrowing base's own acceptance gate:
whether the MI Query Agent and the dashboard's governed envelope report the
SAME position, live.

THE EXPECTATION IS READ FROM PRODUCTION, NOT HARD-CODED. The gate fetches
``GET /mi/borrowing-base`` — the block the React Eligibility & Concentrations
tab renders — and derives what every MI answer must say from it:

* the envelope is AVAILABLE  → each named measure the MI answer delivers must
  equal the envelope's ``measures`` value exactly, and a measure the envelope
  reports NOT_CALCULABLE must come back as the governed not-calculable state,
  never as a number and never as a zero;
* the envelope is UNAVAILABLE → every borrowing-base question must refuse with
  the envelope's OWN reason. Two surfaces, one sentence.

Either way the gate is the same property — the dashboard and MI cannot
disagree — so it holds whatever the deployed facility configuration happens to
be, and it cannot be satisfied by a run that quietly answered a different
question.

UNLIKE the TIME x DIMENSION collector, this one ADJUDICATES. A violated
expectation exits non-zero and names the violation, because the release step
this serves is "stop if anything unexpected moves".

    python -m due_diligence.evidence.mi_api_certification.borrowing_base_acceptance \\
        --base-url https://app.traktinfra.io/api \\
        --portfolio-id ERE/2026-06-30 --expect-commit <sha>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from due_diligence.evidence.mi_api_certification.certify_mi_api import (  # noqa: E402
    _live_asker, preflight,
)

#: The governed sentinel. Repeated as a literal rather than imported: this
#: module runs against a DEPLOYED build and must not be able to pass because
#: the local tree agrees with itself.
NOT_CALCULABLE = "NOT_CALCULABLE"

#: (id, question, measure_id) — the measure each question must report, named in
#: the borrowing-base service's own registry vocabulary. Every one is compared
#: against the dashboard envelope; none is scored on prose.
MEASURES: Tuple[Tuple[str, str, str], ...] = (
    ("B01", "What is the borrowing base?", "borrowing_base"),
    ("B02", "How much eligible collateral do we have?", "eligible_balance"),
    ("B03", "What is the ineligible balance?", "ineligible_balance"),
    ("B04", "How many loans are ineligible?", "ineligible_loan_count"),
    ("B05", "How much is drawn under the facility?", "facility_drawn"),
    ("B06", "What is the borrowing-base headroom?", "borrowing_base_headroom"),
    ("B07", "What is facility utilisation?", "facility_utilisation"),
    ("B08", "What is borrowing-base utilisation?", "borrowing_base_utilisation"),
)

#: Questions the borrowing base must CLAIM and then REFUSE by name — the
#: governed-refusal half of the surface. `contains` is checked case-folded.
REFUSALS: Tuple[Tuple[str, str, str], ...] = (
    ("R01", "What is the borrowing base by region?", "by region"),
    ("R02", "Which loans are ineligible?", "loan-level"),
)

#: Questions that must NOT reach this route, and must keep working. This is the
#: live half of the frozen non-regression gate: generic "headroom" and
#: "utilisation" belong to the concentration-limit owner, and ordinary MI is
#: untouched. `answerable` questions must also come back ok.
CONTROLS: Tuple[Tuple[str, str, bool], ...] = (
    ("N01", "What is the headroom?", True),
    ("N02", "Show risk limit headroom.", True),
    ("N03", "What is our total funded balance?", True),
    ("N04", "Show funded balance by region", True),
    ("N05", "Are we within our concentration limits?", True),
)


def _meta(env: Dict[str, Any]) -> Dict[str, Any]:
    return env.get("metadata") or {}


def _bb(env: Dict[str, Any]) -> Dict[str, Any]:
    return _meta(env).get("borrowingBase") or {}


def _get_json(base_url: str, path: str) -> Dict[str, Any]:
    """One authenticated GET, on the SAME bearer convention as the asker."""
    import os
    import urllib.error
    import urllib.request

    headers = {"Accept": "application/json"}
    bearer = os.environ.get("MI_BEARER", "").strip()
    if bearer:
        headers["Authorization"] = "Bearer " + bearer.removeprefix("Bearer ").strip()
    request = urllib.request.Request(base_url.rstrip("/") + path, headers=headers,
                                     method="GET")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # noqa: PERF203
        return {"__error__": f"HTTP {exc.code}", "__body__": exc.read()[:400].decode(
            "utf-8", "replace")}
    except Exception as exc:  # noqa: BLE001
        return {"__error__": repr(exc)}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--path", default="/mi/query")
    ap.add_argument("--portfolio-id", default=None)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ask = _live_asker(args.base_url, args.path, [], args.portfolio_id)
    reached, authorised, commit = preflight(
        args.base_url, args.path, [], args.portfolio_id)

    # THE DEPLOYED BUILD IS ESTABLISHABLE WITHOUT A CREDENTIAL, and it must be.
    # `preflight` asks a QUESTION first and returns early on 401 — so an expired
    # `MI_BEARER` reported the build as NOT ESTABLISHED even though `/health` is
    # in the API's OPEN_PATHS allowlist and answers unauthenticated. That
    # conflated two independent facts: WHICH BUILD IS SERVING (a deployment
    # fact) and WHETHER WE MAY ASK IT ANYTHING (a credential fact). Reading
    # /health directly keeps them apart, so a stale token can no longer hide a
    # deployment that did or did not land.
    health = _get_json(args.base_url, "/health")
    served = str(((health.get("build") or {}).get("commit") or "")).strip() or None
    if served and not commit:
        commit = served

    print("=" * 74)
    print("BORROWING BASE — LIVE ACCEPTANCE")
    print(f"target      : {args.base_url.rstrip('/')}{args.path}")
    print(f"portfolio   : {args.portfolio_id}")
    print(f"reached     : {reached}")
    print(f"authorised  : {authorised}")
    print(f"deployed    : {commit or 'NOT ESTABLISHED'}")
    print(f"expected    : {args.expect_commit or '(not supplied)'}")
    match = bool(commit and args.expect_commit and commit == args.expect_commit)
    print(f"served      : {served or 'NOT ESTABLISHED'}  "
          f"(GET /health, unauthenticated — open path)")
    print(f"SHA match   : {'YES' if match else 'NO'}")
    print("=" * 74)
    if not str(authorised).startswith("YES"):
        # The deployed build is still REPORTED, because it was established
        # without the credential that failed.
        print(f"deployed build established: {served or 'NO'}"
              + (" — matches --expect-commit" if match else ""))
        # WHICH SIDE OF THE CREDENTIAL FAILED. /health publishes the
        # deployment's own bearer configuration precisely so a 401 can be told
        # apart from "this deployment does not read bearers at all" without
        # attempting a login. Printing it turns one indistinguishable 401 into
        # an actionable one: `mode: swa` means no token can ever authenticate
        # here, `bearerConfigured: false` means the deployment is missing its
        # tenant/audience settings, and a configured bearer mode means the
        # token itself (audience, tenant, scope or expiry) is the problem.
        # Configuration STATES only — never a token, a key or a claim.
        # NESTED UNDER `governance`, and read from there. The first cut of this
        # read it at the top level, got {} because the key lives one level
        # down, and fell through to the "the token is the problem" branch — an
        # ABSENT field presented as a positive finding, which is the one thing
        # a diagnostic must never do. Absence is now reported as absence.
        governance = health.get("governance") or {}
        dashboard_auth = governance.get("dashboardAuth") or {}
        print(f"dashboardAuth: {json.dumps(dashboard_auth, sort_keys=True)}")
        print(f"tenantId     : {governance.get('tenantId')}")
        print(f"platformAuth : {json.dumps(governance.get('platformAuth'), sort_keys=True, default=str)}")
        mode = str(dashboard_auth.get("mode") or "")
        if not dashboard_auth:
            print("DIAGNOSIS: UNAVAILABLE — /health carried no governance."
                  "dashboardAuth block, so this run cannot say whether the "
                  "deployment accepts bearer tokens. Not evidence either way.")
        elif mode and mode not in ("bearer", "both"):
            print(f"DIAGNOSIS: this deployment's dashboard auth mode is "
                  f"{mode!r} — a bearer token is NOT read on this surface, so "
                  "no token value can authenticate. Rotating the secret cannot "
                  "fix this; the deployment setting TRAKT_MI_REACT_AUTH_MODE is "
                  "what decides it.")
        elif dashboard_auth.get("bearerConfigured") is False:
            print("DIAGNOSIS: bearer auth is enabled but NOT CONFIGURED on this "
                  "deployment (missing tenant and/or audience settings), so "
                  "every token is refused regardless of its value.")
        else:
            print("DIAGNOSIS: the deployment reads and is configured for bearer "
                  "tokens, so the refusal is a property of THIS token — "
                  "audience, tenant, delegated scope or expiry.")
        print("VERDICT: NOT EXECUTABLE — auth")
        return 2
    if not match:
        # PROVENANCE IS A HARD GATE. An acceptance that cannot say which build
        # answered is evidence about nothing.
        print("VERDICT: NOT EXECUTABLE — provenance")
        return 2

    record: Dict[str, Any] = {"deployed": commit, "portfolio": args.portfolio_id}
    violations: List[str] = []

    # ---- THE EXPECTATION, read from the dashboard's own envelope ---------- #
    scope = f"?portfolioId={args.portfolio_id}" if args.portfolio_id else ""
    envelope = _get_json(args.base_url, "/mi/borrowing-base" + scope)
    if envelope.get("__error__"):
        print(f"\nVERDICT: NOT EXECUTABLE — /mi/borrowing-base said "
              f"{envelope['__error__']} {envelope.get('__body__', '')}")
        return 2
    available = bool(envelope.get("available"))
    governed = envelope.get("measures") or {}
    facility = envelope.get("facility") or {}
    record["envelope"] = {
        "available": available,
        "reason": envelope.get("reason"),
        "reportingDate": envelope.get("reportingDate"),
        "toRunId": envelope.get("toRunId"),
        "facilityId": facility.get("facilityId"),
        "configVersion": facility.get("configVersion"),
        "reconciles": envelope.get("reconciles"),
        "prototypeAssumptionsUsed": envelope.get("prototypeAssumptionsUsed") or [],
        "measures": governed,
    }
    print("\nDASHBOARD ENVELOPE (GET /mi/borrowing-base — the block React renders)")
    print(f"    available : {available}")
    print(f"    facility  : {facility.get('facilityId') or '(none)'}  "
          f"config {facility.get('configVersion') or '(none)'}")
    print(f"    reporting : {envelope.get('reportingDate')}  run {envelope.get('toRunId')}")
    if not available:
        print(f"    reason    : {envelope.get('reason')}")
    else:
        print(f"    measures  : {json.dumps({k: governed.get(k) for _, _, k in MEASURES}, sort_keys=True)}")
    print("\nEXPECTATION DERIVED: MI must "
          + ("MATCH every governed measure above."
             if available else "REFUSE with the envelope's own reason."))

    # ---- 1. Measures ------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("1. SAME-PIPELINE PARITY — MI answer vs the dashboard envelope")
    rows: Dict[str, Any] = {}
    latencies: List[float] = []
    for qid, question, measure in MEASURES:
        started = time.perf_counter()
        env = ask(question)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        meta, bb = _meta(env), _bb(env)
        route = meta.get("route")
        values = bb.get("values") or {}
        not_calc = bb.get("notCalculable") or {}
        answer = str(env.get("answer") or env.get("error") or "")
        row = {"question": question, "measure": measure, "route": route,
               "ok": env.get("ok"), "latency_s": round(elapsed, 3),
               "mi_value": values.get(measure),
               "mi_not_calculable": measure in not_calc,
               "governed_value": governed.get(measure),
               "answer": answer[:200]}
        rows[qid] = row

        if route != "borrowing_base":
            violations.append(f"{qid}: route {route!r}, expected 'borrowing_base'")
        elif not available:
            # The envelope is unavailable: MI must refuse, in the same words.
            if env.get("ok"):
                violations.append(f"{qid}: answered ok while the envelope is "
                                  f"unavailable — {answer[:120]}")
            elif str(envelope.get("reason") or "")[:60].lower() not in answer.lower():
                violations.append(f"{qid}: refused with a DIFFERENT reason from the "
                                  f"envelope — {answer[:120]}")
        else:
            expected = governed.get(measure)
            if expected == NOT_CALCULABLE:
                if not row["mi_not_calculable"] or _is_number(row["mi_value"]):
                    violations.append(
                        f"{qid}: envelope says {measure} is NOT_CALCULABLE; MI "
                        f"reported {row['mi_value']!r}")
            elif row["mi_value"] != expected:
                violations.append(
                    f"{qid}: {measure} MI {row['mi_value']!r} != envelope {expected!r}")
        print(f"\n--- {qid}  {question}")
        print(f"    route={route}  ok={env.get('ok')}  {row['latency_s']}s")
        print(f"    {measure}: MI={row['mi_value']!r}  envelope={row['governed_value']!r}"
              f"  notCalculable={row['mi_not_calculable']}")
        print(f"    answer: {answer[:170]}")
    record["measures"] = rows

    # ---- 2. Governed refusals --------------------------------------------- #
    print("\n" + "=" * 74)
    print("2. GOVERNED REFUSALS — claimed, then refused by name")
    refusals: Dict[str, Any] = {}
    for qid, question, contains in REFUSALS:
        env = ask(question)
        meta = _meta(env)
        answer = str(env.get("answer") or env.get("error") or "")
        refusals[qid] = {"question": question, "route": meta.get("route"),
                         "ok": env.get("ok"),
                         "controlledRefusal": bool(env.get("controlledRefusal")
                                                   or meta.get("controlledRefusal")),
                         "expect_contains": contains, "answer": answer[:200]}
        if meta.get("route") != "borrowing_base":
            violations.append(f"{qid}: route {meta.get('route')!r}, expected "
                              "'borrowing_base'")
        elif env.get("ok"):
            violations.append(f"{qid}: answered ok; a refusal was required")
        elif contains.lower() not in answer.lower():
            violations.append(f"{qid}: refusal does not name {contains!r} — "
                              f"{answer[:120]}")
        if env.get("artifacts"):
            violations.append(f"{qid}: a refusal shipped {len(env['artifacts'])} "
                              "artifact(s)")
        print(f"\n--- {qid}  {question}")
        print(f"    route={meta.get('route')}  ok={env.get('ok')}  "
              f"refusal={refusals[qid]['controlledRefusal']}")
        print(f"    answer: {answer[:170]}")
    record["refusals"] = refusals

    # ---- 3. Non-regression controls --------------------------------------- #
    print("\n" + "=" * 74)
    print("3. NON-REGRESSION — this route claims nothing generic")
    controls: Dict[str, Any] = {}
    for qid, question, answerable in CONTROLS:
        env = ask(question)
        meta = _meta(env)
        route = meta.get("route")
        controls[qid] = {"question": question, "route": route, "ok": env.get("ok"),
                         "answer": str(env.get("answer") or env.get("error") or "")[:200]}
        if route == "borrowing_base":
            violations.append(f"{qid}: CAPTURED by borrowing_base — {question!r}")
        if answerable and not env.get("ok"):
            violations.append(f"{qid}: no longer answers — {controls[qid]['answer'][:120]}")
        print(f"\n--- {qid}  {question}")
        print(f"    route={route}  ok={env.get('ok')}")
        print(f"    answer: {controls[qid]['answer'][:170]}")
    record["controls"] = controls

    # ---- Verdict ----------------------------------------------------------- #
    if latencies:
        ordered = sorted(latencies)
        record["latency"] = {"median_s": round(ordered[len(ordered) // 2], 3),
                             "slowest_s": round(ordered[-1], 3), "n": len(ordered)}
    record["violations"] = violations
    ready = "YES" if not violations else "NO"
    record["BORROWING_BASE_LIVE_READY"] = ready

    print("\n" + "=" * 74)
    print(f"VIOLATIONS: {len(violations)}")
    for line in violations:
        print(f"  - {line}")
    if latencies:
        print(f"latency: median {record['latency']['median_s']}s  "
              f"slowest {record['latency']['slowest_s']}s  n={record['latency']['n']}")
    print(f"\nBORROWING_BASE_LIVE_READY = {ready}")
    print("=" * 74)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(record, indent=1, default=str),
                                       encoding="utf-8")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
