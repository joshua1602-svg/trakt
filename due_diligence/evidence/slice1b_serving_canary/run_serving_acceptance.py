#!/usr/bin/env python3
"""Slice 1B: did the NEW deterministic result actually become the response?

THE ONE ASSERTION LEFT. Slice 1A's deployed acceptance already proved the
interpretation, the plan, the eligibility gate and the deterministic execution
against this same deployment. What it could not prove is the only thing slice 1B
adds: that for the allow-listed principal the new result IS the `/mi/query`
response. Its adjudicator never reads the `serving` block, and it stores the HTTP
response under a field named `legacy_value` — so on a serving canary it compares
the new figure with itself and reports agreement either way. That is why this
exists, and it is all it does.

WHAT IT REUSES RATHER THAN REBUILDS. `run_acceptance.Sink` (the Kudu VFS reader),
`publish_profile_credentials`, `poll_for`, `scrub`, the frozen 12-case manifest
and its sha256 check, and `certify_mi_api._live_asker` for the authenticated POST.
Nothing here is new infrastructure; the new part is the adjudication of six
serving facts and printing them.

IMPORT-LIGHT ON PURPOSE. Stdlib plus those two acceptance modules, and no product
import at module scope. A previous run of the shadow harness died mid-bank on
`ModuleNotFoundError: No module named 'yaml'` because a helper reached into
`mi_agent`; no acceptance workflow in this estate installs the application's
dependency tree.

PROVENANCE IS A GET. `certify_mi_api.preflight` POSTs a question before reading
the health stamp, and on a live serving canary that POST costs a real Opus
interpretation and serves a new result for a question in no bank. So the commit
is read from `/health` directly — the same `build.commit` stamp, no question
asked, and the run stops before the bank if it does not match.

THE SERVING BLOCK IS AUTHORITATIVE for which path served. The HTTP response is
used only to prove CORRESPONDENCE: that the figure or the grid the caller
received is the one this record says the deterministic engine produced. Neither
is inferred from the other.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

HERE = Path(__file__).resolve().parent
_REPO_ROOT = HERE.parents[2]
_SHADOW = _REPO_ROOT / "due_diligence" / "evidence" / "deployed_acceptance_0399a315"
for _path in (str(_REPO_ROOT), str(_SHADOW)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_acceptance as ra                                           # noqa: E402

from due_diligence.evidence.mi_api_certification.certify_mi_api import (  # noqa: E402
    _live_asker)

#: Agreement tolerance, the same 0.01 the shadow adjudicator uses.
TOLERANCE = 0.01

#: How many of the manifest's eligible cases to ask. A01-A06.
CASES = 6


def served_commit(base_url: str) -> Optional[str]:
    """The deployed build stamp, by GET. No question is asked."""
    bearer = os.environ.get("MI_BEARER", "").strip()
    for candidate in ("/health", "/api/health"):
        try:
            request = urllib.request.Request(
                base_url.rstrip("/") + candidate, method="GET")
            if bearer:
                request.add_header("Authorization", "Bearer "
                                   + bearer.removeprefix("Bearer ").strip())
            with urllib.request.urlopen(request, timeout=30) as response:
                health = json.loads(response.read().decode("utf-8") or "{}")
            build = health.get("build")
            if isinstance(build, Mapping) and build.get("commit"):
                return str(build["commit"]).strip()
        except Exception:                                            # noqa: BLE001
            continue
    return None


def response_figures(envelope: Mapping[str, Any]
                     ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """`({kpi field: figure}, grid rows)` as the CALLER received them."""
    kpis: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    for artefact in (envelope.get("artifacts") or ()):
        for kpi in (artefact.get("kpis") or ()):
            raw = kpi.get("rawValue")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                kpis[str(kpi.get("field"))] = float(raw)
        candidate = artefact.get("rows") or artefact.get("data") or ()
        if candidate and isinstance(candidate[0], Mapping):
            rows = [dict(row) for row in candidate]
    return kpis, rows


def corresponds(record: Mapping[str, Any], envelope: Mapping[str, Any]
                ) -> Tuple[bool, str]:
    """Does the response carry the figure THIS record says was executed?

    Grouped plans are compared cell by cell on the plan's own axes; scalar plans
    on the KPI whose field name the bound spec implies — not the first KPI, which
    is the loan count and read 130 for a question about £37.9MM of balance.
    """
    execution = record.get("execution") or {}
    spec = execution.get("bound_spec") or {}
    kpis, rows = response_figures(envelope)
    cells = execution.get("grouped_cells")

    if cells:
        axes = [str(a) for a in (spec.get("dimensions") or ())]
        if not axes:
            return False, "grouped cells with no axes in the bound spec"
        executed = {tuple(str(cell.get(a)) for a in axes): cell.get("value")
                    for cell in cells}
        received: Dict[Tuple[str, ...], float] = {}
        for row in rows:
            if not all(a in row for a in axes):
                continue
            figures = [v for k, v in row.items()
                       if k not in axes and isinstance(v, (int, float))
                       and not isinstance(v, bool)]
            if figures:
                received[tuple(str(row[a]) for a in axes)] = float(figures[0])
        if not received:
            return False, "the response carries no grid for a grouped plan"
        if set(executed) != set(received):
            return False, (f"grid keys differ: {len(set(executed) ^ set(received))} "
                           f"not in both")
        differing = [key for key, value in executed.items()
                     if value is None
                     or abs(float(value) - received[key]) > TOLERANCE]
        if differing:
            return False, f"{len(differing)} cell(s) differ from the response"
        return True, f"{len(received)} cells match"

    value = execution.get("value")
    if value is None:
        return False, "the record carries no figure to correspond to"
    aggregation = str(spec.get("aggregation") or "")
    field = ("loan_count" if aggregation == "count"
             else f"{spec.get('metric')}_{aggregation}")
    received_value = kpis.get(field)
    if received_value is None:
        return False, f"the response carries no {field!r} figure"
    if abs(received_value - float(value)) > TOLERANCE:
        return False, (f"the response carries {received_value} where the record "
                       f"executed {value}")
    return True, f"{field}={received_value}"


def judge(case: Mapping[str, Any], record: Optional[Mapping[str, Any]],
          envelope: Mapping[str, Any], http_status: Optional[int]
          ) -> Dict[str, Any]:
    """The six serving facts for one case, from the record and the response."""
    out: Dict[str, Any] = {
        "case_id": case["case_id"], "question": case["question"],
        "http_status": http_status, "http_ok": bool(envelope.get("ok")),
        "evidence": record is not None, "model_id": None,
        "principal_matched": None, "eligible": None, "decision": None,
        "response_served_from": None, "execution_ok": None,
        "response_matches_new": None, "note": "", "verdict": None,
    }
    if record is None:
        out["verdict"] = "EVIDENCE_LOST"
        out["note"] = "no new record arrived within the polling window"
        return out

    serving = record.get("serving") or {}
    execution = record.get("execution") or {}
    out["model_id"] = (record.get("model") or {}).get("model_id")
    out["principal_matched"] = bool(serving.get("principal_matched"))
    out["eligible"] = bool((record.get("eligibility") or {}).get("eligible"))
    out["decision"] = serving.get("decision")
    out["response_served_from"] = serving.get("response_served_from")
    out["disposition"] = record.get("disposition")
    out["plan_id"] = (record.get("compiler") or {}).get("plan_id")
    out["execution_ok"] = bool(execution.get("attempted")
                               and not execution.get("error")
                               and execution.get("reconciled"))
    out["orchestration_error"] = record.get("orchestration_error")

    # Silent semantic loss, exactly as the shadow adjudicator measures it: a
    # facet the plan stated that did not reach the executable spec.
    requested = execution.get("requested_semantics") or {}
    spec = execution.get("bound_spec") or {}
    drops = sorted({f.get("field") for f in (requested.get("filters") or ())}
                   - set(spec.get("filters") or {}))
    drops += sorted(set(requested.get("dimensions") or ())
                    - set(spec.get("dimensions") or ()))
    if (requested.get("measure_field") and spec.get("aggregation") != "count"
            and spec.get("metric") != requested.get("measure_field")):
        drops.append(str(requested.get("measure_field")))
    out["semantic_drops"] = drops

    matches, detail = corresponds(record, envelope)
    out["response_matches_new"] = matches
    out["note"] = detail

    if not ra.is_required_model(out["model_id"]):
        out["verdict"] = "MODEL_NOT_OPUS"
    elif not out["principal_matched"]:
        out["verdict"] = "PRINCIPAL_NOT_MATCHED"
    elif not out["eligible"]:
        out["verdict"] = "NOT_ELIGIBLE"
    elif out["decision"] != "NEW" or out["response_served_from"] != "NEW":
        out["verdict"] = "LEGACY_FALLBACK"
        out["note"] = f"reason={serving.get('reason')!r}"
    elif out["orchestration_error"]:
        out["verdict"] = "EXCEPTION_ESCAPED"
    elif not out["execution_ok"]:
        out["verdict"] = "EXECUTION_ERROR"
    elif drops:
        out["verdict"] = "SEMANTIC_DROP"
    elif not matches:
        out["verdict"] = "RESPONSE_DOES_NOT_MATCH_NEW"
    else:
        out["verdict"] = "SERVED_NEW"
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--path", default="/mi/query")
    parser.add_argument("--portfolio-id", required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--evidence-path", required=True)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--poll-timeout", type=float, default=120.0)
    parser.add_argument("--json-out", default="slice1b-serving-acceptance.json")
    args = parser.parse_args(argv)

    bearer = os.environ.get("MI_BEARER", "").strip()
    profile = os.environ.get("AZURE_MI_API_PUBLISH_PROFILE", "").strip()
    secrets = [s for s in (bearer, profile) if len(s) >= ra.MIN_SECRET_LENGTH]
    if not bearer or not profile:
        print("::error::MI_BEARER and AZURE_MI_API_PUBLISH_PROFILE are both "
              "required")
        return 2

    manifest = ra.verify_manifest()
    client_id = args.portfolio_id.split("/", 1)[0]
    eligible = [c for c in manifest["cases"]
                if c["expected_slice1_eligible"]][:CASES]
    report: Dict[str, Any] = {
        "what_this_is": "slice 1B live serving canary — did the NEW result "
                        "become the response, for the canary principal",
        "manifest_sha256": ra.MANIFEST_HASH.read_text().split()[0],
        "expect_commit": args.expect_commit, "portfolio_id": args.portfolio_id,
        "cases_asked": [c["case_id"] for c in eligible], "stages": {},
        "cases": [],
    }

    def stop(stage: str, verdict: str, message: str) -> int:
        """One exit for every gate, so none can forget to record why."""
        report["verdict"], report["stopped_at"] = verdict, stage
        ra._save(report, args.json_out, secrets)
        print(f"::error::{message}")
        return 2

    # -- provenance, by GET, before a single Opus call ------------------------
    commit = served_commit(args.base_url)
    expected = args.expect_commit.strip().lower()
    matches = bool(commit) and (commit.lower().startswith(expected[:7])
                               or expected.startswith(commit.lower()[:7]))
    report["stages"]["provenance"] = {"served_commit": commit,
                                      "matches_expected": matches}
    print(f"expected sha       {args.expect_commit}")
    print(f"served sha         {commit or 'NOT ESTABLISHED'}")
    if not matches:
        return stop("provenance", ra.INCONCLUSIVE,
                    "the deployed commit is not the one under acceptance; no "
                    "question was asked")

    # -- the sink, readable, and its contents BEFORE the bank ----------------
    try:
        scm_host, user, password = ra.publish_profile_credentials(profile)
    except Exception as exc:                                         # noqa: BLE001
        return stop("sink credentials", ra.INCONCLUSIVE, str(exc))
    sink = ra.Sink(scm_host, user, password, args.evidence_path)
    rows, detail = sink.records()
    report["stages"]["sink"] = {"readable": rows is not None, "detail": detail,
                               "records_before_the_run": len(rows or ())}
    print(f"evidence sink      {sink.scm_host} :: "
          f"{ra.vfs_path(args.evidence_path)} — {detail}")
    if rows is None:
        return stop("sink unreadable", ra.INCONCLUSIVE,
                    "the sink cannot be read, so which path served cannot be "
                    "proved; stopping BEFORE spending model calls")
    # Every pre-existing correlation id, so no old record can satisfy a case.
    seen = {row.get("correlation_id") for row in rows}

    # -- the six questions, each asked once ----------------------------------
    ask = _live_asker(args.base_url, args.path, [], args.portfolio_id)
    judged: List[Dict[str, Any]] = []
    for case in eligible:
        envelope = ask(case["question"])
        status = envelope.get("__http_status__") or (
            None if envelope.get("__transport_error__") else 200)
        if status in (401, 403):
            return stop(f"authentication on {case['case_id']}",
                        "AUTH / NOT_EXECUTABLE",
                        f"HTTP {status} on {case['case_id']}; the credential was "
                        f"refused before any semantics ran")
        record, how = ra.poll_for(sink, case["question"], client_id,
                                  interval=args.poll_interval,
                                  timeout=args.poll_timeout, already=seen)
        verdict = judge(case, record, envelope, status)
        verdict["evidence_lookup"] = how
        verdict["raw_serving"] = (record or {}).get("serving")
        judged.append(verdict)
        print(f"  {verdict['case_id']}  http={str(status):<4} "
              f"{str(verdict['decision']):<16}{verdict['verdict']:<28}"
              f"{verdict['note'][:60]}")

    report["cases"] = judged
    served_new = sum(1 for c in judged if c["verdict"] == "SERVED_NEW")
    tallies = {
        "live_questions": len(judged),
        "http_successful": sum(1 for c in judged if c["http_status"] == 200),
        "live_opus_interpretations": sum(1 for c in judged
                                         if ra.is_required_model(c["model_id"])),
        "principal_matches": sum(1 for c in judged if c["principal_matched"]),
        "eligible": sum(1 for c in judged if c["eligible"]),
        "served_new": served_new,
        "legacy_fallbacks": sum(1 for c in judged
                                if c["verdict"] == "LEGACY_FALLBACK"),
        "response_to_new_matches": sum(1 for c in judged
                                       if c["response_matches_new"]),
        "semantic_drops": sum(len(c.get("semantic_drops") or ()) for c in judged),
        "execution_errors": sum(1 for c in judged
                                if c["verdict"] == "EXECUTION_ERROR"),
        "exception_escapes": sum(1 for c in judged
                                 if c["verdict"] == "EXCEPTION_ESCAPED"),
        "evidence_lost": sum(1 for c in judged if c["verdict"] == "EVIDENCE_LOST"),
    }
    report.update(tallies)

    if tallies["evidence_lost"]:
        report["verdict"] = ra.INCONCLUSIVE
        report["fail_reason"] = (f"{tallies['evidence_lost']} case(s) produced no "
                                 f"record; which path served cannot be proved")
    elif served_new == len(eligible) == CASES:
        report["verdict"] = ra.PASS
    else:
        report["verdict"] = ra.FAIL
        report["fail_reason"] = ", ".join(
            f"{c['case_id']}={c['verdict']}" for c in judged
            if c["verdict"] != "SERVED_NEW")

    ra._save(report, args.json_out, secrets)
    for key, value in tallies.items():
        print(f"{key:<28} {value}")
    if report.get("fail_reason"):
        print(f"fail reason        {report['fail_reason']}")
    print(f"verdict            {report['verdict']}")
    print("::notice::set MI_AGENT_PLAN_SERVE back to off on trakt-mi-api now; "
          "it is an app setting and needs ARM access this workflow does not hold")
    return 0 if report["verdict"] == ra.PASS else (
        1 if report["verdict"] == ra.FAIL else 2)


if __name__ == "__main__":
    raise SystemExit(main())
