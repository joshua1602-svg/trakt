#!/usr/bin/env python3
"""The 12-case deployed shadow acceptance, run from CI against the real API.

WHAT IT DOES, IN ORDER, AND WHY THE ORDER MATTERS.

    0  verifies the pre-registered manifest against its committed sha256, so a
       run cannot quietly adjudicate against an edited bank;
    1  proves the SERVED commit through the estate's own
       `certify_mi_api.preflight`, and refuses to continue if it is not the
       commit this acceptance was dispatched to certify;
    2  proves the shadow evidence sink is READABLE from here, before a single
       model call is spent. The brief's phase 2 gate: a run that cannot read the
       evidence can only produce a transcript, never an adjudication;
    3  optionally probes canary isolation with a second, non-canary portfolio;
    4  posts the 12 pre-registered questions and polls the sink for each;
    5  adjudicates offline, writes JSON, and exits non-zero unless everything
       passed.

WHAT IT CANNOT DO, SAID HERE RATHER THAN DISCOVERED IN A LOG.

    IT DOES NOT ENABLE OR DISABLE THE SHADOW. App settings need ARM credentials
    and this repository's mi-api automation holds only a publish profile, which
    grants Kudu and nothing else. The canary must already be configured when this
    runs, and turning it off afterwards is an operator action that this harness
    reports as outstanding rather than silently claiming.

    THERE IS NO INDEPENDENT ORACLE FOR DEPLOYED FIGURES. The manifest
    pre-registers every eligible figure as TRUTH_UNAVAILABLE for that reason. A
    shadow figure matching the legacy answer is recorded as a SIGNAL; it is never
    scored as a pass, because the legacy answer is not the truth standard.

CREDENTIALS. `MI_BEARER` and the publish profile are read from the environment
only — never an argv value, which would put them in process listings and CI logs.
Neither is written to the evidence, and the output is scanned for both before it
is saved.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from due_diligence.evidence.mi_api_certification.certify_mi_api import (  # noqa: E402
    _live_asker, preflight)

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "acceptance_manifest.json"
MANIFEST_HASH = HERE / "acceptance_manifest.sha256"

REQUIRED_MODEL = "claude-opus-5"

# Verdicts for the run as a whole.
PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"

# Per-case classifications, as the brief names them.
EXACT_SEMANTIC_PARITY = "EXACT_SEMANTIC_PARITY"
SEMANTICALLY_EQUIVALENT = "SEMANTICALLY_EQUIVALENT"
NEW_PATH_CORRECT_LEGACY_DIFFERS = "NEW_PATH_CORRECT_LEGACY_DIFFERS"
LEGACY_NEW_DIVERGENCE = "LEGACY_NEW_DIVERGENCE"
JUSTIFIED_INELIGIBLE = "JUSTIFIED_INELIGIBLE"
INTERPRETATION_DEVIATION = "INTERPRETATION_DEVIATION"
ELIGIBILITY_DEFECT = "ELIGIBILITY_DEFECT"
DETERMINISTIC_EXECUTION_DEFECT = "DETERMINISTIC_EXECUTION_DEFECT"
ASYNC_EVIDENCE_LOST = "ASYNC_EVIDENCE_LOST"
TRUTH_UNAVAILABLE = "TRUTH_UNAVAILABLE"
SHADOW_NOT_ACTIVE = "SHADOW_NOT_ACTIVE"

#: Classifications that make the run FAIL rather than merely inconclusive.
DEFECTS = frozenset({ELIGIBILITY_DEFECT, DETERMINISTIC_EXECUTION_DEFECT,
                     LEGACY_NEW_DIVERGENCE})

#: Facets whose LOSS is a silent semantic drop rather than a change.
LOSABLE = ("filters", "dimensions", "measure_field", "statistic",
           "comparison_kind", "period_form", "population_lens",
           "geography_requested")


# --------------------------------------------------------------------------- #
# reading the sink — Kudu VFS, on the credential the deployment already uses
# --------------------------------------------------------------------------- #

def publish_profile_credentials(xml_text: str) -> Tuple[str, str, str]:
    """`(scm_host, username, password)` from an App Service publish profile.

    The MSDeploy profile is the one carrying SCM credentials; its `publishUrl` is
    the SCM host. Raises rather than guessing, because a silently wrong host would
    read as "the sink is not there".
    """
    root = ElementTree.fromstring(xml_text)
    profiles = [p for p in root.iter("publishProfile")]
    chosen = next((p for p in profiles
                   if (p.get("publishMethod") or "").lower() == "msdeploy"), None)
    if chosen is None:
        raise ValueError("the publish profile carries no MSDeploy entry, so it "
                         "grants no SCM credential")
    host = (chosen.get("publishUrl") or "").strip()
    host = host.split(":", 1)[0].strip("/")
    user = (chosen.get("userName") or "").strip()
    password = (chosen.get("userPWD") or "").strip()
    if not (host and user and password):
        raise ValueError("the MSDeploy profile is missing publishUrl, userName "
                         "or userPWD")
    return host, user, password


def vfs_path(sink_path: str) -> str:
    """An absolute app path -> a Kudu VFS path.

    Kudu's VFS root is `/home`, so `/home/LogFiles/x.jsonl` is `LogFiles/x.jsonl`
    and a path outside `/home` is addressed from the filesystem root. Written out
    because getting this wrong reads exactly like a missing file.
    """
    clean = sink_path.strip()
    if clean.startswith("/home/"):
        return clean[len("/home/"):]
    return clean.lstrip("/")


class Sink:
    """Read-only access to the deployed evidence sink."""

    def __init__(self, scm_host: str, user: str, password: str,
                 sink_path: str) -> None:
        self.scm_host = scm_host
        self.sink_path = sink_path
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self._auth = f"Basic {token}"

    @property
    def url(self) -> str:
        # A host given WITH a scheme is used verbatim. That is what lets this be
        # exercised end to end against a local stand-in before it is ever pointed
        # at production, and it covers an App Service whose SCM endpoint is not at
        # the default host. Without a scheme, https is assumed, which is the only
        # thing Kudu serves.
        if self.scm_host.startswith(("http://", "https://")):
            root = self.scm_host.rstrip("/")
        else:
            root = f"https://{self.scm_host}"
        return f"{root}/api/vfs/{vfs_path(self.sink_path)}"

    def read(self) -> Tuple[Optional[str], str]:
        """`(body, detail)`. `body` is None when the sink could not be read."""
        request = urllib.request.Request(self.url, method="GET")
        request.add_header("Authorization", self._auth)
        # Kudu returns 412 without this on some file endpoints.
        request.add_header("If-Match", "*")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", "replace"), "read"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return "", "absent (404) — the sink file does not exist yet"
            return None, (f"HTTP {exc.code} from Kudu VFS; 401/403 usually means "
                          f"SCM basic authentication is disabled on the app")
        except Exception as exc:                                     # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    def records(self) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        body, detail = self.read()
        if body is None:
            return None, detail
        rows = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows, detail


# --------------------------------------------------------------------------- #
# adjudication
# --------------------------------------------------------------------------- #

def _comparable(value: Any) -> Any:
    """Tuples and lists compare equal, recursively — JSON has no tuple."""
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    if isinstance(value, Mapping):
        return {k: _comparable(v) for k, v in value.items()}
    return value


def semantics_of(plan: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The same projection the manifest pre-registered, over a live plan."""
    sys.path.insert(0, str(HERE))
    import preregister
    return preregister._semantics(plan)


def compare_semantics(expected: Mapping[str, Any],
                      live: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    drops, changes = [], []
    for facet in sorted(set(expected) | set(live)):
        want, got = expected.get(facet), live.get(facet)
        if _comparable(want) == _comparable(got):
            continue
        if facet in LOSABLE and bool(want) and not got:
            drops.append(f"{facet}: {want!r} -> {got!r}")
        else:
            changes.append(f"{facet}: {want!r} -> {got!r}")
    return drops, changes


def adjudicate(case: Mapping[str, Any], record: Optional[Mapping[str, Any]],
               legacy: Mapping[str, Any]) -> Dict[str, Any]:
    """One case, judged against its pre-registration and nothing else."""
    out: Dict[str, Any] = {
        "case_id": case["case_id"],
        "question": case["question"],
        "legacy_http_status": legacy.get("http_status"),
        "legacy_ok": legacy.get("ok"),
        "classification": None,
        "notes": [],
        "silent_drops": [],
        "semantic_changes": [],
    }
    if record is None:
        out["classification"] = ASYNC_EVIDENCE_LOST
        out["notes"].append("no evidence record arrived within the polling "
                            "window; this case is inconclusive, not a pass and "
                            "not a semantic failure")
        return out

    compiler = record.get("compiler") or {}
    model = record.get("model") or {}
    eligibility = record.get("eligibility") or {}
    execution = record.get("execution") or {}

    out["correlation_id"] = record.get("correlation_id")
    out["disposition"] = record.get("disposition")
    out["model_id"] = model.get("model_id")
    out["plan_id"] = compiler.get("plan_id")
    out["expected_plan_id"] = case["expected_frozen_plan_id"]

    # --- eligibility is checked first: a mismatch is a defect whatever else --
    expected_eligible = case["expected_slice1_eligible"]
    expected_reason = case["expected_ineligible_reason"]
    live_eligible = eligibility.get("eligible")
    live_reason = eligibility.get("reason") or ""
    if compiler.get("outcome") == "PLAN":
        if live_eligible != expected_eligible or live_reason != expected_reason:
            out["classification"] = ELIGIBILITY_DEFECT
            out["notes"].append(
                f"expected eligible={expected_eligible!r} reason="
                f"{expected_reason!r}; got eligible={live_eligible!r} reason="
                f"{live_reason!r}")
            return out

    # --- interpretation -----------------------------------------------------
    live_semantics = semantics_of(compiler.get("plan"))
    if compiler.get("outcome") == case["expected_interpretation_disposition"]:
        drops, changes = compare_semantics(case["expected_semantics"],
                                           live_semantics)
        out["silent_drops"], out["semantic_changes"] = drops, changes
    else:
        drops, changes = [], [f"disposition: "
                              f"{case['expected_interpretation_disposition']} -> "
                              f"{compiler.get('outcome')}"]
        out["semantic_changes"] = changes

    # --- ineligible controls -----------------------------------------------
    if not expected_eligible:
        if execution.get("attempted"):
            out["classification"] = ELIGIBILITY_DEFECT
            out["notes"].append("an ineligible plan reached the adapter")
            return out
        out["classification"] = (JUSTIFIED_INELIGIBLE
                                 if not out["semantic_changes"]
                                 else INTERPRETATION_DEVIATION)
        out["notes"].append(f"gate reason {live_reason} as pre-registered")
        return out

    # --- eligible: did the plan reach the spec whole? ------------------------
    spec = execution.get("bound_spec") or {}
    requested = execution.get("requested_semantics") or {}
    if execution.get("error"):
        missing = [f for f in ("ticket_bucket", "interest_rate_bucket")
                   if f in json.dumps(spec.get("dimensions") or [])]
        out["classification"] = TRUTH_UNAVAILABLE if missing else \
            DETERMINISTIC_EXECUTION_DEFECT
        out["notes"].append(f"execution error: {execution['error']}")
        if missing:
            out["notes"].append(
                f"the deployed frame does not carry {missing}; pre-registered as "
                f"an acceptable branch and a dataset fact, not a defect")
        return out

    requested_filters = {f.get("field") for f in (requested.get("filters") or ())}
    spec_filters = set(spec.get("filters") or {})
    filter_drops = sorted(requested_filters - spec_filters)
    dimension_drops = sorted(set(requested.get("dimensions") or ())
                             - set(spec.get("dimensions") or ()))
    measure_dropped = bool(
        requested.get("measure_field")
        and spec.get("aggregation") != "count"
        and spec.get("metric") != requested.get("measure_field"))
    out["filter_drops"] = filter_drops
    out["dimension_drops"] = dimension_drops
    out["measure_dropped"] = measure_dropped

    if filter_drops or dimension_drops or measure_dropped:
        out["classification"] = DETERMINISTIC_EXECUTION_DEFECT
        out["notes"].append("a stated facet did not reach the executable spec")
        return out

    # --- grouped grids: reconciled against the engine's OWN receipt ----------
    cells = execution.get("grouped_cells")
    if cells is not None:
        receipt_groups = (execution.get("receipt") or {}).get("row_count")
        out["grouped_cells"] = len(cells)
        out["receipt_row_count"] = receipt_groups
        incomplete = [c for c in cells
                      if any(not str(c.get(d) or "")
                             for d in (spec.get("dimensions") or ()))]
        if receipt_groups is not None and len(cells) != receipt_groups:
            out["classification"] = DETERMINISTIC_EXECUTION_DEFECT
            out["notes"].append(
                f"{len(cells)} cells captured but the receipt reports "
                f"{receipt_groups} groups")
            return out
        if incomplete:
            out["classification"] = DETERMINISTIC_EXECUTION_DEFECT
            out["notes"].append(f"{len(incomplete)} cells are missing a group key")
            return out
        out["notes"].append(
            "grouped grid reconciles with the execution receipt's own group "
            "count; there is no independent oracle for deployed figures")

    # --- the figure: a signal against legacy, never a verdict ----------------
    shadow_value, legacy_value = execution.get("value"), legacy.get("value")
    out["shadow_value"] = shadow_value
    out["legacy_value"] = legacy_value
    out["truth_status"] = case["independent_numerical_truth_status"]
    if (isinstance(shadow_value, (int, float))
            and isinstance(legacy_value, (int, float))):
        agrees = abs(float(shadow_value) - float(legacy_value)) < 0.01
        out["legacy_agreement_signal"] = "AGREES" if agrees else "DIFFERS"
        out["notes"].append(
            "the legacy figure is a SIGNAL, not the oracle: agreement is not a "
            "pass and disagreement is not a defect without independent truth")

    if out["silent_drops"]:
        out["classification"] = DETERMINISTIC_EXECUTION_DEFECT
        out["notes"].append("a pre-registered facet was silently lost")
    elif out["semantic_changes"]:
        out["classification"] = INTERPRETATION_DEVIATION
    elif out["plan_id"] == out["expected_plan_id"]:
        out["classification"] = EXACT_SEMANTIC_PARITY
    else:
        out["classification"] = SEMANTICALLY_EQUIVALENT
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def verify_manifest() -> Dict[str, Any]:
    body = MANIFEST.read_text()
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    recorded = MANIFEST_HASH.read_text().split()[0]
    if digest != recorded:
        raise SystemExit(f"::error::the acceptance manifest has changed since it "
                         f"was hashed ({digest} != {recorded})")
    return json.loads(body)


def poll_for(sink: Sink, question: str, client_id: str, *, interval: float,
             timeout: float, already: set) -> Tuple[Optional[Dict[str, Any]], str]:
    """Wait for this question's record. Absence at first is not a failure."""
    deadline = time.monotonic() + timeout
    detail = "not polled"
    while True:
        rows, detail = sink.records()
        if rows is not None:
            for row in rows:
                request = row.get("request") or {}
                cid = row.get("correlation_id")
                if (request.get("question") == question
                        and str(request.get("client_id") or "").lower()
                        == client_id.lower()
                        and cid not in already):
                    already.add(cid)
                    return row, "matched on question + client"
        if time.monotonic() >= deadline:
            return None, f"timed out after {timeout:.0f}s ({detail})"
        time.sleep(interval)


#: A value shorter than this is not treated as a secret to search for. A real
#: token is long; a one- or two-character "secret" matches half the document and
#: would make the backstop refuse to write legitimate evidence, which is how a
#: safety check turns into an outage. Found by running this harness with a
#: single-character stand-in token.
MIN_SECRET_LENGTH = 8


def scrub(node: Any, secrets: List[str]) -> Any:
    """Last line of defence: no credential may reach the evidence file."""
    if isinstance(node, str):
        for secret in secrets:
            if secret and secret in node:
                return "[REDACTED]"
        return node
    if isinstance(node, Mapping):
        return {k: scrub(v, secrets) for k, v in node.items()
                if not any(t in str(k).lower() for t in
                           ("authorization", "bearer", "password", "userpwd",
                            "secret", "token"))}
    if isinstance(node, list):
        return [scrub(v, secrets) for v in node]
    return node


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--path", default="/mi/query")
    parser.add_argument("--portfolio-id", required=True,
                        help="the CANARY portfolio; its client half must be the "
                             "one id in MI_AGENT_PLAN_SHADOW_CLIENTS")
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--evidence-path", required=True,
                        help="the sink path configured on the app, e.g. "
                             "/home/LogFiles/mi-plan-shadow/evidence.jsonl")
    parser.add_argument("--scm-host", default="",
                        help="override the SCM host; normally taken from the "
                             "publish profile")
    parser.add_argument("--non-canary-portfolio-id", default="",
                        help="optional second portfolio, to prove isolation")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--poll-timeout", type=float, default=60.0)
    parser.add_argument("--json-out", default="deployed-shadow-acceptance.json")
    args = parser.parse_args(argv)

    bearer = os.environ.get("MI_BEARER", "").strip()
    profile = os.environ.get("AZURE_MI_API_PUBLISH_PROFILE", "").strip()
    secrets = [s for s in (bearer, profile) if len(s) >= MIN_SECRET_LENGTH]
    if not bearer:
        print("::error::MI_BEARER is not set")
        return 2
    if not profile:
        print("::error::AZURE_MI_API_PUBLISH_PROFILE is not set; the evidence "
              "sink cannot be read without it")
        return 2

    manifest = verify_manifest()
    client_id = args.portfolio_id.split("/", 1)[0]
    report: Dict[str, Any] = {
        "what_this_is": "deployed slice 1 shadow acceptance, 12 pre-registered "
                        "cases",
        "manifest_sha256": MANIFEST_HASH.read_text().split()[0],
        "manifest_verified_unchanged": True,
        "base_url": args.base_url,
        "path": args.path,
        "canary_portfolio_id": args.portfolio_id,
        "canary_client_id": client_id,
        "expect_commit": args.expect_commit,
        "evidence_path": args.evidence_path,
        "polling": {"interval_seconds": args.poll_interval,
                    "timeout_seconds": args.poll_timeout},
        "shadow_enabled_by_this_harness": False,
        "shadow_disabled_by_this_harness": False,
        "shadow_disable_is_an_operator_action": True,
        "stages": {},
        "cases": [],
    }

    # -- stage 1: provenance ------------------------------------------------
    # `preflight` establishes reachability by POSTING one question, which on a
    # canary client means one shadow interpretation that is NOT part of the bank.
    # So it is sent as the NON-canary portfolio whenever one was supplied, and the
    # cost is accounted for either way rather than left to be noticed in a bill.
    probe_portfolio = args.non_canary_portfolio_id or args.portfolio_id
    probe_is_canary = probe_portfolio == args.portfolio_id
    report["preflight_probe"] = {
        "portfolio_id": probe_portfolio,
        "on_the_canary_client": probe_is_canary,
        "extra_shadow_interpretations": 1 if probe_is_canary else 0,
        "note": ("the probe ran as the canary client, so ONE shadow "
                 "interpretation beyond the 12 was spent; its record is excluded "
                 "from the bank by correlation id"
                 if probe_is_canary else
                 "the probe ran as a non-canary portfolio, so it cost no shadow "
                 "interpretation"),
    }
    reached, authorised, commit = preflight(args.base_url, args.path, [],
                                           probe_portfolio)
    expected = args.expect_commit.strip().lower()
    matches = bool(commit) and (commit.lower().startswith(expected[:7])
                               or expected.startswith(commit.lower()[:7]))
    report["stages"]["provenance"] = {
        "reached": reached, "authorised": authorised,
        "served_commit": commit, "matches_expected": matches}
    print(f"reached            {reached}")
    print(f"authorised         {authorised}")
    print(f"served commit      {commit or 'NOT ESTABLISHED'}")
    if reached != "YES" or authorised != "YES" or not matches:
        report["verdict"] = INCONCLUSIVE
        report["stopped_at"] = "provenance"
        _save(report, args.json_out, secrets)
        print("::error::the deployed commit could not be confirmed as the one "
              "under acceptance; nothing was asked")
        return 2

    # -- stage 2: can the sink be read at all? ------------------------------
    try:
        scm_host, user, password = publish_profile_credentials(profile)
    except Exception as exc:                                         # noqa: BLE001
        report["stages"]["sink"] = {"readable": False, "detail": str(exc)}
        report["verdict"] = INCONCLUSIVE
        report["stopped_at"] = "sink credentials"
        _save(report, args.json_out, secrets)
        print(f"::error::{exc}")
        return 2
    sink = Sink(args.scm_host or scm_host, user, password, args.evidence_path)
    rows, detail = sink.records()
    report["stages"]["sink"] = {
        "scm_host": sink.scm_host, "vfs_path": vfs_path(args.evidence_path),
        "readable": rows is not None, "detail": detail,
        "records_present_before_the_run": len(rows or ())}
    print(f"evidence sink      {sink.scm_host} :: "
          f"{vfs_path(args.evidence_path)} — {detail}")
    if rows is None:
        report["verdict"] = INCONCLUSIVE
        report["stopped_at"] = "sink unreadable"
        _save(report, args.json_out, secrets)
        print("::error::the evidence sink cannot be read from here, so no "
              "adjudication is possible; stopping BEFORE spending model calls")
        return 2
    seen = {row.get("correlation_id") for row in rows}

    # -- stage 3: optional isolation probe ----------------------------------
    if args.non_canary_portfolio_id:
        other = args.non_canary_portfolio_id
        ask_other = _live_asker(args.base_url, args.path, [], other)
        envelope = ask_other(manifest["cases"][0]["question"])
        time.sleep(args.poll_interval * 3)
        after, _ = sink.records()
        new = [r for r in (after or ())
               if r.get("correlation_id") not in seen]
        foreign = [r for r in new
                   if str(((r.get("request") or {}).get("client_id")) or "").lower()
                   != client_id.lower()]
        report["stages"]["canary_isolation"] = {
            "non_canary_portfolio_id": other,
            "http_ok": not envelope.get("__transport_error__"),
            "new_records_for_a_non_canary_client": len(foreign),
            "isolated": not foreign}
        seen |= {r.get("correlation_id") for r in (after or ())}
        print(f"canary isolation   {'PASS' if not foreign else 'FAIL'} "
              f"({len(foreign)} foreign records)")
        if foreign:
            report["verdict"] = FAIL
            report["stopped_at"] = "canary isolation"
            _save(report, args.json_out, secrets)
            print("::error::a non-canary client produced shadow evidence")
            return 1
    else:
        report["stages"]["canary_isolation"] = {
            "isolated": "UNPROVEN — no --non-canary-portfolio-id was supplied; "
                        "every retrieved record is still checked to belong to the "
                        "one canary client"}

    # -- stage 4: the bank ---------------------------------------------------
    ask = _live_asker(args.base_url, args.path, [], args.portfolio_id)
    adjudicated: List[Dict[str, Any]] = []
    for case in manifest["cases"]:
        envelope = ask(case["question"])
        legacy = {
            "ok": bool(envelope.get("ok")),
            "http_status": envelope.get("__http_status__") or (
                None if envelope.get("__transport_error__") else 200),
            "transport_error": bool(envelope.get("__transport_error__")),
            "value": envelope.get("value"),
            "route": envelope.get("route"),
        }
        record, how = poll_for(sink, case["question"], client_id,
                               interval=args.poll_interval,
                               timeout=args.poll_timeout, already=seen)
        verdict = adjudicate(case, record, legacy)
        verdict["evidence_lookup"] = how
        verdict["raw_evidence"] = record
        adjudicated.append(verdict)
        print(f"  {case['case_id']}  {str(verdict.get('disposition')):<22}"
              f"{verdict['classification']}")

    report["cases"] = adjudicated

    # -- stage 5: the verdict ------------------------------------------------
    counts: Dict[str, int] = {}
    for case in adjudicated:
        counts[case["classification"]] = counts.get(case["classification"], 0) + 1
    foreign_clients = sorted({
        str(((c.get("raw_evidence") or {}).get("request") or {}).get("client_id"))
        for c in adjudicated if c.get("raw_evidence")} - {client_id})
    models = sorted({c.get("model_id") for c in adjudicated if c.get("model_id")})

    report["classifications"] = counts
    report["models_returned"] = models
    report["model_substitutions"] = sum(
        1 for c in adjudicated
        if c.get("model_id") and REQUIRED_MODEL not in str(c["model_id"]))
    report["records_from_a_client_other_than_the_canary"] = foreign_clients
    report["served_http_statuses"] = sorted(
        {c["legacy_http_status"] for c in adjudicated})
    report["async_evidence_lost"] = counts.get(ASYNC_EVIDENCE_LOST, 0)

    if not any(c.get("raw_evidence") for c in adjudicated):
        report["verdict"] = INCONCLUSIVE
        report["shadow_not_active"] = (
            "no evidence record arrived for ANY case. The three causes, none of "
            "them a semantic result: MI_AGENT_PLAN_SHADOW is not 'shadow'; the "
            "canary client is not in MI_AGENT_PLAN_SHADOW_CLIENTS; or the sink "
            "path configured on the app is not the one read here")
        counts[SHADOW_NOT_ACTIVE] = len(adjudicated)
    elif any(c["classification"] in DEFECTS for c in adjudicated):
        report["verdict"] = FAIL
    elif report["model_substitutions"] or foreign_clients:
        report["verdict"] = FAIL
    elif report["async_evidence_lost"]:
        report["verdict"] = INCONCLUSIVE
    else:
        report["verdict"] = PASS

    _save(report, args.json_out, secrets)
    print(f"\nclassifications    {counts}")
    print(f"models returned    {models or 'NONE'}")
    print(f"verdict            {report['verdict']}")
    print("::notice::this harness neither enabled nor disabled the shadow. Set "
          "MI_AGENT_PLAN_SHADOW back to off now — it is an app setting and needs "
          "ARM access this workflow does not hold.")
    return 0 if report["verdict"] == PASS else (1 if report["verdict"] == FAIL
                                               else 2)


def _save(report: Dict[str, Any], path: str, secrets: List[str]) -> None:
    clean = scrub(report, secrets)
    body = json.dumps(clean, indent=2, default=str) + "\n"
    for secret in secrets:
        if secret and secret in body:                                # pragma: no cover
            raise SystemExit("::error::a credential reached the evidence; "
                             "refusing to write it")
    Path(path).write_text(body)
    print(f"written            {path}")


if __name__ == "__main__":
    raise SystemExit(main())
