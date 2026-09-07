#!/usr/bin/env python3
"""Live production certification for the MI Query Agent API.

WHAT THIS CERTIFIES, and it is deliberately not "the tests pass". The test
estate runs against the code in a checkout. This runs against a SERVING
INSTANCE over HTTP — the deployed artefact, its own dependency contract, its
own startup command, its own data — and asks the three questions that decide
whether an operator can trust an answer:

    1. does it answer the questions it should answer, with the right figures?
    2. does it REFUSE the questions it cannot answer, rather than widening?
    3. does it never return a confident figure for a population it dropped?

Every expectation here is either an identity the answer must satisfy, or a
figure computed from the response itself — never a number copied from a
previous run, because a certification that asserts last week's totals fails on
the day the book changes and tells you nothing about correctness.

    python -m due_diligence.evidence.mi_api_certification.certify_mi_api
        --base-url https://trakt-mi-api.azurewebsites.net

With no --base-url it runs the SAME suite in-process against
`mi_agent_api.app` through Starlette's TestClient. That is production-faithful
— the same ASGI app, the same routing, the same governance envelope — and it is
what CI and a network-restricted environment can run. It is NOT a substitute
for the live run: it cannot see deployment, packaging, app settings, or the
data the live instance is actually pointed at.

EXIT CODE is the certification verdict: 0 certified, 1 not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# Transport — one interface, two implementations
# --------------------------------------------------------------------------- #
def _live_asker(base_url: str, path: str = "/mi/query",
                headers: Optional[List[str]] = None,
                portfolio_id: Optional[str] = None
                ) -> Callable[[str], Dict[str, Any]]:
    import urllib.error
    import urllib.request

    extra = {"Content-Type": "application/json"}
    # THE TOKEN COMES FROM THE ENVIRONMENT, never an argument. `MI_BEARER` is
    # the variable `migration_phase0/replay_probe.py` already uses, so the two
    # live clients share one convention — and a secret passed as an argv value
    # lands in process listings and CI logs, which is how a bearer token
    # outlives the run that needed it.
    import os as _os

    bearer = _os.environ.get("MI_BEARER", "").strip()
    if bearer:
        extra["Authorization"] = "Bearer " + bearer.removeprefix("Bearer ").strip()
    for raw in (headers or []):
        name, _, value = raw.partition(":")
        if name.strip():
            extra[name.strip()] = value.strip()

    def ask(question: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"question": question}
        if portfolio_id:
            payload["portfolioId"] = portfolio_id
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            base_url.rstrip("/") + path, data=body,
            headers=extra, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:               # noqa: PERF203
            # The SERVICE answered with a status. 401/403 here is the token.
            return {"ok": False, "answer": f"HTTP {exc.code}",
                    "__transport_error__": True, "__http_status__": exc.code}
        except Exception as exc:  # noqa: BLE001
            # Nothing answered. A proxy's own 403 arrives here as a tunnel
            # error and is a REACHABILITY fact, not a credential one — scoring
            # it as auth would send an operator to rotate a working token.
            return {"ok": False, "answer": str(exc),
                    "__transport_error__": True, "__http_status__": None}

    return ask


def _in_process_asker() -> Callable[[str], Dict[str, Any]]:
    import os
    import warnings

    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg

    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    from fastapi.testclient import TestClient

    from mi_agent_api.app import app

    client = TestClient(app)

    def ask(question: str) -> Dict[str, Any]:
        response = client.post("/mi/query",
                               json={"question": question,
                                     "portfolioId": cfg.CLIENT_ID})
        if response.status_code != 200:
            return {"ok": False, "answer": f"HTTP {response.status_code}",
                    "__transport_error__": True}
        return response.json()

    return ask


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #
def _count(envelope: Dict[str, Any]) -> Optional[int]:
    """The loan count the envelope publishes, wherever it publishes it.

    Read from the answer's own rendering as well as the KPI artefacts, because
    the two shapes an answer can take put it in different places and a
    certification that only knew one of them reported "skip" for half the suite
    — which looks like a pass and is not one.
    """
    import re as _re

    summary = envelope.get("executionSummary") or {}
    if isinstance(summary.get("population"), int):
        return summary["population"]
    recon = envelope.get("reconciliation") or {}
    if isinstance(recon.get("records_included"), int):
        return recon["records_included"]
    for artifact in (envelope.get("artifacts") or []):
        # `kpis` is the shape the KPI artefact actually publishes; `items` was
        # the only one this read at first, which is why half the suite reported
        # "skip" — and a skip reads like a pass.
        for item in (artifact.get("kpis") or []) + (artifact.get("items") or []):
            label = str(item.get("label") or "").lower()
            field = str(item.get("field") or "").lower()
            if field == "loan_count" or ("loan" in label and "count" in label):
                try:
                    return int(float(str(item.get("rawValue", item.get("value")))
                                     .replace(",", "")))
                except Exception:  # noqa: BLE001
                    pass
    match = _re.search(r"([\d,]+)\s+loans?\b", str(envelope.get("answer") or ""))
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except Exception:  # noqa: BLE001
            return None
    return None


#: Questions that MUST answer. Each carries an identity the answer has to
#: satisfy, checked against the response rather than a remembered figure.
#: Questions any governed book can answer: a measure, a count, a grouping, a
#: statistic. A semantic refusal here is a certification failure.
MUST_ANSWER = (
    "What is the total balance?",
    "How many loans are there?",
    "Total balance by region",
    "What is the average loan size?",
    "What is the total balance in Scotland?",   # geography, prepositional
    "What is the Scottish balance?",            # geography, adjectival
    "WA LTV",                                   # a governed statistic
)

#: Questions that need a FIELD not every book carries. A book without
#: `erm_product_type` cannot answer a lump-sum question however well it
#: understands one, and a certification that could not say so would be
#: asserting the fixture rather than the product.
#:
#: These are reported separately and do not decide the verdict when they refuse.
#: What they DO decide is that they must never be ANSWERED WRONGLY — an
#: unavailable field must produce a refusal, never a broader figure — which the
#: subset identities below check independently.
CONDITIONAL_ON_FIELDS = (
    ("How many Scottish lump sum loans are there?", "erm_product_type"),
    ("For joint borrowers, chart balance by LTV by age", "borrower_type"),
    ("Total balance by region for joint borrowers with LTV over 50%",
     "borrower_type"),
    ("Compare balance over time", "a reporting date"),
)

#: Questions that MUST refuse. A confident figure here is the failure mode this
#: whole programme exists to prevent: a population the estate could not bind,
#: answered over a broader one.
#: Every term here names something NO governed vocabulary claims — not the
#: registry, not the value catalogue, not the region ladder. That is the bar,
#: and it is not "this book has no such rows": a book without Scottish loans
#: must still UNDERSTAND "Scottish", and a certification that confused the two
#: would fail on one book and pass on another for no semantic reason.
#:
#: "Yorkshire" was in this list on the first run and was WRONG — the region
#: owner carries it as a governed alias for "Yorkshire and The Humber", and the
#: live book binds it to 5 loans. That is the mistake this comment exists to
#: stop being made again. "Cornish" stays: Cornwall is inside South West
#: (England) and no ITL1 value bears its name, so it has no unambiguous referent.
MUST_REFUSE = (
    "What is the platinum balance?",
    "How many platinum loans do we have?",
    "How many platinum lump sum loans are there?",
    "What is the Cornish balance?",
    "Show me the premium loans",
    "What is the distressed balance?",
    "What is the unicorn ratio by region?",
    # Unresolved MATERIAL qualifiers, in each position a restriction can stand.
    "What is the platinum balance by region?",
    "Show me the gold tier loans",
)

#: Pairs that must agree. A narrowed population computed two ways is one
#: population; if the two disagree, one of them is wrong and neither can be
#: trusted. Stated as an identity so it holds on any book.
AGREEING_PAIRS = (
    ("What is the total balance in Scotland?",
     "What is the Scottish balance?"),
    ("What is the total balance for joint borrowers?",
     "What is the total balance for joint borrowers?"),
    ("What is the total balance for lump sum loans?",
     "What is the total balance for Lump Sum products?"),
)

#: A narrowing must never grow the population it narrows.
SUBSET_PAIRS = (
    ("What is the total balance?", "What is the total balance in Scotland?"),
    ("How many loans are there?", "How many loans are there in Scotland?"),
    ("What is the total balance?",
     "What is the total balance for joint borrowers?"),
)


def run_bank(ask: Callable[[str], Dict[str, Any]], path: str
             ) -> Tuple[int, int, int, int, List[str]]:
    """Drive a frozen question bank through the live route, SCORING each answer.

    Accepts a JSON list of strings, a JSON object with a "rows" list of
    ``{"question": ...}``, or a text file of one question per line — the shapes
    the estate's banks are actually written in.

    This USED TO COUNT AND NOTHING ELSE. It reported answered / refused /
    transport-failed, and `main` failed a bank only when the transport broke.
    That is not a certification of anything: a wrong population, a lost filter,
    a wrong aggregation and an unexpected refusal all arrive as ``ok: true`` or
    as an ordinary refusal, and every one of them was counted as a pass. A large
    replay scored that way says only that the service stayed up.

    A frozen bank carries no expectations — its verdicts belong to whoever froze
    it — so what is scored here is what a response can be held to WITHOUT one:
    the coherence checks in `broad.coherence`. A response that claims a
    population larger than the book, drops a filter it parsed, or publishes a
    breakdown whose rows do not add up to the population it says it covered has
    convicted itself, and that is a bank failure.
    """
    from due_diligence.evidence.mi_api_certification import broad as _broad

    raw = Path(path).read_text(encoding="utf-8")
    try:
        questions = questions_from_log(path)
    except Exception:  # noqa: BLE001 - a plain text bank is a bank too
        questions = [line.strip() for line in raw.splitlines()]
    questions = [q for q in questions if q.strip()]

    answered = refused = broken = incoherent = 0
    lines: List[str] = []
    for question in questions:
        envelope = ask(question)
        if envelope.get("__transport_error__"):
            broken += 1
            lines.append(f"  BROKEN {question[:70]}  {envelope.get('answer')}")
            continue
        evidence = _broad.read_evidence(question, envelope)
        problems = _broad.coherence(evidence)
        if problems:
            incoherent += 1
            lines.append(f"  FAIL   {question[:70]}  {'; '.join(problems)[:110]}")
        if evidence.ok:
            answered += 1
        else:
            refused += 1
    return answered, refused, broken, incoherent, lines


#: The estate's own wording for "this book does not report that field". A
#: MUST-ANSWER question refused for THIS reason is a data-coverage limit, not a
#: comprehension failure, and a certification that could not tell the two apart
#: would fail on every book that happens to lack a column — which is how a
#: harness starts asserting the fixture instead of the product.
_DATA_COVERAGE_MARKERS = (
    "not available in this dataset",
    "unavailable in this dataset",
    "does not report it",
    "field is unavailable",
    "no reporting periods are available",
)


def _refused_for_data_coverage(envelope: Dict[str, Any]) -> bool:
    answer = str(envelope.get("answer") or envelope.get("error") or "").lower()
    return any(marker in answer for marker in _DATA_COVERAGE_MARKERS)


#: HTTP statuses that mean the TOKEN failed, never the model. Kept as its own
#: class because `replay_probe` learned the same lesson the hard way: an auth
#: failure scored as a wrong answer reports a regression in a question the
#: service still answers.
_AUTH_STATUSES = (401, 403)


def preflight(base_url: str, path: str, headers: List[str],
              portfolio_id: Optional[str]) -> Tuple[str, str, Optional[str]]:
    """``(reached, authorised, observed_commit)`` before any question is asked.

    The three are reported separately because they fail for different reasons
    and a run that conflates them tells an operator nothing: an unreachable
    endpoint is a network or DNS fact, a 401 is a credential fact, and a
    refused question is a semantic fact.
    """
    import urllib.error
    import urllib.request

    ask = _live_asker(base_url, path, headers, portfolio_id)
    envelope = ask("What is the total balance?")
    if envelope.get("__transport_error__"):
        detail = str(envelope.get("answer") or "")
        status = envelope.get("__http_status__")
        if status in _AUTH_STATUSES:
            # Reached: the service replied, and refused the credential.
            return "YES", f"NO — HTTP {status}", None
        if status is not None:
            return "YES", f"NO — service returned HTTP {status}", None
        # No HTTP status at all: nothing answered.
        return f"NO — {detail}", "NOT REACHED", None

    # WHICH COMMIT IS SERVING. This used to fall back through
    # ("commit", "sha", "revision", "build", "version") and settle on
    # `version=1.0.0` — the application's hand-written version string, identical
    # across every deploy this year. It looked like provenance and established
    # nothing: it could not tell the release being certified from an older build
    # that was never replaced, or from a rollback nobody recorded. Only the
    # immutable build stamp counts now, and its ABSENCE is reported as absence
    # rather than filled in with something that reads like an answer.
    commit = None
    for candidate in ("/health", "/api/health", "/", "/api/"):
        try:
            request = urllib.request.Request(
                base_url.rstrip("/") + candidate, method="GET")
            bearer = __import__("os").environ.get("MI_BEARER", "").strip()
            if bearer:
                request.add_header(
                    "Authorization",
                    "Bearer " + bearer.removeprefix("Bearer ").strip())
            with urllib.request.urlopen(request, timeout=30) as response:
                health = json.loads(response.read().decode("utf-8") or "{}")
            build = health.get("build")
            if isinstance(build, dict) and build.get("commit"):
                commit = str(build["commit"]).strip()
                break
        except Exception:  # noqa: BLE001 - health is a nicety, the stamp is not
            continue
    return "YES", "YES", commit


def questions_from_log(path: str) -> List[str]:
    """The QUESTIONS from a replay-probe telemetry log, and nothing else.

    The frozen banks are saved `/ops/mi-queries` responses, so they carry
    production ANSWER data alongside the questions. This reads the question
    strings only — never an answer, never a figure — so a frozen bank can be
    driven through the live route without its payload being copied anywhere.
    Order is preserved and duplicates are kept: a bank's ordering is part of
    what was frozen.
    """
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = loaded.get("rows") if isinstance(loaded, dict) else loaded
    if isinstance(loaded, dict) and rows is None:
        for key in ("queries", "items", "data", "results"):
            if isinstance(loaded.get(key), list):
                rows = loaded[key]
                break
    out: List[str] = []
    for row in (rows or []):
        if isinstance(row, str):
            out.append(row)
        elif isinstance(row, dict):
            question = row.get("question") or row.get("q") or row.get("text")
            if question:
                out.append(str(question))
    return out


def certify(ask: Callable[[str], Dict[str, Any]]) -> Tuple[bool, List[str]]:
    lines: List[str] = []
    ok = True

    def record(status: str, detail: str) -> None:
        nonlocal ok
        if status == "FAIL":
            ok = False
        lines.append(f"  {status:6} {detail}")

    lines.append("MUST ANSWER (or refuse for a DATA reason, never a semantic one)")
    for question in MUST_ANSWER:
        envelope = ask(question)
        if envelope.get("__transport_error__"):
            record("FAIL", f"{question}  [transport] {envelope.get('answer')}")
        elif envelope.get("ok"):
            record("ok", question)
        elif _refused_for_data_coverage(envelope):
            record("data", f"{question}  — book does not carry the field")
        else:
            record("FAIL", f"{question}  refused semantically: "
                           f"{str(envelope.get('answer'))[:90]}")

    lines.append("CONDITIONAL ON A FIELD (refusal is a data limit, not a fault)")
    for question, needs in CONDITIONAL_ON_FIELDS:
        envelope = ask(question)
        if envelope.get("__transport_error__"):
            record("FAIL", f"{question}  [transport] {envelope.get('answer')}")
        elif envelope.get("ok"):
            record("ok", f"{question}  (book carries {needs})")
        else:
            record("data", f"{question}  refused — needs {needs}: "
                           f"{str(envelope.get('answer'))[:70]}")

    lines.append("MUST REFUSE (a figure here is a silent wrong answer)")
    for question in MUST_REFUSE:
        envelope = ask(question)
        if envelope.get("__transport_error__"):
            record("FAIL", f"{question}  [transport] {envelope.get('answer')}")
        elif envelope.get("ok"):
            record("FAIL", f"{question}  ANSWERED — population not bound")
        else:
            record("ok", f"{question}  refused")

    lines.append("AGREEING PAIRS (one population, two wordings)")
    for first, second in AGREEING_PAIRS:
        a, b = ask(first), ask(second)
        if a.get("ok") != b.get("ok"):
            record("FAIL", f"{first!r} and {second!r} disagree about whether "
                           "they can be answered")
        elif not a.get("ok"):
            record("skip", f"{first!r} / {second!r} both refuse")
        else:
            first_count, second_count = _count(a), _count(b)
            if first_count is None or second_count is None:
                record("skip", f"{first!r} / {second!r} publish no loan count")
            elif first_count != second_count:
                record("FAIL", f"{first!r}={first_count} vs "
                               f"{second!r}={second_count}")
            else:
                record("ok", f"{first!r} == {second!r} ({first_count} loans)")

    lines.append("SUBSET (a narrowing never grows the population)")
    for whole, part in SUBSET_PAIRS:
        a, b = ask(whole), ask(part)
        if not (a.get("ok") and b.get("ok")):
            record("skip", f"{part!r} not answered")
            continue
        whole_count, part_count = _count(a), _count(b)
        if whole_count is None or part_count is None:
            record("skip", f"{part!r} publishes no loan count")
        elif part_count > whole_count:
            record("FAIL", f"{part!r}={part_count} exceeds {whole!r}"
                           f"={whole_count}")
        else:
            record("ok", f"{part_count} <= {whole_count}")

    return ok, lines


#: The BROAD section's own bar, from the certification brief. It is deliberately
#: NOT "how many questions were answered": a book that cannot carry a field must
#: be free to refuse, and a suite tuned for answer rate is a suite that rewards
#: the one failure mode this programme exists to prevent.
def report_broad(ask: Callable[[str], Dict[str, Any]],
                 progress: bool = False
                 ) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Run the broad scored sweep and render it. Returns (passed, lines, facts)."""
    from due_diligence.evidence.mi_api_certification import broad as _broad

    cases = _broad.load_cases()
    session, results, holdout = _broad.run_broad(ask, cases, progress=progress)

    lines: List[str] = ["", "=" * 74,
                        "BROAD SCORED CERTIFICATION", "=" * 74]

    by_class: Dict[str, Dict[str, int]] = {}
    for result in results + holdout:
        bucket = by_class.setdefault(result.cls, {})
        bucket[result.status] = bucket.get(result.status, 0) + 1

    def emit(title: str, rows: List[Any]) -> None:
        lines.append(title)
        for result in rows:
            marker = {"ok": "ok", "FAIL": "FAIL", "data": "data",
                      "n/e": "NOT-EST", "skip": "skip"}[result.status]
            lines.append(f"  {marker:7} {result.case_id}  {result.detail}")

    section_titles = {
        "singles": "COVERAGE MATRIX (A basic · B one filter · C several filters · "
                   "D 1-D grouping · E 2-D grouping · F filtered grouping · "
                   "I temporal/specialist · J adversarial)",
        "outputs": "SAME-TURN MULTI-OUTPUT (every requested output must survive)",
        "output_local": "OUTPUT-LOCAL NARROWING (a narrowing must not cost the breakdown)",
        "equivalence_groups": "SEMANTIC EQUIVALENCE (the relation is scored, not ok:true)",
        "subset_pairs": "SUBSET (a narrowing never grows the population)",
        "algebra": "FILTER ALGEBRA / METAMORPHIC PROPERTIES",
        "reconciliations": "GROUPING RECONCILIATION (parts sum to the whole, this run only)",
        "arithmetic": "NUMERIC IDENTITIES (computed from the responses themselves)",
    }
    index = 0
    for key, _scorer in _broad._SECTIONS:
        count = len(cases.get(key, []))
        emit(section_titles[key], results[index:index + count])
        index += count
    emit("HOLDOUT (written after the suite was fixed; run exactly once)", holdout)

    # A SECOND HOLDOUT, WRITTEN AFTER THE SEMANTIC REPAIR. The first one was
    # written before it and three of its questions are what found the defect, so
    # it can no longer be blind to the change it caused. This one uses head nouns
    # and modifiers that appear nowhere in the repair, the property tests, the
    # six production failures or the first holdout — and half of it is
    # LEGITIMATE language, because a repair that refuses everything would pass a
    # holdout made only of nonsense.
    fresh = [_broad.score_single(session, case)
             for case in cases.get("fresh_holdout", [])]
    for result in fresh:
        bucket = by_class.setdefault(result.cls, {})
        bucket[result.status] = bucket.get(result.status, 0) + 1
    emit("FRESH HOLDOUT (written after the semantic repair; run exactly once)",
         fresh)
    holdout = holdout + fresh

    # ---- the bar ---------------------------------------------------------- #
    silent_wrong = [r for r in results + holdout if r.silent_wrong]
    equivalence_failures = [r for r in results
                            if r.status == "FAIL" and r.cls == "F_equivalence"]
    algebra_failures = [r for r in results if r.status == "FAIL"
                        and r.cls in ("G_subset", "H_filter_algebra")]
    lost_outputs = [r for r in results if r.status == "FAIL"
                    and r.cls in ("G_multi_output", "H_output_local")]
    holdout_wrong = [r for r in holdout if r.silent_wrong]
    server_errors = [s for s in session.statuses if 500 <= s < 600]
    client_errors = [s for s in session.statuses if 400 <= s < 500]

    timings = _broad.latency(session)
    lines += ["", "LIVE PERFORMANCE (measured, not tuned)",
              f"  distinct requests   : {int(timings.get('requests', 0))}",
              f"  wall clock (s)      : {timings.get('total_s', 0.0):.1f}",
              f"  p50 / p95 / max (s) : {timings.get('p50_s', 0.0):.2f} / "
              f"{timings.get('p95_s', 0.0):.2f} / {timings.get('max_s', 0.0):.2f}",
              f"  transport failures  : {len(session.transport_failures)}",
              f"  4xx / 5xx           : {len(client_errors)} / {len(server_errors)}"]

    lines += ["", "COUNTS BY CAPABILITY CLASS"]
    for cls in sorted(by_class):
        bucket = by_class[cls]
        lines.append(f"  {cls:22} ok {bucket.get('ok', 0):3d} · data "
                     f"{bucket.get('data', 0):3d} · NOT-EST "
                     f"{bucket.get('n/e', 0):3d} · FAIL {bucket.get('FAIL', 0):3d}")

    # §7 — WHY EACH REFUSAL HAPPENED, counted. A report that says "data" for
    # everything it did not get an answer to sends an operator to load columns
    # that are already loaded. These four are different conversations.
    classified: Dict[str, int] = {}
    for evidence in session._seen.values():          # noqa: SLF001 - same package
        if evidence.transport_error:
            continue
        reason = _broad.refusal_class(evidence.envelope)
        classified[reason or "ANSWERED"] = classified.get(reason or "ANSWERED", 0) + 1
    lines += ["", "REFUSAL CLASSIFICATION (of the distinct live requests)"]
    for name in ("ANSWERED", _broad.DATA_UNAVAILABLE, _broad.CAPABILITY_UNAVAILABLE,
                 _broad.SEMANTIC_UNRESOLVED, _broad.GOVERNED_REFUSAL):
        lines.append(f"  {name:24} {classified.get(name, 0)}")

    lines += ["", "COUNTS BY SAFETY CLASS",
              f"  silent wrong answers            : {len(silent_wrong)}",
              f"  unexplained equivalence failures: {len(equivalence_failures)}",
              f"  unexplained subset/algebra fails: {len(algebra_failures)}",
              f"  silently lost requested outputs : {len(lost_outputs)}",
              f"  holdout silent wrong answers    : {len(holdout_wrong)}",
              f"  transport failures              : "
              f"{len(session.transport_failures)}",
              f"  5xx responses                   : {len(server_errors)}"]

    lines += ["", "NOT ESTABLISHED — checks the response does not expose enough to decide",
              "  metadata.executionReceipt is absent from the live envelope, so no",
              "  claim here rests on a per-row execution receipt. Population, group",
              "  cells, applied/dropped filters and applied/dropped dimensions ARE",
              "  published, and every check above rests on those."]
    not_established = [r for r in results + holdout if r.status == "n/e"]
    for result in not_established:
        lines.append(f"  {result.case_id}  {result.detail}")

    passed = not (silent_wrong or equivalence_failures or algebra_failures
                  or lost_outputs or session.transport_failures or server_errors)
    facts = {"silent_wrong": len(silent_wrong),
             "equivalence_failures": len(equivalence_failures),
             "algebra_failures": len(algebra_failures),
             "lost_outputs": len(lost_outputs),
             "holdout_wrong": len(holdout_wrong),
             "not_established": len(not_established),
             "transport": len(session.transport_failures),
             "http_5xx": len(server_errors), "http_4xx": len(client_errors),
             "timings": timings, "by_class": by_class,
             "classified": classified,
             "cases": len(results) + len(holdout)}
    return passed, lines, facts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None,
                        help="serving instance, e.g. https://app.traktinfra.io/api "
                             "(the PUBLIC route a real client uses, which also "
                             "certifies routing, auth, and serialization). "
                             "Omitted: run in-process against mi_agent_api.app.")
    parser.add_argument("--path", default="/mi/query",
                        help="endpoint path appended to --base-url "
                             "(default /mi/query).")
    parser.add_argument("--header", action="append", default=[],
                        metavar="NAME:VALUE",
                        help="extra request header, repeatable — e.g. an "
                             "Authorization bearer token for the auth boundary.")
    parser.add_argument("--portfolio-id", default=None,
                        help="portfolioId sent with each question, where the "
                             "deployment selects a book that way.")
    parser.add_argument("--expect-commit", default=None, metavar="SHA",
                        help="the Git commit this deployment is supposed to be "
                             "serving. The run FAILS if the deployed commit "
                             "cannot be established, or differs from this. "
                             "Provenance that cannot tell two builds apart is "
                             "not provenance, and an application version string "
                             "cannot tell two builds apart.")
    parser.add_argument("--report", default=None, metavar="FILE",
                        help="write the report here as well as to stdout, so a "
                             "CI run can keep it as an artifact.")
    parser.add_argument("--broad", dest="broad", action="store_true",
                        default=True,
                        help="run the BROAD scored sweep from broad_cases.json "
                             "in addition to the core suite (default).")
    parser.add_argument("--no-broad", dest="broad", action="store_false",
                        help="core suite only — the release smoke gate on its own.")
    parser.add_argument("--bank", action="append", default=[], metavar="FILE",
                        help="frozen question bank to drive through the same "
                             "live route, repeatable. JSON list, {rows:[...]}, "
                             "or one question per line.")
    args = parser.parse_args(argv)

    if args.base_url:
        target = args.base_url.rstrip("/") + args.path
        ask = _live_asker(args.base_url, args.path, args.header,
                          args.portfolio_id)
    else:
        target = "in-process (mi_agent_api.app via TestClient)"
        ask = _in_process_asker()

    header = ["=" * 74, "MI Query Agent — production certification",
              f"target: {target}", "=" * 74]
    provenance = "not checked (in-process run)"
    if args.base_url:
        reached, authorised, commit = preflight(
            args.base_url, args.path, args.header, args.portfolio_id)
        header.append(f"endpoint reached : {reached}")
        header.append(f"auth passed      : {authorised}")
        header.append(f"deployed commit  : {commit or 'NOT ESTABLISHED'}")
        expected = (args.expect_commit or "").strip()
        if expected:
            header.append(f"expected commit  : {expected}")
            if not commit:
                provenance = ("NO — the service publishes no build stamp, so "
                              "which commit is serving cannot be established")
            elif commit.lower().startswith(expected.lower()[:7]) or \
                    expected.lower().startswith(commit.lower()[:7]):
                provenance = "YES"
            else:
                provenance = (f"NO — serving {commit[:12]}, expected "
                              f"{expected[:12]}")
        else:
            provenance = "not checked (no --expect-commit given)"
        header.append(f"provenance       : {provenance}")
        header.append("=" * 74)
        if not (reached == "YES" and authorised == "YES"):
            print("\n".join(header))
            print("VERDICT: NOT EXECUTABLE — the service was not reached and "
                  "authorised, so no question was put to it.")
            if args.report:
                Path(args.report).write_text("\n".join(header), encoding="utf-8")
            return 2
    print("\n".join(header))
    lines = ["CORE SUITE — the release smoke gate", "-" * 74]
    core_ok, core_lines = certify(ask)
    lines.extend(core_lines)
    for path in args.bank:
        answered, refused, broken, incoherent, bank_lines = run_bank(ask, path)
        lines.append(f"FROZEN BANK {path}")
        lines.append(f"  answered {answered} · refused {refused} · "
                     f"transport-failed {broken} · INCOHERENT {incoherent}")
        lines.extend(bank_lines)
        if broken or incoherent:
            core_ok = False

    broad_ok: Optional[bool] = None
    if args.broad:
        broad_ok, broad_lines, facts = report_broad(
            ask, progress=bool(args.base_url))
        lines.extend(broad_lines)

    body = "\n".join(lines)
    print(body)
    print("-" * 74)
    # The two verdicts are reported SEPARATELY because they answer different
    # questions. The core suite says the release is safe to ship; the broad
    # sweep says the semantics hold across the surface an operator actually
    # uses. A run that passed one and failed the other must not be able to
    # report a single word.
    # PROVENANCE IS PART OF THE VERDICT, not a note beside it. A green
    # certification against a build nobody can identify certifies nothing an
    # operator can act on: "it passed" is only useful if "it" is a commit.
    if args.base_url and (args.expect_commit or "").strip():
        if not str(provenance).startswith("YES"):
            core_ok = False
            lines.append("")
            lines.append(f"  FAIL   deployed-build provenance: {provenance}")
    core_verdict = "PASS" if core_ok else "FAIL"
    print(f"CORE VERDICT : {core_verdict}")
    if broad_ok is not None:
        print(f"BROAD VERDICT: {'BROAD LIVE GO' if broad_ok else 'BROAD NO-GO'}")
    certified = core_ok and (broad_ok is not False)
    verdict = "CERTIFIED" if certified else "NOT CERTIFIED"
    print("VERDICT:", verdict)
    if args.report:
        trailer = [f"CORE VERDICT : {core_verdict}"]
        if broad_ok is not None:
            trailer.append(
                f"BROAD VERDICT: {'BROAD LIVE GO' if broad_ok else 'BROAD NO-GO'}")
        trailer.append(f"VERDICT: {verdict}")
        Path(args.report).write_text(
            "\n".join(header) + "\n" + body + "\n\n" + "\n".join(trailer) + "\n",
            encoding="utf-8")
    if not args.base_url:
        print("NOTE: in-process. This exercises the same ASGI app and the same")
        print("      governance envelope, but it does NOT certify deployment,")
        print("      packaging, app settings, or the data the live instance is")
        print("      pointed at. Re-run with --base-url against the serving")
        print("      instance for a live certification.")
    return 0 if certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
