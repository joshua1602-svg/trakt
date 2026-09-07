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


def _in_process_asker(portfolio_id: Optional[str] = None
                      ) -> Callable[[str], Dict[str, Any]]:
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

    # THE CALLER'S PORTFOLIO, when it named one. This used to send the demo
    # client id unconditionally, which meant `--portfolio-id` was silently
    # ignored in process — and a gate that resolves a DIFFERENT book from the
    # one it was asked about still prints a verdict, which is worse than not
    # running at all.
    selected = portfolio_id or cfg.CLIENT_ID

    def ask(question: str) -> Dict[str, Any]:
        response = client.post("/mi/query",
                               json={"question": question,
                                     "portfolioId": selected})
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


# --------------------------------------------------------------------------- #
# GEOGRAPHY / CONFIG-OWNERSHIP RELEASE ACCEPTANCE
# --------------------------------------------------------------------------- #
#
# WHAT THIS GATE IS FOR. The estate now says, in configuration rather than in
# code, which geography a book reports on:
#
#     ERE is the CLIENT       config/client/config_client_ERE.yaml
#       declares              portfolio.asset_class: equity_release
#     ERM is the ASSET        config/asset/product_defaults_ERM.yaml
#       declares              mi_geography.primary_basis: collateral
#
# so a request for `ERE/2026-06-30` should resolve collateral from the asset
# class default, with no portfolio registry, no environment variable and no
# hand-seeded source_portfolio_id. Every part of that is testable in a checkout
# and none of it proves the DEPLOYED service does it. This does, over the public
# route, against a named commit.
#
# WHY IT IS SMALL. The broad sweep already scores ~150 questions across the
# capability surface and takes the time that implies. This gate asks one thing —
# is the geography configuration the one we shipped, and is it being applied —
# so it asks ten questions and scores them on identities and provenance.
#
# WHAT IT REFUSES TO ACCEPT. HTTP 200. A question that comes back `ok` having
# quietly measured the other geography has failed at precisely the thing this
# architecture exists to prevent, and a gate that read the status line would
# call it a pass.

#: The ten questions, in the order the report prints them.
GEOGRAPHY_QUESTIONS: Tuple[Tuple[str, str], ...] = (
    ("G01", "What is the total balance?"),
    ("G02", "Total balance by region"),
    ("G03", "What is the Scottish balance?"),
    ("G04", "How many loans are there in Scotland?"),
    ("G05", "Loan count by region"),
    ("G06", "Total balance by property region"),
    ("G07", "Total balance by borrower region"),
    ("G08", "Total balance by region for joint borrowers with LTV over 50%"),
    ("G09", "Show total balance and loan count by region"),
    ("G10", "What is the platinum balance by region?"),
)

#: Questions whose geography is UNQUALIFIED: they must resolve the configured
#: basis, and the source must be the asset class default — not the registry, not
#: a client override, and emphatically not `unconfigured`.
_GENERIC_GEOGRAPHY = ("G02", "G05", "G09")

#: The scorer's checks, in report order. Every one must pass for the release to
#: be accepted; they are named separately because they fail for different
#: reasons and an operator needs to know which.
ACCEPTANCE_CHECKS: Tuple[str, ...] = (
    "DEPLOYED_PROVENANCE_PASS",
    "AUTH_PASS",
    "PORTFOLIO_PASS",
    "ASSET_CLASS_PASS",
    "CONFIGURED_BASIS_PASS",
    "GENERIC_REGION_PASS",
    "EXPLICIT_PROPERTY_PASS",
    "EXPLICIT_BORROWER_PASS",
    "NUMERICAL_RECONCILIATION_PASS",
    "SILENT_WRONG_SAFETY_PASS",
)

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "geography_snapshot.json"


def load_snapshot(portfolio_id: Optional[str],
                  path: Optional[str] = None) -> Dict[str, Any]:
    """The frozen truth for ``portfolio_id``, or ``{}`` when none is recorded.

    An absent entry is not a failure: the identities still hold, and a book this
    harness has no snapshot for is gated on them alone rather than on figures
    that were never about it.
    """
    try:
        doc = json.loads(Path(path or _SNAPSHOT_PATH).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing snapshot degrades, never raises
        return {}
    entry = ((doc.get("portfolios") or {}).get(portfolio_id or ""))
    return entry if isinstance(entry, dict) else {}


def _geography_block(envelope: Dict[str, Any]) -> Dict[str, Any]:
    block = (envelope.get("metadata") or {}).get("geographyBasis")
    return block if isinstance(block, dict) else {}


def _configured_basis(envelope: Dict[str, Any]) -> Optional[str]:
    """The basis the CONFIGURATION established, whatever the question asked.

    The envelope names it explicitly only when the question overrode it — that
    is when there are two bases to tell apart. For an unqualified question the
    configured basis IS the primary basis, so this reads the field that is
    there rather than asking the service to publish a duplicate one for the
    convenience of a smoke test.
    """
    block = _geography_block(envelope)
    if "configuredBasis" in block:
        # Present-but-None is a STATEMENT: the configuration established no
        # basis and the question supplied one. Falling through to the stated
        # basis here would report the question's own answer as though the
        # configuration had agreed with it, which is the failure this gate is
        # built to catch.
        return block["configuredBasis"]
    return block.get("primaryBasis")


#: Cache for the field->basis owner. Module-global rather than lru_cache so
#: this file keeps its dependency-free import list.
_BASIS_OF_FIELD: List[Any] = []


def basis_of_field_owner() -> Optional[Callable[[Optional[str]], Optional[str]]]:
    """``mi_agent.mi_geography.basis_of_field``, loaded BY PATH.

    WHY NOT `from mi_agent.mi_geography import basis_of_field`. That is the
    obvious line and it broke the gate in production. Importing it executes
    `mi_agent/__init__.py`, which imports the whole package — including
    `mi_query_validator`, which imports `yaml` — and THIS WORKFLOW INSTALLS NO
    DEPENDENCIES. It is a thin HTTP client by design; the core and broad suites
    never imported the engine, so nothing had ever needed them. The acceptance
    run reached production, authenticated, confirmed the deployed commit, and
    then died on `ModuleNotFoundError: No module named 'yaml'` before scoring a
    single question.

    Loading the module by file path executes THAT FILE and nothing else. Its
    module-level imports are stdlib only (`yaml` is imported lazily inside the
    functions that read configuration, and `basis_of_field` is a dict lookup
    that never reads any). So the field-to-basis mapping keeps exactly ONE
    owner — this reads the same source the engine does — without the harness
    growing a dependency contract it was deliberately built not to have.

    Returns None if the owner cannot be loaded. The caller must treat that as a
    FAILURE, never as "no region field": a gate that cannot tell which basis was
    measured has not checked the thing it exists to check.
    """
    if not _BASIS_OF_FIELD:
        import importlib.util

        source = _REPO_ROOT / "mi_agent" / "mi_geography.py"
        try:
            name = "_trakt_mi_geography_for_acceptance"
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            # REGISTERED BEFORE EXECUTION, and it is not optional. The module
            # declares dataclasses, and `dataclasses._is_type` resolves a
            # field's type through `sys.modules.get(cls.__module__)` — which is
            # None for a module loaded by path and never registered, so the
            # decorator raises `'NoneType' object has no attribute '__dict__'`
            # at import time. Found by running this loader in a subprocess with
            # `yaml` blocked, which is the only way to see what CI sees.
            sys.modules[name] = module
            spec.loader.exec_module(module)          # type: ignore[union-attr]
            _BASIS_OF_FIELD.append(module.basis_of_field)
        except Exception as exc:                     # noqa: BLE001
            print(f"  the field-to-basis owner could not be loaded: {exc}")
            _BASIS_OF_FIELD.append(None)
    return _BASIS_OF_FIELD[0]


#: Cache for the governed region owner, on the same by-path contract.
_REGION_RESOLVE: List[Any] = []


def region_resolver_owner() -> Optional[Callable[..., Any]]:
    """``mi_agent.region_resolution.resolve``, loaded BY PATH.

    WHY THE HARNESS ASKS RATHER THAN MAPS. "Scottish" covers whichever governed
    labels the region ladder says it covers, and that set is a property of the
    estate's taxonomy, not of this gate. A mapping written here would be a
    second taxonomy — able to disagree with the one production actually used,
    and to disagree silently, which is how a certification ends up asserting its
    own fixture.

    Same by-path load as the field-to-basis owner, and for the same reason: this
    module's imports are stdlib only, and the certification workflow installs no
    dependencies. Returns None if it cannot be loaded; the caller must then
    report the question as UNADJUDICATED rather than pass it.
    """
    if not _REGION_RESOLVE:
        import importlib.util

        source = _REPO_ROOT / "mi_agent" / "region_resolution.py"
        try:
            name = "_trakt_region_resolution_for_acceptance"
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)      # type: ignore[union-attr]
            _REGION_RESOLVE.append(module.resolve)
        except Exception as exc:                 # noqa: BLE001
            print(f"  the governed region owner could not be loaded: {exc}")
            _REGION_RESOLVE.append(None)
    return _REGION_RESOLVE[0]


def _group_table(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every returned group, with its label and both measures.

    THE EVIDENCE THAT ADJUDICATES A DISAGREEMENT. A filtered answer and a
    grouped cell that disagree have two possible explanations — one governed
    taxonomy spread over several labels, or two different geography semantics —
    and they are indistinguishable from a total. Printing the labels makes the
    log itself sufficient, which matters when the artefact store is unreachable.
    """
    rows = _acceptance_rows(envelope)
    key = _measured_region_field(envelope)
    if not rows or not key:
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if key not in row:
            continue
        entry: Dict[str, Any] = {"label": str(row[key])}
        for name, candidates in (
                ("balance", ("current_outstanding_balance_sum", "balance")),
                ("count", ("loan_count", "count"))):
            for candidate in candidates:
                if candidate in row:
                    try:
                        entry[name] = float(row[candidate])
                    except (TypeError, ValueError):
                        pass
                    break
        out.append(entry)
    return out


def _resolved_filters(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """The filter the service actually applied, as it reports it."""
    recon = envelope.get("reconciliation") or {}
    if isinstance(recon.get("filters"), dict) and recon["filters"]:
        return recon["filters"]
    spec = envelope.get("spec") or {}
    return spec.get("filters") or {}


def _measured_region_field(envelope: Dict[str, Any]) -> Optional[str]:
    """The region column the answer was actually MEASURED on.

    This is the check that catches a silent substitution, and it deliberately
    does not trust the metadata: `geographyBasis` says which basis was
    RESOLVED, and an answer that resolved `borrower` and then grouped by a
    collateral column would publish exactly the metadata we want to see while
    doing the thing we are trying to forbid. The spec's dimension, and failing
    that the column the rows are actually keyed by, is what was measured.
    """
    basis_of_field = basis_of_field_owner()
    if basis_of_field is None:
        return None

    spec = envelope.get("spec") or {}
    candidates: List[Any] = [spec.get("dimension")]
    candidates.extend(spec.get("dimensions") or [])
    candidates.extend((spec.get("filters") or {}).keys())
    for row in _acceptance_rows(envelope)[:1]:
        candidates.extend(row.keys())
    for candidate in candidates:
        name = str(candidate or "")
        if name and basis_of_field(name):
            return name
    return None


def _measured_basis(envelope: Dict[str, Any]) -> Optional[str]:
    basis_of_field = basis_of_field_owner()
    field = _measured_region_field(envelope)
    return basis_of_field(field) if (basis_of_field and field) else None


def _acceptance_rows(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The grouped rows, from whichever artefact carries them."""
    best: List[Dict[str, Any]] = []
    for artifact in (envelope.get("artifacts") or []):
        rows = artifact.get("rows") or []
        if rows and len(rows) > len(best):
            best = [r for r in rows if isinstance(r, dict)]
    return best


def _cells(envelope: Dict[str, Any], *value_keys: str
           ) -> Dict[str, float]:
    """``{region: value}`` from a grouped answer, or ``{}``.

    The value column is found by name from the candidates given, so a rename in
    one answer shape does not silently score as "no cells", which would read as
    a pass.
    """
    rows = _acceptance_rows(envelope)
    key = _measured_region_field(envelope)
    if not rows or not key:
        return {}
    out: Dict[str, float] = {}
    for row in rows:
        if key not in row:
            continue
        for value_key in value_keys:
            if value_key in row:
                try:
                    out[str(row[key])] = float(row[value_key])
                except (TypeError, ValueError):
                    pass
                break
    return out


def _balance(envelope: Dict[str, Any]) -> Optional[float]:
    """The balance figure for the population this answer measured."""
    recon = envelope.get("reconciliation") or {}
    for key in ("balance_after_filters", "total_balance"):
        value = recon.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for artifact in (envelope.get("artifacts") or []):
        for item in (artifact.get("kpis") or []):
            if str(item.get("field") or "").startswith("current_outstanding_balance"):
                try:
                    return float(item.get("rawValue"))
                except (TypeError, ValueError):
                    pass
    return None


def _close(left: Optional[float], right: Optional[float],
           tolerance: float = 0.02) -> bool:
    """Money compared as money: a penny of float noise is not a discrepancy,
    and a relative term keeps the test meaningful at £1.9bn."""
    if left is None or right is None:
        return False
    return abs(left - right) <= max(tolerance, abs(right) * 1e-9)


def _answered(envelope: Dict[str, Any]) -> bool:
    return bool(envelope.get("ok"))


def _refused_for_missing_field(envelope: Dict[str, Any]) -> bool:
    """A refusal that NAMES the facet it could not apply.

    A book that does not carry a borrower-type field cannot answer a joint
    borrower question however well it understands one. Saying so is correct
    behaviour; the failure mode is answering anyway over a broader population,
    which `_facets_preserved` checks independently.
    """
    if _answered(envelope):
        return False
    if (envelope.get("spec") or {}).get("unavailable_filters"):
        return True
    return _refused_for_data_coverage(envelope)


def geography_acceptance(ask: Callable[[str], Dict[str, Any]], *,
                         portfolio_id: Optional[str] = None,
                         provenance: str = "not checked",
                         reached: str = "YES", authorised: str = "YES",
                         snapshot: Optional[Dict[str, Any]] = None,
                         snapshot_path: Optional[str] = None,
                         ) -> Tuple[Dict[str, Optional[bool]],
                                    List[Dict[str, Any]], List[str]]:
    """Score the geography release acceptance.

    Returns ``(checks, rows, lines)`` — the named verdicts, a machine-readable
    row per question, and the human report. Never raises: a check that cannot be
    established is ``False`` with a stated reason, because "we could not tell"
    must not be able to read as "it passed".
    """
    # Established BEFORE any question is asked. Without the field-to-basis owner
    # the scorer cannot tell which geography an answer was measured on, which is
    # the one thing this gate exists to check — so it fails, loudly, with the
    # reason stated, rather than reporting `measuredBasis: null` ten times and
    # passing everything else.
    owner_available = basis_of_field_owner() is not None

    snap = snapshot if snapshot is not None else load_snapshot(portfolio_id,
                                                               snapshot_path)
    want_class = str(snap.get("expectedAssetClass") or "equity_release")
    want_basis = str(snap.get("expectedPrimaryBasis") or "collateral")

    envelopes: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    lines: List[str] = ["GEOGRAPHY RELEASE ACCEPTANCE", "-" * 74,
                        f"portfolio        : {portfolio_id or '(none sent)'}",
                        f"expected asset   : {want_class}",
                        f"expected basis   : {want_basis}", ""]

    for qid, question in GEOGRAPHY_QUESTIONS:
        envelope = ask(question)
        envelopes[qid] = envelope
        block = _geography_block(envelope)
        rows.append({
            "id": qid, "question": question,
            "answered": _answered(envelope),
            "transportError": bool(envelope.get("__transport_error__")),
            "assetClass": block.get("assetClass"),
            "configuredBasis": _configured_basis(envelope),
            "effectiveBasis": block.get("primaryBasis"),
            "basisSource": block.get("basisSource"),
            "measuredField": _measured_region_field(envelope),
            "measuredBasis": _measured_basis(envelope),
            # PUBLISHED BY THE SERVICE ALL ALONG, and not recorded until a live
            # run refused an explicit basis that the generic question had just
            # been answered on. `supportedBases` is what the guard consults, so
            # it is the field that says whether a refusal was the availability
            # check or something else.
            "supportedBases": block.get("supportedBases"),
            "groups": _group_table(envelope),
            "filters": _resolved_filters(envelope),
            "records": (envelope.get("executionSummary") or {}).get("population"),
            "balance": _balance(envelope),
            "answer": str(envelope.get("answer") or envelope.get("error") or "")[:160],
        })

    checks: Dict[str, Optional[bool]] = {}
    why: Dict[str, str] = {}

    def _record(name: str, ok: bool, reason: str = "") -> None:
        checks[name] = ok
        if reason:
            why[name] = reason

    # ---- provenance and access, decided before any semantics ------------- #
    _record("DEPLOYED_PROVENANCE_PASS", str(provenance).startswith("YES")
            or provenance == "not checked (in-process run)",
            f"provenance: {provenance}")
    _record("AUTH_PASS", reached == "YES" and authorised == "YES",
            f"reached {reached}, authorised {authorised}")

    # ---- the request reached the book it named --------------------------- #
    if portfolio_id:
        named = []
        for qid, envelope in envelopes.items():
            meta = envelope.get("metadata") or {}
            seen = str(meta.get("portfolioId") or meta.get("selectedPortfolio")
                       or meta.get("selectedClient") or "")
            if seen:
                named.append(portfolio_id.split("/")[0].lower() in seen.lower()
                             or seen.lower() in portfolio_id.lower())
        _record("PORTFOLIO_PASS", bool(named) and all(named),
                "no answer named the requested portfolio" if not named
                else "")
    else:
        _record("PORTFOLIO_PASS", False, "no portfolio was requested")

    # ---- the configuration this release ships ---------------------------- #
    classes = {r["assetClass"] for r in rows if r["assetClass"] is not None}
    _record("ASSET_CLASS_PASS", classes == {want_class},
            f"asset classes seen: {sorted(classes) or 'none'}")
    configured = {r["configuredBasis"] for r in rows
                  if r["configuredBasis"] is not None}
    _record("CONFIGURED_BASIS_PASS", configured == {want_basis},
            f"configured bases seen: {sorted(configured) or 'none'}")

    # ---- unqualified "region" means the configured basis ----------------- #
    generic_problems = []
    for qid in _GENERIC_GEOGRAPHY:
        row = next(r for r in rows if r["id"] == qid)
        if not row["answered"]:
            generic_problems.append(f"{qid} did not answer")
            continue
        if row["basisSource"] != "asset_class_default":
            generic_problems.append(
                f"{qid} basisSource={row['basisSource']!r}, expected "
                "asset_class_default")
        if row["effectiveBasis"] != want_basis:
            generic_problems.append(
                f"{qid} resolved {row['effectiveBasis']!r}, expected {want_basis!r}")
        if row["measuredBasis"] not in (None, want_basis):
            generic_problems.append(
                f"{qid} MEASURED {row['measuredBasis']!r} on "
                f"{row['measuredField']!r}")
    if not owner_available:
        generic_problems.insert(0, "the field-to-basis owner "
                                   "(mi_agent/mi_geography.py) could not be "
                                   "loaded, so what was MEASURED is unknown")
    _record("GENERIC_REGION_PASS", not generic_problems,
            "; ".join(generic_problems))

    # ---- an explicitly stated basis is honoured, both ways --------------- #
    prop = next(r for r in rows if r["id"] == "G06")
    prop_problems = []
    if not prop["answered"]:
        prop_problems.append("G06 did not answer")
    if prop["basisSource"] != "explicit_query":
        prop_problems.append(f"G06 basisSource={prop['basisSource']!r}")
    if prop["configuredBasis"] != want_basis:
        prop_problems.append(f"G06 configured={prop['configuredBasis']!r}")
    if prop["effectiveBasis"] != "collateral":
        prop_problems.append(f"G06 effective={prop['effectiveBasis']!r}")
    if prop["answered"] and prop["measuredBasis"] != "collateral":
        prop_problems.append(f"G06 measured={prop['measuredBasis']!r}")
    _record("EXPLICIT_PROPERTY_PASS", not prop_problems, "; ".join(prop_problems))

    # THE ONE THAT MATTERS MOST. A borrower-region question answered on the
    # collateral column, because collateral is what the asset configures, is a
    # confident answer to a question nobody asked. A refusal is acceptable —
    # the book may not carry an obligor geography — and a substitution is not.
    borrower = next(r for r in rows if r["id"] == "G07")
    borrower_problems = []
    if borrower["configuredBasis"] != want_basis:
        borrower_problems.append(f"configured={borrower['configuredBasis']!r}")
    if borrower["answered"]:
        if borrower["basisSource"] != "explicit_query":
            borrower_problems.append(f"basisSource={borrower['basisSource']!r}")
        if borrower["effectiveBasis"] != "borrower":
            borrower_problems.append(f"effective={borrower['effectiveBasis']!r}")
        if borrower["measuredBasis"] != "borrower":
            borrower_problems.append(
                f"SILENT SUBSTITUTION: measured {borrower['measuredBasis']!r} "
                f"on {borrower['measuredField']!r}")
    elif not _refused_for_missing_field(envelopes["G07"]):
        # A refusal is fine, but it has to be a governed one that says why.
        borrower_problems.append("refused without naming what it could not apply")
    _record("EXPLICIT_BORROWER_PASS", not borrower_problems,
            "; ".join(borrower_problems))

    # ---- identities, and then the snapshot ------------------------------- #
    recon_problems: List[str] = []
    total = _balance(envelopes["G01"])
    by_region = _cells(envelopes["G02"], "current_outstanding_balance_sum",
                       "balance", "value")
    counts_by_region = _cells(envelopes["G05"], "loan_count", "count", "value")

    if not by_region:
        recon_problems.append("G02 published no regional cells")
    elif not _close(sum(by_region.values()), total):
        recon_problems.append(
            f"G02 cells sum to {sum(by_region.values()):.2f}, G01 total is "
            f"{total if total is None else format(total, '.2f')}")

    # WHICH GROUPED LABELS "Scottish" COVERS is the region ladder's decision,
    # not this gate's. Matching the literal label "Scotland" was wrong: a book
    # whose labels are finer-grained than ITL1 spreads Scotland over several of
    # them, and comparing a filtered total against one of those rows would
    # report a defect that is really a granularity difference.
    resolve = region_resolver_owner()
    scots_balance = _balance(envelopes["G03"])
    scots_count = _count(envelopes["G04"])
    scots_note = "not adjudicated"
    if resolve is None:
        recon_problems.append(
            "the governed region owner could not be loaded, so the Scottish "
            "answer cannot be reconciled against the grouped labels")
    elif by_region:
        try:
            members = {str(v) for v in resolve("Scottish", list(by_region))}
        except Exception as exc:                 # noqa: BLE001
            members = set()
            recon_problems.append(f"the region owner could not resolve "
                                  f"'Scottish' over the returned labels: {exc}")
        scots_note = (f"'Scottish' resolves to {sorted(members) or 'no label'} "
                      f"of {len(by_region)} returned")
        if members:
            grouped_balance = sum(by_region.get(m, 0.0) for m in members)
            if not _close(scots_balance, grouped_balance):
                recon_problems.append(
                    f"G03 Scottish balance {scots_balance} != the governed "
                    f"Scottish membership of G02 {grouped_balance} "
                    f"({sorted(members)})")
            if counts_by_region:
                grouped_count = sum(counts_by_region.get(m, 0.0)
                                    for m in members)
                if scots_count is None or int(grouped_count) != int(scots_count):
                    recon_problems.append(
                        f"G04 Scotland count {scots_count} != the governed "
                        f"Scottish membership of G05 {grouped_count}")
        else:
            recon_problems.append(
                "the region owner resolved 'Scottish' to none of the returned "
                "labels, so the filtered answer cannot be reconciled")
    if not counts_by_region:
        recon_problems.append("G05 published no regional cells")

    # ERE's configured basis IS collateral, so "by property region" and "by
    # region" are the same question asked two ways and must agree cell for cell.
    prop_cells = _cells(envelopes["G06"], "current_outstanding_balance_sum",
                        "balance", "value")
    if want_basis == "collateral" and by_region and prop_cells:
        disagreeing = [region for region in set(by_region) | set(prop_cells)
                       if not _close(by_region.get(region), prop_cells.get(region))]
        if disagreeing:
            recon_problems.append(
                f"G06 disagrees with G02 for {sorted(disagreeing)[:5]}")

    # G08: answered means every facet survived; refused-for-coverage is allowed.
    g08 = envelopes["G08"]
    if _answered(g08):
        spec = g08.get("spec") or {}
        grouped = bool(spec.get("dimension") or spec.get("dimensions"))
        filters = {str(k).lower() for k in (spec.get("filters") or {})}
        if not grouped:
            recon_problems.append("G08 answered without the region grouping")
        if not any("loan_to_value" in f or "ltv" in f for f in filters):
            recon_problems.append("G08 answered without the LTV filter")
        if not any("borrower" in f for f in filters):
            recon_problems.append("G08 answered without the joint borrower filter")
    elif not _refused_for_missing_field(g08):
        recon_problems.append("G08 refused without naming what it could not apply")

    # G09 asked for two outputs and must return both.
    g09_rows = _acceptance_rows(envelopes["G09"])
    if _answered(envelopes["G09"]):
        keys = set().union(*(set(r) for r in g09_rows)) if g09_rows else set()
        if not any("balance" in str(k).lower() for k in keys):
            recon_problems.append("G09 dropped the balance")
        if not any("count" in str(k).lower() for k in keys):
            recon_problems.append("G09 dropped the loan count")
    else:
        recon_problems.append("G09 did not answer")

    # A NUMERIC FIXTURE IS APPLIED ONLY IF ONE IS RECORDED, and one is not.
    # The figures that used to live here were demo-derived and keyed to the live
    # portfolio without ever being checked against it, so the gate failed the
    # deployed service for not matching a book it does not hold. They are gone
    # rather than corrected, and deliberately not replaced with what the API
    # returned — a service that supplies its own expectations cannot fail.
    #
    # A future fixture must carry the identity of the dataset it was computed
    # from; a run against a different dataset is STALE_FIXTURE, not a numerical
    # failure. Until then the scoring above is entirely relational.
    snapshot_note = "no numeric fixture: scored on cross-response identities"
    truth_total = snap.get("totalBalance") if snap else None
    if truth_total is not None:
        identity = snap.get("datasetIdentity")
        observed = ((envelopes["G01"].get("reconciliation") or {})
                    .get("dataset_identity"))
        if identity and observed and str(identity) != str(observed):
            snapshot_note = (f"STALE_FIXTURE: recorded for dataset {identity}, "
                             f"served {observed} — numeric truth not applied")
        else:
            snapshot_note = f"numeric fixture for {portfolio_id}"
            if not _close(total, truth_total):
                recon_problems.append(
                    f"total balance {total} != fixture {truth_total}")
            for region, fixed in (snap.get("regions") or {}).items():
                if by_region and not _close(by_region.get(region),
                                            fixed.get("balance")):
                    recon_problems.append(
                        f"{region} balance {by_region.get(region)} != fixture "
                        f"{fixed.get('balance')}")
                cell = counts_by_region.get(region) if counts_by_region else None
                if cell is not None and int(cell) != int(
                        fixed.get("loanCount", -1)):
                    recon_problems.append(
                        f"{region} loan count {cell} != fixture "
                        f"{fixed.get('loanCount')}")
    _record("NUMERICAL_RECONCILIATION_PASS", not recon_problems,
            "; ".join(recon_problems))

    # ---- an unresolved qualifier must not quietly disappear -------------- #
    g10 = envelopes["G10"]
    safety_problems = []
    if _answered(g10):
        safety_problems.append("G10 answered a question containing 'platinum'")
    elif total is not None and _close(_balance(g10), total):
        safety_problems.append("G10 refused but published the whole-book figure")
    _record("SILENT_WRONG_SAFETY_PASS", not safety_problems,
            "; ".join(safety_problems))

    # ---- the report ------------------------------------------------------ #
    for row in rows:
        verdict = "answered" if row["answered"] else "REFUSED "
        lines.append(f"  {row['id']}  {verdict}  {row['question']}")
        lines.append(f"        configured={row['configuredBasis']}  "
                     f"effective={row['effectiveBasis']}  "
                     f"source={row['basisSource']}  "
                     f"measured={row['measuredBasis']} ({row['measuredField']})")
        lines.append(f"        supportedBases={row['supportedBases']}")
        if row["records"] is not None or row["balance"] is not None:
            lines.append(f"        records={row['records']}  "
                         f"balance={row['balance']}")
        if row["filters"]:
            lines.append(f"        filters={row['filters']}")
        for group in row["groups"]:
            lines.append(f"          group {group.get('label')!r:>34}  "
                         f"count={group.get('count')}  "
                         f"balance={group.get('balance')}")
        if not row["answered"]:
            lines.append(f"        {row['answer']}")
    lines.append("")
    lines.append(f"  Scottish membership : {scots_note}")
    lines.append(f"  {snapshot_note}")
    lines.append("")
    for name in ACCEPTANCE_CHECKS:
        state = checks.get(name)
        mark = "PASS" if state else "FAIL"
        detail = why.get(name, "")
        lines.append(f"  {mark}  {name}" + (f"  — {detail}" if detail else ""))
    return checks, rows, lines


def acceptance_markdown(payload: Dict[str, Any]) -> str:
    """The acceptance result as a GitHub job summary.

    Rendered from the SAME payload the JSON artefact carries, so the summary a
    reviewer reads and the file a machine reads cannot disagree — which they
    would the moment two renderers existed.
    """
    ready = payload.get("TIME_X_DIMENSION_READY", "NO")
    badge = "✅" if ready == "YES" else "❌"
    out = [f"## {badge} Geography release acceptance — "
           f"TIME_X_DIMENSION_READY = **{ready}**", "",
           f"_{payload.get('reason', '')}_", "",
           "| | |", "|---|---|",
           f"| portfolio | `{payload.get('portfolio') or '(none)'}` |",
           f"| expected SHA | `{payload.get('expectedCommit') or 'not given'}` |",
           f"| deployed SHA | `{payload.get('deployedCommit') or 'NOT ESTABLISHED'}` |",
           f"| reached service | {payload.get('reached')} |",
           f"| authorised | {payload.get('authorised')} |",
           f"| provenance | {payload.get('provenance')} |", ""]
    questions = payload.get("questions") or []
    if questions:
        out += ["### Questions", "",
                "| id | question | outcome | configured | effective | source | "
                "measured | records | key result |",
                "|---|---|---|---|---|---|---|---|---|"]
        for row in questions:
            out.append(
                f"| {row.get('id')} | {row.get('question')} | "
                f"{'answered' if row.get('answered') else '**refused**'} | "
                f"{row.get('configuredBasis')} | {row.get('effectiveBasis')} | "
                f"{row.get('basisSource')} | {row.get('measuredBasis')} | "
                f"{row.get('records')} | {row.get('balance')} |")
        out.append("")
    out += ["### Checks", "", "| check | verdict |", "|---|---|"]
    checks = payload.get("checks") or {}
    for name in ACCEPTANCE_CHECKS:
        out.append(f"| {name} | {'✅ PASS' if checks.get(name) else '❌ FAIL'} |")
    out.append("")
    return "\n".join(out)


def acceptance_verdict(checks: Dict[str, Optional[bool]]
                       ) -> Tuple[str, str]:
    """``(TIME_X_DIMENSION_READY, reason)``.

    An infrastructure blockage is NOT a semantic verdict. A run that could not
    reach or authenticate against the service has not found anything wrong with
    the geography configuration, and reporting it as though it had would send
    somebody to debug the wrong system.
    """
    if not checks.get("AUTH_PASS") or not checks.get("DEPLOYED_PROVENANCE_PASS"):
        return "NO", "NOT_EXECUTABLE"
    failed = [name for name in ACCEPTANCE_CHECKS if not checks.get(name)]
    if failed:
        return "NO", "FAILED: " + ", ".join(failed)
    return "YES", "all acceptance checks passed"


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
    parser.add_argument("--geography-acceptance", dest="geography",
                        action="store_true", default=False,
                        help="run the GEOGRAPHY / CONFIG-OWNERSHIP release "
                             "acceptance instead of the core and broad suites: "
                             "ten questions scored on configuration provenance, "
                             "explicit-basis semantics and numerical identity. "
                             "This is the release gate for the ERE-client / "
                             "ERM-asset ownership model, and it is deliberately "
                             "small — the broad sweep is not rerun for it.")
    parser.add_argument("--acceptance-json", default=None, metavar="FILE",
                        help="write the acceptance verdicts and per-question "
                             "rows here as JSON, for a CI job summary.")
    parser.add_argument("--summary-markdown", default=None, metavar="FILE",
                        help="write the acceptance result as GitHub-flavoured "
                             "markdown here, for $GITHUB_STEP_SUMMARY.")
    parser.add_argument("--snapshot", default=None, metavar="FILE",
                        help="override the frozen snapshot contract "
                             "(geography_snapshot.json).")
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
        ask = _in_process_asker(args.portfolio_id)

    header = ["=" * 74, "MI Query Agent — production certification",
              f"target: {target}", "=" * 74]
    provenance = "not checked (in-process run)"
    reached, authorised = "YES", "YES"
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
            if args.geography and (args.acceptance_json
                                   or args.summary_markdown):
                # An infrastructure blockage still has to be REPORTED as one.
                # Writing nothing here would leave the job summary silent, and
                # a silent gate is indistinguishable from a passing one.
                checks = {name: False for name in ACCEPTANCE_CHECKS}
                ready, reason = acceptance_verdict(checks)
                payload = {
                    "portfolio": args.portfolio_id, "expectedCommit": expected,
                    "deployedCommit": commit, "reached": reached,
                    "authorised": authorised, "provenance": provenance,
                    "checks": checks, "questions": [],
                    "TIME_X_DIMENSION_READY": ready, "reason": reason,
                }
                if args.acceptance_json:
                    Path(args.acceptance_json).write_text(
                        json.dumps(payload, indent=1), encoding="utf-8")
                if args.summary_markdown:
                    Path(args.summary_markdown).write_text(
                        acceptance_markdown(payload), encoding="utf-8")
            return 2
    print("\n".join(header))

    if args.geography:
        checks, rows, geo_lines = geography_acceptance(
            ask, portfolio_id=args.portfolio_id, provenance=provenance,
            reached=reached, authorised=authorised,
            snapshot_path=args.snapshot)
        ready, reason = acceptance_verdict(checks)
        body = "\n".join(geo_lines)
        print(body)
        print("-" * 74)
        print(f"TIME_X_DIMENSION_READY = {ready}")
        print(f"reason: {reason}")
        payload = {
            "portfolio": args.portfolio_id,
            "expectedCommit": (args.expect_commit or "").strip() or None,
            "deployedCommit": commit if args.base_url else None,
            "reached": reached, "authorised": authorised,
            "provenance": provenance, "checks": checks, "questions": rows,
            "TIME_X_DIMENSION_READY": ready, "reason": reason,
        }
        if args.acceptance_json:
            Path(args.acceptance_json).write_text(
                json.dumps(payload, indent=1, default=str), encoding="utf-8")
        if args.summary_markdown:
            Path(args.summary_markdown).write_text(
                acceptance_markdown(payload), encoding="utf-8")
        if args.report:
            Path(args.report).write_text(
                "\n".join(header) + "\n" + body
                + f"\n\nTIME_X_DIMENSION_READY = {ready}\nreason: {reason}\n",
                encoding="utf-8")
        # 2 is "we could not run it", and it must never read as a pass.
        return 0 if ready == "YES" else (2 if reason == "NOT_EXECUTABLE" else 1)

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
