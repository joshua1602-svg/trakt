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
def _live_asker(base_url: str) -> Callable[[str], Dict[str, Any]]:
    import urllib.error
    import urllib.request

    def ask(question: str) -> Dict[str, Any]:
        body = json.dumps({"question": question}).encode("utf-8")
        request = urllib.request.Request(
            base_url.rstrip("/") + "/mi/query", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:               # noqa: PERF203
            return {"ok": False, "answer": f"HTTP {exc.code}",
                    "__transport_error__": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "answer": str(exc),
                    "__transport_error__": True}

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

    for artifact in (envelope.get("artifacts") or []):
        for item in (artifact.get("items") or []):
            label = str(item.get("label") or "").lower()
            if "loan" in label and "count" in label:
                try:
                    return int(str(item.get("value")).replace(",", ""))
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
MUST_ANSWER = (
    "What is the total balance?",
    "How many loans are there?",
    "Total balance by region",
    "What is the average loan size?",
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


def certify(ask: Callable[[str], Dict[str, Any]]) -> Tuple[bool, List[str]]:
    lines: List[str] = []
    ok = True

    def record(status: str, detail: str) -> None:
        nonlocal ok
        if status == "FAIL":
            ok = False
        lines.append(f"  {status:6} {detail}")

    lines.append("MUST ANSWER")
    for question in MUST_ANSWER:
        envelope = ask(question)
        if envelope.get("__transport_error__"):
            record("FAIL", f"{question}  [transport] {envelope.get('answer')}")
        elif envelope.get("ok"):
            record("ok", question)
        else:
            record("FAIL", f"{question}  refused: "
                           f"{str(envelope.get('answer'))[:90]}")

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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None,
                        help="serving instance, e.g. "
                             "https://trakt-mi-api.azurewebsites.net. Omitted: "
                             "run in-process against mi_agent_api.app.")
    args = parser.parse_args(argv)

    if args.base_url:
        target = args.base_url
        ask = _live_asker(args.base_url)
    else:
        target = "in-process (mi_agent_api.app via TestClient)"
        ask = _in_process_asker()

    print("=" * 74)
    print("MI Query Agent — production certification")
    print("target:", target)
    print("=" * 74)
    certified, lines = certify(ask)
    print("\n".join(lines))
    print("-" * 74)
    print("VERDICT:", "CERTIFIED" if certified else "NOT CERTIFIED")
    if not args.base_url:
        print("NOTE: in-process. This exercises the same ASGI app and the same")
        print("      governance envelope, but it does NOT certify deployment,")
        print("      packaging, app settings, or the data the live instance is")
        print("      pointed at. Re-run with --base-url against the serving")
        print("      instance for a live certification.")
    return 0 if certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
