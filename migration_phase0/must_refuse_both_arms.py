#!/usr/bin/env python3
"""migration_phase0/must_refuse_both_arms.py — do the three still refuse?

    "What changed?"                 names no period to compare over
    "Show me the trend."            names no measure
    "Compare us with the market."   there is NO MARKET DATA in this platform

The third is the worst outcome available from any of this work: answering it
means presenting a whole-book figure as a comparison against something the
estate has never held.

WHY BOTH ARMS. These three refuse on the deterministic parser and the frozen
CFO bank has expected that since it was written. They were re-measured here for
Stage 3 and found to ANSWER — all three — and it took a stashed working tree to
establish that Stage 3 had not caused it.

The cause is that an `ANTHROPIC_API_KEY` was in the environment. `_mi_llm_config`
enables the shipped free-form LLM parser arm on `auto` whenever a key is
present, and that arm emits a whole `MIQuerySpec`: it supplies the missing
period and the missing measure itself, so no governed default is ever recorded
and the guards that fire on a recorded default never see one. That is the same
mechanism the Opus acceptance run walked through, and it is not a Stage 3
regression — it is the arrangement the concept-proposal split exists to replace.

    python -m migration_phase0.must_refuse_both_arms [--json out.json]

EXITS NON-ZERO if the DETERMINISTIC arm stops refusing. The LLM arm's result is
printed in full and does not fail the run, because it is a known-open finding
with its own owner — failing here would only teach the estate to stop running
this. It is recorded, by name, every time.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

MUST_REFUSE = (
    ("What changed?", "names no period to compare over"),
    ("Show me the trend.", "names no measure"),
    ("Compare us with the market.", "there is no market data in this platform"),
)

PORTFOLIO = "client_001/mi_2026_06"
AS_OF = "2026-06-30"


class MeasurementError(RuntimeError):
    """The measurement could not run. Never absorbed into a pass."""


def _ask(client, question: str) -> Dict[str, Any]:
    body = client.post("/mi/query", json={"question": question,
                                          "portfolioId": PORTFOLIO,
                                          "asOfDate": AS_OF}).json()
    meta = body.get("metadata") or {}
    return {"question": question, "ok": body.get("ok"),
            "refused": not body.get("ok"),
            "parser": (meta.get("parserProvenance") or {}).get("parser_used"),
            "route": meta.get("route"),
            "answer": (body.get("answer") or "")[:400]}


def _arm(force: str) -> Dict[str, Any]:
    """One arm, in a subprocess, because `_mi_llm_config` reads the environment
    and an arm measured after another arm has imported the app is not an arm."""
    import subprocess

    env = dict(os.environ)
    env["MI_AGENT_LLM_PARSER"] = force
    code = (
        "import json,sys,os,warnings,logging\n"
        "warnings.simplefilter('ignore'); logging.disable(logging.ERROR)\n"
        "sys.path.insert(0, %r)\n"
        "from fastapi.testclient import TestClient\n"
        "from mi_agent_api.app import app\n"
        "from mi_agent_api import datasets as D\n"
        "cfg = D._mi_llm_config()\n"
        "c = TestClient(app)\n"
        "rows=[]\n"
        "for q,_ in %r:\n"
        "    r=c.post('/mi/query', json={'question':q,'portfolioId':%r,'asOfDate':%r}).json()\n"
        "    m=r.get('metadata') or {}\n"
        "    rows.append({'question':q,'ok':r.get('ok'),'refused': not r.get('ok'),\n"
        "                 'parser':(m.get('parserProvenance') or {}).get('parser_used'),\n"
        "                 'route':m.get('route'),'answer':(r.get('answer') or '')[:400]})\n"
        "print(json.dumps({'config':{'enabled':cfg.enabled,'available':cfg.available,\n"
        "                            'model':cfg.model,'status':cfg.status},'rows':rows}))\n"
        % (str(_REPO), MUST_REFUSE, PORTFOLIO, AS_OF))
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True)
    tail = (out.stdout or "").strip().splitlines()
    if not tail:
        raise MeasurementError("arm %r produced no result: %s"
                               % (force, (out.stderr or "")[-400:]))
    return json.loads(tail[-1])


def run() -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    env = os.environ.get("MI_COMPLETENESS_FIXTURE", "/tmp/cfo_env")
    if not Path(env, "onboarding_output").is_dir():
        raise MeasurementError(
            "MEASUREMENT INVALID - fixture root %r has no onboarding_output" % env)
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = "%s/onboarding_output" % env
    os.environ["TRAKT_PORTFOLIO_REGISTRY"] = "%s/portfolio_registry.yaml" % env
    os.environ.setdefault("MI_AGENT_PIPELINE_ROOT",
                          str(_REPO / "tests" / "fixtures" / "pipeline_history_5w"))
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"

    deterministic = _arm("off")
    llm: Dict[str, Any]
    if os.environ.get("ANTHROPIC_API_KEY"):
        llm = _arm("on")
    else:
        llm = {"config": {"status": "not measured - no key in the environment"},
               "rows": []}
    return {"deterministic": deterministic, "llm_arm": llm,
            "why": {q: why for q, why in MUST_REFUSE}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="out")
    args = ap.parse_args(argv)

    result = run()
    ok = True
    print("must-refuse, deterministic arm  (config: %s)"
          % result["deterministic"]["config"]["status"])
    for row in result["deterministic"]["rows"]:
        mark = "REFUSED" if row["refused"] else "*** ANSWERED ***"
        print("  %-32s %-16s parser=%s" % (row["question"], mark, row["parser"]))
        if not row["refused"]:
            ok = False
            print("       %s" % row["answer"][:200])

    print("\nmust-refuse, LLM parser arm     (config: %s)"
          % result["llm_arm"]["config"].get("status"))
    if not result["llm_arm"]["rows"]:
        print("  not measured — no ANTHROPIC_API_KEY in the environment")
    answered = [r for r in result["llm_arm"]["rows"] if not r["refused"]]
    for row in result["llm_arm"]["rows"]:
        mark = "REFUSED" if row["refused"] else "*** ANSWERED ***"
        print("  %-32s %-16s parser=%s route=%s" % (row["question"], mark,
                                                    row["parser"], row["route"]))
        if not row["refused"]:
            print("       %s" % row["answer"][:200])
    if answered:
        print("\n  OPEN FINDING — %d of %d answer on the LLM arm. The free-form "
              "parser supplies the missing element itself, so no governed "
              "default is recorded and the guard that fires on one never sees "
              "it. Recorded, not failed: this instrument's job is to keep it "
              "visible." % (len(answered), len(result["llm_arm"]["rows"])))

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print("  wrote %s" % args.out)
    print("\n%s" % ("DETERMINISTIC ARM HOLDS" if ok
                    else "DETERMINISTIC ARM FAILED — a must-refuse question answered"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
