#!/usr/bin/env python3
"""Are the stage-movement ANSWERS right, or only present?

WHY THIS EXISTS. `live_pipeline_probe.py` reported 36 of 36 PASS on the shipped
stage-movement bank, and PASS there means the question DELIVERED rather than
refused. It does not mean the figure is correct: the probe records route,
outcome and artefact shape, and deliberately records no figures at all, so
nothing in that result speaks to arithmetic. Reporting it as "answers correctly
on the real book" was an overstatement, and this is the check that was missing.

The bank's own `must` assertions cannot close the gap either — `must: ['2']` was
written against `tests/fixtures/pipeline_transition_2w`, where the answer is 2.
Against a live book it asserts nothing.

WHAT THIS DOES INSTEAD. It recomputes each transition from the two governed
weekly extracts using PLAIN PANDAS — an outer join on the case identifier,
compare prior stage to latest stage, count — and compares that against the
figure the agent published. The production engine (`movement_detail
.stage_transition_events`) is never called, because a check that reuses the
thing it is checking proves only that the code is self-consistent.

Loading the extracts DOES use the app's own loader. Loading is not computing:
reading a different file, or preparing it differently, would make a mismatch
mean nothing.

IT PRINTS BOOLEANS. Question id, whether the figures matched, and the event
class compared. No counts, no balances, no case identifiers — the output is
safe to paste anywhere, which is the same discipline every probe here follows.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ["MI_AGENT_AUTH_ENABLED"] = "false"


def _locate_app_root() -> None:
    if os.path.isdir("mi_agent_api"):
        sys.path.insert(0, os.getcwd())
        return
    for p in glob.glob("/tmp/*/mi_agent_api"):
        root = os.path.dirname(p)
        if os.path.isfile(os.path.join(root, "mi_agent_api", "app.py")):
            sys.path.insert(0, root)
            return
    raise SystemExit("could not find mi_agent_api; run from the extracted app")


#: The nine shipped subtypes, each as one business question, with the event this
#: check recomputes for it. Only shapes this file can verify INDEPENDENTLY are
#: listed: a question whose answer cannot be re-derived from the extracts by
#: plain arithmetic does not belong here, because an unverifiable row reported
#: as a pass is exactly the problem this file exists to fix.
CHECKS = [
    ("SM01", "How many cases moved from KFI to Application?",
     ("move", "KFI", "APPLICATION", "count")),
    ("SM02", "How much balance moved from Application to Offer?",
     ("move", "APPLICATION", "OFFER", "amount")),
    ("SM03", "How many cases moved from Offer to Completion?",
     ("move", "OFFER", "COMPLETED", "count")),
    ("SM04", "How much balance moved from Offer to Completion?",
     ("move", "OFFER", "COMPLETED", "amount")),
    ("SM05", "How many new cases entered KFI?",
     ("new", None, "KFI", "count")),
    ("SM06", "How many cases stayed in Application?",
     ("stayer", "APPLICATION", "APPLICATION", "count")),
    ("SM07", "What was the amount change on cases that stayed in Application?",
     ("stayer", "APPLICATION", "APPLICATION", "change")),
]


def _independent(cur, pri, event, src, dst, measure):
    """Recompute one figure from the extracts, without the governed engine."""
    import pandas as pd
    from mi_agent_api.movement_detail import CASE_KEY, MEASURE, STAGE

    def shape(df):
        if df is None or df.empty or CASE_KEY not in df.columns:
            return pd.DataFrame(columns=["stage", "amount"])
        out = pd.DataFrame({
            "case": df[CASE_KEY].astype(str).str.strip(),
            "stage": (df[STAGE].astype(str).str.strip().str.upper()
                      if STAGE in df.columns else ""),
            "amount": pd.to_numeric(df.get(MEASURE), errors="coerce").fillna(0.0),
        })
        out = out[out["case"].str.lower().isin(
            {"", "nan", "none", "null"}) == False]
        return out.groupby("case").agg(stage=("stage", "last"),
                                       amount=("amount", "sum"))

    lat, old = shape(cur), shape(pri)
    both = lat.join(old, how="outer", rsuffix="_prior")
    in_lat = both["stage"].notna()
    in_pri = both["stage_prior"].notna()

    if event == "new":
        sel = in_lat & ~in_pri & (both["stage"] == dst)
    elif event == "stayer":
        sel = in_lat & in_pri & (both["stage"] == both["stage_prior"]) \
            & (both["stage"] == src)
    else:
        sel = (in_lat & in_pri & (both["stage_prior"] == src)
               & (both["stage"] == dst))

    rows = both[sel]
    if measure == "count":
        return float(len(rows))
    if measure == "amount":
        return float(rows["amount"].sum())
    return float(rows["amount"].sum() - rows["amount_prior"].fillna(0.0).sum())


def _figures(envelope):
    """Every number the answer published, from its artefact rows."""
    out = []
    for art in envelope.get("artifacts") or []:
        for row in art.get("rows") or []:
            if isinstance(row, dict):
                for v in row.values():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        out.append(float(v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--tolerance", type=float, default=0.51,
                    help="absolute tolerance; counts are exact, money rounds")
    args = ap.parse_args()

    _locate_app_root()
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    from mi_agent_api import pipeline_contract as pipeline_mod
    from mi_agent_api.movement_detail import select_pair

    # THE SAME ROOT THE ROUTE USES. Reading MI_AGENT_ONBOARDING_OUTPUT_ROOT
    # directly finds nothing: the weekly extracts live under a pipeline root
    # derived from MI_AGENT_PIPELINE_URI and MATERIALISED to local scratch, and
    # the onboarding root holds funded cuts instead. A reconciliation pointed at
    # a different root cannot disagree with the answer meaningfully — it can
    # only fail to find anything, which is what it did.
    from mi_agent_api import datasets as ds_mod

    root = ds_mod._materialise_pipeline_root(ds_mod._pipeline_root())
    client_id = os.environ.get("MI_AGENT_CLIENT_ID") or "client_001"
    print("pipeline root resolved:", "yes" if root else "NO")
    print("client id             :", client_id)
    inv = pipeline_mod.weekly_extract_inventory(root, client_id)
    extracts = inv.get("extracts", []) or []
    print("weekly extracts found :", len(extracts))
    cur_e, pri_e = select_pair(extracts, None)
    if cur_e is None or pri_e is None:
        raise SystemExit(
            "no governed weekly extract PAIR to reconcile against (%d extract(s) "
            "found). The agent answers transitions from a pair, so if it is "
            "answering and this cannot find one, the roots differ — check "
            "MI_AGENT_PIPELINE_ROOT / MI_AGENT_PIPELINE_URI." % len(extracts))
    print("comparing             :", pri_e.get("pipeline_extract_date"),
          "->", cur_e.get("pipeline_extract_date"))
    print()
    cur, _ = pipeline_mod.load_prepared_pipeline(cur_e)
    pri, _ = pipeline_mod.load_prepared_pipeline(pri_e)

    client = TestClient(app, raise_server_exceptions=False)
    rows, agree = [], 0
    print("%-6s %-9s %-34s %s" % ("id", "answered", "recomputed", "verdict"))
    print("-" * 74)
    for qid, question, (event, src, dst, measure) in CHECKS:
        res = client.post("/mi/query", json={"question": question})
        env = res.json()
        expected = _independent(cur, pri, event, src, dst, measure)
        published = _figures(env)
        # The answer is CORRECT if the independently computed figure is among
        # the numbers it published. Matching on presence rather than position
        # keeps this robust to how a route lays its artefact out; the figure
        # itself still has to be there.
        hit = any(abs(p - expected) <= args.tolerance for p in published)
        ok = bool(env.get("ok"))
        verdict = "MATCH" if (ok and hit) else ("MISMATCH" if ok else "REFUSED")
        agree += verdict == "MATCH"
        print("%-6s %-9s %-34s %s"
              % (qid, ok, "%s %s->%s (%s)" % (event, src, dst, measure), verdict))
        rows.append({"id": qid, "event": event, "source": src,
                     "destination": dst, "measure": measure,
                     "answered": ok, "verdict": verdict,
                     "figures_published": len(published)})

    print()
    print("independently reconciled: %d of %d" % (agree, len(CHECKS)))
    if agree < len(CHECKS):
        print("a MISMATCH means the agent published a figure this file could "
              "not reproduce from the extracts. Investigate before trusting "
              "the 36/36.")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"reconciled": agree, "total": len(CHECKS),
                       "extract_dates": [pri_e.get("pipeline_extract_date"),
                                         cur_e.get("pipeline_extract_date")],
                       "rows": rows}, fh, indent=1, default=str)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
