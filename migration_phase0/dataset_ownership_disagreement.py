#!/usr/bin/env python3
"""migration_phase0/dataset_ownership_disagreement.py

READ-ONLY. Reproduces, from EXECUTED production, exactly where the active
workspace tab changes what a natural-language MI question MEANS.

Three readers answer "which dataset is this question about?" today:

    workspace.resolve_active_view(question, tab)   question, THEN THE TAB
    chat_routing._dataset_for(question, view)      a wider tape vocabulary
    projection._dataset(qi, caller_dataset)        question, then the caller's

The first is the only one production actually loads a frame from, and it is the
one that consults the tab. This instrument puts all three side by side with what
the served envelope reports, per question per tab, so the tab's influence is
visible as a difference rather than argued from the source.

    python -m migration_phase0.dataset_ownership_disagreement
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: (case, question, why it is here)
CASES: Tuple[Tuple[str, str, str], ...] = (
    ("F-BAL", "What is the funded balance?", "funded, named"),
    ("F-CNT", "What is the funded loan count?", "funded, named"),
    ("F-ACQ", "What is the acquired funded balance?",
     "funded dataset, ACQUIRED population — the two must stay separate"),
    ("F-DIR", "What is the direct funded balance?",
     "funded dataset, DIRECT population"),
    ("P-CASE", "How many cases are there?", "pipeline by tape vocabulary only"),
    ("P-APP", "How many applications are there?", "pipeline by tape vocabulary only"),
    ("P-KFI", "How many KFIs are there?", "pipeline by tape vocabulary only"),
    ("P-OFF", "How many offers are there?", "pipeline by tape vocabulary only"),
    ("P-AMT", "What is the pipeline amount?", "pipeline, named"),
    ("X-FCA", "Forecast application volumes next quarter",
     "FORECAST with pipeline vocabulary — precedence case"),
    ("X-FCC", "Forecast completions over the next 3 months",
     "FORECAST with completion vocabulary — precedence case"),
    ("X-FCP", "How much of the forecast comes from pipeline?",
     "FORECAST with the word pipeline — the corpus case that must not regress"),
    ("D-DIS", "What is the balance by seasoning segment excluding pipeline cases?",
     "B21 — a DISCLAIMED tape word must not select"),
    ("N-BAL", "What is the total balance?",
     "NO dataset vocabulary at all — today the tab alone decides"),
)

TABS: Tuple[Any, ...] = (None, "funded", "pipeline", "forecast")


def _env() -> str:
    import logging
    import warnings
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def main() -> int:
    client_id = _env()
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from mi_agent_api import chat_routing as cr
    from mi_agent_api import workspace as ws
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(client_id)
    print("=" * 100)
    print("DATASET OWNERSHIP — where the workspace tab changes what a question MEANS")
    print("=" * 100)
    print(f"{'case':<7} {'tab':<9} {'active_view':<11} {'_dataset_for':<13} "
          f"{'served':<9} route")

    rows: List[Dict[str, Any]] = []
    tab_sensitive: List[str] = []
    for case, question, why in CASES:
        served: Dict[Any, str] = {}
        print(f"\n{case}  {question!r}\n      ({why})")
        for tab in TABS:
            view = ws.resolve_active_view(question, tab)
            # Runnable at BOTH commits. `_dataset_for` is the second owner this
            # remediation retires, so after the change there is nothing to read
            # and the column says so rather than the instrument dying.
            legacy = getattr(cr, "_dataset_for", None)
            ds_for = legacy(question, view) if legacy else "(retired)"
            env = execute_governed_mi_query(
                MiQueryRequest(question=question, dataset_context=tab),
                ctx).result or {}
            md = env.get("metadata") or {}
            recon = env.get("reconciliation") or {}
            served_ds = recon.get("dataset") or md.get("datasetContext")
            served[tab] = str(served_ds)
            print(f"      {str(tab):<9} {view:<11} {ds_for:<13} "
                  f"{str(served_ds):<9} {md.get('route') or '-'}")
            rows.append({
                "case": case, "question": question, "tab": tab,
                "resolveActiveView": view,
                "datasetFor": ds_for,
                "servedDataset": served_ds,
                "route": md.get("route"),
                "ok": env.get("ok"),
                "answer": (env.get("answer") or "")[:160],
            })
        if len(set(served.values())) > 1:
            tab_sensitive.append(case)
            print(f"      >>> TAB-SENSITIVE: served dataset differs across tabs "
                  f"{sorted(set(served.values()))}")

    out = _REPO / "migration_phase0" / "DATASET_OWNERSHIP_DISAGREEMENT.json"
    out.write_text(json.dumps({"rows": rows}, indent=2, default=str))

    print("\n" + "=" * 100)
    print(f"cases                       : {len(CASES)}  x {len(TABS)} tabs "
          f"= {len(rows)} executions")
    print(f"TAB-SENSITIVE cases         : {len(tab_sensitive)}  {tab_sensitive}")
    print(f"written                     : {out.relative_to(_REPO)}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
