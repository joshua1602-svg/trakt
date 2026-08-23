#!/usr/bin/env python3
"""migration_phase0/dataset_conflation_check.py — is the disclaimer honoured?

READ-ONLY. Conversion 1 §2, and it runs BEFORE any production change.

    "The balance by seasoning segment excluding pipeline cases"

B21 recorded that a bare substring test made the clause RULING A VIEW OUT the
thing that selected it. Two owners read that vocabulary — `resolve_active_view`
picks the frame, `chat_routing._dataset_for` picks the dataset — and the
target-state closure found the second still deriving its own answer.

So the question this settles is not "is the contract tidy" but **does a user
asking to EXCLUDE pipeline cases get an answer computed over pipeline cases**.
If they do, that is a live wrong number and the conversion stops.

Traced end to end: resolved dataset, selected data, executed calculation,
returned answer — and the same question with the disclaimer REMOVED, as the
control that shows the disclaimer is what moved it.

    python -m migration_phase0.dataset_conflation_check
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.simplefilter("ignore")
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: (case, question, the dataset the SENTENCE asks for)
CASES: Tuple[Tuple[str, str, str], ...] = (
    ("the disclaimer", "The balance by seasoning segment excluding pipeline cases",
     "funded"),
    ("control: no mention", "The balance by seasoning segment", "funded"),
    ("control: asks for it", "The pipeline balance by seasoning segment",
     "pipeline"),
    ("control: other disclaimer", "The balance by vintage, ignoring the forecast",
     "funded"),
)


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        _REPO / "migration_phase0" / "DATASET_CONFLATION.json"))
    args = ap.parse_args(argv)
    client_id = _env()

    from mi_agent_api import chat_routing as routing
    from mi_agent_api.workspace import resolve_active_view, view_named_by_question
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.datasets import semantics_path
    from question_interpretation import projection
    from trakt_core.context import ExecutionContext

    semantics = load_mi_semantics(semantics_path())
    ctx = ExecutionContext.for_internal(client_id)

    rows: List[Dict[str, Any]] = []
    print("=" * 108)
    print("DATASET CONFLATION — is an EXCLUSION honoured, or does it select?")
    print("=" * 108)
    for case, question, asks_for in CASES:
        named = view_named_by_question(question)
        view = resolve_active_view(question, None)
        # The two owners, asked separately.
        dataset_for = routing._dataset_for(question, view)
        claim = projection.project(question, semantics=semantics).dataset
        result = execute_governed_mi_query(
            MiQueryRequest(question=question), ctx).result or {}
        metadata = result.get("metadata") or {}
        answer = (result.get("answer") or "").strip().replace("\n", " ")
        row = {
            "case": case, "question": question, "asksFor": asks_for,
            "viewNamedByQuestion": named,
            "resolveActiveView": view,
            "chatRoutingDatasetFor": dataset_for,
            "contractDataset": claim.dataset,
            "contractProvenance": claim.provenance,
            "route": metadata.get("route"),
            "ok": bool(result.get("ok")),
            "controlledRefusal": bool(result.get("controlledRefusal")),
            "answer": answer[:260],
            "honoured": (view == asks_for and dataset_for == asks_for
                         and claim.dataset == asks_for),
        }
        rows.append(row)
        print(f"\n--- {case}: {question!r}")
        print(f"    the sentence asks for      : {asks_for}")
        print(f"    view_named_by_question     : {named}")
        print(f"    resolve_active_view        : {view}")
        print(f"    chat_routing._dataset_for  : {dataset_for}")
        print(f"    contract claim             : {claim.dataset} "
              f"({claim.provenance})")
        print(f"    route / ok                 : {metadata.get('route')} / "
              f"{result.get('ok')}")
        print(f"    answer                     : {answer[:180]}")
        print(f"    HONOURED                   : {row['honoured']}")

    print("\n" + "=" * 108)
    breached = [r for r in rows if not r["honoured"]]
    if breached:
        print("STOP — LIVE PRODUCT DEFECT. A requested dataset was not honoured:")
        for row in breached:
            print(f"   {row['case']}: asked {row['asksFor']}, "
                  f"view={row['resolveActiveView']} "
                  f"_dataset_for={row['chatRoutingDatasetFor']} "
                  f"contract={row['contractDataset']}")
    else:
        print("Every case resolves the dataset the sentence asks for, at every "
              "owner.\nThe disclaimer is honoured; this is contract "
              "representation only.")
    print("=" * 108)

    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {Path(args.out).relative_to(_REPO)}")
    return 1 if breached else 0


if __name__ == "__main__":
    sys.exit(main())
