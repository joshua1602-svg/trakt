#!/usr/bin/env python3
"""migration_phase0/portfolio_identity_map.py — the portfolio identity chain.

READ-ONLY. Executes the real chain for every context the governed registry
publishes, and asks the question Phase 1D exists to answer:

    Can MI resolve the labels React actually shows the client?

    storage name -> governed portfolio id -> source type -> React label
                 -> MI semantic value -> governed scope -> execution filter

React's selector is built from ``portfolio_context.context_index()`` — the same
registry MI resolves against (frontend/mi-agent-ui/src/state/useWorkspace.ts:300,
"derived from the governed hierarchy ... exactly one source of portfolio truth").
So the two share an identity MODEL. What this measures is whether they share an
identity VOCABULARY: React renders ``label``, and MI's lens layer reads words.

    python -m migration_phase0.portfolio_identity_map
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

warnings.simplefilter("ignore")
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def main(argv: Optional[Sequence[str]] = None) -> int:
    client_id = _env()
    from mi_agent import portfolio_lens as lens_mod
    from mi_agent_api import evolution as evolution_mod
    from mi_agent_api import portfolio_context as ctx_mod

    index = ctx_mod.context_index(client_id=client_id)
    contexts = index["contexts"]
    root = os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"]
    frames = evolution_mod.funded_frames(root, client_id, None)
    df = frames[-1]["df"]
    balance = "current_outstanding_balance"

    print("=" * 112)
    print("PORTFOLIO IDENTITY MAP — what React shows vs what MI can resolve")
    print("=" * 112)
    print(f"\ndefault context: {index.get('default_context_id')!r}\n")

    rows: List[Dict[str, Any]] = []
    unresolvable: List[str] = []
    for ctx in contexts:
        context_id = ctx["context_id"]
        label = ctx["label"]
        scope = ctx_mod.resolve_context(context_id, discover_pipeline=False).scope
        scoped = evolution_mod._scope_frame_lens(df, scope.filters or None)

        # The question a client would type, using the label React renders.
        question = f"Summarise the {label}"
        mi_lens = lens_mod.resolve_lens(question)
        mi_context = lens_mod.context_id(mi_lens)
        mi_scope = ctx_mod.resolve_context(mi_context, discover_pipeline=False).scope
        mi_scoped = evolution_mod._scope_frame_lens(df, mi_scope.filters or None)

        agrees = tuple(mi_scope.portfolio_ids) == tuple(scope.portfolio_ids)
        if not agrees:
            unresolvable.append(label)

        rows.append({
            "reactLabel": label, "contextId": context_id,
            "contextKind": ctx["context_kind"],
            "governedIds": list(scope.portfolio_ids),
            "rows": int(len(scoped)),
            "balance": round(float(scoped[balance].sum()), 2) if balance in scoped else None,
            "question": question,
            "miLensName": mi_lens.name,
            "miContextId": mi_context,
            "miGovernedIds": list(mi_scope.portfolio_ids),
            "miRows": int(len(mi_scoped)),
            "miBalance": (round(float(mi_scoped[balance].sum()), 2)
                          if balance in mi_scoped else None),
            "miFellBackToTotal": bool(mi_scope.fell_back_to_total),
            "miResolvesToTheSamePopulation": agrees,
        })

    print(f"{'React label':26s} {'context_id':18s} {'governed ids':40s} {'rows':>7s}")
    print("-" * 112)
    for r in rows:
        print(f"{r['reactLabel']:26s} {r['contextId']:18s} "
              f"{str(r['governedIds']):40s} {r['rows']:7d}")

    print(f"\n\nASKED BY LABEL — \"Summarise the <React label>\"\n")
    print(f"{'React label':26s} {'MI lens':10s} {'MI ids':40s} {'rows':>7s}  same?")
    print("-" * 112)
    for r in rows:
        mark = "YES" if r["miResolvesToTheSamePopulation"] else "NO"
        print(f"{r['reactLabel']:26s} {r['miLensName']:10s} "
              f"{str(r['miGovernedIds']):40s} {r['miRows']:7d}  {mark}"
              + ("   <- fell back to Total" if r["miFellBackToTotal"] else ""))

    print("\n" + "-" * 112)
    print(f"React labels MI CANNOT resolve to the same population: "
          f"{len(unresolvable)} of {len(rows)}")
    for label in unresolvable:
        print(f"    {label!r}")

    out = _REPO / "migration_phase0" / "PORTFOLIO_IDENTITY_MAP.json"
    out.write_text(json.dumps({"defaultContextId": index.get("default_context_id"),
                               "rows": rows, "unresolvableLabels": unresolvable},
                              indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
