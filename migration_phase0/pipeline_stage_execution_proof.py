#!/usr/bin/env python3
"""migration_phase0/pipeline_stage_execution_proof.py — can a governed stage
predicate reproduce the shipped weekly stage series?

READ-ONLY. C6 prerequisite §7. The Stage vocabulary is governed and the Stage
claim is represented, but representation is not execution: nothing had proved
that `Predicate("pipeline_stage", "eq", "<STAGE>")`, run through the ONE
governed predicate executor, selects exactly the cases the shipped funnel
selects — week by week, on the real five-week extract history.

The shipped rule is an inline mask inside `evolution.pipeline_funnel_evolution`:

    stage_col.str.upper() == stage

The governed rule is `mi_query_executor.governed_predicate_mask`, which
casefolds and STRIPS. If the two ever select different cases, a converted route
would silently move a stage series, so this compares selected CASE IDS, not just
counts.

Fails loudly if the fixture or the governed semantics are missing: a stage proof
that silently measured zero weeks would be the vacuous pass this programme keeps
finding.

    python -m migration_phase0.pipeline_stage_execution_proof [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: The canonical five. WITHDRAWN is included so its coverage is REPORTED rather
#: than quietly omitted — the brief allows it to stay representation-only, but
#: only if that is stated.
STAGES = ("KFI", "APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN")

#: The stages the brief requires to execute.
REQUIRED = ("KFI", "APPLICATION", "OFFER")


def _boot():
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg


def _case_ids(frame) -> List[str]:
    for column in ("case_identifier", "case_id", "loan_identifier",
                   "borrower_identifier"):
        if column in frame.columns:
            return sorted(str(v) for v in frame[column].tolist())
    return sorted(str(i) for i in frame.index)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = _boot()
    from analytics_lib.numeric import coerce_numeric
    from migration_phase0.assurance_semantics import load_assurance_semantics
    from mi_agent_api import pipeline_contract as pipeline_mod
    from mi_agent.population import Predicate, apply_population
    from mi_agent_api import evolution as evolution_mod

    semantics = load_assurance_semantics()
    # THE deterministic five-week fixture, and the client it was built for.
    #
    # NOT the demo store: measured, `datasets._pipeline_discovery_root()`
    # resolves to the platform blob mirror, which carries ZERO weekly extracts
    # in this environment — every stage and funnel question there answers "no
    # weekly pipeline extracts are available", with ok=True. A stage proof run
    # against that root would have compared an empty series with an empty series
    # and reported agreement. The fixture is the only place stage history exists,
    # so it is the only place stage execution can be proved.
    root = str(_REPO / "tests" / "fixtures" / "pipeline_history_5w")
    client = "client_001"
    if not Path(root).is_dir():
        raise SystemExit("ASSURANCE INVALID - the five-week fixture is missing")
    # THE SAME inventory the shipped funnel walks, so the two cannot be reading
    # different weeks — which is the one way this proof could pass vacuously.
    inventory = pipeline_mod.weekly_extract_inventory(root, client)
    extracts = list(inventory.get("extracts") or ())
    if not extracts:
        raise SystemExit("ASSURANCE INVALID - no governed weekly extracts; the "
                         "five-week pipeline fixture is not present")

    frames: List[Dict[str, Any]] = []
    for ext in extracts:
        try:
            frame, _ = pipeline_mod.load_prepared_pipeline(ext)
        except Exception as exc:  # noqa: BLE001 - a load failure INVALIDATES
            raise SystemExit(
                f"ASSURANCE INVALID - a governed extract failed to load: {exc}")
        frames.append({"week": ext.get("pipeline_extract_date"), "df": frame})
    if not frames:
        raise SystemExit("ASSURANCE INVALID - no prepared weekly frames")

    balance = "current_outstanding_balance"
    rows: List[Dict[str, Any]] = []
    for entry in frames:
        frame, week = entry["df"], entry["week"]
        if "pipeline_stage" not in frame.columns:
            raise SystemExit(
                f"ASSURANCE INVALID - week {week} carries no pipeline_stage column")
        stage_col = frame["pipeline_stage"].astype(str)
        bal = coerce_numeric(frame[balance]) if balance in frame.columns else None
        for stage in STAGES:
            shipped_mask = stage_col.str.upper() == stage
            shipped = frame[shipped_mask]
            governed, evidence = apply_population(
                frame, [Predicate("pipeline_stage", "eq", stage)], semantics)
            usable = governed is not None and evidence.is_usable
            rows.append({
                "week": week, "stage": stage,
                "shipped_count": int(shipped_mask.sum()),
                "governed_count": (int(len(governed)) if usable else None),
                "ids_identical": bool(usable and _case_ids(shipped) == _case_ids(governed)),
                "shipped_value": (round(float(bal[shipped_mask].sum()), 2)
                                  if bal is not None else None),
                "governed_value": (round(float(coerce_numeric(governed[balance]).sum()), 2)
                                   if usable and bal is not None else None),
                "usable": usable,
            })

    print("=" * 92)
    print("PIPELINE STAGE HISTORICAL EXECUTION — governed predicate vs shipped mask")
    print("=" * 92)
    print(f"weeks: {len(frames)}  ({', '.join(str(f['week']) for f in frames)})")
    print(f"\n{'week':<14}{'stage':<13}{'shipped':>9}{'governed':>10}"
          f"{'ids':>6}{'shipped £':>18}{'governed £':>18}")
    for row in rows:
        print(f"{str(row['week']):<14}{row['stage']:<13}{row['shipped_count']:>9}"
              f"{str(row['governed_count']):>10}{'OK' if row['ids_identical'] else 'DIFF':>6}"
              f"{(row['shipped_value'] if row['shipped_value'] is not None else 0):>18,.2f}"
              f"{(row['governed_value'] if row['governed_value'] is not None else 0):>18,.2f}")

    print(f"\n{'stage':<13}{'weeks':>7}{'ids identical':>16}{'non-empty weeks':>18}"
          f"{'EXECUTES':>10}")
    verdicts = {}
    for stage in STAGES:
        group = [r for r in rows if r["stage"] == stage]
        identical = sum(1 for r in group if r["ids_identical"])
        non_empty = sum(1 for r in group if r["shipped_count"] > 0)
        # NON-VACUOUS: a stage that selects nothing in every week proves nothing,
        # however cleanly the two paths "agree" on the empty set.
        executes = identical == len(group) and non_empty > 0
        verdicts[stage] = executes
        print(f"{stage:<13}{len(group):>7}{identical:>16}{non_empty:>18}"
              f"{('YES' if executes else 'NO'):>10}")

    missing = [s for s in REQUIRED if not verdicts.get(s)]
    print("\n" + "=" * 92)
    if missing:
        print(f"VERDICT: STAGE EXECUTION NOT PROVEN for {', '.join(missing)}")
    else:
        extra = [s for s in STAGES if s not in REQUIRED and verdicts.get(s)]
        rep_only = [s for s in STAGES if s not in REQUIRED and not verdicts.get(s)]
        print("VERDICT: STAGE EXECUTION PROVEN for " + ", ".join(REQUIRED)
              + (f"; also {', '.join(extra)}" if extra else "")
              + (f"; representation-only (no shipped series to reproduce): "
                 f"{', '.join(rep_only)}" if rep_only else ""))
    print("=" * 92)

    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "verdicts": verdicts},
                                        indent=2, default=str), encoding="utf-8")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
