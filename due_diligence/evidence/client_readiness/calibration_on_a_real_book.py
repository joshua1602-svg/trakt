"""E4: run the 252-case calibration bank against a REAL book, not build_fixture.

What build_fixture is
---------------------
``mi_agent.mi_query_harness.build_fixture`` makes 400 rows by drawing every
column independently from a uniform distribution over its range. It has no
nulls, no correlation between any two fields, no skew, no vintages that relate
to balances, and no relationship between LTV and valuation. It is a SHAPE, not a
book — which is the right instrument for asking "does this query parse and
execute", and the wrong one for asking "is the answer right on client data".

Every one of the 252 calibration cases has been graded against it.

What this does
--------------
Runs the same bank, unchanged, against the prepared funded tape of a real book
from the client-readiness pack, and reports the delta case by case. The bank file
is not edited: a case that fails here fails because the data underneath it
changed, which is the finding.

    python3 .../calibration_on_a_real_book.py --root <pack root> [--client alderbridge]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))


def _real_frame(root: Path, client: str, reporting_date: str):
    import pandas as pd
    from mi_agent_api.funded_prep import prepare_funded_mi_dataset

    path = (Path(root) / "processed" / "platform" / client / reporting_date
            / "platform_canonical_typed.csv")
    raw = pd.read_csv(path, low_memory=False)
    prepared, _report = prepare_funded_mi_dataset(raw)
    return prepared


def _run(df, semantics) -> Dict[str, Any]:
    from mi_agent import mi_calibration
    results, _summary = mi_calibration.run_bank(df=df, semantics=semantics)
    by_id = {}
    for r in results:
        by_id[r.id] = {
            "passed": bool(r.ok),
            "failures": list(r.failures or []),
            "known_gap": r.known_gap,
            "category": r.category,
            "question": r.question,
        }
    return by_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="a store root holding processed/platform/<client>/<date>/")
    parser.add_argument("--client", default="alderbridge")
    parser.add_argument("--reporting-date", default="2026-06-30")
    parser.add_argument("--out", default=str(_HERE / "calibration_real_book.json"))
    args = parser.parse_args(argv)

    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    from mi_agent.mi_query_harness import build_fixture
    from mi_agent.mi_query_validator import load_mi_semantics

    semantics = load_mi_semantics(
        str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

    synthetic = _run(build_fixture(), semantics)
    real_frame = _real_frame(Path(args.root), args.client, args.reporting_date)
    real = _run(real_frame, semantics)

    print(f"build_fixture      : {len(synthetic)} cases, "
          f"{sum(1 for v in synthetic.values() if v['passed'])} passed")
    print(f"{args.client:<19}: {len(real_frame):,} loans, {len(real)} cases, "
          f"{sum(1 for v in real.values() if v['passed'])} passed")

    regressed = [cid for cid in real
                 if synthetic.get(cid, {}).get("passed") and not real[cid]["passed"]]
    recovered = [cid for cid in real
                 if not synthetic.get(cid, {}).get("passed") and real[cid]["passed"]]
    print(f"\npassed on build_fixture, FAILS on the real book: {len(regressed)}")
    for cid in regressed:
        row = real[cid]
        print(f"  {cid:<14} {row['question'][:64]}")
        for f in row["failures"][:2]:
            print(f"      {str(f)[:150]}")
    print(f"\nfailed on build_fixture, PASSES on the real book: {len(recovered)}")
    for cid in recovered:
        print(f"  {cid:<14} {real[cid]['question'][:64]}")

    Path(args.out).write_text(json.dumps(
        {"client": args.client, "reportingDate": args.reporting_date,
         "loans": int(len(real_frame)),
         "syntheticPassed": sum(1 for v in synthetic.values() if v["passed"]),
         "realPassed": sum(1 for v in real.values() if v["passed"]),
         "regressed": {c: real[c] for c in regressed},
         "recovered": {c: real[c] for c in recovered}},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":                  # pragma: no cover - CLI
    raise SystemExit(main())
