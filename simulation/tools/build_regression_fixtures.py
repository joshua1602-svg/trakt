#!/usr/bin/env python3
"""Refresh the committed simulation regression fixtures.

    python -m simulation.tools.build_regression_fixtures

A fixture is a case's MANIFEST plus its independently computed EXPECTED TRUTH —
never its generated data files. Two properties follow:

* the fixtures stay small (a few JSON files, not hundreds of CSVs);
* ``tests/test_simulation_regression_fixtures.py`` regenerates the economics
  from the manifest and asserts the expected truth is byte-identical, so a
  change to the generator that silently alters a case's economics fails
  immediately with the case and seed named.

Deliberately narrow, per the framework's own rule about not committing bulk
generated data: one clean case per asset family, one schema-drift case, and the
two intended-failure guard cases.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .. import manifests as M
from ..history import generate_history
from ..reference_truth import build_expected_truth

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "simulation"

#: Blocks pinned VERBATIM. Small, and the first thing anyone reads when a case
#: moves — the headline economics, the regime expectation, the reporting dates.
_VERBATIM_EXCLUDES = ("periods",)


def condense(truth: Dict[str, Any]) -> Dict[str, Any]:
    """Pin a case's economics without committing every period.

    ``periods`` is ~97% of a raw expected-truth file and the reason the fixtures
    were 3,266 lines. It is replaced by a digest over its canonical JSON: a
    single changed figure in any period still fails the comparison, so the
    regression signal is unchanged.

    What IS lost is the diff. A failure says "this case's per-period economics
    moved" and names the case and seed; it no longer shows which month and which
    figure. Regenerate locally to see that:

        python -m simulation.runner reproduce --case <case_id> --seed <seed>

    Everything else is kept verbatim, so the headline numbers stay reviewable in
    the diff of a pull request.
    """
    out: Dict[str, Any] = {k: v for k, v in truth.items() if k not in _VERBATIM_EXCLUDES}
    for key in _VERBATIM_EXCLUDES:
        if key not in truth:
            continue
        canonical = json.dumps(truth[key], sort_keys=True, default=str)
        out[f"{key}_digest"] = {
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "count": len(truth[key]) if isinstance(truth[key], (list, dict)) else 1,
        }
    return out

#: The cases whose economics are pinned. Every one has a stated reason.
PINNED: Dict[str, str] = {
    "equity_release_clean_origination_v1":
        "clean equity-release baseline — the whole pathway's reference case",
    "bridge_clean_short_duration_v1":
        "clean bridge baseline — short-duration first-charge economics",
    "asset_finance_clean_sme_equipment_v1":
        "clean asset-finance baseline — equipment finance as a subtype",
    "bridge_maturity_wall_v1":
        "schema drift: a header renamed mid-history must still resolve",
    "bridge_unsupported_schema_guard_v1":
        "intended failure: an unresolvable source schema must be refused",
    "equity_release_missing_mandatory_guard_v1":
        "intended failure: a withheld mandatory column must block at Gate 2",
}


def build(root: Path = FIXTURE_ROOT) -> List[Path]:
    root.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    index = []
    for case_id, reason in PINNED.items():
        case = M.get_case(case_id)
        truth = condense(build_expected_truth(case, generate_history(case)))
        directory = root / case_id
        directory.mkdir(parents=True, exist_ok=True)
        written.append(M.write_manifest(case, directory / "manifest.json"))
        path = directory / "expected_truth.json"
        path.write_text(json.dumps(truth, indent=2, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
        written.append(path)
        index.append({"case_id": case_id, "seed": case.seed,
                      "asset_class": case.asset_class,
                      "simulation_version": case.simulation_version,
                      "manifest_hash": case.content_hash(), "reason": reason})
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    written.append(index_path)
    return written


def main() -> int:  # pragma: no cover - developer entry point
    for path in build():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
