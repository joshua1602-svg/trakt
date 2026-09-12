#!/usr/bin/env python3
"""Assert the live change_form gate against a result file, and print the report.

Separate from the runner on purpose. The runner MEASURES and must finish writing
its evidence whatever it finds; this DECIDES, and its exit status is what fails
the CI job. Keeping them apart means a gate that trips cannot truncate the
evidence that explains why.

The thresholds are written here as literals rather than read from the result, so
a result file cannot assert its own pass mark.

    exit 0   gate met
    exit 1   gate not met
    exit 2   PROVIDER_BLOCKED — the bank did not complete, so there is no verdict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

#: The gate, as stated in the brief this run was commissioned under.
REQUIRED_CASES = 18
REQUIRED = {
    "change_form_total": 18,
    "material_summary": 4,
    "metric_delta": 6,
    "attribution": 4,
    "level_comparison": 4,
    "explicit_measure_preserved": 14,
    "scope_preserved": 4,
}
#: Must be exactly zero.
REQUIRED_ZERO = (
    "material_summary_specific_measure_invented",
    "unnecessary_clarifications",
    "invented_filters",
    "invented_dimensions",
    "silent_semantic_drops",
)
#: Temporal is REPORTED and not gated in v2. The deterministic owners do not
#: honour a `current` anchor for material_summary or attribution; that is an
#: independent temporal-contract defect, reported separately and not repaired in
#: this sprint. Counting it here would make a temporal failure read as a
#: change_form failure.

_COLS = ("ID", "EXP_FORM", "ACT_FORM", "EXP_MEASURE", "ACT_MEASURE",
         "EXP_SCOPE", "ACT_SCOPE", "PERIOD", "VERDICT", "CLASS")


def _num(value: Any) -> int:
    """`"17 / 18"` -> 17. A count is already an int."""
    if isinstance(value, int):
        return value
    return int(str(value).split("/")[0].strip())


def _table(rows: List[Mapping[str, Any]]) -> str:
    out = [list(_COLS)]
    for r in rows:
        measures = [m.get("canonical") for m in (r.get("measures") or ())
                    if m.get("canonical")]
        out.append([
            str(r.get("id", "")),
            str(r.get("expected_change_form") or "-"),
            str(r.get("change_form") or "NONE"),
            str(r.get("expected_measure") or "none required"),
            ",".join(measures) if measures else "NONE",
            str(r.get("expected_scope") or "-"),
            str(r.get("lens") or "-"),
            str(r.get("period_result") or "-"),
            str(r.get("verdict", "")),
            ",".join(r.get("failure_classes") or ()) or "-",
        ])
    widths = [max(len(row[i]) for row in out) for i in range(len(_COLS))]
    lines = []
    for n, row in enumerate(out):
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip())
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)

    if not args.result.exists():
        print(f"::error::no result file at {args.result}: the run did not reach "
              f"the end of the bank")
        print("LIVE_INTERPRETATION_GATE = PROVIDER_BLOCKED")
        return 2

    report = json.loads(args.result.read_text(encoding="utf-8"))
    totals: Dict[str, Any] = dict(report.get("totals") or {})
    rows = list(report.get("cases") or ())
    by_form = dict(totals.get("by_change_form") or {})

    print(f"BANK_ID     = {report.get('bank_id')}")
    print(f"BANK_SHA256 = {report.get('bank_sha256')}")
    print(f"BASELINE    = {report.get('baseline_sha')}")
    print(f"MODEL       = {report.get('model')}")
    served = sorted({r.get("served_model") for r in rows if r.get("served_model")})
    print(f"SERVED      = {', '.join(served) if served else 'none recorded'}")
    print(f"MODE        = {report.get('mode')}")
    print()
    print(_table(rows))
    print()

    failures: List[str] = []

    # -- the bank must have COMPLETED. A partial run has no verdict. --------- #
    attempted = _num(totals.get("attempted", 0))
    interpreted = _num(totals.get("interpreted", 0))
    provider = _num(report.get("provider_failures", 0))
    if report.get("mode") != "live":
        failures.append(f"mode is {report.get('mode')!r}, not 'live': a dry run "
                        f"is not a measurement")
    if provider or attempted < REQUIRED_CASES or interpreted < REQUIRED_CASES:
        print(f"attempted={attempted} interpreted={interpreted} "
              f"provider_failures={provider}")
        print(f"::error::the bank did not complete: {interpreted} of "
              f"{REQUIRED_CASES} questions were interpreted")
        print("LIVE_INTERPRETATION_GATE = PROVIDER_BLOCKED")
        return 2

    # -- the primary gate and its breakdown --------------------------------- #
    checks = [
        ("CHANGE_FORM_TOTAL", _num(totals.get("change_form_total", 0)),
         REQUIRED["change_form_total"]),
        ("MATERIAL_SUMMARY", _num(by_form.get("material_summary", 0)),
         REQUIRED["material_summary"]),
        ("METRIC_DELTA", _num(by_form.get("metric_delta", 0)),
         REQUIRED["metric_delta"]),
        ("ATTRIBUTION", _num(by_form.get("attribution", 0)),
         REQUIRED["attribution"]),
        ("LEVEL_COMPARISON", _num(by_form.get("level_comparison", 0)),
         REQUIRED["level_comparison"]),
        ("EXPLICIT_MEASURE_PRESERVED",
         _num(totals.get("explicit_measure_preserved", 0)),
         REQUIRED["explicit_measure_preserved"]),
        ("SCOPE_PRESERVED", _num(totals.get("scope_preserved", 0)),
         REQUIRED["scope_preserved"]),
    ]
    for name, got, want in checks:
        mark = "ok  " if got >= want else "FAIL"
        print(f"  {mark} {name} = {got} / {want}")
        if got < want:
            failures.append(f"{name} = {got}, required {want}")

    for key in REQUIRED_ZERO:
        got = _num(totals.get(key, 0))
        mark = "ok  " if got == 0 else "FAIL"
        print(f"  {mark} {key.upper()} = {got} / 0")
        if got:
            failures.append(f"{key} = {got}, required 0")

    print()
    if failures:
        for f in failures:
            print(f"::error::{f}")
        print("LIVE_INTERPRETATION_GATE = FAIL")
        return 1
    print("LIVE_INTERPRETATION_GATE = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
