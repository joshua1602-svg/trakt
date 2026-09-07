#!/usr/bin/env python3
"""Run a synthetic borrower base through the PRODUCTION borrowing-base path.

Nothing here calculates anything. It builds the governed inputs — a facility
configuration, an activated Schedule 8 concentration configuration, a prepared
canonical frame — and then calls exactly the service the dashboard calls, so
the JSON it writes is the wire payload the React workspace would receive from a
live API, not a hand-built mock.

    python scripts/render_borrowing_base_envelope.py \
        --tape synthetic_borrowing_base.csv --out-dir preview_data

Two states are emitted, because they are the two things worth seeing:

  prototype.json  the facility as configured TODAY — no approved Eligible
                  Mortgage Loan definition, no facility statement. Every loan
                  eligible under the prototype assumption, and headroom
                  reported as NOT_CALCULABLE rather than invented.

  governed.json   the same book under a facility whose eligibility criteria
                  have been approved and whose drawn balance has been supplied.
                  A real three-way eligibility split and real headroom.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

SCHEDULE_8 = (REPO / "tests" / "concentration_tests" / "fixtures"
              / "warehouse_facility_schedule_8.txt")
CLIENT_ID = "ere_funding_uk"
REPORTING_DATE = "2025-11-30"
RUN_ID = "mi_2025_11"

PROTOTYPE_FACILITY: Dict[str, Any] = {
    "client_id": CLIENT_ID,
    "facility_id": "ERE_WAREHOUSE_01",
    "facility_label": "ERE Warehouse Facility 01",
    "facility_type": "warehouse",
    "currency": "GBP",
    "commitment": 250_000_000,
    "advance_rate": 1.03,
    "concentration_denominator_floor": 33_000_000,
    "current_drawn_amount": None,
    "environment": "prototype",
    "eligibility": {
        "rule_version": "prototype-0",
        "rules": [],
        "prototype_assume_financing_portfolio_eligible": True,
    },
    "concentration": {"population": "eligible_mortgage_loans",
                      "borrowing_base_treatment": "monitor_only"},
}

GOVERNED_FACILITY: Dict[str, Any] = {
    **PROTOTYPE_FACILITY,
    "current_drawn_amount": 118_500_000,
    "current_drawn_amount_as_of": REPORTING_DATE,
    "environment": "production",
    "eligibility": {
        "rule_version": "ere-warehouse-2025-11",
        "rules": [
            {"rule_id": "no_material_arrears",
             "description": "No more than 30 days past due",
             "field": "days_past_due", "operator": "max", "value": 30,
             "reason_code": "more_than_30_days_past_due",
             "source_reference": "Facility Agreement cl. 4.2(c)"},
            {"rule_id": "max_original_ltv",
             "description": "Original loan to value must not exceed 50%",
             "field_role": "ltv_original", "operator": "max", "value": 50,
             "reason_code": "original_ltv_above_facility_limit",
             "source_reference": "Facility Agreement cl. 4.2(f)"},
        ],
        "prototype_assume_financing_portfolio_eligible": False,
    },
}


def _write_facility(directory: Path, facility: Dict[str, Any]) -> Path:
    path = directory / "funding_facilities.yaml"
    path.write_text(yaml.safe_dump(
        {"schema_version": "1.0.0", "config_version": "demo-2025-11-30",
         "facilities": [facility]}), encoding="utf-8")
    os.environ["TRAKT_FUNDING_FACILITIES_PATH"] = str(path)
    os.environ["TRAKT_CLIENT_CONFIG_DIR"] = str(directory / "no_client_configs")
    return path


def _activate_schedule_8() -> None:
    """Extract, approve and activate the supplied Schedule 8, as OCC would."""
    from apps.blob_trigger_app.storage import open_storage
    from operations_control.concentration import ConcentrationGovernanceService
    from operations_control.stores import OpsLayout, OpsStore

    service = ConcentrationGovernanceService(
        OpsStore(open_storage(), OpsLayout(container="operations-control")))
    proposals = service.extract(
        CLIENT_ID, actor="Demo Operator",
        source_reference="Warehouse Facility Schedule 8",
        text=SCHEDULE_8.read_text(encoding="utf-8"))
    approved = 0
    for proposal in proposals:
        # Net WAC carries an unresolved concern and is NOT approved: the
        # schedule defines it against a Series Fixed Rate that lives elsewhere
        # in the facility. Approving it would be inventing the answer.
        if proposal.concern_codes:
            continue
        service.approve(CLIENT_ID, proposal.proposal_id, actor="Demo Operator",
                        effective_date="2025-01-01",
                        comments="Reviewed against Schedule 8.")
        approved += 1
    service.activate(CLIENT_ID, actor="Demo Operator",
                     reason="Schedule 8 initial activation")
    print(f"  activated {approved} of {len(proposals)} extracted limits "
          f"({len(proposals) - approved} held for confirmation)")


def build(tape: Path, facility: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
    from mi_agent_api import concentration_tests_api as conc_mod
    from mi_agent_api.funded_prep import prepare_funded_mi_dataset

    _write_facility(workspace, facility)
    raw = pd.read_csv(tape, low_memory=False)
    frame, report = prepare_funded_mi_dataset(raw)
    eligibility = report["borrowing_base_eligibility"]
    print(f"  eligibility: {eligibility.get('status_counts')} "
          f"(rule version {eligibility.get('eligibility_rule_version')})")

    original = conc_mod._resolve_frames
    conc_mod._resolve_frames = (
        lambda output_root, client_id, to_run_id, scope=None:
        (frame, None, REPORTING_DATE, None, RUN_ID))
    try:
        envelope = conc_mod.compute_concentration_tests(None, CLIENT_ID, None)
    finally:
        conc_mod._resolve_frames = original
    return envelope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        os.environ["TRAKT_RUNTIME_MODE"] = "test"
        os.environ["TRAKT_STORAGE_BACKEND"] = "file"
        os.environ["TRAKT_LOCAL_BLOB_ROOT"] = str(workspace / "blob")
        print("Activating Schedule 8 through the governance service…")
        _activate_schedule_8()

        for name, facility in (("prototype", PROTOTYPE_FACILITY),
                               ("governed", GOVERNED_FACILITY)):
            print(f"\n{name}:")
            envelope = build(args.tape, facility, workspace)
            base = envelope.get("borrowingBase") or {}
            print(f"  eligible £{base.get('eligibleCurrentBalance', 0):,.0f} "
                  f"· borrowing base {base.get('availableBorrowingBase')} "
                  f"· headroom {base.get('borrowingBaseHeadroom')}")
            summary = envelope.get("summary") or {}
            print(f"  concentrations: {summary.get('breaches')} breach, "
                  f"{summary.get('warnings')} warning, "
                  f"{summary.get('passes')} pass, "
                  f"{summary.get('unavailable')} unavailable")
            out = args.out_dir / f"{name}.json"
            out.write_text(json.dumps(envelope, indent=2, default=str),
                           encoding="utf-8")
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
