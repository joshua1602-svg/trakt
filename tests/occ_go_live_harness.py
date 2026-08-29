"""Shared harness for the OCC go-live acceptance tests.

Everything here drives the REAL production route with the REAL agents: a raw
file is written to the governed blob prefix, the Event Grid intake picks it up,
and the Operations Control Centre runs Gate 1 -> Gate 2 -> Gate 3 (and, for a
regime-required source, Gate 4 -> Gate 5). No stub adapters — the point of these
tests is that the wiring works, so stubbing the thing being wired would prove
nothing.

Runs are slow by nature (real onboarding over a real tape), so the fixtures in
:mod:`tests.test_occ_go_live_e2e` are module-scoped and each rehearsal is
performed once and asserted many times.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Valid LEIs (check digits included) for two unrelated lenders.
LEI_A = "549300ABCDE123456702"
LEI_B = "213800ABCDE123456701"

CONTAINER = "raw-v2"


def bootstrap(tmp: Path, monkeypatch) -> Dict[str, Any]:
    """A complete file-backed Trakt: blob storage, OCC store, source registry."""
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp / "blob"))
    monkeypatch.setenv("TRAKT_OPS_MEMORY_ROOT", str(tmp / "memory"))
    monkeypatch.setenv("TRAKT_OPS_STAGING_ROOT", str(tmp / "staging"))
    monkeypatch.setenv("TRAKT_SOURCE_REGISTRY_URI",
                       "blob://trakt-state/registry/source_registry.yaml")
    monkeypatch.setenv("TRAKT_OPS_OPERATORS", json.dumps(
        {"tok-all": {"name": "Root Operator", "clients": ["*"], "role": "admin"}}))
    # The resolver is a recommender in production, and these tests assert
    # deterministic outcomes, so no provider is reachable from here.
    monkeypatch.setenv("TRAKT_LLM_ENABLED", "0")

    from apps.blob_trigger_app.source_registry import SourceRegistry
    from apps.blob_trigger_app.storage import Storage
    from operations_control.engine import OpsEngine
    from operations_control.onboarding.service import OnboardingService
    from operations_control.stores import OpsLayout, OpsStore

    storage = Storage(tmp / "blob")
    store = OpsStore(storage, OpsLayout("operations-control"))
    registry = SourceRegistry(os.environ["TRAKT_SOURCE_REGISTRY_URI"],
                              storage=storage)
    return {"tmp": tmp, "storage": storage, "store": store, "registry": registry,
            "svc": OnboardingService(store, registry_factory=lambda: registry),
            "engine": OpsEngine(store)}


def onboard(env, *, client_id: str, name: str, portfolio_id: str, lei: str,
            products) -> Dict[str, Any]:
    """Create and activate a client through the OCC onboarding case only.

    Every value here is an answer an operator types. If this function ever needs
    a code change to onboard a new lender, the sprint's first question is a NO.
    """
    svc, by = env["svc"], "Operator"
    case = svc.start_new_client(by=by)
    cid = case.case_id
    svc.save_step(case_id=cid, step="client", by=by, payload={
        "client_id": client_id, "client_name": name, "jurisdiction": "GB",
        "reporting_currency": "GBP", "time_zone": "Europe/London"})
    case = svc.save_step(case_id=cid, step="entities", by=by, payload={
        "entities": [{"legal_name": f"{name} Limited",
                      "roles": ["originator", "servicer", "reporting_entity"],
                      "lei": lei, "country_of_establishment": "GB"}]})
    entity = case.items("entities")[0]["entity_id"]
    svc.save_step(case_id=cid, step="contacts", by=by, payload={
        "reporting_contact_name": "R Reporter",
        "reporting_contact_email": f"reporting@{client_id.lower()}.example",
        "operational_contact_name": "O Ops",
        "operational_contact_email": f"ops@{client_id.lower()}.example",
        "reporting_contact_phone": "+44-2000000000"})
    svc.save_step(case_id=cid, step="portfolios", by=by, payload={
        "portfolios": [{"portfolio_id": portfolio_id,
                        "display_name": f"{name} Direct",
                        "portfolio_type": "direct",
                        "asset_class": "equity_release", "structure": "spv",
                        "owning_entity": entity,
                        "period_convention": "calendar_month_end"}]})
    svc.save_step(case_id=cid, step="reporting", by=by,
                  payload={"products": list(products)})
    svc.save_step(case_id=cid, step="sources", by=by, payload={
        "sources": [{"source_key": f"{portfolio_id}/funded",
                     "portfolio_id": portfolio_id, "dataset": "funded",
                     "cadence": "monthly", "source_party": f"{name} platform",
                     "delivery_channel": "sftp", "file_format": "csv"}]})
    if "esma_annex2" in products:
        svc.save_step(case_id=cid, step="regime", by=by, payload={"regime": {
            "esma_annex2": {"originator_name": f"{name} Limited",
                            "originator_legal_entity_identifier": lei,
                            "originator_establishment_country": "GB"}}})
    svc.approve(case_id=cid, by="Administrator", reason="go-live rehearsal")
    return svc.activate(case_id=cid, by="Administrator")


def arrive(env, *, client_id: str, portfolio_id: str, period: str, tape: Path,
           filename: str = "LoanExtract.csv", book: str = "direct") -> Dict[str, Any]:
    """Land a raw file at the governed blob prefix and fire the OCC intake.

    This is the production entrypoint: the same path the Event Grid trigger in
    the root ``function_app`` takes, with the Azure download swapped for a copy.
    """
    from apps.blob_trigger_app import occ_intake
    blob = f"{client_id}/{book}/funded/monthly/{portfolio_id}/{period}/{filename}"
    dest = env["tmp"] / "blob" / CONTAINER / blob
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tape, dest)

    def download(container: str, blob_path: str, dest_dir):
        out = Path(dest_dir) / blob_path.rsplit("/", 1)[-1]
        shutil.copy(env["tmp"] / "blob" / container / blob_path, out)
        return out

    return occ_intake.handle_arrival(CONTAINER, blob, download=download)


def settle(env, client_id: str, workflow_id: str, timeout: int = 600):
    """Wait for the background execution thread to reach a resting state."""
    store = env["store"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = store.load_workflow(client_id, workflow_id)
        if run and run.status not in ("running", "queued", "received"):
            return run
        time.sleep(0.5)
    return store.load_workflow(client_id, workflow_id)


def name_the_file(env, client_id: str, role: str = "loan_extract") -> int:
    """Answer every open "what is this file?" question. Returns how many."""
    from operations_control.contracts import KIND_FILE_ROLE
    qs = [d for d in env["store"].open_decisions(client_id)
          if d["kind"] == KIND_FILE_ROLE]
    for d in qs:
        env["engine"].resolve_decision(
            client_id=client_id, decision_id=d["decision_id"], action="approve",
            actor="Operator", value=role, scope="file", actor_is_admin=True)
    return len(qs)


def answer_mapping_queue(env, client_id: str, workflow_id: str, *,
                         mapping: Dict[str, str],
                         static: Optional[Dict[str, str]] = None,
                         not_applicable_reason: str = "not held in this book"
                         ) -> Dict[str, int]:
    """Play the operator through Gate 1's review queue.

    ``mapping`` is ``{canonical_field: source column}`` — what a person reads off
    the lender's data dictionary. Anything not named is marked not applicable.
    """
    engine, store = env["engine"], env["store"]
    counts = {"mapped": 0, "static": 0, "not_applicable": 0}
    for d in store.open_decisions(client_id, workflow_id):
        target = (d.get("subject") or {}).get("target_field") or ""
        if target in mapping:
            engine.resolve_decision(
                client_id=client_id, decision_id=d["decision_id"],
                action="approve", actor="Operator",
                value="provide_source_mapping", source_column=mapping[target],
                scope="portfolio", actor_is_admin=True)
            counts["mapped"] += 1
        elif static and target in static:
            engine.resolve_decision(
                client_id=client_id, decision_id=d["decision_id"],
                action="amend", actor="Operator", value=static[target],
                scope="portfolio", actor_is_admin=True)
            counts["static"] += 1
        else:
            engine.resolve_decision(
                client_id=client_id, decision_id=d["decision_id"],
                action="approve", actor="Operator", value="mark_not_applicable",
                reason=not_applicable_reason, scope="portfolio",
                actor_is_admin=True)
            counts["not_applicable"] += 1
    return counts


# --------------------------------------------------------------------------- #
# Reading what actually happened
# --------------------------------------------------------------------------- #

def orchestrator_state(run) -> Dict[str, Any]:
    root = Path(run.staging_root or "")
    if not root.exists():
        return {}
    hits = sorted(root.rglob("run_state.json"))
    return json.loads(hits[-1].read_text(encoding="utf-8")) if hits else {}


def gate_steps(run) -> Dict[str, str]:
    """``{step: status}`` for the portfolio — the proof of which gates ran."""
    state = orchestrator_state(run)
    portfolio = (state.get("portfolios") or [{}])[0]
    return {k: v.get("status") for k, v in (portfolio.get("steps") or {}).items()}


def artefact(run, name: str) -> Optional[Path]:
    root = Path(run.staging_root or "")
    if not root.exists():
        return None
    hits = sorted(root.rglob(name))
    return hits[-1] if hits else None


def read_csv(run, name: str) -> Optional[pd.DataFrame]:
    p = artefact(run, name)
    return pd.read_csv(p) if p is not None else None


def central_tape(run) -> Optional[pd.DataFrame]:
    return read_csv(run, "18_central_lender_tape.csv")


def validated_canonical(run) -> Optional[pd.DataFrame]:
    return read_csv(run, "31_transformed_canonical_tape.csv")


def validation_issues(run) -> pd.DataFrame:
    df = read_csv(run, "43_validation_issues.csv")
    return df if df is not None else pd.DataFrame()


def blocking_validation_fields(run) -> List[str]:
    df = validation_issues(run)
    if df.empty or "severity" not in df:
        return []
    return sorted(df[df["severity"] == "error"]["field"].astype(str).unique())


def handoff(run) -> Dict[str, Any]:
    p = artefact(run, "24_onboarding_handoff_manifest.json")
    return json.loads(p.read_text(encoding="utf-8")) if p else {}


# --------------------------------------------------------------------------- #
# Synthetic lender tapes — two unrelated equity release books, GBP, monthly
# --------------------------------------------------------------------------- #

def alpha_tape(path: Path, period: str = "2025-11-30", rows: int = 30) -> Path:
    """Client A's vocabulary: terse core-system codes."""
    import numpy as np
    rng = np.random.default_rng(11)
    df = pd.DataFrame({
        "ACCT_REF": [f"ACC{100000 + i}" for i in range(rows)],
        "CUST_ID": [f"C{9000 + i}" for i in range(rows)],
        "COMPLETION_DT": (pd.to_datetime("2016-01-01")
                          + pd.to_timedelta(rng.integers(0, 3000, rows), "D")
                          ).strftime("%Y-%m-%d"),
        "ADVANCE_AMT": rng.integers(70000, 220000, rows),
        "BAL_OS": rng.integers(95000, 380000, rows).astype(float).round(2),
        "INT_RATE_PA": rng.uniform(3.1, 6.2, rows).round(3),
        "PROP_VAL_ORIG": rng.integers(230000, 850000, rows),
        "PROP_VAL_CURR": rng.integers(250000, 930000, rows),
        "VAL_DT": period,
        "PROP_PCODE": rng.choice(["BT1 5GS", "G1 2AA", "NE1 4XX"], rows),
        "PROP_TYPE_CD": rng.choice(["DET", "SEMI", "TER", "FLAT"], rows),
        "TENURE_CD": rng.choice(["FH", "LH"], rows),
        "AGE_YNGST": rng.integers(61, 92, rows),
        "STATUS_CD": rng.choice(["LIVE", "REDEEMED"], rows, p=[0.9, 0.1]),
        "SNAPSHOT_DT": period,
        "CCY": "GBP",
        "ARREARS_AMT": 0.0,
    })
    df.to_csv(path, index=False)
    return path


#: Client A's data dictionary, as an operator would read it.
ALPHA_MAPPING = {
    "loan_identifier": "ACCT_REF",
    "account_status": "STATUS_CD",
    "current_outstanding_balance": "BAL_OS",
    "current_principal_balance": "BAL_OS",
    "original_principal_balance": "ADVANCE_AMT",
    "current_interest_rate": "INT_RATE_PA",
    "origination_date": "COMPLETION_DT",
    "current_valuation_amount": "PROP_VAL_CURR",
    "original_valuation_amount": "PROP_VAL_ORIG",
    "valuation_date": "VAL_DT",
    "tenure": "TENURE_CD",
    "youngest_borrower_age": "AGE_YNGST",
    "collateral_type": "PROP_TYPE_CD",
    "collateral_geography": "PROP_PCODE",
    "reporting_date": "SNAPSHOT_DT",
    "cut_off_date": "SNAPSHOT_DT",
    "arrears_balance": "ARREARS_AMT",
    "currency": "CCY",
}

#: Management statuses a lifetime-mortgage servicer really uses. "Probate -
#: awaiting sale" is the one that matters: it is not in any ESMA enum, and it
#: must survive to MI exactly as written.
BETA_STATUSES = (["Live"] * 20 + ["Probate - awaiting sale"] * 4
                 + ["Redeemed"] * 3 + ["In possession"] * 2 + ["Moved to LTC"])


def beta_tape(path: Path, period: str = "2026-01-31") -> Path:
    """Client B's vocabulary: prose headers, a policy book, richer statuses."""
    import numpy as np
    rng = np.random.default_rng(7)
    rows = len(BETA_STATUSES)
    df = pd.DataFrame({
        "Policy Number": [f"POL{700000 + i}" for i in range(rows)],
        "Borrower Ref": [f"BR{4000 + i}" for i in range(rows)],
        "Drawdown Date": (pd.to_datetime("2015-01-01")
                          + pd.to_timedelta(rng.integers(0, 3500, rows), "D")
                          ).strftime("%Y-%m-%d"),
        "Initial Advance": rng.integers(60000, 240000, rows),
        "Loan Balance": rng.integers(90000, 420000, rows).astype(float).round(2),
        "Interest Rate %": rng.uniform(2.9, 6.4, rows).round(3),
        "Original Valuation": rng.integers(220000, 900000, rows),
        "Latest Valuation": rng.integers(240000, 980000, rows),
        "Valuation Date": period,
        "Postcode": rng.choice(["EH1 2AB", "CF10 1AA", "M1 4BT"], rows),
        "Dwelling Type": rng.choice(["Detached", "Semi", "Flat"], rows),
        "Ownership": rng.choice(["Freehold", "Leasehold"], rows),
        "Age Of Youngest Life": rng.integers(62, 91, rows),
        "Policy Status": list(BETA_STATUSES),
        "Reporting Month End": period,
        "Currency Code": "GBP",
        "Arrears": 0.0,
    })
    df.to_csv(path, index=False)
    return path


BETA_MAPPING = {
    "loan_identifier": "Policy Number",
    "account_status": "Policy Status",
    "current_outstanding_balance": "Loan Balance",
    "current_principal_balance": "Loan Balance",
    "original_principal_balance": "Initial Advance",
    "current_interest_rate": "Interest Rate %",
    "origination_date": "Drawdown Date",
    "current_valuation_amount": "Latest Valuation",
    "original_valuation_amount": "Original Valuation",
    "valuation_date": "Valuation Date",
    "tenure": "Ownership",
    "youngest_borrower_age": "Age Of Youngest Life",
    "collateral_type": "Dwelling Type",
    "collateral_geography": "Postcode",
    "reporting_date": "Reporting Month End",
    "cut_off_date": "Reporting Month End",
    "arrears_balance": "Arrears",
    "currency": "Currency Code",
}
