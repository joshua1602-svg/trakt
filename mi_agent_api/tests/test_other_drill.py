"""tests/test_other_drill.py

The capped "Other" bar bucket carries the shown top-N categories so a drill on
"Other" executes as ``<dim> NOT IN [shown]`` and recovers the underlying rows,
instead of matching an opaque "Other" label (which matched nothing).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent_api.adapters import adapt_workflow_result

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


def _df(n_brokers: int = 15, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    brokers = [f"Broker {i:02d}" for i in range(n_brokers)]
    return pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(n)],
        "current_outstanding_balance": rng.uniform(50_000, 400_000, n).round(2),
        "broker_channel": rng.choice(brokers, n),
    })


def test_bar_capped_other_carries_shown_categories():
    sem = load_mi_semantics(_SEMANTICS)
    df = _df()
    wf = run_mi_agent_query("balance by broker", df, sem)
    resp = adapt_workflow_result(wf, portfolio_id="client_001/mi_2025_11")
    chart = next(a for a in resp["artifacts"] if a["type"] == "chart")
    # Capped to top-10 (9 shown + Other).
    assert any(str(r.get("broker_channel")) == "Other" for r in chart["rows"])
    shown = chart["otherCategories"]["broker_channel"]
    assert shown and "Other" not in shown
    assert len(shown) == 9


def test_other_drill_not_in_recovers_excluded_rows():
    sem = load_mi_semantics(_SEMANTICS)
    df = _df()
    wf = run_mi_agent_query("balance by broker", df, sem)
    resp = adapt_workflow_result(wf, portfolio_id="client_001/mi_2025_11")
    chart = next(a for a in resp["artifacts"] if a["type"] == "chart")
    shown = chart["otherCategories"]["broker_channel"]

    # Drill "Other" -> broker_channel NOT IN shown.
    drilled = run_mi_agent_query(
        "balance by broker", df, sem,
        extra_filters={"broker_channel": {"op": "not_in", "value": shown}})
    assert drilled["ok"], drilled.get("error")
    qr = drilled["query_result"].to_dict()
    got_brokers = {str(r["broker_channel"]) for r in (qr.get("data") or [])}
    # Only the excluded (non-shown) brokers remain, and none of the shown ones.
    assert got_brokers and got_brokers.isdisjoint(set(shown))
    expected = set(df["broker_channel"].unique()) - set(shown)
    assert got_brokers == expected


def test_in_filter_keeps_only_named_categories():
    sem = load_mi_semantics(_SEMANTICS)
    df = _df()
    keep = ["Broker 00", "Broker 01"]
    wf = run_mi_agent_query("balance by broker", df, sem,
                            extra_filters={"broker_channel": {"op": "in", "value": keep}})
    qr = wf["query_result"].to_dict()
    got = {str(r["broker_channel"]) for r in (qr.get("data") or [])}
    assert got == set(keep)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
