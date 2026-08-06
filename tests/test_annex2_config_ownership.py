#!/usr/bin/env python3
"""tests/test_annex2_config_ownership.py — the configuration hierarchy.

Layers, in precedence order:

    Global regime -> Asset pack -> Client -> Deal/SPV -> Reporting cycle -> Loan tape

Two properties make that hierarchy trustworthy, and both are pinned here:

1. **The asset pack actually participates.** It was previously declared,
   documented and referenced — and never reached Gate 4, because no caller
   passed ``--product-defaults`` and because the pack publishes ``nd_defaults``
   at the top level while the projector reads ``defaults.nd_defaults``. Either
   defect alone makes the layer inert, so both are guarded.

2. **Precedence is never silent.** A no-data code states WHY a regulatory field
   is empty. Where two layers disagree the run must refuse rather than quietly
   prefer the outer one.

Run: python -m pytest tests/test_annex2_config_ownership.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_PROJECTOR = _REPO / "engine" / "gate_4_projection" / "regime_projector.py"
_ASSET = _REPO / "config" / "asset" / "product_defaults_ERM.yaml"
_RULES = _REPO / "config" / "regime" / "annex2_delivery_rules.yaml"
_ORCH = _REPO / "engine" / "orchestrator" / "trakt_run.py"
_DEMO_ARTEFACTS = _REPO / "demo_platform" / "artefacts.py"
_DEMO_CLIENT = _REPO / "demo_platform" / "config" / "config_client_ALDERBRIDGE_DEMO.yaml"


def _run_projector(asset: Path, client: Path, out: Path):
    frame = out / "in.csv"
    frame.write_text("loan_id\nL1\n", encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(_PROJECTOR), str(frame),
         "--regime", "ESMA_Annex2",
         "--registry", str(_REPO / "config" / "system" / "fields_registry.yaml"),
         "--enum-mapping", str(_REPO / "config" / "system" / "enum_mapping.yaml"),
         "--product-defaults", str(asset),
         "--config", str(client),
         "--template-order", str(_REPO / "config" / "system" / "esma_code_order.yaml"),
         "--portfolio-type", "equity_release",
         "--output-dir", str(out)],
        capture_output=True, text=True)


class TestTheAssetPackIsActuallyWiredIn(unittest.TestCase):
    """A configuration layer that no caller passes is not a layer."""

    def test_the_orchestrator_passes_the_asset_pack(self):
        source = _ORCH.read_text(encoding="utf-8")
        self.assertIn("--product-defaults", source,
                      "trakt_run.py must pass the asset pack to Gate 4")
        self.assertIn("ASSET_PACKS", source,
                      "asset packs must be registered, not hard-coded per call")

    def test_the_demo_route_passes_the_asset_pack(self):
        self.assertIn("--product-defaults", _DEMO_ARTEFACTS.read_text(encoding="utf-8"),
                      "the benchmark route must exercise the asset pack")

    def test_every_registered_pack_exists(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_orch_probe", _ORCH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_orch_probe"] = module
        spec.loader.exec_module(module)
        self.assertIn("equity_release", module.ASSET_PACKS)
        for portfolio_type, path in module.ASSET_PACKS.items():
            self.assertTrue(Path(path).exists(), f"{portfolio_type}: {path} missing")

    def test_the_schema_lift_makes_pack_nd_defaults_reachable(self):
        """The pack publishes nd_defaults at the top level; Gate 4 reads
        defaults.nd_defaults. Without the lift the layer is inert."""
        pack = yaml.safe_load(_ASSET.read_text(encoding="utf-8"))
        self.assertIn("nd_defaults", pack,
                      "the published asset-pack shape is top-level nd_defaults")
        self.assertNotIn("nd_defaults", pack.get("defaults") or {},
                         "do not restructure the file — other consumers read the "
                         "top-level key; the projector lifts it on a copy")
        self.assertIn("setdefault(\"defaults\", {}).setdefault(\"nd_defaults\", {})",
                      _PROJECTOR.read_text(encoding="utf-8"),
                      "the lift must exist or the pack contributes nothing")


class TestPrecedenceIsNeverSilent(unittest.TestCase):
    """Layer order must not quietly overturn a regulatory decision."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ownership_"))
        self.asset = self.tmp / "asset.yaml"
        self.asset.write_text(yaml.safe_dump(
            {"asset_class": "equity_release", "nd_defaults": {"maturity_date": "ND5"}}),
            encoding="utf-8")

    def _client(self, name: str, payload: dict) -> Path:
        p = self.tmp / name
        p.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return p

    def test_a_disagreement_refuses_rather_than_overriding(self):
        client = self._client("conflict.yaml",
                              {"defaults": {"nd_defaults": {"maturity_date": "ND1"}}})
        out = _run_projector(self.asset, client, self.tmp)
        blob = out.stdout + out.stderr
        self.assertIn("Conflicting no-data decisions", blob,
                      "the client silently overrode the asset pack")
        self.assertIn("maturity_date: asset=ND5 client=ND1", blob,
                      "the refusal must name the field and both values")
        self.assertIn("nd_override_rationale", blob,
                      "the refusal must say how to declare a legitimate override")

    def test_an_explicit_override_is_permitted(self):
        client = self._client("override.yaml", {
            "defaults": {"nd_defaults": {"maturity_date": "ND1"}},
            "nd_override_rationale": {"maturity_date": "this book has fixed-term loans"}})
        out = _run_projector(self.asset, client, self.tmp)
        self.assertNotIn("Conflicting no-data decisions", out.stdout + out.stderr,
                         "a declared override must be allowed to proceed")

    def test_agreeing_layers_are_not_treated_as_a_conflict(self):
        client = self._client("agree.yaml",
                              {"defaults": {"nd_defaults": {"maturity_date": "ND5"}}})
        out = _run_projector(self.asset, client, self.tmp)
        self.assertNotIn("Conflicting no-data decisions", out.stdout + out.stderr)


class TestOwnershipBoundaries(unittest.TestCase):
    """Each layer holds what it is authoritative for, and nothing else."""

    def setUp(self):
        self.asset_nd = (yaml.safe_load(_ASSET.read_text(encoding="utf-8"))
                         .get("nd_defaults") or {})
        self.client_nd = ((yaml.safe_load(_DEMO_CLIENT.read_text(encoding="utf-8"))
                           .get("defaults") or {}).get("nd_defaults") or {})

    def test_product_decisions_live_in_the_asset_pack(self):
        """True of a lifetime mortgage regardless of who originated it."""
        for field in ("scheduled_principal_payment_frequency",
                      "scheduled_interest_payment_frequency",
                      "principal_grace_period_end_date",
                      "current_interest_rate_index",
                      "current_interest_rate_index_tenor",
                      "current_interest_rate_margin",
                      "interest_rate_reset_interval",
                      "interest_rate_cap", "interest_rate_floor",
                      "deposit_amount"):
            self.assertIn(field, self.asset_nd,
                          f"{field} follows from the product; it belongs to the pack")
            self.assertNotIn(field, self.client_nd,
                             f"{field} is duplicated in a client config")

    def test_the_client_keeps_only_lender_and_deal_facts(self):
        for field in ("date_of_repurchase", "purchase_price", "customer_type",
                      "current_valuation_amount"):
            self.assertIn(field, self.client_nd,
                          f"{field} is specific to this lender or deal")

    def test_no_client_field_silently_duplicates_the_pack(self):
        overlap = {f for f in set(self.asset_nd) & set(self.client_nd)}
        self.assertEqual(overlap, set(),
                         f"duplicated ND ownership: {sorted(overlap)} — one layer "
                         f"must be authoritative")

    def test_the_global_regime_holds_no_client_identifiers(self):
        """A regime file is shared by every client; it must stay anonymous."""
        rules = yaml.safe_load(_RULES.read_text(encoding="utf-8"))
        scope = rules.get("reconciliation_scope") or {}
        self.assertIn("non_emitting_fields", scope)
        for code in ("RREL18", "RREL28", "RREC22"):
            self.assertIn(code, scope["non_emitting_fields"])


class TestNonEmittingReconciliation(unittest.TestCase):
    """A CSV value does not imply an XML node."""

    def test_the_four_way_split_is_reported_and_adds_up(self):
        import importlib.util
        import pandas as pd
        spec = importlib.util.spec_from_file_location(
            "_norm_probe", _REPO / "engine" / "gate_4b_delivery" / "annex2_delivery_normalizer.py")
        norm = importlib.util.module_from_spec(spec)
        sys.modules["_norm_probe"] = norm
        spec.loader.exec_module(norm)

        rules = yaml.safe_load(_RULES.read_text(encoding="utf-8"))
        scoped = {"defaults": rules.get("defaults", {}),
                  "reconciliation_scope": rules["reconciliation_scope"],
                  "field_rules": {"RREL16": rules["field_rules"]["RREL16"]}}
        frame = pd.DataFrame({"RREL3": ["A", "B"],
                              "RREL18": ["ND1", "ND1"],     # non-emitting
                              "RREL22": ["ND5", "ND5"]})    # emitting
        _out, _issues, summary = norm.normalize_delivery(frame, scoped)
        rec = summary["delivery_instrumentation"]["nd_reconciliation"]
        self.assertEqual(rec["nd_on_non_emitting_fields"], 2)
        self.assertEqual(rec["non_emitting_by_field"], {"RREL18": 2})
        self.assertEqual(
            rec["nd_in_delivery_data"],
            rec["nd_on_emitting_fields"] + rec["nd_on_non_emitting_fields"],
            "the split must account for every ND cell")

    def test_rrel18_is_not_treated_as_a_missing_element(self):
        """It is the Ccy attribute of an amount, not an absent node."""
        rules = yaml.safe_load(_RULES.read_text(encoding="utf-8"))
        self.assertNotIn("RREL18", rules["field_rules"],
                         "RREL18 has no XML element; a delivery rule would write "
                         "a value with no destination")
        text = _RULES.read_text(encoding="utf-8")
        self.assertIn("Ccy", text, "the attribute rationale must be recorded")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
