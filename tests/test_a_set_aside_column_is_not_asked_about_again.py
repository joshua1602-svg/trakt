"""The operator set the column aside. The workflow asked about it anyway.

ERE's pack is three workbooks that overlap: an interest rate in two, a
valuation in two, a balance in two. In the Client Onboarding screen the
operator kept one of each and set the duplicates aside — ``Latest Property
Value``, ``Loan Interest Rate`` — column by column, and committed. The live
workflow then put eleven questions to them, most of them::

    Which source column is the authoritative source for
    'current_valuation_amount'?

about the very duplicates they had removed.

THREE THINGS WERE WRONG, AND EACH HID THE NEXT.

1. The commit records "Do not use" as ``resolution="approve"`` with
   ``resolved_value=NOT_USED``. Promotion read only an AMENDMENT's answer, so
   an approved set-aside fell through to its own proposed field and was
   promoted as that mapping — the removal arrived in the governed store as a
   live rule for the field it had been removed from.
2. Even written correctly, nothing said the column fed NOTHING: retiring a
   rule leaves no rule, and coverage finds sources by NAME against the alias
   registry, so it found the column again.
3. And nothing carried the store's rules to coverage at all: they were
   projected into a client-memory directory the workflow's onboarding never
   opens, and the overrides file is read only by the tape builder, after
   coverage has already asked its questions.

These tests hold each link: the commit's real shape promotes a set-aside, the
set-aside supersedes the mapping in a real rule store, the engine reads it
back, and coverage — on ERE-shaped data — no longer asks.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pytest
import yaml

from engine.onboarding_agent import target_coverage as tcov
from operations_control.occ_agent import mapping_promotion as mp
from operations_control.occ_agent import staging as _staging

LOAN = "LoanExtract One - OMNI 2026_09_01.xlsx"
PROP = "PropertyExtract - Omni 2026_09_01.xlsx"


def _committed(column, target, *, action, source_file=PROP, amended_to=""):
    """A mapping decision exactly as `confirm_mappings` leaves it."""
    staged = {"action": action, "target_field": amended_to or target}
    return {"decision_id": f"d_{column}", "decision_type": "mapping_proposal",
            "subject": {"source_file": source_file, "source_column": column,
                        "target_field": target},
            "status": "approved",
            "resolution": _staging.RESOLUTION[action],
            "resolved_value": _staging.resolved_value(staged),
            "resolved_by": "operator", "resolved_at": "2026-09-18T10:00:00Z"}


def _set_aside(column, target, source_file=PROP):
    return _committed(column, target, action=_staging.ACTION_NOT_USED,
                      source_file=source_file)


def _confirmed(column, target, source_file=PROP):
    return _committed(column, target, action=_staging.ACTION_CONFIRM,
                      source_file=source_file)


# --------------------------------------------------------------------------- #
# 1. Promotion reads the commit's real shape
# --------------------------------------------------------------------------- #
class TestASetAsideIsNotPromotedAsAMapping:

    def test_the_committed_set_aside_asserts_no_mapping(self):
        assert mp.mapping_of(_set_aside("Latest Property Value",
                                         "current_valuation_amount")) is None

    def test_it_never_becomes_a_rule_for_the_field_it_was_taken_from(self):
        rules = mp.rules_from([_set_aside("Latest Property Value",
                                          "current_valuation_amount")],
                              client_id="ERE", portfolio_id="direct_001",
                              workflow_id="ONB-2026-0010")
        assert all(r.payload.get("canonical_field") != "current_valuation_amount"
                   for r in rules)

    def test_it_becomes_a_rule_that_it_feeds_nothing(self):
        [rule] = mp.rules_from([_set_aside("Latest Property Value",
                                           "current_valuation_amount")],
                               client_id="ERE", portfolio_id="direct_001",
                               workflow_id="ONB-2026-0010")
        assert mp.is_set_aside(rule)
        assert rule.kind == "field_mapping"
        assert rule.payload["source_column"] == "Latest Property Value"
        assert rule.payload["canonical_field"] == ""
        assert rule.payload["source_files"] == [PROP]

    def test_a_confirmation_beside_it_is_still_a_mapping(self):
        rules = mp.rules_from([
            _set_aside("Latest Property Value", "current_valuation_amount"),
            _confirmed("Latest Valuation", "current_valuation_amount")],
            client_id="ERE", portfolio_id="direct_001", workflow_id="w")
        mapped = [(r.payload["source_column"], r.payload["canonical_field"])
                  for r in rules if not mp.is_set_aside(r)]
        assert mapped == [("Latest Valuation", "current_valuation_amount")]

    def test_a_column_mapped_in_one_file_is_not_set_aside_by_another(self):
        """One rule per column: taking a field away from a column an operator
        mapped would be the worse error of the two."""
        rules = mp.rules_from([
            _set_aside("Loan Interest Rate", "current_interest_rate",
                       source_file=PROP),
            _confirmed("Loan Interest Rate", "current_interest_rate",
                       source_file=LOAN)],
            client_id="ERE", portfolio_id="direct_001", workflow_id="w")
        assert not any(mp.is_set_aside(r) for r in rules)

    def test_the_activation_count_counts_mappings_not_removals(self):
        """The intent tells the operator how many mappings become standing
        rules; a removal is not one."""
        import inspect
        from operations_control.occ_agent import service
        src = inspect.getsource(service.OccAgentService._intent)
        assert "is_set_aside" in src


# --------------------------------------------------------------------------- #
# 2. In a real rule store, the set-aside supersedes the mapping
# --------------------------------------------------------------------------- #
@pytest.fixture()
def rules(tmp_path):
    from apps.blob_trigger_app.storage import Storage
    from operations_control.rules import RuleStore
    from operations_control.stores import OpsLayout, OpsStore
    return RuleStore(OpsStore(Storage(tmp_path / "blob"),
                              OpsLayout("operations-control")))


def _live_mapping(rules, column, field):
    from operations_control.rules import RuleRecord
    return rules.approve(RuleRecord(
        rule_id="", version=0, kind="field_mapping", scope="portfolio",
        client_id="ERE", portfolio_id="direct_001",
        payload={"source_column": column, "canonical_field": field},
        approved_by="operator"))


class TestTheStoreSaysWhatTheOperatorSaid:

    def _promote(self, rules, decisions):
        return mp.promote(rules, decisions, client_id="ERE",
                          portfolio_id="direct_001",
                          workflow_id="ONB-2026-0010")

    def test_the_mapping_is_superseded_by_the_set_aside(self, rules):
        old = _live_mapping(rules, "Latest Property Value",
                            "current_valuation_amount")
        self._promote(rules, [_set_aside("Latest Property Value",
                                         "current_valuation_amount")])
        [current] = [r for r in rules.applicable(client_id="ERE",
                                                 portfolio_id="direct_001")
                     if r.payload.get("source_column") == "Latest Property Value"]
        assert mp.is_set_aside(current)
        assert current.rule_id == old.rule_id and current.version == 2
        history = rules.history("ERE", old.rule_id)
        assert [h.payload.get("canonical_field") for h in history] == [
            "current_valuation_amount", ""]

    def test_promoting_twice_does_not_withdraw_the_set_aside(self, rules):
        """Carrying a case forward re-runs promotion. The set-aside it wrote
        the first time is the column's rule, not a mapping to withdraw."""
        _live_mapping(rules, "Latest Property Value", "current_valuation_amount")
        decisions = [_set_aside("Latest Property Value",
                                "current_valuation_amount")]
        self._promote(rules, decisions)
        self._promote(rules, decisions)
        current = [r for r in rules.applicable(client_id="ERE",
                                               portfolio_id="direct_001")
                   if r.payload.get("source_column") == "Latest Property Value"]
        assert len(current) == 1 and mp.is_set_aside(current[0])

    def test_no_mapping_rule_is_left_for_the_overrides_file(self, rules):
        """`_write_approved_overrides_file` writes only rules with a field."""
        _live_mapping(rules, "Latest Property Value", "current_valuation_amount")
        self._promote(rules, [_set_aside("Latest Property Value",
                                         "current_valuation_amount")])
        with_field = [r for r in rules.applicable(client_id="ERE",
                                                  portfolio_id="direct_001")
                      if r.payload.get("canonical_field")]
        assert with_field == []


# --------------------------------------------------------------------------- #
# 3. The engine reads it back and hands it to Gate 1
# --------------------------------------------------------------------------- #
class TestTheEngineCarriesItToGateOne:

    def test_set_asides_are_read_from_the_governed_rules(self, rules):
        from types import SimpleNamespace
        from operations_control.engine import OpsEngine
        _live_mapping(rules, "Latest Valuation", "current_valuation_amount")
        mp.promote(rules, [_set_aside("Latest Property Value",
                                      "current_valuation_amount")],
                   client_id="ERE", portfolio_id="direct_001", workflow_id="w")
        fake = SimpleNamespace(rules=rules)
        run = SimpleNamespace(client_id="ERE", portfolio_id="direct_001",
                              delivery={})
        assert OpsEngine._set_aside_columns(fake, run) == [
            (PROP, "Latest Property Value")]

    def test_a_set_aside_with_no_file_applies_to_every_file(self, rules):
        from types import SimpleNamespace
        from operations_control.engine import OpsEngine
        rules.approve(mp.set_aside_rule("Month Run", [], {},
                                        client_id="ERE",
                                        portfolio_id="direct_001",
                                        workflow_id="w"))
        run = SimpleNamespace(client_id="ERE", portfolio_id="direct_001",
                              delivery={})
        assert OpsEngine._set_aside_columns(SimpleNamespace(rules=rules),
                                            run) == [("*", "Month Run")]

    def test_the_real_adapter_passes_them_to_the_workflow(self):
        import inspect
        from engine.orchestrator_agent import adapters
        src = inspect.getsource(adapters.RealAgentAdapters)
        assert "set_aside_columns=self.set_aside_columns" in src


# --------------------------------------------------------------------------- #
# 4. Coverage, on ERE-shaped data, stops asking
# --------------------------------------------------------------------------- #
class TestCoverageLeavesThemOut:

    @pytest.mark.parametrize("pairs,kept", [
        ([(PROP, "Latest Property Value")], ["Latest Valuation"]),
        ([("*", "latest  property VALUE")], ["Latest Valuation"]),
        ([(LOAN, "Latest Property Value")],
         ["Latest Valuation", "Latest Property Value"]),
        ([], ["Latest Valuation", "Latest Property Value"]),
    ])
    def test_without_set_asides(self, pairs, kept):
        rows = [{"source_file": PROP, "source_column": "Latest Valuation"},
                {"source_file": PROP, "source_column": "Latest Property Value"}]
        assert [r["source_column"]
                for r in tcov.without_set_asides(rows, pairs)] == kept

    def _decisions(self, set_aside):
        from engine.onboarding_agent.llm_assisted_mapping import \
            run_llm_assisted_mapping
        warnings.simplefilter("ignore")
        n = 20
        loan = pd.DataFrame({
            "Loan Policy Number": [f"L{i}" for i in range(n)],
            "Current Interest Rate": [5.1] * n,
            "Loan Interest Rate": [5.1] * n,
            "Current Balance": [100000.0 + i for i in range(n)]})
        prop = pd.DataFrame({
            "Loan Policy Number": [f"L{i}" for i in range(n)],
            "Latest Valuation": [300000.0 + i for i in range(n)],
            "Latest Property Value": [300000.0 + i for i in range(n)]})
        out = Path(tempfile.mkdtemp())
        run_llm_assisted_mapping(dataframes={LOAN: loan, PROP: prop},
                                 output_dir=str(out), mode="mi_only",
                                 client_id="direct_001", run_id="run",
                                 set_aside_columns=set_aside)
        doc = yaml.safe_load(
            (out / "34_target_first_decisions.yaml").read_text())
        return [(d.get("decision_type"), d.get("target_field"))
                for d in (doc.get("decisions") or [])]

    def test_duplicates_are_asked_about_until_they_are_set_aside(self):
        asked = self._decisions(None)
        assert ("source_priority_confirmation",
                "current_valuation_amount") in asked
        assert ("source_priority_confirmation",
                "current_interest_rate") in asked

    def test_once_set_aside_they_are_not(self):
        asked = self._decisions([(PROP, "Latest Property Value"),
                                 (LOAN, "Loan Interest Rate")])
        assert not [d for d in asked
                    if d[1] in ("current_valuation_amount",
                                "current_interest_rate")]


# --------------------------------------------------------------------------- #
# 5. A case activated before this fix can carry its choices forward
# --------------------------------------------------------------------------- #
class TestAnActivatedCaseCarriesItsChoicesForward:

    def _service(self, rules, *, mode="live", activated="v1"):
        from types import SimpleNamespace
        from operations_control.occ_agent import adapters as _adapters
        audits = []
        run = SimpleNamespace(case_ref="ONB-2026-0010", open_decisions=[
            _set_aside("Latest Property Value", "current_valuation_amount"),
            _confirmed("Latest Valuation", "current_valuation_amount")])
        svc = SimpleNamespace(
            adapter=SimpleNamespace(
                mode=_adapters.MODE_LIVE if mode == "live" else "synthetic",
                engine=SimpleNamespace(rules=rules)),
            facts=lambda _c: SimpleNamespace(client_id="ERE",
                                             portfolio_id="direct_001"),
            _audit=lambda *a, **k: audits.append((a, k)))
        case = SimpleNamespace(run=run,
                               case=SimpleNamespace(activated_version=activated))
        return svc, case, audits

    def test_it_writes_the_set_aside_and_is_audited(self, rules):
        from operations_control.occ_agent.service import OccAgentService
        _live_mapping(rules, "Latest Property Value", "current_valuation_amount")
        svc, case, audits = self._service(rules)
        written = OccAgentService.carry_mappings_forward(svc, case,
                                                         actor="operator")
        assert [w["source_column"] for w in written
                if w.get("set_aside") == "true"] == ["Latest Property Value"]
        assert audits and audits[0][0][1] == "mappings_carried_forward"
        live = {r.payload["source_column"]: r.payload["canonical_field"]
                for r in rules.applicable(client_id="ERE",
                                          portfolio_id="direct_001")}
        assert live == {"Latest Property Value": "",
                        "Latest Valuation": "current_valuation_amount"}

    @pytest.mark.parametrize("mode,activated", [("synthetic", "v1"),
                                                ("live", None)])
    def test_it_is_refused_outside_a_live_activated_case(self, rules, mode,
                                                         activated):
        from operations_control.engine import OpsError
        from operations_control.occ_agent.service import OccAgentService
        svc, case, _ = self._service(rules, mode=mode, activated=activated)
        with pytest.raises(OpsError):
            OccAgentService.carry_mappings_forward(svc, case, actor="operator")
