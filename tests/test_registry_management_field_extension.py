"""A genuinely new management concept can enter the canonical registry.

Phase E of the go-live sprint. When a lender's file carries something useful
that no canonical field means — not a different word for an existing concept,
an actual new one — the answer is neither "force it into the nearest field" nor
"drop it". It becomes a proposed canonical management field, an administrator
approves it, and it follows the normal registry structure.

The risk being tested is the regulatory one: Trakt has complete ESMA Annex
coverage, and a new management field must not disturb it. So this asserts the
shape of the addition and, more importantly, that the regulatory projection
surface is byte-for-byte the same before and after.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from engine.gate_3_validation import validate_canonical as vc

REGISTRY = "config/system/fields_registry.yaml"

#: What an administrator would add for "the lender tracks which of its own
#: servicing teams owns the case" — real management value, no ESMA meaning.
NEW_FIELD = "servicing_team"
NEW_FIELD_SPEC = {
    "allowed_values": None,
    "category": "analytics",
    "description": "Servicing team that owns the case in the lender's own "
                   "operating model.",
    "format": "string",
    "portfolio_type": "common",
    "layer": "core",
    "core_canonical": False,
}


@pytest.fixture(scope="module")
def registry():
    return yaml.safe_load(open(REGISTRY, encoding="utf-8"))


@pytest.fixture(scope="module")
def extended(registry):
    doc = copy.deepcopy(registry)
    doc["fields"][NEW_FIELD] = copy.deepcopy(NEW_FIELD_SPEC)
    return doc


class TestTheProposalHasTheRightShape:
    def test_it_is_management_only(self):
        assert NEW_FIELD_SPEC["category"] == "analytics"
        assert "regime_mapping" not in NEW_FIELD_SPEC, (
            "a management field must not claim a regulatory code")

    def test_it_is_not_promoted_into_the_core_contract_by_default(self):
        """core_canonical: false unless deliberately promoted."""
        assert NEW_FIELD_SPEC["core_canonical"] is False

    def test_it_carries_the_things_a_reader_needs(self):
        for key in ("description", "format", "portfolio_type", "layer"):
            assert NEW_FIELD_SPEC.get(key), f"missing {key}"


class TestItDoesNotDisturbTheRegulatorySurface:
    def test_no_esma_code_gains_a_second_owner(self, extended):
        owners = {}
        for name, spec in extended["fields"].items():
            for regime, mapping in (spec or {}).get("regime_mapping", {}).items():
                code = (mapping or {}).get("code")
                if code:
                    owners.setdefault((regime, code), []).append(name)
        clashes = {k: v for k, v in owners.items() if len(v) > 1}
        assert not clashes, f"a regime code gained a second canonical owner: {clashes}"

    def test_the_annex_field_sets_are_unchanged(self, registry, extended):
        def codes(doc):
            return {regime: sorted(
                (m or {}).get("code", "")
                for s in doc["fields"].values()
                for r, m in (s or {}).get("regime_mapping", {}).items()
                if r == regime)
                for regime in ("ESMA_Annex2", "ESMA_Annex3", "ESMA_Annex8")}
        assert codes(extended) == codes(registry)

    def test_it_adds_nothing_to_the_core_canonical_contract(self, registry,
                                                            extended):
        def core(doc):
            return sorted(n for n, s in doc["fields"].items()
                          if (s or {}).get("core_canonical"))
        assert core(extended) == core(registry)


class TestItBehavesLikeAnyOtherManagementField:
    def test_it_is_in_scope_for_an_equity_release_portfolio(self, extended):
        scoped = vc.select_fields_for_portfolio(extended, "equity_release")
        assert NEW_FIELD in scoped

    def test_an_absent_value_never_blocks(self, extended):
        """A management field is not a gate. Absent means absent."""
        import pandas as pd
        scoped = vc.select_fields_for_portfolio(extended, "equity_release")
        required = vc.get_core_required_fields(scoped)
        assert NEW_FIELD not in required
        df = pd.DataFrame({"loan_identifier": ["L1"]})
        violations = vc.validate_core_presence(
            df, required, extended["fields"], "equity_release")
        assert not [v for v in violations if v.field == NEW_FIELD]
