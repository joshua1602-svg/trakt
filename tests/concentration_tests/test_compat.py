"""Controlled migration of the legacy limit stores: everything becomes a
PROPOSAL awaiting human approval — never a silently active control."""

from __future__ import annotations

from mi_agent.concentration_tests.compat import (
    migrate_legacy_defaults,
    proposals_from_extracted_config,
    proposals_from_legacy_limits,
)
from mi_agent.concentration_tests.models import (
    MATCH_MATCHED,
    MATCH_UNSUPPORTED,
    PROPOSAL_PENDING_APPROVAL,
)


class TestLegacyPythonDict:
    def test_the_shipped_legacy_config_converts(self):
        from config.client.risk_limits_config import ALL_LIMITS
        proposals = proposals_from_legacy_limits(ALL_LIMITS, "client_a")
        assert len(proposals) == len(ALL_LIMITS)
        by_metric = {}
        for p in proposals:
            by_metric.setdefault(p.proposed_metric_id, []).append(p)
        # The regional grid maps onto ONE library metric with parameters —
        # not twelve bespoke calculators.
        assert len(by_metric.get("geo_region_share", [])) == 12
        assert "borrower_largest_share" in by_metric
        assert "borrower_max_loans" in by_metric
        assert "rate_type_share" in by_metric
        # Nothing arrives active.
        for p in proposals:
            assert p.status != "approved"

    def test_region_wrapper_ids_map_to_readable_labels(self):
        proposals = proposals_from_legacy_limits(
            {"max_region_ukh_pct": {"limit_value": 10.0, "direction": "max",
                                    "description": "East Anglia limit"}},
            "client_a")
        assert proposals[0].parameters == {"regions": ["East Anglia"]}
        assert proposals[0].match_outcome == MATCH_MATCHED

    def test_unmapped_limits_become_implementation_requests(self):
        proposals = proposals_from_legacy_limits(
            {"max_exotic_thing": {"limit_value": 1.0, "direction": "max",
                                  "description": "Something bespoke"}},
            "client_a")
        assert proposals[0].match_outcome == MATCH_UNSUPPORTED


class TestExtractedYaml:
    def test_the_shipped_extracted_config_converts(self):
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(Path(
            "config/clients/client_001/risk_limits_extracted.yaml"
        ).read_text(encoding="utf-8"))
        proposals = proposals_from_extracted_config(cfg, "client_001")
        assert len(proposals) == len(cfg["limits"])
        # Source wording travels as evidence.
        assert all(p.evidence.source_text for p in proposals)
        # The needs_review joint-borrower limit stays ambiguous.
        joint = next(p for p in proposals
                     if p.proposed_metric_id == "borrower_joint_share")
        assert joint.status != PROPOSAL_PENDING_APPROVAL


class TestMigrationIsGoverned:
    def test_migration_persists_proposals_and_is_idempotent(self, service,
                                                            store, lib):
        limits = {"max_region_ukh_pct": {"limit_value": 10.0,
                                         "direction": "max",
                                         "description": "East Anglia"}}
        first = migrate_legacy_defaults("client_a", store, lib, limits=limits)
        assert len(first) == 1
        second = migrate_legacy_defaults("client_a", store, lib, limits=limits)
        assert second == []  # already proposed — never duplicated
        # Migrated proposals still need the human gate before activation.
        import pytest
        from operations_control.concentration import ConcentrationError
        with pytest.raises(ConcentrationError):
            service.activate("client_a", actor="Alice")
