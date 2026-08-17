#!/usr/bin/env python3
"""tests/test_concentration_dimension_fallback.py — the P6 fallback guard.

``_dimension_role`` now falls back to a field role of the SAME NAME when the
explicit table has no entry. That is what lets a new composition dimension be
added as configuration rather than as code — and it means 36 declared role
names became resolvable that previously returned ``None``.

A dimension that previously returned ``None`` produced an *unavailable* test. If
one of those names appears as a ``dimension`` in a live configuration, that
test could start MEASURING, and a client's RAG state could move with no
configuration change and no operator approval.

No committed configuration is affected today — the audit that established this
is recorded in the commit that introduced the fallback. This file keeps it that
way: a future
configuration that uses one of these names as a governed ``dimension`` and
reaches metric parameters fails here rather than changing a RAG state quietly.

Run: python -m pytest tests/test_concentration_dimension_fallback.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mi_agent.concentration_tests import compat  # noqa: E402
from mi_agent.concentration_tests.library import load_library  # noqa: E402
from mi_agent.concentration_tests.metrics import (  # noqa: E402
    _DIMENSION_ROLES,
    _dimension_role,
)

#: Trees that never carry a governed concentration configuration.
# "out/" is gitignored run output, not configuration. Scanning it made this test
# fail on whatever a developer had last generated there — an MI harness
# transcript that merely MENTIONS a dimension is not a configuration using one.
_SKIP_PREFIXES = ("demo-video", "frontend", "landing-page", ".git",
                  "node_modules", "out/")


def _newly_resolvable() -> list:
    lib = yaml.safe_load(
        (_REPO / "config" / "risk" / "concentration_test_library.yaml")
        .read_text(encoding="utf-8"))
    return sorted(r for r in (lib.get("field_roles") or {})
                  if r not in _DIMENSION_ROLES)


def _candidate_configs():
    for path in _REPO.rglob("*"):
        if path.suffix not in (".yaml", ".yml", ".json") or not path.is_file():
            continue
        rel = path.relative_to(_REPO).as_posix()
        if any(rel.startswith(p) for p in _SKIP_PREFIXES):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "dimension" not in text:
            continue
        try:
            data = (yaml.safe_load(text) if path.suffix in (".yaml", ".yml")
                    else json.loads(text))
        except Exception:  # noqa: BLE001 - unparseable config is not our concern
            continue
        yield path, data


def _walk_dimensions(node):
    if isinstance(node, dict):
        value = node.get("dimension")
        if isinstance(value, str):
            yield node, value
        for child in node.values():
            yield from _walk_dimensions(child)
    elif isinstance(node, list):
        for child in node:
            yield from _walk_dimensions(child)


class TestFallbackIsASupersetNotAReplacement(unittest.TestCase):
    """Every pre-existing dimension must behave exactly as before."""

    @classmethod
    def setUpClass(cls):
        cls.lib = load_library()

    def test_the_explicit_table_is_consulted_first(self):
        for dimension, expected in _DIMENSION_ROLES.items():
            self.assertEqual(_dimension_role(self.lib, dimension), expected,
                             f"{dimension} changed meaning")

    def test_an_undeclared_name_still_resolves_to_nothing(self):
        self.assertIsNone(_dimension_role(self.lib, "not_a_governed_role"))
        self.assertIsNone(_dimension_role(self.lib, ""))
        self.assertIsNone(_dimension_role(self.lib, None))

    def test_a_declared_role_resolves_to_itself(self):
        self.assertEqual(_dimension_role(self.lib, "manufacturer"), "manufacturer")


class TestNoCommittedConfigurationIsAffected(unittest.TestCase):
    """The audit, executable.

    If this fails, a configuration has started using one of the newly
    resolvable names as a governed dimension. That is not automatically wrong —
    but it must be an explicit, approved decision, not a silent RAG change.
    """

    @classmethod
    def setUpClass(cls):
        cls.newly = set(_newly_resolvable())

    def test_the_audited_scope_is_still_thirty_six_names(self):
        self.assertEqual(len(self.newly), 36,
                         "the newly-resolvable set changed; re-audit which "
                         "configurations could now resolve a dimension that "
                         "previously returned None")

    def test_no_configuration_feeds_a_newly_resolvable_dimension_to_a_metric(self):
        offenders = []
        for path, data in _candidate_configs():
            for node, dimension in _walk_dimensions(data):
                if dimension not in self.newly:
                    continue
                rel = path.relative_to(_REPO).as_posix()
                # The one known hit is an EXTRACTED limit awaiting approval; it
                # is allowed, and the test below pins why it is harmless.
                if rel == "config/clients/client_001/risk_limits_extracted.yaml":
                    continue
                offenders.append((rel, dimension,
                                  node.get("limit_id") or node.get("test_id")))
        self.assertEqual(offenders, [],
                         "a configuration now uses a newly-resolvable dimension; "
                         "confirm the RAG impact before allowing it — a test "
                         "that was 'unavailable' may now report a breach")

    def test_the_known_hit_cannot_reach_metric_parameters(self):
        """`borrower_structure` in the extracted limits is inert."""
        config = yaml.safe_load(
            (_REPO / "config" / "clients" / "client_001"
             / "risk_limits_extracted.yaml").read_text(encoding="utf-8"))
        proposals = compat.proposals_from_extracted_config(
            config=config, client_id="audit")
        target = None
        for proposal in proposals:
            data = (dataclasses.asdict(proposal)
                    if dataclasses.is_dataclass(proposal) else vars(proposal))
            if "joint_borrower" in json.dumps(data, default=str):
                target = data
                break
        self.assertIsNotNone(target, "the audited limit is no longer produced")
        # The dimension never reaches the metric...
        self.assertNotIn("dimension", target.get("parameters") or {})
        # ...and the proposal cannot evaluate without an operator approving it.
        self.assertEqual(target.get("status"), "pending_confirmation")
        self.assertEqual(target.get("match_outcome"), "ambiguous")

    def test_the_extracted_limit_has_no_threshold_to_compare_against(self):
        config = yaml.safe_load(
            (_REPO / "config" / "clients" / "client_001"
             / "risk_limits_extracted.yaml").read_text(encoding="utf-8"))
        limit = next(l for l in config["limits"]
                     if l.get("dimension") == "borrower_structure")
        self.assertIsNone(limit.get("limit_value"))
        self.assertTrue(limit.get("needs_review"))


class TestCompatNeverForwardsAnExtractedDimension(unittest.TestCase):
    """The conversion path reads category, not dimension."""

    def test_the_only_dimension_compat_sets_is_a_pre_existing_one(self):
        source = (_REPO / "mi_agent" / "concentration_tests" / "compat.py").read_text(
            encoding="utf-8")
        import re
        emitted = set(re.findall(r'"dimension":\s*"([a-z_]+)"', source))
        self.assertTrue(emitted <= set(_DIMENSION_ROLES),
                        f"compat emits dimensions outside the explicit table: "
                        f"{sorted(emitted - set(_DIMENSION_ROLES))}")

    def test_compat_does_not_read_the_extracted_dimension_key(self):
        source = (_REPO / "mi_agent" / "concentration_tests" / "compat.py").read_text(
            encoding="utf-8")
        self.assertNotIn('lim.get("dimension")', source)
        self.assertNotIn('lim["dimension"]', source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
