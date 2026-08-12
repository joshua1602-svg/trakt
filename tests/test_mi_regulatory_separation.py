#!/usr/bin/env python3
"""tests/test_mi_regulatory_separation.py

The MI runtime does not depend on the regulatory / Annex 2 / XML delivery chain.

This replaces ``test_no_regulatory_or_annex2_files_modified``, which lived in six
near-identical copies across the phase 3/4/5/6/6c/8a MI test modules. Each ran
``git diff --name-only main`` and failed if any changed path started with
``engine/gate_``, ``config/regime/``, ``config/delivery/`` or contained
"annex2". They were retired for two reasons.

They asserted a property of a DIFF, not of the code. On main the diff is empty
and they pass; on any branch they inspect whatever the local ``main`` ref happens
to point at, which in a fresh clone lags by however many commits were merged
since. Two runs of the same commit can disagree.

And their subject no longer exists. "The Phase 3 change set must not touch
regulatory/Annex 2/XML logic" is a scope statement about one piece of work. Once
that work merged, the assertion silently became "no branch may ever change
regulatory code" — enforced from six MI test files, which is not where anyone
would look for it. A person fixing a Gate 1 defect sees six unrelated MI tests go
red and no explanation of what they did wrong.

What the phases actually promised is asserted here instead, once, as a permanent
property of the source: the MI packages do not import the projection or delivery
chain, and they do not read the regulatory configuration. That holds on every
branch, on a shallow clone, and with no git at all — and unlike the diff check it
would still fail if an MI module genuinely reached into the regulatory chain,
which is the thing worth preventing.

Complements ``test_governance_dependency_direction.py`` (which pins ``trakt_core``
and the ``mi_agent_api`` service modules) rather than repeating it.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Packages that make up the MI runtime the phase work built.
MI_PACKAGES = ("mi_agent", "analytics_lib")

#: The regulatory chain: projection (Gate 4), delivery (Gate 4b/5) and the XML
#: agent. Gates 1-3 (alignment, transform, validation) are NOT here — they are
#: the shared onboarding path, and MI legitimately consumes their output.
FORBIDDEN_MODULE_PREFIXES = (
    "engine.gate_4", "engine.gate_5",
    "engine.delivery_xml_agent", "engine.projection_agent",
)

#: Regulatory configuration an MI module must not read.
FORBIDDEN_CONFIG_PATHS = ("config/regime/", "config/delivery/")

FORBIDDEN_NAME_SUBSTRINGS = ("annex2", "annex_2", "annex12")


def _source_files():
    for pkg in MI_PACKAGES:
        for path in sorted((_REPO_ROOT / pkg).rglob("*.py")):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if "/tests/" in rel or path.name.startswith("test_"):
                continue
            yield rel, path


def _imported_modules(path: Path) -> set:
    """Every module imported anywhere in the file, including inside a function
    body — a deferred import is still a dependency."""
    names: set = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - not expected in a healthy tree
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(("." * node.level) + (node.module or ""))
    return names


class TestMiDoesNotDependOnTheRegulatoryChain(unittest.TestCase):

    def test_the_packages_under_test_exist(self):
        """A typo in MI_PACKAGES would make every assertion below vacuous."""
        for pkg in MI_PACKAGES:
            self.assertTrue((_REPO_ROOT / pkg).is_dir(), pkg)
        self.assertGreater(len(list(_source_files())), 20)

    def test_no_mi_module_imports_projection_or_delivery(self):
        offenders = []
        for rel, path in _source_files():
            for module in _imported_modules(path):
                low = module.lower()
                if any(low.startswith(p) for p in FORBIDDEN_MODULE_PREFIXES) or \
                        any(s in low for s in FORBIDDEN_NAME_SUBSTRINGS):
                    offenders.append(f"{rel} imports {module}")
        self.assertEqual(offenders, [], "MI runtime reached into the "
                                        f"regulatory chain: {offenders}")

    def test_no_mi_module_reads_the_regulatory_configuration(self):
        """Paths only.

        Deliberately NOT a search for the word "annex2" in the text: the MI query
        spec carries ``regulatory_annex2`` as a ROUTE ID, which is how the
        interpreter recognises that route in order to hand it off and how
        ``validate_state_for_route`` refuses an MI state on it. Naming the route
        you decline to serve is separation, not a dependency on it. What must not
        appear is a path into the regulatory configuration.
        """
        offenders = []
        for rel, path in _source_files():
            text = path.read_text(encoding="utf-8")
            for needle in FORBIDDEN_CONFIG_PATHS:
                if needle in text:
                    offenders.append(f"{rel} references {needle}")
        self.assertEqual(offenders, [], "MI runtime referenced regulatory "
                                        f"configuration: {offenders}")

    def test_the_regulatory_route_is_named_but_not_served(self):
        """The route id is present precisely so MI can refuse it — asserted here
        so the exemption above is a stated fact rather than a gap in the scan."""
        from mi_agent.states import validate_state_for_route
        issue = validate_state_for_route("total_funded", "regulatory_annex2")
        self.assertIsNotNone(issue)


if __name__ == "__main__":
    unittest.main()
