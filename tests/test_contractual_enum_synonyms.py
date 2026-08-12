#!/usr/bin/env python3
"""Do the contractual analytics and the Annex 2 delivery path agree on enums?

``analytics_lib/contractual.py`` restates the ESMA enum synonyms rather than
reading ``config/regime/annex2_delivery_rules.yaml``, because MI runtime must not
depend on the regulatory delivery chain — see
``tests/test_mi_regulatory_separation.py``.

Restating creates the risk the coupling was there to prevent: a synonym added to
one and not the other. This file removes that risk in the direction that
matters. It lives in ``tests/``, which the separation scan excludes, so the test
may read the delivery configuration even though the module under test may not.

Why the comparison is deliberately one-directional
--------------------------------------------------
Every synonym the delivery path recognises must be recognised by the analytics
too — otherwise a loan Trakt can *deliver* is a loan Trakt cannot *schedule*,
and the gap would show up as quietly missing WAL coverage rather than as an
error. The reverse is allowed: the analytics may recognise labels the delivery
rules do not, because the analytics see source data that has not yet been
mapped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics_lib.contractual import _ENUM_SYNONYMS

DELIVERY_RULES = _REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml"


def _delivery_enum_map(code: str) -> dict:
    """The delivery path's own map for one code, found wherever it is nested."""
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(DELIVERY_RULES.read_text(encoding="utf-8")) or {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == code and isinstance(value, dict):
                    return value
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = walk(item)
                if found is not None:
                    return found
        return None

    entry = walk(doc) or {}
    raw = (entry.get("transform") or {}).get("enum_map") or {}
    return {str(k).strip().lower(): str(v).strip().upper()
            for k, v in raw.items()}


@pytest.mark.skipif(not DELIVERY_RULES.exists(),
                    reason="regulatory delivery configuration not deployed — "
                           "which is exactly the deployment shape the "
                           "analytics must keep working in")
@pytest.mark.parametrize("code", sorted(_ENUM_SYNONYMS))
def test_every_delivered_synonym_is_recognised_by_the_analytics(code):
    delivered = _delivery_enum_map(code)
    if not delivered:
        pytest.skip(f"{code} carries no enum_map in the delivery rules")

    restated = _ENUM_SYNONYMS[code]
    missing = {label: target for label, target in delivered.items()
               if label not in restated}
    assert not missing, (
        f"{code}: the delivery path maps {sorted(missing)} but "
        "analytics_lib.contractual does not. A loan Trakt can deliver but "
        "cannot schedule loses its contractual WAL silently — add the synonym "
        "to _ENUM_SYNONYMS.")

    disagreeing = {label: (target, restated[label])
                   for label, target in delivered.items()
                   if label in restated and restated[label] != target}
    assert not disagreeing, (
        f"{code}: delivery and analytics map the same label to different "
        f"codes: {disagreeing}")


def test_the_analytics_do_not_read_the_regulatory_configuration():
    """The point of restating the map. Asserted here so the reason survives
    the next person who thinks reading the yaml would be tidier."""
    source = (_REPO_ROOT / "analytics_lib" / "contractual.py").read_text(
        encoding="utf-8")
    assert "config/regime/" not in source
    assert "annex2_delivery_rules" not in source


def test_normalisation_works_with_no_regulatory_configuration(monkeypatch):
    """The regression the old design would have failed.

    The previous implementation read the delivery yaml and fell back to an empty
    map on OSError, so a deployment without ``config/regime/`` silently stopped
    recognising "French" and quietly produced fewer schedules. Nothing this
    module does may depend on that file being present.
    """
    from analytics_lib.contractual import (normalise_amortisation,
                                           normalise_rate_type)

    monkeypatch.chdir(Path(__file__).parent)  # no config/regime/ from here
    assert normalise_amortisation("French") == "FRXX"
    assert normalise_amortisation("Bullet") == "BLLT"
    assert normalise_rate_type("Fixed") == "FXRL"
