#!/usr/bin/env python3
"""tests/test_registry_governance.py

Governance guards for the MI semantic registry:

1. GENERATOR EQUALITY — the checked-in ``mi_semantics_field_registry.yaml`` must
   be exactly what ``build_mi_semantics_registry.py`` produces (hand-edits to
   the generated artifact silently disappear on the next regeneration; the
   ``borrower_type`` dimension was nearly lost this way).
2. SYNONYM UNIQUENESS — no governed business synonym may map to more than one
   field; duplicated synonyms make NL resolution dependent on dict order.
3. METADATA HONESTY — the metadata counts must match the actual entries.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.build_mi_semantics_registry import DEFAULT_SOURCE, build_registry
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def checked_in():
    return load_mi_semantics(_SEMANTICS)


@pytest.fixture(scope="module")
def generated():
    return build_registry(DEFAULT_SOURCE)


def test_checked_in_registry_matches_generator(checked_in, generated):
    """Regenerating the registry must be a no-op (bar the timestamp). A diff
    here means the YAML was hand-edited without updating CURATION — the edit
    WILL be lost on the next regeneration."""
    assert checked_in["fields"] == generated["fields"], (
        "mi_semantics_field_registry.yaml has drifted from "
        "build_mi_semantics_registry.py. Update CURATION in the build script "
        "and regenerate (python -m mi_agent.build_mi_semantics_registry) — "
        "never hand-edit the YAML."
    )
    skip = {"generated_at"}
    meta_a = {k: v for k, v in checked_in["metadata"].items() if k not in skip}
    meta_b = {k: v for k, v in generated["metadata"].items() if k not in skip}
    assert meta_a == meta_b


def test_metadata_counts_match_actual_entries(checked_in):
    m = checked_in["metadata"]
    fields = checked_in["fields"]
    tiers = defaultdict(int)
    derived = virtual = 0
    for entry in fields.values():
        tiers[entry.get("mi_tier")] += 1
        derived += 1 if entry.get("derived") else 0
        virtual += 1 if entry.get("virtual") else 0
    assert m["field_count"] == len(fields)
    assert m["core_field_count"] == tiers["core"]
    assert m["extended_field_count"] == tiers["extended"]
    assert m["derived_field_count"] == derived
    assert m["virtual_field_count"] == virtual


def test_no_synonym_maps_to_two_fields(checked_in):
    owners = defaultdict(list)
    for key, entry in checked_in["fields"].items():
        for syn in entry.get("synonyms") or []:
            owners[str(syn).strip().casefold()].append(key)
    dups = {syn: keys for syn, keys in owners.items() if len(keys) > 1}
    assert not dups, (
        f"duplicated business synonyms (resolution becomes order-dependent): {dups}"
    )


def test_no_synonym_shadows_another_field_key(checked_in):
    """A synonym that equals a DIFFERENT field's key is a resolution trap."""
    keys = set(checked_in["fields"])
    bad = []
    for key, entry in checked_in["fields"].items():
        for syn in entry.get("synonyms") or []:
            norm = str(syn).strip().casefold().replace(" ", "_")
            if norm in keys and norm != key:
                bad.append((key, syn, norm))
    assert not bad, bad


def test_borrower_type_is_generated_not_hand_edited(generated):
    """borrower_type (the single-vs-joint dimension funded prep materialises)
    must come from CURATION so a regeneration cannot delete it."""
    entry = generated["fields"].get("borrower_type")
    assert entry is not None, "borrower_type missing from the generated registry"
    assert entry["role"] == "dimension"
    assert entry["mi_tier"] == "core"
    assert "single vs joint" in [s.lower() for s in entry["synonyms"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
