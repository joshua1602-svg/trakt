"""tests/test_assurance_semantics_loading.py — assurance cannot pass on no semantics.

A C6 trace was written against three loader names that do not exist on
`mi_service`. The probe loop caught the AttributeError, fell through to `{}`, and
the instrument measured an EMPTY registry while exiting 0 and printing well-formed
numbers. Two committed instruments carried the same helper.

The failure mode is not "an instrument crashed". It is "an instrument succeeded
and the number was meaningless", which no exit code can distinguish. These tests
make the distinction structural.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from migration_phase0.assurance_semantics import (  # noqa: E402
    REQUIRED_FIELDS, AssuranceSemanticsError, load_assurance_semantics)

#: The loader names the broken helper guessed at. None exists.
GUESSED_NAMES = ("load_semantics", "_load_semantics", "semantics_for")


def test_the_guessed_loader_names_still_do_not_exist():
    """The premise. If any of these ever becomes real, the old helper would
    start working and this test should be revisited deliberately."""
    from mi_agent_api import mi_service
    for name in GUESSED_NAMES:
        assert not callable(getattr(mi_service, name, None)), name


def test_the_assurance_loader_delegates_to_production():
    """No migration-specific notion of 'loaded semantics'."""
    import inspect

    src = inspect.getsource(load_assurance_semantics)
    assert "load_mi_semantics" in src
    assert "semantics_path" in src
    # No fallback of any kind.
    assert "or {}" not in src and "= {}" not in src


def test_real_semantics_load_and_carry_the_required_governed_fields():
    semantics = load_assurance_semantics()
    fields = semantics["fields"]
    for name in REQUIRED_FIELDS:
        assert name in fields, name
    assert len(fields) > 50, len(fields)


def test_a_missing_semantics_file_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("MI_AGENT_SEMANTICS", str(tmp_path / "nope.yaml"))
    with pytest.raises(AssuranceSemanticsError) as exc:
        load_assurance_semantics()
    assert "does not exist" in str(exc.value)


def test_an_empty_registry_fails_loudly(monkeypatch, tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("fields: {}\nmetadata: {}\n", encoding="utf-8")
    monkeypatch.setenv("MI_AGENT_SEMANTICS", str(p))
    with pytest.raises(AssuranceSemanticsError) as exc:
        load_assurance_semantics()
    assert "no 'fields'" in str(exc.value)


def test_a_partially_complete_registry_fails_loudly(monkeypatch, tmp_path):
    """THE case a `len(semantics) > 0` check cannot catch: 117 of 118 fields
    present, and the one missing is the field the instrument measures."""
    import yaml

    from mi_agent_api.data_source import semantics_path

    full = yaml.safe_load(Path(semantics_path()).read_text(encoding="utf-8"))
    full["fields"].pop(REQUIRED_FIELDS[-1], None)
    p = tmp_path / "partial.yaml"
    p.write_text(yaml.safe_dump(full), encoding="utf-8")
    monkeypatch.setenv("MI_AGENT_SEMANTICS", str(p))

    with pytest.raises(AssuranceSemanticsError) as exc:
        load_assurance_semantics()
    assert "materially incomplete" in str(exc.value)
    assert REQUIRED_FIELDS[-1] in str(exc.value)
    assert len(full["fields"]) > 50  # non-empty, and still refused


# --------------------------------------------------------------------------- #
# The estate-wide control: no instrument may reintroduce the pattern
# --------------------------------------------------------------------------- #
def _instrument_sources():
    for path in sorted((_REPO / "migration_phase0").glob("*.py")):
        yield path, path.read_text(encoding="utf-8")


def test_no_assurance_instrument_guesses_a_loader_name():
    offenders = []
    for path, text in _instrument_sources():
        if path.name == "assurance_semantics.py":
            continue  # it names them to prove they are absent
        for name in GUESSED_NAMES:
            if f'"{name}"' in text or f"'{name}'" in text:
                offenders.append(f"{path.name}:{name}")
    assert offenders == [], offenders


def test_every_semantics_dependent_instrument_uses_the_assurance_loader():
    """An instrument that PARSES with semantics must have got them from the one
    loader. Keyed on the production calls that consume a semantics registry, so
    a new instrument cannot opt out by naming its variable something else."""
    # Bare-name consumers, and the one method call, keyed on its RECEIVER.
    # Two false positives were needed to get this right: a substring check
    # flagged an instrument that names these three in a label map, and matching
    # the attribute `parse` alone flagged three that call `ast.parse` — Python's
    # own parser. A guard that fires on the wrong thing is not a stricter guard.
    bare = {"requested_dimension_terms", "detect_requested_facets"}
    offenders = []
    for path, text in _instrument_sources():
        if path.name == "assurance_semantics.py":
            continue
        tree = ast.parse(text)
        calls = False
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            func = n.func
            if isinstance(func, ast.Name) and func.id in bare:
                calls = True
                break
            if (isinstance(func, ast.Attribute) and func.attr == "parse"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "ParsedQuestion"):
                calls = True
                break
            if isinstance(func, ast.Attribute) and func.attr in bare:
                calls = True
                break
        if not calls:
            continue
        if "load_assurance_semantics" in text or "load_mi_semantics" in text:
            continue
        offenders.append(path.name)
    assert offenders == [], (
        "these instruments consume semantics but never load them authoritatively: %s"
        % offenders)


def test_no_instrument_swallows_a_loader_failure_into_an_empty_registry():
    """A broad `except` around the load is how the original defect hid. Flags any
    instrument whose module text pairs a semantics load with a bare fallback."""
    offenders = []
    for path, text in _instrument_sources():
        if path.name == "assurance_semantics.py":
            continue
        if "load_assurance_semantics" not in text and "load_mi_semantics" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            loads = any("load_mi_semantics" in ast.dump(n) or
                        "load_assurance_semantics" in ast.dump(n)
                        for n in ast.walk(node))
            if not loads:
                continue
            for handler in node.handlers:
                for stmt in ast.walk(handler):
                    if isinstance(stmt, (ast.Dict, ast.Constant)) and not isinstance(
                            getattr(stmt, "value", None), str):
                        offenders.append(f"{path.name}:{node.lineno}")
                        break
    assert offenders == [], offenders
