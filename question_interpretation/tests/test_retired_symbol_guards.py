"""Retired symbols, checked STRUCTURALLY — and the guard against checking them
by substring.

Four times in this programme a guard of the form "module X must not mention Y"
has fired on a MENTION rather than a USE:

  1. an estate-wide "no route re-reads the question" guard flagged a label map;
  2. the same guard flagged `ast.parse` while looking for `ParsedQuestion.parse`;
  3. the C6 filter guard failed against its own docstring, which explains what
     it replaced;
  4. the C6 INDEPENDENT AUDIT failed against three docstrings recording that
     `_FUNNEL_KEYWORDS` had been removed.

The fourth is the instructive one: the audit was checking compliance with the
rule it was breaking. A comment explaining why a symbol went is the opposite of
a problem — it is the record the next reader needs — so a guard that punishes it
trains people to delete the explanation.

These tests fix the rule in the estate rather than in each author's memory: a
retired symbol is one the AST no longer BINDS or READS, and prose about it is
always allowed.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Set, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Symbols removed by a conversion, and the module that used to own each.
#: Adding a row here is how a conversion records what it retired.
RETIRED: Tuple[Tuple[str, str], ...] = (
    ("_FUNNEL_KEYWORDS", "mi_agent_api/chat_routing.py"),
    ("_mask", "mi_agent/population.py"),
)

#: Modules that must not IMPORT a retired symbol from anywhere. Production is
#: not enough: C6 retired `_FUNNEL_KEYWORDS` and the estate stayed green
#: because its last consumer was an ASSURANCE INSTRUMENT, whose ImportError
#: only surfaced two tests deep. An instrument that cannot import is an
#: instrument that cannot measure.
CONSUMER_ROOTS: Tuple[str, ...] = ("migration_phase0", "mi_agent", "mi_agent_api",
                                   "mi_workflows", "question_interpretation")


def _bound_and_read(path: Path) -> Tuple[Set[str], Set[str]]:
    """Every name the module BINDS, and every name it READS. Docstrings and
    comments contribute neither, which is the whole point."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound: Set[str] = set()
    read: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            read.add(node.id)
        elif isinstance(node, ast.Attribute):
            read.add(node.attr)
    return bound, read


@pytest.mark.parametrize("symbol,module", RETIRED)
def test_a_retired_symbol_is_neither_bound_nor_read(symbol, module):
    bound, read = _bound_and_read(_REPO_ROOT / module)
    assert symbol not in bound, f"{module} still defines {symbol}"
    assert symbol not in read, f"{module} still uses {symbol}"


@pytest.mark.parametrize("symbol,module", RETIRED)
def test_prose_about_a_retired_symbol_is_allowed(symbol, module):
    """The control on the control.

    A retired symbol SHOULD still be named in the comment explaining why it
    went. If this ever starts failing, someone has replaced the structural check
    with a substring one — again."""
    text = (_REPO_ROOT / module).read_text(encoding="utf-8")
    bound, read = _bound_and_read(_REPO_ROOT / module)
    if symbol in text:
        assert symbol not in bound | read, (
            f"{symbol} appears in {module} as CODE, not only as prose")


def test_no_estate_guard_checks_a_symbol_by_substring():
    """The assurance instruments must not regress to `"name" in source`.

    Scoped to the guards that assert about symbols, because a substring test is
    perfectly reasonable elsewhere — the failure mode is specifically using one
    to decide whether a symbol is still in use.
    """
    audit = _REPO_ROOT / "migration_phase0" / "c6_independent_audit.py"
    if not audit.exists():
        pytest.skip("the C6 audit is not present")
    tree = ast.parse(audit.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        # `"_SOMETHING" not in source` / `in source` where the left side is a
        # dunder-ish private symbol name and the right is a whole-file string.
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(
                node.ops[0], (ast.In, ast.NotIn)):
            left, right = node.left, node.comparators[0]
            if (isinstance(left, ast.Constant) and isinstance(left.value, str)
                    and left.value.startswith("_")
                    and isinstance(right, ast.Name)
                    and right.id in ("source", "text", "body", "src")):
                offenders.append(left.value)
    assert not offenders, (
        f"substring symbol checks found: {offenders}. Use the AST: a symbol is "
        "retired when nothing binds or reads it, and prose about it is fine.")


@pytest.mark.parametrize("symbol,module", RETIRED)
def test_nothing_anywhere_still_imports_a_retired_symbol(symbol, module):
    """The consumer sweep, across instruments as well as production.

    `_FUNNEL_KEYWORDS` was deleted and every production module was clean, yet
    `migration_phase0/route_ownership_evolution.py` still did
    `from mi_agent_api.chat_routing import _FUNNEL_KEYWORDS`. The full suite
    caught it only through two unrelated-looking failures in
    test_assurance_measurement_failure.py.
    """
    offenders = []
    for root in CONSUMER_ROOTS:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                        a.name == symbol for a in node.names):
                    offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, f"{symbol} is still imported by: {sorted(set(offenders))}"


def test_an_assurance_instrument_restores_the_environment_it_repoints():
    """State leaked between measurements is a measurement fault.

    `route_ownership_evolution` repoints MI_AGENT_PIPELINE_ROOT at the five-week
    fixture. It used to leave it repointed, so a later test read the fixture AS
    production and reported that production had acquired weekly extracts.
    """
    path = _REPO_ROOT / "migration_phase0" / "route_ownership_evolution.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    sets = {n.slice.value for n in ast.walk(tree)
            if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
            and isinstance(n.value, ast.Attribute) and n.value.attr == "environ"
            and isinstance(n.slice, ast.Constant)}
    assert sets, "the instrument no longer repoints anything — update this control"
    source = path.read_text(encoding="utf-8")
    assert "_saved_env" in source and "os.environ.pop" in source, (
        f"{sorted(sets)} are set but never restored")
