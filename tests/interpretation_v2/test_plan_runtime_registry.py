"""The population perimeter's single declaration surface — P0 Change 1.

These pin the property the P0 design exists to establish: that "which populations
can this estate execute, and by whom" has one answer, that every runtime states
its own part of it, and that no runtime can be added, or no base added to the
vocabulary, without the tree saying where it goes.

Two of them guard hazards that are easy to reintroduce:

* the GENERIC FUNDED EXECUTOR'S set must never widen. Its comment used to invite
  a future owner to "declare `pipeline` here", and doing so would admit that base
  into funded-only runtimes through the funded population gate.
* every production call to `check_population_base` must name whose declaration it
  checks. The argument defaults to the generic executor's set, so a call without
  it silently checks against funded — the inheritance the design removes.
"""

import ast
import pathlib
import re

import pytest

from mi_agent import plan_runtime_adapter as adapter
from mi_agent import plan_runtime_registry as registry
from mi_agent.interpretation_v2.vocabulary import POPULATION_BASES

_REPO = pathlib.Path(__file__).resolve().parents[2]
_CANARY = _REPO / "mi_agent" / "plan_serving_canary.py"


# --------------------------------------------------------------------------- #
# every runtime declares, uniformly
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("runtime", registry.GOVERNED_RUNTIMES,
                         ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_every_runtime_declares_what_it_executes_and_what_it_loads(runtime):
    assert isinstance(runtime.EXECUTABLE_POPULATIONS, frozenset)
    assert runtime.EXECUTABLE_POPULATIONS, "an empty set executes nothing"
    assert isinstance(runtime.EXECUTION_POPULATION, str)
    # A runtime that loads a population it does not claim to execute would pass
    # its own gate and fail the mismatch guard on every request.
    assert runtime.EXECUTION_POPULATION in runtime.EXECUTABLE_POPULATIONS


@pytest.mark.parametrize("runtime", registry.GOVERNED_RUNTIMES,
                         ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_every_declared_population_is_a_governed_base(runtime):
    assert runtime.EXECUTABLE_POPULATIONS <= POPULATION_BASES


def test_the_old_per_module_names_are_gone():
    """Four names for one idea is what made the set unenumerable. A module that
    grows a second name for its perimeter has reintroduced the problem."""
    for runtime in registry.GOVERNED_RUNTIMES:
        source = pathlib.Path(runtime.__file__).read_text(encoding="utf-8")
        assert not re.search(r"^(SLICE_2_)?POPULATION_BASES\s*=", source, re.M), (
            f"{runtime.__name__} declares its perimeter under an old name")


# --------------------------------------------------------------------------- #
# the estate's set is derivable, and covers the vocabulary exactly
# --------------------------------------------------------------------------- #

def test_every_governed_base_is_executed_or_deliberately_refused():
    """The headline property. A base added to the vocabulary with no owner and no
    explicit refusal fails here, instead of failing at request time as a
    POPULATION_NOT_EXECUTABLE nobody decided on."""
    assert (registry.executable_populations()
            | registry.DELIBERATELY_UNEXECUTED) == POPULATION_BASES


def test_no_base_is_both_executed_and_deliberately_refused():
    assert not (registry.executable_populations()
                & registry.DELIBERATELY_UNEXECUTED)


def test_whole_book_stays_refused_until_limits_are_established():
    assert "whole_book" in registry.DELIBERATELY_UNEXECUTED
    assert "whole_book" not in registry.executable_populations()


# --------------------------------------------------------------------------- #
# the partition around the funded population gate
# --------------------------------------------------------------------------- #

def test_the_funded_gate_admits_exactly_what_its_runtimes_declare():
    assert registry.FUNDED_GATE_POPULATIONS == registry.executable_populations(
        registry.FUNDED_RUNTIMES)
    assert registry.FUNDED_GATE_POPULATIONS == frozenset({"funded"})


def test_a_population_owning_runtime_never_owns_the_funded_book():
    """A runtime above the gate that also executed `funded` would carry funded
    plans past the gate that guards the funded runtimes."""
    for runtime in registry.POPULATION_OWNING_RUNTIMES:
        assert not (runtime.EXECUTABLE_POPULATIONS
                    & registry.FUNDED_GATE_POPULATIONS), runtime.__name__


def test_every_runtime_is_listed_exactly_once():
    names = [m.__name__ for m in registry.GOVERNED_RUNTIMES]
    assert len(names) == len(set(names))


def test_the_generic_executor_speaks_for_the_funded_book_alone():
    """The hazard the corrected comment describes. Widening this set would admit
    a base into funded-only runtimes via the funded gate."""
    assert adapter.EXECUTABLE_POPULATIONS == frozenset({"funded"})
    assert adapter.EXECUTION_POPULATION == "funded"
    assert adapter in registry.FUNDED_RUNTIMES


# --------------------------------------------------------------------------- #
# the canary honours the registry
# --------------------------------------------------------------------------- #

def _canary_alias(module) -> str:
    """The name the canary imports a runtime under."""
    tail = module.__name__.rsplit(".", 1)[-1]
    source = _CANARY.read_text(encoding="utf-8")
    match = re.search(rf"^from mi_agent import {tail} as (\w+)$", source, re.M)
    assert match, f"the canary does not import {tail}"
    return match.group(1)


def test_every_population_owning_runtime_is_dispatched_above_the_funded_gate():
    """Listing a runtime in POPULATION_OWNING_RUNTIMES without a dispatch branch
    above the gate would leave its plans refused by the gate — the exact failure
    this module was written to end."""
    source = _CANARY.read_text(encoding="utf-8")
    gate = source.index("executable=runtime_registry.FUNDED_GATE_POPULATIONS")
    for runtime in registry.POPULATION_OWNING_RUNTIMES:
        assert hasattr(runtime, "claims"), runtime.__name__
        dispatch = source.find(f"{_canary_alias(runtime)}.claims(plan)")
        assert dispatch != -1, f"{runtime.__name__} is never dispatched"
        assert dispatch < gate, (
            f"{runtime.__name__} is dispatched BELOW the funded gate, which "
            f"would refuse its plans POPULATION_NOT_EXECUTABLE")


def _production_sources():
    for root in ("mi_agent", "mi_agent_api"):
        for path in sorted((_REPO / root).rglob("*.py")):
            if "tests" in path.parts:
                continue
            yield path


def test_every_production_population_check_names_its_declaration():
    """`executable` defaults to the generic funded executor's set. A production
    call without it silently checks against funded, which is the inheritance
    the P0 design removes."""
    offenders = []
    for path in _production_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else getattr(func, "id", None))
            if name != "check_population_base":
                continue
            if not any(k.arg == "executable" for k in node.keywords):
                offenders.append(f"{path.relative_to(_REPO)}:{node.lineno}")
    assert not offenders, (
        "check_population_base called without executable=: " + ", ".join(offenders))


def test_the_production_population_checks_exist():
    """Guards the guard: if both call sites vanished, the test above would pass
    vacuously."""
    calls = 0
    for path in _production_sources():
        calls += path.read_text(encoding="utf-8").count("check_population_base(")
    assert calls >= 2
