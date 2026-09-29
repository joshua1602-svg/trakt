"""A capability's SEMANTIC MODEL: one file declaring what each of its figures
is, and where the owner publishes it (P0 design §16.2).

`config/mi/semantic_model/<capability>.yaml` holds, per measure, the definition
the interpreter reads, the path of the figure in the owner's published output
and the POPULATION the figure is measured over; per dimension, its definition
and governed values. The vocabulary reads
the first half, the capability's runtime the second — one entry, so what the
model is told a figure is and what executes for it cannot drift apart.

This module loads and VALIDATES the file and READS A PATH. It never computes:
a path is a sequence of keys, a member is a named path, a row is found by its
key. The owner computed every number it returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

_ROOT = Path(__file__).resolve().parents[1].joinpath("config", "mi", "semantic_model")

#: Units a figure may be stated in. Presentation reads these; nothing converts.
UNITS = frozenset({"gbp", "count", "gbp_per_month", "gbp_per_year", "month"})

#: Dimensions a measure may be broken down by without declaring them in its own
#: file: the governed registry dimensions and region, which every capability
#: shares (the owner publishes the breakdown; the model already knows the name).
SHARED_DIMENSIONS = frozenset({"canonical_region_reporting", "ltv_bucket"})


class SemanticModelError(ValueError):
    """The file does not describe a servable model; the message says why."""


@dataclass(frozen=True)
class View:
    name: str
    label: str
    owner: str
    inputs: Mapping[str, Mapping[str, str]]
    available_when: Mapping[str, str]


@dataclass(frozen=True)
class Dimension:
    name: str
    definition: str
    values: Tuple[str, ...]


@dataclass(frozen=True)
class Measure:
    name: str
    label: str
    unit: str
    view: str
    definition: str
    operations: Tuple[str, ...]
    periods: Tuple[str, ...]
    grains: Tuple[str, ...] = ()
    value: Optional[str] = None
    inputs: Tuple[str, ...] = ()
    context: Mapping[str, str] = field(default_factory=dict)
    explain: str = ""
    series: Mapping[str, str] = field(default_factory=dict)
    by: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    decision: str = ""
    #: The population the figure is measured over — the one a plan asking for
    #: it must name, and the one its receipt proves. The capability's own
    #: population unless the file says otherwise (see `_validate`).
    population: str = ""


@dataclass(frozen=True)
class SemanticModel:
    capability: str
    population: str
    views: Mapping[str, View]
    dimensions: Mapping[str, Dimension]
    measures: Mapping[str, Measure]

    def measure(self, name: str) -> Optional[Measure]:
        return self.measures.get(name)

    def populations(self) -> Tuple[str, ...]:
        """Every population a figure in this model is measured over, the
        capability's own first."""
        others = sorted({m.population for m in self.measures.values()}
                        .difference({self.population}))
        return (self.population, *others)


# --------------------------------------------------------------------------- #
# reading the owner's output
# --------------------------------------------------------------------------- #

_MISSING = object()


def read(payload: Any, path: str, default: Any = None) -> Any:
    """The value at a dotted `path` in an owner's output, or `default`.

    Keys only — `forecastBridge.fundedBalance` is `payload["forecastBridge"]
    ["fundedBalance"]`. A missing key is `default`, never a guess.
    """
    node: Any = payload
    for key in str(path).split("."):
        if not isinstance(node, Mapping):
            return default
        node = node.get(key, _MISSING)
        if node is _MISSING:
            return default
    return node


def has(payload: Any, path: str) -> bool:
    """Does the owner's output carry `path` at all?"""
    return read(payload, path, _MISSING) is not _MISSING


# --------------------------------------------------------------------------- #
# loading and validating
# --------------------------------------------------------------------------- #

def _tuple(raw: Any) -> Tuple[str, ...]:
    return tuple(str(x) for x in (raw or ()))


def _text(raw: Any) -> str:
    return " ".join(str(raw or "").split())


def _validate(doc: Mapping[str, Any], capability: str) -> SemanticModel:
    if int(doc.get("version") or 0) != 1:
        raise SemanticModelError(f"{capability}: unsupported version {doc.get('version')!r}")
    if str(doc.get("capability") or "") != capability:
        raise SemanticModelError(
            f"{capability}: the file declares capability {doc.get('capability')!r}")
    home = str(doc.get("population") or "")
    if not home:
        raise SemanticModelError(f"{capability}: the file declares no population")

    views: Dict[str, View] = {}
    for name, row in (doc.get("views") or {}).items():
        if not row.get("owner"):
            raise SemanticModelError(f"view {name!r} names no owner")
        views[name] = View(name=name, label=_text(row.get("label")) or name,
                           owner=str(row["owner"]),
                           inputs=dict(row.get("inputs") or {}),
                           available_when=dict(row.get("available_when") or {}))

    dimensions: Dict[str, Dimension] = {}
    for name, row in (doc.get("dimensions") or {}).items():
        definition = _text(row.get("definition"))
        if not definition:
            raise SemanticModelError(f"dimension {name!r} has no definition")
        dimensions[name] = Dimension(name=name, definition=definition,
                                     values=_tuple(row.get("values")))

    measures: Dict[str, Measure] = {}
    for name, row in (doc.get("measures") or {}).items():
        where = f"measure {name!r}"
        definition = _text(row.get("definition"))
        if not definition:
            raise SemanticModelError(f"{where} has no definition")
        view = str(row.get("view") or "")
        if view not in views:
            raise SemanticModelError(f"{where} reads undeclared view {view!r}")
        unit = str(row.get("unit") or "")
        if unit not in UNITS:
            raise SemanticModelError(f"{where} has unit {unit!r}, not one of {sorted(UNITS)}")
        inputs = _tuple(row.get("inputs"))
        unknown = set(inputs).difference(views[view].inputs)
        if unknown:
            raise SemanticModelError(f"{where} uses inputs {sorted(unknown)} its view does not declare")
        by = dict(row.get("by") or {})
        for dim, binding in by.items():
            if dim not in dimensions and dim not in SHARED_DIMENSIONS:
                raise SemanticModelError(f"{where} is broken down by undeclared {dim!r}")
            shapes = [k for k in ("members", "rows", "map", "columns") if k in binding]
            if len(shapes) != 1:
                raise SemanticModelError(
                    f"{where} by {dim!r} must be ONE of members/rows/map/columns, not {shapes}")
            governed = set(dimensions[dim].values) if dim in dimensions else set()
            named = set(binding.get("members") or ()).union(binding.get("columns") or ())
            outside = named.difference(governed)
            if governed and outside:
                raise SemanticModelError(
                    f"{where} by {dim!r} names {sorted(outside)} outside its "
                    f"governed values {sorted(governed)}")
        if not (row.get("value") or row.get("series") or by):
            raise SemanticModelError(f"{where} publishes no value, series or breakdown")
        # A figure measured over ANOTHER population than the capability's is one
        # of the view's dated inputs, read alone: the pipeline's exclusions from
        # forecast weighting are a property of the pipeline extract, not of the
        # forecast that composes it. Anything else would let a receipt claim a
        # population the figure was not measured on.
        population = str(row.get("population") or home)
        if population != home and (population not in views[view].inputs
                                   or inputs != (population,)):
            raise SemanticModelError(
                f"{where} is measured over {population!r}, so it must be one of "
                f"its view's inputs and read that input alone (inputs: {list(inputs)})")
        measures[name] = Measure(
            name=name, label=_text(row.get("label")) or name, unit=unit, view=view,
            definition=definition, operations=_tuple(row.get("operations")),
            periods=_tuple(row.get("periods")), grains=_tuple(row.get("grains")),
            value=row.get("value"), inputs=inputs,
            context=dict(row.get("context") or {}), explain=_text(row.get("explain")),
            series=dict(row.get("series") or {}), by=by,
            decision=str(row.get("decision") or ""), population=population)
    return SemanticModel(capability=capability, population=home, views=views,
                         dimensions=dimensions, measures=measures)


@lru_cache(maxsize=8)
def load(capability: str) -> SemanticModel:
    """The validated semantic model for `capability`. Raises if it is invalid:
    a model that cannot be served must not be shown to the interpreter."""
    path = _ROOT.joinpath(f"{capability}.yaml")
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _validate(doc, capability)
