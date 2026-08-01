"""mi_agent.concentration_tests.library — the governed metric registry.

Loads ``config/risk/concentration_test_library.yaml`` into typed definitions,
validates the registry itself (unique ids, known evaluators, closed
vocabularies) and validates client parameters against a metric's declared
parameter schema. No pandas here; evaluation lives in ``metrics``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .models import OPERATORS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY_PATH = REPO_ROOT / "config" / "risk" / "concentration_test_library.yaml"

CATEGORIES = (
    "geography", "property_value", "loan_balance", "borrower", "rate_product",
    "ltv", "performance", "composition", "external_index", "primitive",
)

IMPLEMENTATION_STATUSES = ("implemented", "interface_only", "not_implemented")

#: Evaluator names that mi_agent.concentration_tests.metrics registers. The
#: registry test asserts every implemented metric names one of these.
KNOWN_EVALUATORS = (
    "share_of_balance", "share_of_balance_numeric", "postcode_area_share",
    "largest_group_share", "distinct_group_count", "weighted_average",
    "field_average", "field_maximum", "field_minimum", "largest_by_field_share",
    "top_n_share", "max_count_per_group", "multi_loan_borrower_share",
    "joint_borrower_share", "arrears_share", "dimension_share",
    "vintage_share", "external_index_ratio", "filtered_share",
)

_PARAM_TYPES = ("number", "integer", "string", "list_string", "enum", "date",
                "boolean", "filter_list")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LibraryError(ValueError):
    """The registry itself is invalid — a deployment problem, not client data."""


@dataclass(frozen=True)
class FieldRole:
    role: str
    candidates: Tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    canonical_name: str
    display_name: str
    category: str
    description: str
    evaluator: Optional[str]
    unit: str
    output_precision: int
    supported_operators: Tuple[str, ...]
    aggregation: str
    numerator: str
    denominator_options: Tuple[str, ...]
    required_roles: Tuple[str, ...]
    optional_roles: Tuple[str, ...]
    parameters: Dict[str, Dict[str, Any]]
    aliases: Tuple[str, ...]
    composable: bool
    implementation_status: str
    version: str
    asset_classes: Tuple[str, ...] = ("cross_asset",)
    owner: str = ""

    @property
    def implemented(self) -> bool:
        return self.implementation_status == "implemented"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "canonical_name": self.canonical_name,
            "display_name": self.display_name,
            "category": self.category,
            "description": self.description,
            "evaluator": self.evaluator,
            "unit": self.unit,
            "output_precision": self.output_precision,
            "supported_operators": list(self.supported_operators),
            "aggregation": self.aggregation,
            "numerator": self.numerator,
            "denominator_options": list(self.denominator_options),
            "required_roles": list(self.required_roles),
            "optional_roles": list(self.optional_roles),
            "parameters": self.parameters,
            "aliases": list(self.aliases),
            "composable": self.composable,
            "implementation_status": self.implementation_status,
            "version": self.version,
            "asset_classes": list(self.asset_classes),
        }


@dataclass
class ConcentrationLibrary:
    library_version: str
    schema_version: str
    owner: str
    field_roles: Dict[str, FieldRole]
    metrics: Dict[str, MetricDefinition] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def get(self, metric_id: str) -> Optional[MetricDefinition]:
        return self.metrics.get(metric_id)

    def all(self) -> List[MetricDefinition]:
        return list(self.metrics.values())

    def implemented(self) -> List[MetricDefinition]:
        return [m for m in self.metrics.values() if m.implemented]

    def by_category(self, category: str) -> List[MetricDefinition]:
        return [m for m in self.metrics.values() if m.category == category]

    def role(self, role: str) -> Optional[FieldRole]:
        return self.field_roles.get(role)

    # ------------------------------------------------------------------ #
    def find_by_alias(self, text: str) -> List[Tuple[MetricDefinition, str]]:
        """Metrics whose alias appears in ``text`` (case-insensitive), longest
        alias first so the most specific phrasing wins."""
        lowered = " ".join(str(text).lower().split())
        hits: List[Tuple[MetricDefinition, str]] = []
        for m in self.metrics.values():
            for alias in m.aliases:
                if alias and alias.lower() in lowered:
                    hits.append((m, alias))
        hits.sort(key=lambda pair: (-len(pair[1]), pair[0].metric_id))
        return hits

    # ------------------------------------------------------------------ #
    def validate_parameters(self, metric_id: str,
                            params: Dict[str, Any]) -> List[str]:
        """Validate client parameters against the metric's schema.

        Returns plain-English problems; an empty list means valid. Unknown
        parameters are errors — a parameter the schema does not declare cannot
        silently steer a calculation.
        """
        metric = self.get(metric_id)
        if metric is None:
            return [f"Unknown metric '{metric_id}'."]
        problems: List[str] = []
        schema = metric.parameters or {}
        for name in params:
            if name not in schema:
                problems.append(
                    f"Parameter '{name}' is not declared by metric "
                    f"'{metric_id}'.")
        for name, spec in schema.items():
            required = bool(spec.get("required"))
            present = name in params and params[name] not in (None, "", [])
            if required and not present:
                problems.append(f"Required parameter '{name}' is missing.")
                continue
            if not present:
                continue
            problems.extend(self._check_value(name, spec, params[name]))
        return problems

    def _check_value(self, name: str, spec: Dict[str, Any],
                     value: Any) -> List[str]:
        ptype = spec.get("type", "string")
        problems: List[str] = []
        if ptype == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"Parameter '{name}' must be a number.")
            else:
                minimum = spec.get("minimum")
                if minimum is not None and value < minimum:
                    problems.append(
                        f"Parameter '{name}' must be at least {minimum}.")
        elif ptype == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                problems.append(f"Parameter '{name}' must be an integer.")
            else:
                minimum = spec.get("minimum")
                if minimum is not None and value < minimum:
                    problems.append(
                        f"Parameter '{name}' must be at least {minimum}.")
        elif ptype == "string":
            if not isinstance(value, str):
                problems.append(f"Parameter '{name}' must be text.")
        elif ptype == "list_string":
            if (not isinstance(value, (list, tuple)) or not value
                    or not all(isinstance(v, str) and v.strip() for v in value)):
                problems.append(
                    f"Parameter '{name}' must be a non-empty list of values.")
        elif ptype == "enum":
            values = spec.get("values") or []
            if value not in values:
                problems.append(
                    f"Parameter '{name}' must be one of {values}, "
                    f"got {value!r}.")
        elif ptype == "date":
            if not isinstance(value, str) or not _DATE_RE.match(value):
                problems.append(
                    f"Parameter '{name}' must be an ISO date (YYYY-MM-DD).")
        elif ptype == "boolean":
            if not isinstance(value, bool):
                problems.append(f"Parameter '{name}' must be true or false.")
        elif ptype == "filter_list":
            problems.extend(self._check_filters(name, value))
        return problems

    def _check_filters(self, name: str, value: Any) -> List[str]:
        """A filter is {role, values} or {role, min and/or max} over a governed
        field role. Anything expression-shaped is rejected."""
        problems: List[str] = []
        if not isinstance(value, (list, tuple)) or not value:
            return [f"Parameter '{name}' must be a non-empty list of filters."]
        for i, f in enumerate(value):
            if not isinstance(f, dict):
                problems.append(f"Filter {i + 1} must be an object.")
                continue
            role = f.get("role")
            if not role or role not in self.field_roles:
                problems.append(
                    f"Filter {i + 1} must name a governed field role "
                    f"(got {role!r}).")
            has_values = isinstance(f.get("values"), (list, tuple)) and f.get("values")
            has_range = ("min" in f) or ("max" in f)
            if not has_values and not has_range:
                problems.append(
                    f"Filter {i + 1} must declare 'values' or a numeric "
                    "'min'/'max' range.")
            extras = set(f) - {"role", "values", "min", "max"}
            if extras:
                problems.append(
                    f"Filter {i + 1} carries unsupported keys {sorted(extras)} "
                    "— formulas and expressions are not accepted.")
        return problems


# --------------------------------------------------------------------------- #
# Loading + registry validation
# --------------------------------------------------------------------------- #
def _tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def validate_registry(doc: Dict[str, Any]) -> List[str]:
    """Registry-level validation, returned as plain-English problems."""
    problems: List[str] = []
    metrics = doc.get("metrics") or []
    roles = doc.get("field_roles") or {}
    seen: set = set()
    seen_aliases: Dict[str, str] = {}
    for m in metrics:
        mid = m.get("metric_id") or ""
        if not mid:
            problems.append("A metric is missing metric_id.")
            continue
        if mid in seen:
            problems.append(f"Duplicate metric_id '{mid}'.")
        seen.add(mid)
        if m.get("category") not in CATEGORIES:
            problems.append(f"{mid}: unknown category {m.get('category')!r}.")
        status = m.get("implementation_status")
        if status not in IMPLEMENTATION_STATUSES:
            problems.append(f"{mid}: unknown implementation_status {status!r}.")
        evaluator = m.get("evaluator")
        if status in ("implemented", "interface_only"):
            if evaluator not in KNOWN_EVALUATORS:
                problems.append(
                    f"{mid}: evaluator {evaluator!r} is not a registered "
                    "deterministic evaluator.")
        ops = _tuple(m.get("supported_operators"))
        if not ops or any(op not in OPERATORS for op in ops):
            problems.append(f"{mid}: supported_operators must be within {OPERATORS}.")
        for role in list(_tuple(m.get("required_roles"))) + list(_tuple(m.get("optional_roles"))):
            if role not in roles:
                problems.append(f"{mid}: undeclared field role '{role}'.")
        params = m.get("parameters")
        if params is None:
            problems.append(f"{mid}: parameters block missing (use {{}} for none).")
        else:
            for pname, spec in (params or {}).items():
                if spec.get("type", "string") not in _PARAM_TYPES:
                    problems.append(
                        f"{mid}: parameter '{pname}' has unknown type "
                        f"{spec.get('type')!r}.")
                if spec.get("type") == "enum" and not spec.get("values"):
                    problems.append(
                        f"{mid}: enum parameter '{pname}' declares no values.")
        for alias in _tuple(m.get("aliases")):
            key = alias.lower()
            if key in seen_aliases and seen_aliases[key] != mid:
                problems.append(
                    f"{mid}: alias '{alias}' already claimed by "
                    f"{seen_aliases[key]}.")
            seen_aliases[key] = mid
    return problems


_CACHE: Dict[str, ConcentrationLibrary] = {}


def load_library(path: Optional[Path] = None, *, use_cache: bool = True
                 ) -> ConcentrationLibrary:
    p = Path(path) if path else DEFAULT_LIBRARY_PATH
    key = str(p)
    if use_cache and key in _CACHE:
        return _CACHE[key]
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    problems = validate_registry(doc)
    if problems:
        raise LibraryError(
            "concentration_test_library.yaml is invalid: " + "; ".join(problems))
    roles = {
        name: FieldRole(role=name, candidates=_tuple(spec.get("candidates")),
                        description=str(spec.get("description") or ""))
        for name, spec in (doc.get("field_roles") or {}).items()
    }
    lib = ConcentrationLibrary(
        library_version=str(doc.get("library_version") or "0"),
        schema_version=str(doc.get("schema_version") or "0"),
        owner=str(doc.get("owner") or ""),
        field_roles=roles,
    )
    for m in doc.get("metrics") or []:
        definition = MetricDefinition(
            metric_id=m["metric_id"],
            canonical_name=str(m.get("canonical_name") or m["metric_id"]),
            display_name=str(m.get("display_name") or m["metric_id"]),
            category=m["category"],
            description=str(m.get("description") or "").strip(),
            evaluator=m.get("evaluator"),
            unit=str(m.get("unit") or "percent"),
            output_precision=int(m.get("output_precision", 2)),
            supported_operators=_tuple(m.get("supported_operators")),
            aggregation=str(m.get("aggregation") or ""),
            numerator=str(m.get("numerator") or ""),
            denominator_options=_tuple(m.get("denominator_options")),
            required_roles=_tuple(m.get("required_roles")),
            optional_roles=_tuple(m.get("optional_roles")),
            parameters=dict(m.get("parameters") or {}),
            aliases=_tuple(m.get("aliases")),
            composable=bool(m.get("composable", False)),
            implementation_status=m["implementation_status"],
            version=str(m.get("version") or "1.0"),
            asset_classes=_tuple(m.get("asset_classes")) or ("cross_asset",),
        )
        lib.metrics[definition.metric_id] = definition
    if use_cache:
        _CACHE[key] = lib
    return lib


def reset_cache() -> None:
    _CACHE.clear()
