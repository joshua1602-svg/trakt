#!/usr/bin/env python3
"""Sprint 1: the agent tool registry.

The registry's job is to be the ONE place a Trakt capability becomes
agent-callable, and to hold no business logic while doing it. The tests that
matter most are therefore not about the data structure — they are:

* ``test_the_covenant_tool_calls_the_same_implementation_the_ui_calls``
  There is exactly one covenant calculation. The tool reaches it; it does not
  reimplement it.
* ``test_the_projection_carries_every_value_verbatim``
  The tool re-keys the existing envelope and changes no number. A projection
  that rounded, defaulted or recomputed would be a second calculator wearing a
  registry's clothes.
* ``test_importing_the_registry_loads_no_web_framework``
  The catalogue, the OpenAPI generator and a future MCP server all import this;
  none of them should pay for FastAPI or pandas.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import trakt_tools
from trakt_core.context import (
    KNOWN_CAPABILITIES,
    SCOPE_MI_QUERY,
    SCOPE_RISK_READ,
    ExecutionContext,
)
from trakt_core.entitlement import (
    EntitlementStore,
    Grant,
    resolve_entitlements,
)
from trakt_core.resource import KIND_SOURCE_PORTFOLIO, ResourceRef
from trakt_tools import registry as registry_mod
from trakt_tools.schema import SchemaError, assert_supported_schema, object_schema, validate
from trakt_tools.spec import ToolSpec

PORTFOLIO_A = ResourceRef(tenant_id="ERE", kind=KIND_SOURCE_PORTFOLIO,
                          resource_id="direct_001")


# --------------------------------------------------------------------------- #
# The registry contract
# --------------------------------------------------------------------------- #
def test_the_covenant_tool_is_registered_with_the_risk_capability():
    spec = trakt_tools.get("evaluate_covenants")
    assert spec is not None
    assert spec.required_capability == SCOPE_RISK_READ
    assert spec.version
    assert spec.capability_id == "tool.evaluate_covenants"


def test_every_registered_capability_is_in_the_known_vocabulary():
    """The grant verbs and the tool verbs are the SAME set, deliberately.

    A tool requiring a capability that no grant can name would be unreachable;
    one requiring a capability outside the vocabulary would need a translation
    layer, which is where the two sets drift apart.
    """
    for spec in trakt_tools.all_tools():
        assert spec.required_capability in KNOWN_CAPABILITIES


def test_the_covenant_tool_calls_the_same_implementation_the_ui_calls():
    """One covenant calculation, two callers.

    ``GET /mi/concentration-tests`` calls
    ``concentration_tests_api.compute_concentration_tests``. So must the tool —
    otherwise an agent and the workspace could report different limit usage for
    the same book, and the agent's number would be the one a counterparty saw.
    """
    from mi_agent_api import concentration_tests_api as conc_mod
    from trakt_tools.handlers import covenants as handler_mod

    assert callable(conc_mod.compute_concentration_tests)

    handler_source = Path(handler_mod.__file__).read_text(encoding="utf-8")
    assert "conc_mod.compute_concentration_tests" in handler_source

    # The React route reaches the same function. Asserted against the source
    # rather than by importing the app, which would pull the whole plotting
    # stack in for a one-line fact.
    route_source = (_REPO_ROOT / "mi_agent_api" / "app.py").read_text(encoding="utf-8")
    assert "conc_mod.compute_concentration_tests" in route_source

    # And the handler defines no evaluator of its own.
    assert "def compute_concentration_tests" not in handler_source


def test_the_projection_carries_every_value_verbatim():
    """Re-keying is allowed. Recomputing is not.

    Feeds the projection deliberately awkward values — a zero, a ``None``, a
    long float, a negative — and asserts each arrives unchanged. Rounding or
    defaulting any of them would make the tool a second calculator.
    """
    from trakt_tools.handlers.covenants import _project_summary, _project_test

    evaluated = {
        "testId": "T1", "metricId": "share_of_balance", "displayName": "South East",
        "category": "geographic", "currentValue": 0.0, "priorValue": None,
        "threshold": 33.333333333333336, "operator": "<=", "unit": "percent",
        "utilization": 0.0, "headroom": -1.5, "breachAmount": None,
        "status": "pass", "loansInNumerator": 0, "totalLoans": 4821,
        "missingFields": [], "numeratorValue": 0.0, "denominatorValue": 1e9,
        "provenance": {"approvedBy": "A. Operator", "sourceReference": "Schedule 8"},
        "definition": {"numerator": "balance in region"},
        "configurationVersion": 7,
    }
    out = _project_test(evaluated)

    assert out["current_value"] == 0.0 and out["current_value"] is not None
    assert out["prior_value"] is None
    assert out["threshold"] == 33.333333333333336      # not rounded
    assert out["headroom"] == -1.5                     # sign preserved
    assert out["breach_amount"] is None                # NOT defaulted to 0
    assert out["total_loans"] == 4821
    assert out["denominator_value"] == 1e9
    assert out["utilisation"] == 0.0                   # renamed, not recomputed
    assert out["provenance"]["approved_by"] == "A. Operator"
    assert out["definition"]["numerator"] == "balance in region"
    assert out["configuration_version"] == 7

    summary = _project_summary({"overallStatus": "breach", "breaches": 2,
                                "passes": 0, "closestToLimit": None})
    assert summary["overall_status"] == "breach"
    assert summary["breaches"] == 2
    assert summary["passes"] == 0
    assert summary["closest_to_limit"] is None


def test_importing_the_registry_loads_no_web_framework():
    """The catalogue route, the OpenAPI generator and a future MCP server all
    import this. None of them should drag in FastAPI or pandas to read a schema.

    A subprocess, not a module check, because the rest of this suite has already
    imported both by the time it runs.
    """
    code = (
        "import sys; import trakt_tools; "
        "trakt_tools.all_tools(); "
        "loaded = [m for m in ('fastapi', 'starlette', 'pandas') if m in sys.modules]; "
        "print(','.join(loaded))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO_ROOT),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", (
        f"importing trakt_tools loaded {out.stdout.strip()}")


# --------------------------------------------------------------------------- #
# ToolSpec validation — the guards that make a bad tool a startup failure
# --------------------------------------------------------------------------- #
def _minimal_schema(**overrides):
    schema = object_schema(
        description="x",
        properties={"resource": {"type": "string"}},
        required=["resource"])
    schema.update(overrides)
    return schema


def test_a_tool_must_require_a_known_capability():
    with pytest.raises(ValueError, match="KNOWN_CAPABILITIES"):
        ToolSpec(name="t", version="1.0.0", description="d",
                 input_schema=_minimal_schema(), output_schema=_minimal_schema(),
                 required_capability="invented:verb", handler=lambda a, i: {})


def test_a_tool_must_declare_a_required_resource_argument():
    """A tool with no resource cannot be authorised, and a resource that is
    optional could be omitted — which would be authorisation by omission."""
    no_resource = object_schema(description="x", properties={"other": {"type": "string"}})
    with pytest.raises(ValueError, match="no such property"):
        ToolSpec(name="t", version="1.0.0", description="d",
                 input_schema=no_resource, output_schema=_minimal_schema(),
                 required_capability=SCOPE_RISK_READ, handler=lambda a, i: {})

    optional_resource = object_schema(
        description="x", properties={"resource": {"type": "string"}})
    with pytest.raises(ValueError, match="must mark"):
        ToolSpec(name="t", version="1.0.0", description="d",
                 input_schema=optional_resource, output_schema=_minimal_schema(),
                 required_capability=SCOPE_RISK_READ, handler=lambda a, i: {})


def test_a_schema_keyword_the_executor_cannot_enforce_is_rejected():
    """A published constraint the server does not apply is worse than no
    constraint: the contract would promise something untrue."""
    with pytest.raises(ValueError, match="unsupported schema keywords"):
        ToolSpec(name="t", version="1.0.0", description="d",
                 input_schema=_minimal_schema(oneOf=[{"type": "string"}]),
                 output_schema=_minimal_schema(),
                 required_capability=SCOPE_RISK_READ, handler=lambda a, i: {})


def test_a_tool_must_carry_a_version():
    with pytest.raises(ValueError, match="version"):
        ToolSpec(name="t", version="", description="d",
                 input_schema=_minimal_schema(), output_schema=_minimal_schema(),
                 required_capability=SCOPE_RISK_READ, handler=lambda a, i: {})


def test_duplicate_registration_is_refused_not_silently_replaced():
    spec = ToolSpec(name="evaluate_covenants", version="9.9.9", description="d",
                    input_schema=_minimal_schema(), output_schema=_minimal_schema(),
                    required_capability=SCOPE_RISK_READ, handler=lambda a, i: {})
    registry_mod._ensure_loaded()
    with pytest.raises(ValueError, match="already registered"):
        registry_mod.register(spec)


# --------------------------------------------------------------------------- #
# The published catalogue
# --------------------------------------------------------------------------- #
def _context(capabilities, *, organisation="a2a_test_agent"):
    store = EntitlementStore(
        [Grant(organisation_id=organisation, resource_ref=PORTFOLIO_A,
               capabilities=frozenset(capabilities))],
        configured=True, validate=False)
    entitlements = store.resolve(organisation)
    return ExecutionContext(
        tenant_id="ERE", actor_id="agent-1", actor_type="service",
        channel="enterprise_agent",
        scopes=frozenset(capabilities), organisation_id=organisation,
        entitlements=entitlements)


def test_the_catalogue_is_narrowed_to_what_this_caller_can_actually_use():
    """An agent given a tool it will always be refused wastes a call and then
    reasons about a refusal. Telling it only what it can use is the difference
    between a tool list and a capability statement."""
    entitled = trakt_tools.catalogue(_context([SCOPE_RISK_READ]))
    assert [t["name"] for t in entitled["tools"]] == ["evaluate_covenants"]

    # Holding a DIFFERENT capability hides the tool entirely.
    other = trakt_tools.catalogue(_context([SCOPE_MI_QUERY]))
    assert [t["name"] for t in other["tools"]] == []


def test_the_catalogue_publishes_the_closed_set_of_resources():
    """So an agent never has to guess an identifier and be refused."""
    payload = trakt_tools.catalogue(_context([SCOPE_RISK_READ]))
    assert payload["resources"][SCOPE_RISK_READ] == [PORTFOLIO_A.key]
    assert payload["organisation_id"] == "a2a_test_agent"
    assert payload["tenant_id"] == "ERE"


def test_the_catalogue_publishes_full_json_schemas():
    payload = trakt_tools.catalogue(_context([SCOPE_RISK_READ]))
    entry = payload["tools"][0]
    assert entry["input_schema"]["type"] == "object"
    assert "resource" in entry["input_schema"]["properties"]
    assert entry["input_schema"]["required"] == ["resource"]
    assert entry["output_schema"]["type"] == "object"
    assert entry["agent_guidance"]


# --------------------------------------------------------------------------- #
# The schema subset
# --------------------------------------------------------------------------- #
def test_defaults_are_applied_server_side():
    """So an agent may omit an optional argument and still get the documented
    behaviour, and a handler never re-decides a default the schema published."""
    schema = object_schema(
        description="x",
        properties={"resource": {"type": "string"},
                    "category": {"type": ["string", "null"], "default": None},
                    "limit": {"type": "integer", "default": 10}},
        required=["resource"])
    value, errors = validate({"resource": "ERE/portfolio/x"}, schema)
    assert errors == []
    assert value == {"resource": "ERE/portfolio/x", "category": None, "limit": 10}


def test_an_unrecognised_argument_is_reported_not_ignored():
    schema = object_schema(description="x",
                           properties={"resource": {"type": "string"}},
                           required=["resource"])
    _value, errors = validate({"resource": "a/b/c", "sneaky": 1}, schema)
    assert any("sneaky" in e for e in errors)


def test_a_boolean_is_never_accepted_as_a_number():
    """`isinstance(True, int)` is True in Python; a schema saying `integer` must
    not quietly accept `true`."""
    schema = object_schema(description="x",
                           properties={"resource": {"type": "string"},
                                       "n": {"type": "integer"}},
                           required=["resource"])
    _value, errors = validate({"resource": "a/b/c", "n": True}, schema)
    assert any("n:" in e for e in errors)


def test_required_enum_and_pattern_are_all_enforced():
    schema = object_schema(
        description="x",
        properties={"resource": {"type": "string", "pattern": r"^[^/]+/[^/]+/[^/]+$"},
                    "mode": {"type": "string", "enum": ["a", "b"]}},
        required=["resource", "mode"])

    _v, errors = validate({}, schema)
    assert any("resource: required" in e for e in errors)
    assert any("mode: required" in e for e in errors)

    _v, errors = validate({"resource": "not-a-ref", "mode": "z"}, schema)
    assert any("does not match" in e for e in errors)
    assert any("must be one of" in e for e in errors)


def test_assert_supported_schema_rejects_a_subschema_additionalproperties():
    with pytest.raises(SchemaError, match="additionalProperties"):
        assert_supported_schema({"type": "object",
                                 "additionalProperties": {"type": "string"}})
