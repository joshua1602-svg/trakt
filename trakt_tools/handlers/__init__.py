"""trakt_tools.handlers — the registered tools.

Importing this package registers every tool. Each handler module keeps its heavy
imports (pandas, storage, the analytical engines) *inside* the handler function,
so importing the registry to publish the catalogue or generate the OpenAPI
document costs nothing.

Adding a tool is: write the handler over an EXISTING implementation, declare its
schemas, register it here. If a tool would need new calculation logic, it is not
a tool yet — the calculation belongs in the domain, with the UI and the agent
both calling it.
"""

from __future__ import annotations

from trakt_core.context import SCOPE_LOAN_READ, SCOPE_RISK_READ

from ..registry import register
from ..spec import ToolSpec
from . import covenants as _covenants
from . import loans as _loans
from . import provenance as _provenance

register(ToolSpec(
    name="evaluate_covenants",
    version="1.0.0",
    description=(
        "Evaluate the operator-approved concentration and covenant tests for a "
        "portfolio resource. Returns each test's current value, threshold, "
        "operator, utilisation, headroom, breach amount, status and movement "
        "against the prior governed reporting period, plus the configuration "
        "version and approver that produced them."),
    agent_guidance=(
        "Use this to find out whether a book is within its concentration limits "
        "and covenants, and by how much. Every number is computed by Trakt from "
        "governed data under an operator-approved definition — never estimate or "
        "recompute one yourself. Check 'source': only 'approved_configuration' is "
        "operator-approved."),
    input_schema=_covenants.INPUT_SCHEMA,
    output_schema=_covenants.OUTPUT_SCHEMA,
    required_capability=SCOPE_RISK_READ,
    handler=_covenants.evaluate_covenants,
))

# --------------------------------------------------------------------------- #
# Loan retrieval. The BATCH form is the primitive and the single-loan form is a
# wrapper over it — measured, twenty loans one at a time cost 21x a single
# batched call, and an agent given only a single-loan tool will call it per loan.
# --------------------------------------------------------------------------- #
register(ToolSpec(
    name="get_loans",
    version="1.0.0",
    description=(
        "Retrieve several loans from a portfolio resource in one call. Returns a "
        "flat, typed projection of canonical fields per loan, in the order "
        f"requested, plus an explicit list of identifiers that matched nothing. "
        f"At most {_loans.MAX_LOAN_IDS} loans per call."),
    agent_guidance=(
        "This is the loan-retrieval primitive: to look at more than one loan, "
        "call this ONCE with a list rather than calling get_loan repeatedly — it "
        "is the same work in a single request. Ask for only the fields you need. "
        "For a whole portfolio do not retrieve rows at all: use stratify, "
        "concentration or rank_loans, which return aggregates."),
    input_schema=_loans.GET_LOANS_INPUT,
    output_schema=_loans.GET_LOANS_OUTPUT,
    required_capability=SCOPE_LOAN_READ,
    handler=_loans.get_loans,
))

register(ToolSpec(
    name="get_loan",
    version="1.0.0",
    description=("Retrieve ONE loan. A convenience wrapper over get_loans — same "
                 "implementation, same governed frame."),
    agent_guidance=(
        "Use only for a single loan you are already investigating. For several "
        "loans call get_loans once instead."),
    input_schema=_loans.GET_LOAN_INPUT,
    output_schema=_loans.GET_LOAN_OUTPUT,
    required_capability=SCOPE_LOAN_READ,
    handler=_loans.get_loan,
))

# --------------------------------------------------------------------------- #
# Provenance. Same batch-first shape, same reason.
# --------------------------------------------------------------------------- #
register(ToolSpec(
    name="explain_values",
    version="1.0.0",
    description=(
        "Return the evidence behind several values: the source dataset and field, "
        "snapshot identity and content hash, mapping method and version, any "
        "derivation rule, validation status, and the calculation method where the "
        f"value is calculated. At most {_provenance.MAX_REQUESTS} per call."),
    agent_guidance=(
        "Use this to justify the figures you report. It is the primitive: to "
        "evidence several values call it ONCE with a list. Provenance is bound to "
        "the snapshot the value was read from, so an answer always states which "
        "dataset it came from."),
    input_schema=_provenance.EXPLAIN_VALUES_INPUT,
    output_schema=_provenance.EXPLAIN_VALUES_OUTPUT,
    required_capability=SCOPE_LOAN_READ,
    handler=_provenance.explain_values,
))

register(ToolSpec(
    name="explain_value",
    version="1.0.0",
    description=("Return the evidence behind ONE value. A convenience wrapper "
                 "over explain_values."),
    agent_guidance=(
        "Use only for a single figure. To evidence several, call explain_values "
        "once instead."),
    input_schema=_provenance.EXPLAIN_VALUE_INPUT,
    output_schema=_provenance.EXPLAIN_VALUE_OUTPUT,
    required_capability=SCOPE_LOAN_READ,
    handler=_provenance.explain_value,
))

__all__ = ["_covenants", "_loans", "_provenance"]
