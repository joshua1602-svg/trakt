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
from . import analysis as _analysis
from . import covenants as _covenants
from . import history as _history
from . import loans as _loans
from . import movement as _movement
from . import provenance as _provenance
from . import readiness as _readiness

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

# --------------------------------------------------------------------------- #
# Aggregate analysis. These exist so an agent stops answering population
# questions by retrieving the population: an agent given only get_loans will
# compute a weighted-average LTV in its own context window, and that answer has
# no governed definition, no provenance and no reproducibility. Every one of
# these wraps an EXISTING deterministic implementation — analytics_lib for the
# maths, the concentration and period-change engines for the rest.
# --------------------------------------------------------------------------- #
register(ToolSpec(
    name="portfolio_summary",
    version="1.0.0",
    description=(
        "The headline shape of a portfolio in one call: loan count, total "
        "balance, balance-weighted average LTV and interest rate, arrears, and "
        "the composition by status, IFRS 9 stage, region and collateral type."),
    agent_guidance=(
        "Start here for any book you have not seen. It answers most first "
        "questions in a single call, and tells you which dimensions are worth "
        "stratifying. Never compute a portfolio average by retrieving loans."),
    input_schema=_analysis.PORTFOLIO_SUMMARY_INPUT,
    output_schema=_analysis.PORTFOLIO_SUMMARY_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_analysis.portfolio_summary,
))

register(ToolSpec(
    name="stratify",
    version="1.0.0",
    description=(
        "Break a portfolio down by one canonical dimension: loan count, balance, "
        "balance share and optional balance-weighted averages per category, "
        "ordered by balance."),
    agent_guidance=(
        "Use this for any 'by region', 'by product', 'by stage' question. It is "
        f"bounded to {_analysis.MAX_GROUPS} categories and says so when it "
        "truncates. Do not retrieve loans and group them yourself — the answer "
        "would be yours rather than Trakt's."),
    input_schema=_analysis.STRATIFY_INPUT,
    output_schema=_analysis.STRATIFY_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_analysis.stratify,
))

register(ToolSpec(
    name="concentration",
    version="1.0.0",
    description=(
        "How concentrated a portfolio is in one dimension: the combined balance "
        "and count share of the top-N groups, with the contributing groups."),
    agent_guidance=(
        "This reports the measured SHARE, not a pass or fail. Whether a share "
        "breaches a limit is an operator-approved covenant test — call "
        "evaluate_covenants for that, which carries the approved threshold, its "
        "approver and its configuration version."),
    input_schema=_analysis.CONCENTRATION_INPUT,
    output_schema=_analysis.CONCENTRATION_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_analysis.concentration,
))

register(ToolSpec(
    name="rank_loans",
    version="1.0.0",
    description=(
        "The N loans at the top or bottom of a metric, as identifiers and the "
        "ranked value — without retrieving the population. Reports how much of "
        f"the book carries the metric at all. At most {_analysis.MAX_RANK} loans."),
    agent_guidance=(
        "This is how you find the loans worth looking at. Rank first, then call "
        "get_loans ONCE with the identifiers you want to investigate. Never "
        "retrieve a population to sort it yourself."),
    input_schema=_analysis.RANK_LOANS_INPUT,
    output_schema=_analysis.RANK_LOANS_OUTPUT,
    required_capability=SCOPE_LOAN_READ,
    handler=_analysis.rank_loans,
))

register(ToolSpec(
    name="data_completeness",
    version="1.0.0",
    description=(
        "How populated each canonical field is across a portfolio: populated "
        "count, missing count and completeness percentage, worst first."),
    agent_guidance=(
        "The first diligence question about any tape: what is actually there. "
        "Use 'max_completeness_pct' to see only the gaps. A field being absent "
        "from the tape is different from being present and empty — this "
        "distinguishes them."),
    input_schema=_analysis.DATA_COMPLETENESS_INPUT,
    output_schema=_analysis.DATA_COMPLETENESS_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_analysis.data_completeness,
))

register(ToolSpec(
    name="list_validation_exceptions",
    version="1.0.0",
    description=(
        "Which canonical fields failed or warned in this snapshot's validation, "
        "with the rules that fired and the error counts. Field-level, which is "
        "the grain validation is recorded at."),
    agent_guidance=(
        "Read 'lineage_available' first. When it is false, an empty list means "
        "the validation outcome is UNKNOWN for this snapshot — it does not mean "
        "the tape is clean."),
    input_schema=_analysis.LIST_VALIDATION_EXCEPTIONS_INPUT,
    output_schema=_analysis.LIST_VALIDATION_EXCEPTIONS_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_analysis.list_validation_exceptions,
))

# --------------------------------------------------------------------------- #
# The two follow-up questions every governed figure provokes: which loans, and
# has it moved. Both wrap the engines the React workspace already calls.
# --------------------------------------------------------------------------- #
register(ToolSpec(
    name="covenant_drillthrough",
    version="1.0.0",
    description=(
        "The loans contributing to one covenant or concentration test's "
        "numerator, with the numerator and denominator they reconcile to."),
    agent_guidance=(
        "Call this after evaluate_covenants reports a breach, using the same "
        "test_id and the same as_of_run_id. Check 'reconciles': when it is "
        "false, the population and the reported figure disagree and you must not "
        "attribute the figure to these loans. Do not reconstruct a covenant's "
        "population yourself — it is defined by a registered metric, not by a "
        "field comparison."),
    input_schema=_movement.DRILLTHROUGH_INPUT,
    output_schema=_movement.DRILLTHROUGH_OUTPUT,
    required_capability=SCOPE_LOAN_READ,
    handler=_movement.covenant_drillthrough,
))

register(ToolSpec(
    name="period_change",
    version="1.0.0",
    description=(
        "What changed between two governed reporting periods: metric movements, "
        "distribution shifts and the balance bridge, with the period resolution "
        "that says why the two periods are comparable."),
    agent_guidance=(
        "Use this for any 'has it got worse', 'since last quarter' or 'what "
        "moved' question. Trakt resolves which periods are comparable — do NOT "
        "compare a figure from one call against one you remember from another, "
        "because the population may have changed between them. Read "
        "'limitations' before quoting a movement."),
    input_schema=_movement.PERIOD_CHANGE_INPUT,
    output_schema=_movement.PERIOD_CHANGE_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_movement.period_change,
))

# --------------------------------------------------------------------------- #
# Securitisation readiness. The framework is published rather than implied: an
# agent that does not know what a readiness review covers will invent a
# checklist, and the invented parts are the ones nobody can reproduce.
#
# The load-bearing distinction across all five is between a FACT Trakt measured,
# an EXTERNAL CRITERION somebody supplied, and a TRAKT SCREENING THRESHOLD that
# only means "look at this". Every result carries the authority it came from.
# --------------------------------------------------------------------------- #
register(ToolSpec(
    name="readiness_framework",
    version="1.0.0",
    description=(
        "The Securitisation Readiness Framework: every metric a first-pass "
        "review assesses, which tool supplies each fact, which are only "
        "meaningful against supplied criteria, and Trakt's deterministic "
        "coverage of the framework overall and by category."),
    agent_guidance=(
        "Call this FIRST when assessing readiness — it tells you what to "
        "investigate and what Trakt can actually measure, so you neither invent "
        "a checklist nor assume a gap. Read each metric's "
        "'agent_investigation_guidance' and 'type': MEASURE_ONLY means report "
        "the number and form no view; EXTERNAL_CRITERIA means it is only "
        "meaningful against a supplied rulebook."),
    input_schema=_readiness.FRAMEWORK_INPUT,
    output_schema=_readiness.FRAMEWORK_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_readiness.readiness_framework,
))

register(ToolSpec(
    name="readiness_metrics",
    version="1.0.0",
    description=(
        "Measure framework facts using the governed concentration-test metric "
        "library — the same 39 registered evaluators the Risk Limits workspace "
        "uses. Batch-first."),
    agent_guidance=(
        "Ask for several metrics in ONE call. Each result names its "
        "calculation_source and its numerator and denominator, so you can cite "
        "how a figure was produced. A metric declared in the library but "
        "without an evaluator returns unavailable — report that, never an "
        "estimate."),
    input_schema=_readiness.METRICS_INPUT,
    output_schema=_readiness.METRICS_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_readiness.readiness_metrics,
))

register(ToolSpec(
    name="valuation_age_profile",
    version="1.0.0",
    description=(
        "How old and how good the collateral evidence is: balance-weighted "
        "valuation age distribution, physical/indexed method mix, and the share "
        "of balance with no usable valuation."),
    agent_guidance=(
        "Every LTV in the book rests on these observations, so read this before "
        "quoting any LTV metric. A stale valuation makes the reported LTV "
        "unreliable rather than making the loan bad — that is a data-quality "
        "finding, and it calls for different remediation from a credit one. "
        "Loans with no valuation are silently absent from every LTV figure."),
    input_schema=_readiness.VALUATION_AGE_INPUT,
    output_schema=_readiness.VALUATION_AGE_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_readiness.valuation_age_profile,
))

register(ToolSpec(
    name="regulatory_readiness",
    version="1.0.0",
    description=(
        "Whether a regulatory submission could be produced from this tape: "
        "coverage of the regime's required loan-level fields, separating fields "
        "that may NOT report a No-Data value (hard blockers) from those that "
        "may."),
    agent_guidance=(
        "Read 'blocking_gaps' first — those fields admit no No-Data fallback, "
        "so no valid submission exists until they are resolved. Fields "
        "permitting ND1-ND4 weaken the disclosure without blocking it. Never "
        "let a clean regulatory result imply a sound portfolio, or the "
        "reverse: they are different questions."),
    input_schema=_readiness.REGULATORY_INPUT,
    output_schema=_readiness.REGULATORY_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_readiness.regulatory_readiness,
))

register(ToolSpec(
    name="evaluate_rule_packs",
    version="1.0.0",
    description=(
        "Apply every configured rulebook to the SAME measured facts: warehouse "
        "criteria, Trakt's internal screening thresholds, and any supplied "
        "securitisation criteria. Each fact is computed once and every pack "
        "reads it."),
    agent_guidance=(
        "Check 'authority' on every result before reporting it. An external "
        "pack returns pass/breach and is a real requirement; Trakt's screening "
        "pack returns clear/flag and means only 'worth investigating'. NEVER "
        "describe a screening flag as a breach, and never add the two counts "
        "together. If no securitisation criteria pack is configured, the "
        "assessment cannot conclude the book meets them — say so."),
    input_schema=_readiness.RULE_PACKS_INPUT,
    output_schema=_readiness.RULE_PACKS_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_readiness.evaluate_rule_packs,
))

# --------------------------------------------------------------------------- #
# Observed historical behaviour. Four SHAPES of question rather than one tool
# per metric, plus transition_analysis wrapping the Risk Monitor's existing
# migration matrix. Every calculation lives in analytics_lib so React and MI
# Query reach the same arithmetic.
# --------------------------------------------------------------------------- #
register(ToolSpec(
    name="portfolio_history",
    version="1.0.0",
    description=(
        "How named measures have moved across governed snapshots — balance, "
        "loan count, weighted-average LTV, arrears at 30/60/90 days, defaults, "
        "high-LTV share, largest-region share and stale-valuation share. One "
        "call covers every measure and every period."),
    agent_guidance=(
        "Use this for any 'how has X changed' question. Ask for several "
        "measures at once — it costs one pass per snapshot, not one per "
        "measure. There is no expression language: ask for a named measure. A "
        "single-period window means no movement can be observed, which is an "
        "absence of history rather than evidence of stability."),
    input_schema=_history.HISTORY_INPUT,
    output_schema=_history.HISTORY_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_history.portfolio_history,
))

register(ToolSpec(
    name="prepayment_analysis",
    version="1.0.0",
    description=(
        "Observed prepayment and redemption: single-monthly mortality and its "
        "annualised form, from the canonical unscheduled-principal field, with "
        "per-period detail."),
    agent_guidance=(
        "The rate is measured from unscheduled principal and full redemptions, "
        "over the OPENING balance of each period. Scheduled amortisation, "
        "maturity and default-related reduction are excluded — they are not "
        "prepayment. Never infer a rate from the change in total balance, and "
        "if this returns unavailable, report that rather than estimating."),
    input_schema=_history.PREPAYMENT_INPUT,
    output_schema=_history.PREPAYMENT_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_history.prepayment_analysis,
))

register(ToolSpec(
    name="loss_analysis",
    version="1.0.0",
    description=(
        "Observed cumulative and period losses and recoveries, reported on "
        "every defensible denominator: original balance, opening balance, "
        "current balance, and recoveries over losses."),
    agent_guidance=(
        "Several loss denominators are legitimate and each answers a different "
        "question, so all are returned and none is chosen for you. State which "
        "one you are quoting. Losses come from the governed field and are never "
        "inferred from balance movement."),
    input_schema=_history.LOSS_INPUT,
    output_schema=_history.LOSS_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_history.loss_analysis,
))

register(ToolSpec(
    name="cohort_comparison",
    version="1.0.0",
    description=(
        "One cohort against the REST of the portfolio on the same measures — "
        "by vintage, region, LTV band or any canonical field. Bounded to two "
        "groups."),
    agent_guidance=(
        "This is how a flagged cohort becomes a finding. 'The high-LTV cohort "
        "has 4% arrears' is a number; '4.0% against 1.2% for the rest of the "
        "book' is something a reviewer can act on. Always compare before "
        "concluding, and never retrieve loans to do it yourself."),
    input_schema=_history.COHORT_INPUT,
    output_schema=_history.COHORT_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_history.cohort_comparison,
))

register(ToolSpec(
    name="transition_analysis",
    version="1.0.0",
    description=(
        "How loans moved between states across two governed snapshots: "
        "deterioration, cures, entries and exits. Wraps the Risk Monitor's "
        "existing migration matrix."),
    agent_guidance=(
        "Use this to tell a rising arrears figure caused by NEW delinquency "
        "from one caused by existing cases worsening — they call for different "
        "responses. Cures appear as improving transitions. A loan absent from "
        "one snapshot is an entry or an exit, not a state change."),
    input_schema=_history.TRANSITION_INPUT,
    output_schema=_history.TRANSITION_OUTPUT,
    required_capability=SCOPE_RISK_READ,
    handler=_history.transition_analysis,
))

__all__ = ["_analysis", "_covenants", "_history", "_loans", "_movement",
           "_provenance", "_readiness"]
