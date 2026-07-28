"""mi_workflows — the governed analytical workflow layer.

The layer the MI Query Agent architecture reserves ABOVE the Recogniser
Registry (docs/mi_query_agent_architecture.md §11): deterministic, multi-step
analytical workflows grounded in the Business Semantics Registry, dispatched
through the existing recogniser chain and returned through the existing
governed result envelope.

Structure
---------
``semantics``
    The runtime loader for ``config/business_semantics_registry.yaml`` and the
    ``semantics_resolver`` implementation for the seam prepared in
    ``mi_agent_api.dependencies.CapabilityDependencies``.

``engine``
    The shared deterministic calculation engine: governed aggregation
    (sum / average / weighted average / share), governed distributions,
    comparison arithmetic, the mixed-currency guard, unit handling and
    directionality interpretation. Workflow-neutral by design — every governed
    workflow computes through these primitives so two workflows can never
    disagree about what a weighted average or a share is.

``portfolio_risk_comparison``
    The Portfolio Risk Comparison workflow: deterministic comparison of
    multiple governed portfolio scopes at one reporting date.

``concentration_analysis``
    The Concentration Analysis workflow: deterministic measurement of how
    exposure is distributed across governed dimensions (and governed
    single-name identifiers) for one portfolio scope at one reporting date.
    Measures concentration; never judges whether it is acceptable.

Dependency direction: ``mi_agent_api`` (adapters) → ``mi_workflows`` →
``mi_agent`` / ``trakt_core``. Nothing in this package imports a web
framework, performs I/O beyond reading governed registry files, calls an LLM,
or renders output — adapters present, workflows decide.
"""
