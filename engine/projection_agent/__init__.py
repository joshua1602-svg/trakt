"""
engine.projection_agent
========================

Trakt Projection Agent v1.

The stage after Validation. It consumes the Validation Agent output package::

    output/validation/40_validation_manifest.json
    output/validation/41_validation_results.csv
    output/validation/42_validation_readiness.json
    output/validation/43_validation_issues.csv
    output/validation/44_validation_lineage.json
    output/validation/46_projection_blocker_diagnostics.csv
    output/transformation/31_transformed_canonical_tape.csv
    output/transformation/32_transformation_field_contract.csv
    output/transformation/34_transformation_lineage.json
    output/handoff/24_onboarding_handoff_manifest.json

and produces a governed **projection package** under ``output/projection/``
(artefacts 50..56) — a long, explicit Annex 2 *target frame* plus projection
readiness, issues and a blocker-resolution report.

It is a **projection** stage, not a delivery or XML agent. It does NOT:

  * re-run Gate 1 / Transformation / Validation, or mutate their artefacts;
  * invoke the frozen Gate 5 XML builder or produce any XML;
  * run Gate 4b delivery normalisation (precision / regex / boolean-XSD / preflight);
  * claim XML readiness, or silently resolve operator / config decisions or
    invent ND-values / defaults / source mappings.

It reuses the frozen Gate 4 ESMA-code ordering primitives and the authoritative
``annex2_delivery_rules.yaml`` regime contract through a clean, non-raising
adapter (:mod:`engine.projection_agent.gate4_adapter`).

See ``docs/projection_agent_v1_review.md`` for the Gate 4 / 4b / 5 review.
"""

from engine.projection_agent.projection_agent import (  # noqa: F401
    build_projection_package,
    ValidationHandoffError,
)

__all__ = ["build_projection_package", "ValidationHandoffError"]
