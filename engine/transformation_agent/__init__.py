"""
engine.transformation_agent
===========================

Trakt Transformation Agent v1.

The deterministic bridge between the Onboarding Agent and the Validation Agent.

It consumes the *governed canonical onboarding handoff package* produced by the
Onboarding Agent::

    output/central/18_central_lender_tape.csv
    output/handoff/24_onboarding_handoff_manifest.json
    output/handoff/26_onboarding_handoff_field_contract.csv / .json
    output/handoff/27_onboarding_handoff_lineage.json

and produces a normalized, validation-ready transformed canonical package under
``output/transformation/`` (artefacts 30..35).

It is NOT a projection or delivery agent. It does NOT:

  * re-run raw Gate 1 canonicalisation on the central tape;
  * perform fuzzy source matching / source discovery;
  * mutate any Onboarding Agent artefact;
  * generate Annex 2 XML input or claim XML readiness.

It reuses the existing deterministic ``engine.gate_2_transform`` logic
(type normalisation, canonical enum normalisation, config defaults) through a
clean adapter (:mod:`engine.transformation_agent.gate2_adapter`) so the Gate 2
engine can consume the onboarding central tape + handoff contract without
assuming raw Gate 1 outputs.
"""

from engine.transformation_agent.transformation_agent import (  # noqa: F401
    build_transformation_package,
    HandoffValidationError,
)

__all__ = ["build_transformation_package", "HandoffValidationError"]
