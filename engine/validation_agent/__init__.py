"""
engine.validation_agent
=======================

Trakt Validation Agent v1.

The control gate after Transformation. It consumes the Transformation Agent
output package::

    output/transformation/30_transformation_manifest.json
    output/transformation/31_transformed_canonical_tape.csv
    output/transformation/32_transformation_field_contract.csv
    output/transformation/34_transformation_lineage.json
    output/transformation/35_transformation_issues.csv

validates the transformed canonical values + the transformation issue
classifications, and produces a governed validation-readiness package under
``output/validation/`` (artefacts 40..45) for the Projection Agent.

It is a control gate, not a projection or delivery agent. It does NOT:

  * re-run raw Gate 1 canonicalisation or source discovery / fuzzy matching;
  * mutate any Onboarding or Transformation artefact;
  * produce projection output or Annex 2 XML, or claim XML readiness;
  * silently resolve operator decisions or add enum mappings / defaults.

It reuses the deterministic ``engine.gate_3_validation`` primitives through a
clean adapter (:mod:`engine.validation_agent.rules_adapter`).
"""

from engine.validation_agent.validation_agent import (  # noqa: F401
    build_validation_package,
    TransformationHandoffError,
)

__all__ = ["build_validation_package", "TransformationHandoffError"]
