"""
engine.delivery_xml_agent
=========================

Trakt Delivery/XML Agent v1.

The stage after Projection. It consumes the Projection package::

    output/projection/50_projection_manifest.json
    output/projection/51_projected_annex2_target_frame.csv
    output/projection/52_projection_field_contract.csv
    output/projection/55_projection_issues.csv
    output/projection/56_projection_blocker_resolution.csv

and produces a governed **delivery package** under ``output/delivery_xml/``
(artefacts 60..64): a delivery-facing view of the projected Annex 2 target frame,
a delivery-readiness report, delivery issues by category, and lineage.

It is a **delivery normalisation / readiness** stage, not (yet) an XML producer.
It does NOT:

  * re-run or mutate Onboarding / Transformation / Validation / Projection;
  * invoke the frozen Gate 5 XML builder or the Gate 4b mutator;
  * silently fill blocked / missing values, or let any XML builder override an
    upstream decision;
  * generate production XML, or claim XML readiness unless every delivery
    readiness gate passes (XML preview is hard-gated behind ``--allow-xml-preview``
    AND the readiness gates).

See ``docs/delivery_xml_agent_v1_review.md`` for the Gate 4b / Gate 5 review.
"""

from engine.delivery_xml_agent.delivery_xml_agent import (  # noqa: F401
    build_delivery_package,
    ProjectionHandoffError,
)

__all__ = ["build_delivery_package", "ProjectionHandoffError"]
