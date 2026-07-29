"""operations_control.annex2 — governed wrappers around the proven Annex 2
delivery route.

The authoritative components are invoked, never reimplemented:

  * projector   — engine/gate_4_projection/regime_projector.py
                  (via the existing assembler_agent.build_regime_command seam)
  * normaliser  — engine/gate_4b_delivery/annex2_delivery_normalizer.py
  * XML builder — engine/gate_5_delivery/xml_builder_annex2.py
  * XSD         — config/system/DRAFT1auth.099.001.04_1.3.0.xsd (lxml XMLSchema,
                  as executed inside the builder)

The OCC only invokes, tracks, translates, persists, governs and requires
approval. Builder automatic fills are instrumented and surfaced (I4), never
altered.
"""
