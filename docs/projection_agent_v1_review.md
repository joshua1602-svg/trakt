# Projection Agent v1 — Targeted Review & Adapter Plan

This document is the **required first step** for Projection Agent v1: a review of
the existing frozen Gate 4 / 4b / 5 projection and delivery stack, and a plan for
how (and how much of) it should be reused under the new agentic pipeline.

```text
Onboarding Agent → Transformation Agent → Validation Agent → Projection Agent → Delivery / XML / XSD
                                                              ^^^^^^^^^^^^^^^^  (this PR — v1)
```

Projection Agent v1 is a **review + adapter layer**. The immediate goal is **not**
XML generation. It bridges:

```text
Validation package + transformed canonical tape  →  Projection Agent  →  projected Annex 2 target frame (NOT XML)
```

---

## Frozen components reviewed

| Component | File | Role |
| --- | --- | --- |
| Gate 4 — regime projector | `engine/gate_4_projection/regime_projector.py` | canonical CSV → ESMA-coded "projected" CSV |
| Gate 4b — Annex 2 delivery normaliser | `engine/gate_4b_delivery/annex2_delivery_normalizer.py` | projected CSV → schema-ready "delivery" CSV (precision/regex/boolean/ND preflight) |
| Gate 5 — XML builder Annex 2 | `engine/gate_5_delivery/xml_builder_annex2.py` | delivery CSV → ESMA `auth.099` XML tree |
| Reference artefacts | `DRAFT1auth.099.001.04_*.xsd / .xml / .xlsx` | ESMA Annex 2 XSD, sample XML, workbook — **reference only** |

---

## 1. What does the existing Gate 4 regime projector do?

`regime_projector.py::project_to_regime` takes the **canonical truth set**
(`*_canonical_typed.csv`) and produces a **wide, ESMA-coded** regime projection:

1. selects canonical fields that carry a `regime_mapping` for the target regime
   (read from `fields_registry.yaml`, *not* from `annex2_delivery_rules.yaml`);
2. orders them by ESMA template order (`esma_code_order.yaml`) — fallback is
   mandatory-then-alphabetic;
3. applies enum mappings (canonical enum value → ESMA code) via
   `enum_mapping.yaml` **and** the shared `engine.enum_agent` resolver;
4. applies `nd_defaults` from the **client** config (blank cells only);
5. applies a UK geography override (`country == GB/UK → GBZZZ`);
6. renames canonical columns → ESMA codes (`loan_identifier → LI`, etc.);
7. runs Annex 2 post-projection **guards** (header constants RREL1/RREL6,
   RREC2/RREL3/RREL5 backfill, RREC9 collateral backfill, ScrtstnIdr derivation).

## 2. What inputs does it expect?

* `canonical_typed.csv` — the **old Gate 2** canonical output (positional CLI arg);
* `fields_registry.yaml` with a per-field `regime_mapping.<regime>` block carrying
  `code`, `priority`, `allowed_nd_codes`;
* `enum_mapping.yaml` keyed by regime;
* a **client** config YAML (`config_client_*.yaml`) for `defaults.nd_defaults`,
  `regime_overrides`, `portfolio`;
* `esma_code_order.yaml` for ordering.

## 3. What outputs does it produce?

* `<stem>_<regime>_projected.csv` — wide ESMA-coded frame (one column per code);
* `<stem>_<regime>_projection_report.json` — enum/ND/missing-field report.

## 4. Does it project into canonical names, ESMA codes, XML records, or delivery structures?

Into **ESMA codes in a wide CSV** (one column per code, one row per loan). It does
**not** emit XML records or a delivery structure — but it *does* bake in some
delivery-ish concerns (header constants, ScrtstnIdr generation, RREC backfill)
that arguably belong to a later delivery stage.

## 5. What assumptions does it make about Gate 1/2/3 outputs?

* the input is the **frozen Gate 2** `canonical_typed.csv` (already typed/derived);
* canonical column names match `fields_registry.yaml`;
* the regime field set is declared in `fields_registry.regime_mapping`;
* enum values are already normalised enough for `enum_mapping.yaml` + the
  `enum_agent` reviewer to resolve (it will **raise** on unreviewed enum
  candidates unless `allow_unreviewed=True`);
* ND policy comes from a **client** config file.

## 6. Which assumptions are now outdated under the agentic pipeline?

| Old assumption | New reality |
| --- | --- |
| Input is `canonical_typed.csv` from Gate 2 | Input is `output/transformation/31_transformed_canonical_tape.csv` from the Transformation Agent |
| Regime field set lives in `fields_registry.regime_mapping` | Authoritative regime contract is now `config/regime/annex2_delivery_rules.yaml::field_rules` (the same file Gate 4b consumes), keyed by `projected_source_field` |
| ND/defaults come from a client config | ND/default eligibility is now expressed in `annex2_delivery_rules.yaml` (`nd_allowed`, `default_allowed`, `default_value`) and asset config `product_defaults_ERM.yaml` (`defaults`, `nd_defaults`) |
| Enum resolution may call `enum_agent` and **raise** on unreviewed values | The Projection Agent must be **non-raising / conservative**: unresolved items are carried forward as issues, never guessed |
| Header constants / ScrtstnIdr / RREC backfill happen "in projection" | These are **delivery-normalisation / XML-shaping** concerns and must move to the Delivery/XML Agent — the Projection Agent must not pretend to own them |
| Output is a wide ESMA-coded CSV ready for XML | Output is a **projection package** (long, explicit, auditable target frame) that is explicitly *not* XML and does not claim XML readiness |

## 7. What does Gate 4b Annex 2 delivery normaliser do?

`annex2_delivery_normalizer.py` reads the **projected** CSV and, per field rule in
`annex2_delivery_rules.yaml`, produces a **delivery-ready** CSV:

* `derive` (months-between-dates, first-non-blank);
* `default_value` fill where `default_allowed`;
* `securitisation_id` generation;
* **ND restriction** enforcement (`ND value not allowed for field`);
* boolean → `xsd_lowercase_true_false`;
* `enum_map` / `geography_map` application;
* LEI / `regex` validators;
* numeric **precision** (`total_digits` / `fraction_digits`, half-up rounding);
* a **hard-gate preflight** — any error → non-zero exit.

This is genuinely a **delivery normalisation** stage (XSD-shaped values + preflight),
distinct from projection.

## 8. What does Gate 5 XML builder do?

`xml_builder_annex2.py` walks the **workbook XML path** for each ESMA code and
builds the `auth.099` XML tree with `lxml`: record anchoring (`UndrlygXpsrRcrd`),
ND tag handling (`NODATA*`), choice-branch selection, namespace `urn:esma:xsd:DRAFT1auth.099.001.04`.
It assumes a **delivery-ready** CSV and an XML record structure that may now be
outdated (see §"Record-structure risk" below). **Out of scope for this PR.**

## 9. Which components should be reused immediately?

| Reused now (through the adapter) | Why it is safe |
| --- | --- |
| Gate 4 `load_template_order` + `order_fields_by_template` | pure, deterministic ESMA-code ordering; no enum-agent, no raising |
| The **concept** of `apply_nd_defaults` (blank-only ND fill) | re-implemented in long form against the new config sources, conservative |
| `annex2_delivery_rules.yaml::field_rules` metadata (`projected_source_field`, `nd_allowed`, `default_allowed`, `default_value`, `transform.enum_map`/`geography_map`) | already the authoritative new-model regime contract; Gate 4b reads the same |

## 10. Which components should be deferred to Delivery Agent / XML Agent?

* **All of Gate 5** (XML building, paths, namespaces, ND tags, choice branches).
* Gate 4b **delivery normalisation**: numeric precision, regex/LEI enforcement,
  boolean `xsd_lowercase_true_false`, ND-restriction *hard* preflight,
  derive/securitisation-id generation.
* Gate 4 **post-projection guards**: header constants (RREL1/RREL6),
  ScrtstnIdr derivation, RREC2/RREC9/RREL3/RREL5 backfill — these encode a
  specific XML record shape and belong with delivery.
* Gate 4 **enum_agent** review/raise path.

## 11. What adapter is needed between Validation outputs and the Gate 4 projector?

A thin, non-raising adapter (`engine/projection_agent/gate4_adapter.py`) that:

1. reads the **new** regime contract from `annex2_delivery_rules.yaml::field_rules`
   (keyed by `projected_source_field`) into a rich projection index, instead of
   `fields_registry.regime_mapping`;
2. reuses the frozen Gate 4 ordering primitives to order fields by the
   `esma_code_order.yaml` `Record:` list;
3. applies **only explicit, safe** value transforms (`enum_map`, `geography_map`)
   and ND/default fills — never the enum-agent resolver, never guessed mappings;
4. consumes the transformed canonical tape (`31_*`), the validation issues
   (`43_*`) and the projection blocker diagnostics (`46_*`) rather than a Gate 2
   `canonical_typed.csv`.

## 12. Minimum Projection Agent v1 that is safe and useful

Consume the Validation package → emit a **long, explicit, auditable Annex 2 target
frame** (one row per loan × ESMA field) plus projection readiness, issues and a
**blocker-resolution** report that shows whether each validation projection-blocker
was reduced or carried forward. Resolve only:

* `materialised_projection_pending` (project the transformed value, + safe enum map);
* `nd_or_default_rule_pending` (apply an *allowed* ND/default);
* `source_mapping_pending` **only** where an explicit regime rule exists.

Carry forward operator/config dependencies, unresolved source mappings, and
delivery/XML structure issues. **Never** generate XML; **never** claim XML readiness.

---

## Specific review points (frozen-code behaviour)

| Concern | Where it lives today | v1 decision |
| --- | --- | --- |
| Annex 2 **code ordering** | Gate 4 `order_fields_by_template` + `esma_code_order.yaml` (`Record:` list) | **reuse** for target-frame ordering |
| **RREL / RREC split** | Implicit in code prefixes; Gate 5 anchors `UndrlygXpsrRcrd` | v1 records `record_group` (RREL=loan/exposure, RREC=collateral) as a column; does **not** build nested records |
| **Field naming** | Gate 4 renames canonical → ESMA code | v1 keeps **both** `canonical_field` and `esma_code` columns (auditable, no lossy rename) |
| **ND values** | `nd_allowed` per rule; client config `nd_defaults`; Gate 4b enforces restriction | v1 applies ND **only** where `nd_allowed`/config allows; sets `nd_applied`; does not hard-fail |
| **Default values** | `default_allowed`/`default_value`; asset `defaults`/`nd_defaults` | v1 applies explicit configured defaults only; sets `default_applied`; never invents one |
| **Enum normalization** | Gate 4 `enum_mapping.yaml` + enum_agent; Gate 4b `transform.enum_map` | v1 uses **only** explicit `transform.enum_map`/`geography_map` from the regime rule; unmapped → carried as issue |
| **Date formatting** | Gate 4b/Gate 5 (XSD) | **deferred** to delivery |
| **Numeric formatting / precision** | Gate 4b `apply_precision` | **deferred** to delivery |
| **XML path / record structure** | Gate 5 workbook PATH | **deferred** to XML agent |
| **Multiple record groups** | Gate 5 record anchoring | **deferred**; v1 tags `record_group` only |
| **Collateral vs loan records** | RREL vs RREC | tagged via `record_group`; not split into nested records |
| **XSD-driven field constraints** | XSD + Gate 4b validators | **deferred** to delivery/XSD validation |

### Record-structure risk (called out per the task)

The frozen Gate 5 builder assumes a **single flat record per row** anchored at
`UndrlygXpsrRcrd`, with header-level fields (RREL1/RREL6) forced to pool-level
constants and collateral (RREC) fields living on the *same* row as the loan
(RREL) fields. Under the agentic model the canonical tape is one row per
underlying exposure, but ESMA `auth.099` genuinely nests **collateral records**
under each exposure and carries **pool/header** fields once. **v1 deliberately
does not freeze this shape.** It emits a long frame tagging each value with its
`record_group` so the Delivery/XML Agent can later decide whether to retain or
restructure the old flat-record shape. This PR **identifies** the question; it
does **not** solve it.

---

## What remains for the Delivery / XML Agent (explicit hand-off)

1. Delivery normalisation (Gate 4b): precision, regex/LEI, boolean XSD casing,
   derive/securitisation-id, ND-restriction hard preflight.
2. Annex 2 record shaping: header/pool constants, RREL↔RREC nesting, choice
   branches, ND tag mapping.
3. XML building (Gate 5) against `auth.099` + XSD validation.
4. Deciding whether the legacy flat-record shape is retained or restructured.

`ready_for_xml_delivery` is **always `false`** in this PR.

</invoke>
