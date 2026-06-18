# Phase 0 — Risk Field Registry & Analytics Alias Changes

**Status:** Implemented (config/data only). Adds the generic, asset-class-neutral
canonical risk fields and analytics aliases required by the target MI risk
monitor and M&A diligence risk views, per the Phase 0 scope of the target
architecture and build plan. (That target architecture document —
`docs/mi_mna_target_architecture_and_build_plan.md` — is tracked separately and
will be merged in a separate PR; it is intentionally not included in this
Phase 0-only PR.)

**Date:** 2026-06-18

**Scope guardrails honoured:**
- No source code refactored. No charts migrated. No agents or snapshot layer built.
- Only two files changed: `config/system/fields_registry.yaml` and
  `config/system/aliases_analytics.yaml`.
- Existing bank/SME-specific fields **retained** (source fidelity); they now
  **alias into** the new generic canonical fields rather than being deleted or
  overloaded.

---

## 1. New risk fields added to the canonical field registry

Added to `config/system/fields_registry.yaml` (registry now 493 fields). All are
`category: analytics`, `portfolio_type: common` (asset-class neutral),
`core_canonical: false` (risk data is supplemental, not present on every tape;
surfaced to MI/M&A via the `include_unmapped_analytics: true` consumers, not as
blocking mandatory fields). No `regime_mapping` (these are not ESMA regulatory
fields), consistent with the existing `bank_internal_*` analytics fields.

| Field | format | layer | Purpose |
|---|---|---|---|
| `internal_risk_grade` | list | performance | Generic internal/obligor risk grade (e.g. A–G, 1–10). |
| `internal_risk_score` | decimal | performance | Numeric internal/behavioural/application score. |
| `internal_risk_stage` | list | performance | Internal monitoring/watchlist stage (distinct from IFRS 9). |
| `ifrs9_stage` | list | performance | IFRS 9 impairment stage (Stage 1/2/3). |
| `probability_of_default` | decimal | performance | PD as decimal (0–1). |
| `pd_bucket` | list | performance | Banded PD (derived). |
| `loss_given_default` | decimal | performance | LGD as decimal (0–1). |
| `lgd_bucket` | list | performance | Banded LGD (derived). |
| `exposure_at_default` | decimal | performance | EAD currency amount. |
| `ead_bucket` | list | performance | Banded EAD (derived). |
| `risk_grade_previous` | list | performance | Risk grade at prior period (migration input). |
| `risk_grade_current` | list | performance | Risk grade at current period (migration input). |
| `pd_previous` | decimal | performance | PD at prior period (migration input). |
| `pd_current` | decimal | performance | PD at current period (migration input). |
| `lgd_previous` | decimal | performance | LGD at prior period (migration input). |
| `lgd_current` | decimal | performance | LGD at current period (migration input). |
| `ifrs9_stage_previous` | list | performance | IFRS 9 stage at prior period (migration input). |
| `ifrs9_stage_current` | list | performance | IFRS 9 stage at current period (migration input). |
| `rating_review_date` | date | product | Date of last internal rating/credit review. |
| `risk_model_version` | string | product | Version of the model producing grade/PD/LGD. |
| `risk_model_source` | string | product | Source/system/provider of the risk model. |

**Total: 21 new canonical risk fields.**

---

## 2. Existing fields retained (unchanged)

The following existing fields were **kept as-is** for source/regulatory fidelity.
They are bank/SME-specific and are now treated as *source labels* that map into
the new generic fields (§3), not as the generic risk model:

- `bank_internal_rating` (analytics, sme)
- `bank_internal_rating_prior_to_default` (analytics, sme)
- `bank_internal_loss_given_default_lgd_estimate` (analytics, sme)
- `bank_internal_loss_given_default_lgd_estimate_down_turn` (analytics, sme)
- `last_internal_obligor_rating_review` (analytics, sme)
- `corporate_guarantor_bank_internal_1_year_probability_default` (analytics, common)
- `corporate_guarantor_last_internal_rating_review` (analytics, common)

> Per the brief, these were **not** treated as sufficient for the MI/M&A risk
> model and were **not** overloaded as the generic canonical model. No existing
> field definitions were modified.

---

## 3. Existing fields mapped/aliased into new canonical fields

The superseded bank/SME-specific concepts now map into the generic fields via
human-readable aliases in `aliases_analytics.yaml`, so source tapes using those
labels resolve to the generic canonical model:

| Legacy / source concept | Maps to new canonical field | Alias added |
|---|---|---|
| `bank_internal_rating` | `internal_risk_grade` | "bank internal rating" |
| `bank_internal_loss_given_default_lgd_estimate` | `loss_given_default` | "bank internal lgd estimate", "internal lgd estimate", "lgd estimate" |
| `last_internal_obligor_rating_review` | `rating_review_date` | "last internal obligor rating review", "last internal rating review" |

> `corporate_guarantor_*` fields were intentionally left independent — they are a
> distinct guarantor-level concept, not the loan-level obligor risk model.

---

## 4. Analytics aliases added

Added to `config/system/aliases_analytics.yaml` (the **analytics** alias tier —
not the mandatory/regulatory alias files). **98 aliases across 21 canonical
fields.** Highlights (full list in the file):

- **internal_risk_grade** — internal rating · internal risk grade · risk grade ·
  credit grade · credit rating · obligor rating · borrower rating · account
  rating · loan rating · bank internal rating
- **internal_risk_score** — risk score · credit score · internal score ·
  behavioural score · application score · scorecard score
- **internal_risk_stage** — risk stage · internal stage · credit stage ·
  monitoring stage · watchlist stage
- **ifrs9_stage** — IFRS9 stage · IFRS 9 stage · stage · impairment stage ·
  credit impairment stage
- **probability_of_default** — PD · probability of default · default probability ·
  lifetime PD · 12 month PD · 12m PD · one year PD · point in time PD ·
  through the cycle PD
- **loss_given_default** — LGD · loss given default · lgd estimate · recovery
  assumption · recovery rate · loss severity · bank internal lgd estimate ·
  internal lgd estimate
- **exposure_at_default** — EAD · exposure at default · default exposure ·
  exposure amount · outstanding exposure
- **rating_review_date** — rating review date · last rating review · last internal
  rating review · last internal obligor rating review · last obligor rating
  review · credit review date · risk review date
- Plus sensible aliases for the derived buckets (`pd_bucket`, `lgd_bucket`,
  `ead_bucket`), the migration pairs (`*_previous` / `*_current`), and the model
  metadata (`risk_model_version`, `risk_model_source`).

> Note: per the brief's suggested list, broad aliases such as bare **"stage"**
> (→ `ifrs9_stage`) and **"recovery rate"** (→ `loss_given_default`, the inverse
> of LGD) were included as requested. These are deliberately broad source-label
> mappings; the transformation/enum layer is responsible for value semantics.
> Flag for review if a future asset class needs "stage" to resolve elsewhere.

---

## 5. Chart library — no new chart type needed

**Assessment outcome: no chart addition required.** The MI Agent's permissible
chart library already supports the chart types the new risk views need.

- `mi_agent/mi_query_spec.py` → `CHART_TYPES = {bar, line, scatter, bubble,
  heatmap, treemap, none}`.
- **Line charts already exist** (`mi_agent/mi_chart_factory.py::_build_line`,
  `mi_query_executor.py::_execute_line`) — cohort/vintage trend views use the
  **existing** `line` type. No Streamlit chart code copied or migrated.
- Risk **migration** views (grade/PD/IFRS 9 transitions) and concentration views
  are served by the existing `heatmap`, `bar`, and `treemap` types.

**One enhancement noted for a later phase (not a new chart type):** the current
`_build_line` renders a **single series** (`showlegend=False`, one trace).
True **multi-line cohort curves** (one line per vintage/segment) would need a
small **additive enhancement to the existing `line` type** — an optional
series/`color` grouping producing multiple traces — handled when cohort curves
are built (build-plan Phase 4). This stays within the governed MI Agent chart
factory; it is **not** a new chart type and **not** a Streamlit migration.

---

## 6. Validation performed

- Both YAML files parse (`yaml.safe_load`).
- Registry: 493 fields total; all 21 new fields present; **no duplicate keys**.
- Every new field uses **exactly** the canonical analytics field key-set
  (`allowed_values`, `category`, `format`, `portfolio_type`, `layer`,
  `core_canonical`) — schema-consistent with existing fields.
- Formats/categories/layers drawn only from the registry's existing vocabulary.
- Aliases: 21 canon entries, 98 aliases, no empty lists, no duplicate canon keys,
  legacy field names retained in the registry.

> Existing engine test suites (`tests/test_onboarding_alias_integration.py`) could
> not be executed in this environment due to **pre-existing missing dependencies**
> (`pandas`, `rapidfuzz`) unrelated to these changes; validation was therefore
> done structurally against the registry/alias schema.

---

## 7. Follow-ons (not in Phase 0 scope)

- Wire the new fields into the **MI semantic registry curation** (`mi_agent/
  build_mi_semantics_registry.py` `CURATION`) so they appear as MI dimensions,
  and add `amortisation_type` (already canonical) at the same time.
- Define **bucket edges** for `pd_bucket` / `lgd_bucket` / `ead_bucket` in the
  proposed `config/mi/buckets.yaml`.
- Add the **snapshot-header date fields** (`reporting_date`, `cut_off_date`,
  `upload_timestamp`) and **segmentation ids** (`portfolio_id`, `spv_id`,
  `acquired_portfolio_id`) when the snapshot layer is designed (build-plan
  Phase 2) — these belong to the snapshot model, not the per-loan field registry.
