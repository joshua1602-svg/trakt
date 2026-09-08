# Phase 2 pre-flight — the seven produced-but-unregistered pipeline fields

Requested before Phase 2 lands. This is the inventory, the collisions and the
expected before/after. **No code change is proposed as accepted here** — the
wide regression diff runs before anything in this file is applied.

The one exception already taken, in Phase 1: `pipeline_case_age_days`, because
P048 is a SILENT WRONG ANSWER and Phase 1's acceptance is `SUSPECT_ANSWER = 0`.

---

## A. The seven fields

Runtime evidence is `prepare_pipeline_mi_dataset` run on
`tests/fixtures/pipeline_transition_2w`; the counts are non-null rows out of 10.

| # | Runtime name (prepared pipeline frame) | Populated | Declared in | MI registry |
|---|---|---|---|---|
| 1 | `product_type` | 10/10 | `pipeline_field_contract.yaml` → `funded_correlated_fields` | absent (registry says `erm_product_type`) |
| 2 | `pipeline_case_age_days` | 10/10 | `pipeline_specific_fields` | **added in Phase 1** |
| 3 | `expected_funded_amount` | 10/10 | `pipeline_specific_fields` | absent |
| 4 | `weighted_expected_funded_amount` | 9/10 | `pipeline_specific_fields` | absent (near-name `forecast_funded_balance`) |
| 5 | `completion_probability` | 9/10 | `pipeline_specific_fields` | absent (near-name `forecast_funding_probability`) |
| 6 | `expected_completion_month` | 10/10 | `buckets` | absent (near-name `forecast_funding_date`) |
| 7 | `days_to_expected_completion` | 10/10 | `pipeline_specific_fields` | absent |

## B. Canonical semantic name each should bind to

| Runtime name | Should bind to | Shape of the fix |
|---|---|---|
| `product_type` | **`erm_product_type`** (existing) | **column alias**, no registry change |
| `pipeline_case_age_days` | itself | registry entry (done, Phase 1) |
| `expected_funded_amount` | itself | registry entry |
| `weighted_expected_funded_amount` | itself, **or** repoint `forecast_funded_balance` | decision required — see D |
| `completion_probability` | itself, **or** repoint `forecast_funding_probability` | decision required — see D |
| `expected_completion_month` | itself | registry entry; a month is not `forecast_funding_date` |
| `days_to_expected_completion` | itself | registry entry |

### Why product is an alias and not a registry entry

`mi_agent_api/pipeline_prep.py::_apply_group_aliases` **already exists for
exactly this**, and already carries two of the three:

```python
if "collateral_geography" in out.columns and "geographic_region_obligor" not in out.columns:
    out["geographic_region_obligor"] = out["collateral_geography"]
if "broker_channel" in out.columns and "origination_channel" not in out.columns:
    out["origination_channel"] = out["broker_channel"]
```

That is the whole explanation for why pipeline **region** and **broker** verified
in the bank and pipeline **product** did not: two of the three aliases were
declared and the third was never added. The fix is a third clause of the same
shape, materialising `erm_product_type` from `product_type`. It only ADDS a
column to the pipeline frame — no registry change, no new synonym, nothing
renamed, and `erm_product_type` already owns "product" and "product type".

The contract even declares the correspondence already —
`product_type.funded_correlation: [erm_product_type, erm_sub_product_type]` —
and `mi_agent_pptx/pipeline_prep.py:93` reads it. The MI query surface does not.

## C. Consumers of each runtime name

Files naming the field, excluding tests and `migration_phase0/`. This is the
blast surface of any RENAME, which is why no rename is proposed.

| Field | Consumers |
|---|---|
| `product_type` | `onboarding_context` (8), `analytics/static_pools_core` (6), `simulation/assets/asset_finance` (5), `simulation/assets/bridge` (3), `mi_agent_api/snapshots` (3), `onboarding_agent/product_profile` (3), `analytics/generate_pptx_client` (3), `simulation/dialects/vocabulary` (2) |
| `pipeline_case_age_days` | `pipeline_prep` (4), `forecast_bridge` (2), contract (1) |
| `expected_funded_amount` | `analytics/tab_pipeline` (8), `pipeline_prep` (6), contract (4), `workspace` (3), `pipeline_contract` (3), `analytics/pipeline_expected_funding` (3), `central_tape_builder` (2), `mi_workflows/analytical/executors` (1) |
| `weighted_expected_funded_amount` | `pipeline_prep` (9), `pipeline_contract` (6), `evolution` (5), `mi_workflows/analytical/executors` (3), `mi_agent_pptx/pipeline_prep` (3), `workspace` (3), `trakt_notifications/portfolio_update` (1), `mi_agent_pptx/metric_resolver` (1) |
| `completion_probability` | `pipeline_prep` (11), contract (6), `mi_agent_pptx/pipeline_prep` (5), `concentration_tests/forward` (4), `mi_agent_pptx/metric_resolver` (2), `workspace` (2), `forecast_bridge` (2), `mi_workflows/analytical/executors` (1) |
| `expected_completion_month` | `concentration_tests/forward` (7), `pipeline_prep` (6), `workspace` (2), `pipeline_contract` (2), contract (2) |
| `days_to_expected_completion` | `pipeline_prep` (4), contract (1) |

`product_type` is the widest and the most general — it is a name several
unrelated subsystems use for their own product column. That is a second reason
the pipeline fix is an added alias column rather than a rename.

## D. Alias / canonical collisions

Existing registry synonyms that a new entry would sit next to. Every one of
these is a real collision to decide, not a hypothetical.

| Existing field | Synonyms | Collides with |
|---|---|---|
| `erm_product_type` | `product`, `product type` | **1** — and is why the alias approach adds no synonym at all |
| `forecast_funded_balance` | `forecast funded balance`, `expected funded balance` | **3/4** — "expected funded **balance**" vs "expected funded **amount**" is one concept with two registry names |
| `forecast_funding_probability` | `funding probability`, `conversion probability`, `expected conversion` | **5** |
| `forecast_funding_date` | `forecast funding date`, `expected funding date` | **6** |
| `probability_of_default` | `probability of default`, `default probability` | **5** — a bare synonym `probability` must NOT be added; the live refusal *"'probability' is not a governed measure"* is this |
| `origination_date` | **`completion date`** | **6** — the funded book's completion, not the pipeline's expected one |
| `number_of_days_in_arrears` | `days in arrears` | **2** — adjacent to "days in pipeline"; distinct, but the neighbourhood is crowded |
| `arrears_bucket` | `days in arrears band` | **2** — same |

**The unresolved one is 4/5/6.** `forecast_funded_balance`,
`forecast_funding_probability` and `forecast_funding_date` are registry entries
marked `virtual: True, source_criteria: ["forecast"]` — a *state layer* that is
not the prepared pipeline frame. So the estate has two vocabularies for the
same three ideas, and Phase 2 must pick one rather than register a fourth and
fifth name beside them. That choice is a decision, not a mechanical alignment,
and it is the reason clusters E and F are held for an explicit Phase 3 call
even though their economics already exist.

## E. Expected before → after

| Question | Before | After (product alias) |
|---|---|---|
| pipeline amount by product type | *"'Product Type' is not available in this dataset."* | grouped sum by product |
| pipeline case count by product type | same refusal | grouped count |
| WA LTV by product type (pipeline) | same refusal | grouped weighted average |
| average borrower age by product (pipeline) | same refusal | grouped mean |
| largest pipeline amount by product | same refusal | ranked sum |
| WA pipeline rate by product (P028) | same refusal | grouped weighted average |
| **funded** product questions | 9 verified | unchanged — no funded column is touched |

Forecast and timing (P041–P047) are NOT in this table. They stay refused until
the section-D naming decision is made.

## F. Gate

The wide regression diff runs against the Phase 1 commit, on an identical file
list, before the product alias or any registry entry in this file is applied.
The alias adds a column and changes no existing one, so the expected failure-set
delta is empty; that expectation is what the diff exists to falsify.
