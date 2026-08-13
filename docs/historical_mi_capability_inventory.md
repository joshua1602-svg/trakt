# Historical MI capability inventory

*Sprint 2.5B, Part 1. Traced before writing any production code, because the
sprint's first non-negotiable is to reuse the MI registry rather than build a
second analytics layer beside it.*

Filenames were not trusted. Every entry below was traced to the actual
calculation function and its consumers.

---

## What already exists, and is genuinely reusable

| Capability | Implementation | Real? | React use | Agent access | Action |
|---|---|---|---|---|---|
| **Status / DPD transition matrix** | `mi_agent/risk_monitor/migration.py::migration_matrix(baseline, current, dimension)` | ✅ real, tested | Risk Monitor | ❌ none | **REUSE — expose** |
| **Per-loan movement** | `…migration.py::per_loan_movement` | ✅ real | Risk Monitor | ❌ | **REUSE — expose** |
| **Transition classification** | `…migration.py::classify_change` | ✅ real | Risk Monitor | ❌ | REUSE |
| **Cohort / vintage tables** | `analytics_lib/cohort.py::cohort_table`, `add_cohort_period`, `months_on_book` | ✅ real | MI workflows | ❌ | **REUSE — expose** |
| **Two-period metric change** | `mi_agent/period_change/calculations.py::metric_change` | ✅ real | MI Query | ✅ `period_change` | REUSE |
| **Distribution change** | `…/distribution.py::distribution_change` | ✅ real | MI Query | ✅ `period_change` | REUSE |
| **Balance bridge** | `…/bridge.py::balance_bridge` | ✅ real | MI Query | ✅ `period_change` | REUSE |
| **Period resolution** | `…/periods.py`, `selection.py` | ✅ real | MI Query | ✅ | REUSE |
| **Snapshot discovery** | `mi_agent_api/snapshots.py::discover_snapshots`, `find_prior_run` | ✅ real | MI | indirect | **REUSE** |
| **Business semantics registry** | `config/business_semantics_registry.yaml` — 242 fields carrying `temporality`, `aggregation`, `direction`, `workflow` | ✅ real | MI Query | indirect | **REUSE as the metric registry** |
| **Concentration / stratification** | `analytics_lib/{stratify,concentration}.py` | ✅ real | MI, agent | ✅ | REUSE |

The registry already models exactly the distinction this sprint needs:

```
temporality:  point_in_time  198 fields
              period_flow     13 fields   ← flows within a period
              cumulative       5 fields   ← running totals
              static_baseline 26 fields
```

**That is the single most useful finding in this inventory.** The canonical model
already knows which fields are flows and which are stocks, so the historical
metrics do not need a new vocabulary — they need to read the one that exists.

---

## What looks like a capability and is not

| Apparent capability | Reality |
|---|---|
| `analytics_lib/migration.py` | **Explicit stub.** `transition_matrix` and `deterioration_flags` both `return None` with "Not implemented in Phase 1". The real implementation is in `risk_monitor`. Two modules, one name, one of them empty — worth knowing before wiring anything to it. |
| "Prepayment Speed (CPR)" in `config/asset/static_pools_config_erm.yaml` | A **chart title**. The underlying spec is `metric: prepayment_amount, agg: sum` — cumulative redemptions in £, plotted. There is **no CPR or SMM calculation anywhere in the repository.** Treating this as an existing methodology would have been the single easiest mistake in this sprint. |
| `mi_agent_pptx/cohorts.py` | Presentation only. `adapt_formation` / `adapt_progression` *adapt a payload*; they compute nothing. |
| `mi_agent_api/pipeline_history.py::_cohort_progression` | Real, but it is the **origination funnel** (KFI → Application → Offer → Funded), not loan performance over time. Different question. |

---

## Canonical fields that support the missing metrics

These exist in the registry and settle what can honestly be built.

**Prepayment and redemption** — the model already separates scheduled from
unscheduled, which is what makes a defensible methodology possible:

| Field | Temporality | Meaning |
|---|---|---|
| `unscheduled_principal_collections` | period_flow | **unscheduled** principal — the prepayment numerator |
| `redemptions_received_in_period` | period_flow | redemptions in the period |
| `total_scheduled_principal_interest_due` | period_flow | scheduled amount due |
| `total_scheduled_principal_interest_paid` | period_flow | scheduled amount paid |
| `cumulative_prepayments` | cumulative | running total |
| `prepayment_fee`, `prepayment_interest_excess_shortfall` | period_flow | associated charges |

**Loss and recovery:**

| Field | Temporality |
|---|---|
| `allocated_losses` | cumulative |
| `cumulative_recoveries` | cumulative |
| `recoveries_in_period` | period_flow |
| `default_amount`, `default_date` | point_in_time |

Both are genuinely supported. Neither has a calculation.

---

## The gaps this sprint must fill

| Gap | Why it is genuinely missing | Fields available? |
|---|---|---|
| **Multi-period time series** | `period_change` is strictly pairwise. A 12-month trend is a different query shape, not a loop over pairs | ✅ (snapshots) |
| **Observed prepayment / redemption rate** | No calculation exists. Fields do | ✅ |
| **Loss and recovery rates** | No calculation exists. Fields do | ✅ |
| **Cohort comparison (A vs rest)** | `cohort_table` produces one table; comparing a flagged cohort against the remainder is the shape an investigation needs | ✅ |
| **DPD transition exposure** | `migration_matrix` exists and no governed tool reaches it | ✅ |

---

## The architectural conclusion

Roughly **two thirds of the historical capability this sprint needs already
exists** — transitions, cohorts, period change, balance bridge, snapshot
discovery, and a registry that already distinguishes flows from stocks. What is
missing is a multi-period series, three genuinely absent calculations, and
governed exposure for two capabilities React already uses.

So the work is: **expose what exists, add only what does not, and put the new
calculations in `analytics_lib` where React and MI Query can reach them too.**
Nothing about this sprint justifies a securitisation-specific analytics layer.
