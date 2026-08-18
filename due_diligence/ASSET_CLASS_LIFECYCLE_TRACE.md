# Asset-Class Lifecycle — End-to-End Trace

**Date:** 18 August 2026 · **Branch:** `claude/mi-query-agent-review-n8d33r` · **HEAD:** `0197f25`
**Status:** trace only — no code or configuration was changed.

**Intended architecture (as stated):** the OCC onboarding agent establishes a portfolio's
asset class; that selection determines the applicable asset-specific configuration and
persists as governed portfolio metadata into every downstream MI view and the MI Agent.
Equity Release MI = common semantic core + ER-specific semantics; Bridge MI = common core +
Bridge-specific semantics.

**Finding:** the first hop of that chain exists and the last hop exists. **Every hop in
between is missing.** `asset_class` has no representation on the governed portfolio model,
no path through canonical processing, and the two MI consumers that need it read two
different, unconnected, empty sources.

---

## 1. The chain, hop by hop

| # | Hop | Status | Evidence |
| --- | --- | --- | --- |
| 1 | OCC determines asset class | **EXISTS** | `engine/onboarding_agent/onboarding_context.py:92`; `operations_control/onboarding/inference.py:180` |
| 2 | Persisted as onboarding artefact | **EXISTS** | `27_onboarding_context.json` (`onboarding_context.py:377`) |
| 3 | Reaches anything outside onboarding | **BROKEN** | no reader outside `engine/onboarding_agent/` |
| 4 | Survives canonical processing | **BROKEN** | not a canonical field; no tape column |
| 5 | Lands on governed portfolio metadata | **BROKEN** | `PortfolioRecord` has no such field |
| 6 | Reaches the MI service | **BROKEN** | consequence of 3–5 |
| 7 | Controls MI field/metric availability | **PARTIAL** | one live gate, two inert ones |
| 8 | Portfolio comparison reads the same metadata | **BROKEN** | reads a different source again |

## 2. Hop 1 — where OCC determines it

Two independent implementations, both real:

* **`engine/onboarding_agent/onboarding_context.py:92`** — token-signal scoring across file
  names, column names and redacted samples. Vocabulary: `equity_release_mortgage`,
  `residential_mortgage`, `consumer_loan`, `sme_loan`. Defaults to `equity_release_mortgage`
  when no signal fires, with `asset_signal_strength: 0` recorded so consumers can tell a
  guess from evidence.
* **`operations_control/onboarding/inference.py:180-205`** — per-portfolio inference from
  headers into the onboarding case; `occ_agent/derive.py:150` reads
  `portfolio.get("asset_class")` into `ExecutionFacts`, which drives regime and outcome.

## 3. Hops 2–3 — persistence, and where the trail ends

`write_context_artifacts()` writes `27_onboarding_context.json`. Complete reader list:

```
engine/onboarding_agent/onboarding_context.py          (writer)
engine/onboarding_agent/streamlit_onboarding_workbench.py
engine/onboarding_agent/onboarding_orchestrator.py     (manifest listing only)
tests/test_onboarding_asset_regime_resolver.py
```

No module under `mi_agent/`, `mi_agent_api/`, `mi_workflows/`, `trakt_core/` or
`demo_platform/` reads it. The OCC-agent case's `portfolios[].asset_class` likewise never
leaves the onboarding case store.

## 4. Hop 4 — canonical processing

`asset_class` is not a canonical field anywhere:

* absent from `config/system/fields_registry.yaml` (5,810 lines);
* absent from `mi_agent/mi_semantics_field_registry.yaml`;
* not an entry in `config/business_semantics_registry.yaml` (it appears only as the
  `asset_applicability` *attribute* of other entries).

Verified at runtime against the live fixture: `"asset_class" in df.columns` → **False**.

## 5. Hop 5 — the governed portfolio model has no slot

`trakt_core.portfolio.PortfolioRecord` fields, read at runtime:

```
forecast_treatment · has_runoff_profile · label · originates ·
pipeline_data_available · portfolio_id · portfolio_type ·
present_in_data · reporting_dates · row_count · runoff_profile_id
```

`portfolio_type` is `direct` / `acquired` — **origination provenance, not asset class**.
`simulation/models.py:27` states the distinction explicitly: *"Funded provenance INSIDE one
SPV."*

The designed metadata seam **does** exist — `mi_agent/portfolio_metadata.py`, env
`TRAKT_PORTFOLIO_REGISTRY` → `config/client/portfolio_registry.yaml`, already carrying
origination capability, forecast treatment and supplied runoff curves per portfolio — but
its `_ALLOWED_KEYS` (line 43) **excludes `asset_class`**, so the overlay could not carry it
even if OCC wrote it.

Runtime state on the demonstration client:

```
governed portfolio registry file:            None
portfolio metadata overlay for alderbridge:  {} (empty)
PortfolioRecord exposes asset class:         False
```

## 6. Hop 7 — three asset-class gates in MI, two inert

| Gate | Live? | Evidence |
| --- | --- | --- |
| `config/mi/stratification_catalogue.yaml:41` — `youngest_borrower_age: asset_classes: [equity_release]` | **No** | the only loader of this file is `mi_agent_pptx/registry_loader.py`, which never reads `asset_classes`. The key is unconsumed. |
| `config/mi/mi_equity_release_uk_applicability.yaml:33` — `meta.asset_class: equity_release_mortgage` | **No** (for MI Agent) | read by `engine/onboarding_agent/target_coverage.py` and `mi_agent_pptx` only |
| BSR `asset_applicability` + `business_semantics.applies_to_asset()` (line 159) | **Yes** | the sole runtime gate — and it is fed nothing |

`applies_to_asset` is conservative by design: an empty asset-class list admits only
`cross_asset` entries, *"so an equity-release field is never reported for an unidentified
book."* That behaviour is correct. The defect is that the book is always unidentified.

## 7. Hop 8 — the two MI consumers read different sources

| Consumer | Source it reads | Runtime value |
| --- | --- | --- |
| `mi_agent/period_change/selection.py:279` | `PortfolioScopeRef.asset_classes` ← `mi_agent_api/period_change_route.resolve_asset_classes(explicit)` | `()` — **no production caller passes an argument** |
| `mi_workflows/portfolio_risk_comparison.py:307` | `_asset_classes(frame)` — an `asset_class` **dataframe column** | `()` — column does not exist |

Two mechanisms, no shared source of truth, both empty.

## 8. Why `shared_asset=None` for `alp_origination` / `alp_acquired`

```
_asset_classes(frame_a) -> ()          # no asset_class column on the tape
_asset_classes(frame_b) -> ()
_shared_asset_class((), ()) -> None
  -> _comparability_decision(youngest_borrower_age, shared_asset=None, ...)
     -> selected=False, "asset-specific field and the portfolios do not
        declare a single shared governed asset class"
  -> metric_comparisons = []            # B25
```

## 9. Does the Alderbridge fixture bypass OCC persistence?

Partly — **and fixing the fixture alone would fix nothing.**

`demo_platform/config/config_client_ALDERBRIDGE_DEMO.yaml:21` already declares
`portfolio.asset_class: equity_release` — the BSR-correct value. It is consumed by
`engine/gate_4_projection/regime_projector.py:1022` and passed to the pipeline as
`--master-config`. **There is no path from it to MI**, because of §4 and §5.

`demo_platform/onboarding.py` runs the real production profiling and header-mapping
components, but the contract it records stamps only `source_portfolio_id`,
`source_portfolio_type` and `portfolio_cohort` — never asset class.

## 10. Equity release vs bridge — what actually differs

| | Equity release | Bridge |
| --- | --- | --- |
| Asset configuration | `config/asset/product_defaults_ERM.yaml`, `config/asset/static_pools_config_erm.yaml`, ER profile at `product_profiles.yaml:66` | **none** |
| MI applicability config | `config/mi/mi_equity_release_uk_applicability.yaml` | **none** |
| MI stratification | 1 ER-only dimension (`youngest_borrower_age`) | **none** |
| BSR entries | 12 `equity_release`-only fields | **0** |
| Where it exists at all | production config + registry | `simulation/` only (`ASSET_BRIDGE`, `simulation/assets/bridge.py`) — **not imported by any MI module** |

`config/client/portfolio_reference_example.yaml:47` mentions "Client A — Bridging" as an
illustrative portfolio *name*; it carries no asset-class semantics.

**The "common core + asset-specific" composition mechanism does not yet exist in MI.** The
only compositional primitive is the BSR's `applies_to_asset`, and it receives no input.

## 11. Four incompatible vocabularies

| Source | Values |
| --- | --- |
| `engine/onboarding_agent/onboarding_context.py:34` | `equity_release_mortgage`, `residential_mortgage`, `consumer_loan`, `sme_loan` |
| BSR + `config/asset` + demo client config | `equity_release`, `residential_mortgage`, `sme`, `commercial_real_estate`, `equipment_leasing`, `corporate` |
| `simulation/models.py:26` | `equity_release`, `bridge`, `asset_finance` |
| `mi_agent_api/period_change_route.KNOWN_ASSET_CLASSES` | the BSR six |

`config/asset/product_profiles.yaml:66` already absorbs part of this by matching on
`[equity_release_mortgage, equity_release, lifetime_mortgage]` — evidence that the mismatch
is known and has been worked around rather than resolved.

## 12. Verdict — production architecture defect, not an incomplete fixture

Four independent proofs:

1. `PortfolioRecord` has no `asset_class` field — there is no slot to persist into;
2. `portfolio_metadata._ALLOWED_KEYS` excludes it — the designed overlay cannot carry it;
3. the two MI consumers read *different* sources, neither of which is governed portfolio
   metadata;
4. `resolve_asset_classes()` has **zero production callers** — the parameter was built as an
   extension point and never wired.

The fixture's under-declaration is a secondary symptom, not the cause.

## 13. Smallest correction restoring one source of truth

OCC already owns this metadata. The correction gives its output a path — it does **not** add
a second declaration.

1. **`trakt_core.PortfolioRecord` += `asset_class`**, normalised to the BSR vocabulary.
2. **`mi_agent/portfolio_metadata._ALLOWED_KEYS` += `"asset_class"`** — the overlay seam then
   carries it end to end.
3. **OCC writes it** into the existing governed portfolio registry
   (`TRAKT_PORTFOLIO_REGISTRY` → `config/client/portfolio_registry.yaml`), beside the
   origination / forecast / runoff facts it already writes per portfolio. One writer, one
   file, one source of truth.
4. **Both MI consumers read it from there**: `scope_ref_from_lens()` populates
   `asset_classes` from the resolved `PortfolioScope`; `portfolio_risk_comparison`
   `_asset_classes()` prefers the scope's classes, keeping the tape column as a fallback.
5. **One documented mapping** at the OCC boundary (`equity_release_mortgage` →
   `equity_release`), making the BSR vocabulary canonical.

Steps 1–2 are roughly two lines. Steps 3–4 are the real work, and are what makes
"Equity Release MI = common core + ER semantics; Bridge MI = common core + Bridge semantics"
function — today the mechanism exists but is starved.

**Explicitly not recommended:** adding `asset_class` as a canonical tape column, or a second
declaration in the demo client config. Both would create a competing source of truth for
metadata OCC already owns.

**B25 resolves as a consequence of step 4, not as a special case.**
