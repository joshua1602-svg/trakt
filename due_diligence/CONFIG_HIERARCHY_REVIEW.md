# Trakt Configuration Ownership & Standardization Review

## 1) Executive summary

### Overall assessment
The repository has **meaningful layered foundations** (system registry + asset/config/regime/client directories and a multi-mode orchestrator), but it is currently **partially layered and still ERM-shaped** in runtime defaults, dashboard wiring, risk logic, and chart selection.

### Current state classification
- **Mostly configurable?** Partial.
- **Partially layered but still ERM-shaped?** Yes.
- **Material drift risk?** Yes, especially for charting, client constants, and implicit defaults in runtime/UI entrypoints.

### Client 1 (ERE) onboarding readiness
The repo is operational for ERE, but **not yet clean enough for low-drift onboarding**. It should be treated as **borderline-ready** pending ownership-boundary tightening and standardization of hardcoded assumptions.

---

## 2) Config ownership model (target)

This section defines what should belong in each layer.

### A. Platform config
**Purpose**
- Cross-asset, cross-client, stable defaults.
- Global runtime flags and ontology.
- Validation behavior defaults and naming conventions.

**Allowed contents**
- Platform metadata and versioning.
- Default stage ontology and gate behavior.
- Default validation severities/policies.
- Feature toggles that are truly global.

**Should NOT contain**
- Client identity (legal names, LEI).
- Asset-specific business assumptions (ERM-only risk vocabulary).
- Regime-specific ESMA field behavior.
- Client branding or chart labels.

### B. Asset config
**Purpose**
- Asset-family behavior extension (e.g., equity release vs SME).

**Allowed contents**
- Asset defaulting/mapping behavior.
- Asset-specific derived metric behavior.
- Asset-specific risk and scenario concepts.
- Asset-level optional field enablement.

**Should NOT contain**
- Client legal identity constants.
- Regime filing rules and XML behavior.
- Forked full chart universe (should be extension pack only).

### C. Client config
**Purpose**
- Deterministic client identity and tenant-specific constants.

**Allowed contents**
- Legal name(s), LEI/entity IDs.
- Branding (logo/theme/typography).
- Deterministic source mapping constants specific to the client.
- Client-level enable/disable toggles where justified.

**Should NOT contain**
- Canonical field semantics.
- Broad regime logic.
- Asset family defaults that apply to all clients.

### D. Regime config
**Purpose**
- Reporting regime behavior and output rules.

**Allowed contents**
- Regime field constraints and code ordering.
- Deterministic filing/output rules.
- XML/reporting behavior by regime.

**Should NOT contain**
- Client identity constants unless truly mandated by client legal contract and not representable as client injection.
- Asset-wide defaults unrelated to filing logic.

### E. Chart config
**Purpose**
- Layered chart selection and presentation without forking.

**Allowed contents**
- `standard_chart_pack`: common MI set and section/tab defaults.
- `asset_chart_pack`: asset-specific chart additions/toggles.
- `client_chart_overrides`: minimal label/order/enablement overrides.

**Should NOT contain**
- Hardcoded chart fallback definitions in app code as primary mechanism.
- Duplicated chart definitions per client.

### F. Runtime/resolved config
**Purpose**
- Deterministic effective config snapshot per run.

**Allowed contents**
- Fully merged config artifact by precedence.
- Hashes/versions and source provenance.
- Validation diagnostics for conflicting overrides.

**Should NOT contain**
- Business logic itself.
- Ad-hoc mutable state not tied to a run.

---

## 3) Current-state inventory (reviewed files/modules)

### Core orchestrator and runtime entrypoints
- `engine/orchestrator/trakt_run.py`
  - Current purpose: pipeline mode routing and gate orchestration.
  - Contains concerns: mode schema policy, config defaults, runtime flags, file path defaults.
  - Placement assessment: mostly platform-runtime, but with ERM/client default leakage (`equity_release`, ERM master-config default).

- `function_app.py`
  - Current purpose: blob-triggered orchestration in Azure.
  - Contains concerns: mode routing from path, environment var binding for regime/config.
  - Placement assessment: acceptable integration layer; mode routing belongs here, but underlying defaults should remain resolver-driven.

### System/platform-like config
- `config/system/fields_registry.yaml`
  - Current purpose: canonical field semantics, formats, regime mappings, applicability by portfolio type.
  - Placement assessment: correct (strong single source of truth candidate).

- `config/system/esma_code_order.yaml`
  - Current purpose: deterministic ESMA field ordering.
  - Placement assessment: regime-supporting platform asset; acceptable.

- `config/system/enum_mapping.yaml`, `enum_synonyms*.yaml`, `aliases_*.yaml`
  - Current purpose: alias/enum normalization and deterministic mapping support.
  - Placement assessment: mostly correct; should remain centrally governed.

- `config/system/config.py`
  - Current purpose: legacy constants/theme/path defaults in Python module.
  - Placement assessment: mixed; this duplicates concerns better owned by declarative platform/client/chart layers.

### Asset config
- `config/asset/product_defaults_ERM.yaml`
  - Current purpose: ERM default and ND-default behavior.
  - Placement assessment: correct as asset config.

- `config/asset/static_pools_config_erm.yaml`
  - Current purpose: static pools chart definitions for ERM.
  - Placement assessment: partial/acceptable as asset chart pack seed, but naming and loading are hardwired.

- `config/asset/issue_policy.yaml`
  - Current purpose: issue severity/policy controls.
  - Placement assessment: mixed; likely platform policy with potential asset overlays.

### Client config
- `config/client/config_client_ERM_UK.yaml`
  - Current purpose: client identity, pipeline flags, transformations, enrichment, branding.
  - Placement assessment: mixed; identity/branding are correct, but pipeline behavior and some defaults may be over-broad for client layer.

- `config/client/config_client_annex12.yaml`
  - Current purpose: annex12 overlay incl. deal constants and period values.
  - Placement assessment: mixed client+regime concern; acceptable short-term, risky long-term if replicated per client/regime.

- `config/client/risk_limits_config.py`
  - Current purpose: risk limit thresholds as Python constants.
  - Placement assessment: client-level intent but non-standardized storage format (code, not declarative config).

### Regime/reporting
- `config/regime/annex12_rules.yaml`
  - Current purpose: Annex12 field behavior (choice/monetary/boolean/repeatable/sign logic).
  - Placement assessment: correct regime layer.

- `config/regime/annex12_field_constraints.yaml`
  - Current purpose: field-level format + ND permissions for Annex12.
  - Placement assessment: correct regime layer.

- `engine/gate_4_projection/regime_projector.py`
  - Current purpose: regime projection from canonical, template ordering, enum mappings.
  - Placement assessment: largely appropriate regime runtime.

- `engine/gate_4_projection/annex12_projector.py`
  - Current purpose: Annex12 deterministic projection with strict validation.
  - Placement assessment: appropriate regime runtime, but currently injects some client constants from master config directly.

- `engine/gate_5_delivery/xml_builder_investor.py`
  - Current purpose: Annex12 XML generation with mapping/rules.
  - Placement assessment: appropriate regime output layer.

- `engine/gate_5_delivery/xml_builder.py`
  - Current purpose: generic Jinja XML builder (regulatory path).
  - Placement assessment: regime output utility; maturity/consistency with other output builders should be monitored.

### Dashboard/chart layer
- `analytics/streamlit_app_erm.py`
  - Current purpose: ERM dashboard UI and tab logic.
  - Contains concerns: client config loading, theme fallback, chart/tab wiring, static pools loading path, chart fallbacks.
  - Placement assessment: intentionally asset app, but currently embeds many concerns that should become layered config.

- `analytics/charts_plotly.py`
  - Current purpose: shared chart rendering helpers and theme defaults.
  - Placement assessment: good shared rendering utility, but theme defaults duplicate client/theme concerns.

- `analytics/static_pools_core.py`
  - Current purpose: product-agnostic static pools computation engine.
  - Placement assessment: good reusable core.

### Risk/controls layer
- `analytics/risk_monitor.py`
  - Current purpose: risk metric computation and rule checks.
  - Contains concerns: UK region map, expected canonical columns, imported client risk limit constants.
  - Placement assessment: mixed; engine is reusable-ish, but includes UK/ERM assumptions that should be externalized.

---

## 4) Categorization matrix (concern-by-concern)

| Concern | Lives now | Should live | Assessment |
|---|---|---|---|
| `portfolio_type` default | `trakt_run.py` arg default | Runtime resolver / required input; optional platform default | Misplaced (hardcoded baseline) |
| Master config default path | `trakt_run.py` | Runtime resolver profile selection | Misplaced |
| Client legal name / LEI | `config_client_ERM_UK.yaml`, partly `config_client_annex12.yaml` | Client config only | Duplicated / drift risk |
| Reporting date constants | client master + annex overlay | Client deterministic constants + controlled regime derivation | Duplicated / non-standardized |
| Branding/theme | client YAML + Streamlit defaults + chart module defaults | Client config (+ standard theme fallback in platform layer) | Duplicated |
| Field semantics/format | `fields_registry.yaml` | Field registry | Correctly placed |
| Aliases for header mapping | `config/system/aliases_*.yaml` + learned aliases | Platform/registry support | Correctly placed (governance needed) |
| Enum semantics/mappings | system enum files + regime projector | Platform+regime support | Mostly correct |
| ESMA code ordering | `config/system/esma_code_order.yaml` | Regime/platform reporting layer | Correctly placed |
| Annex12 constraints/rules | `config/regime/annex12_*` | Regime config | Correctly placed |
| Deterministic Annex12 deal constants | client annex12 overlay | Client config injected into regime projection | Acceptable temporarily; should standardize split |
| XML/reporting output logic | projector/xml builder modules | Regime runtime layer | Mostly correct |
| Static pool chart definitions | `config/asset/static_pools_config_erm.yaml` | Asset chart pack | Correct direction, partial |
| Chart selection/sections/tabs | mostly in Streamlit code | Standard chart pack + asset pack + client overrides | Misplaced/non-standardized |
| Chart labels/order overrides | hardcoded in UI logic | Client chart overrides | Missing |
| UK region mapping | `risk_monitor.py` constant dict | Asset config (or region mapping config keyed by geography) | Misplaced |
| ERM-specific risk dimensions | risk code and limits config | Asset config (+ optional client override) | Mixed |
| Risk limits thresholds | `risk_limits_config.py` Python | Client config (declarative), possibly asset defaults | Non-standardized |
| Scenario assumptions | mostly app/runtime modules | Asset config default + client overrides | Mixed |
| Dashboard tab enablement | implicit in Streamlit script | Chart/platform/client enablement config | Non-standardized |
| Asset-specific terminology | hardcoded strings in UI | Asset config / chart pack labels | Mixed |

---

## 5) Non-standardized input / hardcoding review (high importance)

### A. Hardcoded ERM/ERE baselines in runtime
1. `trakt_run.py` default `portfolio-type` is `equity_release`.
2. `trakt_run.py` default `master-config` points to ERM UK client file.

**Risk**: second asset onboarding will inherit ERM behavior unless explicitly overridden.

### B. Hardcoded ERM dashboard wiring
1. Main dashboard entrypoint itself is ERM-named (`streamlit_app_erm.py`).
2. Client config loader checks ERM UK filename directly.
3. Static pools chart config filename is hardcoded (`static_pools_config_erm.yaml`).
4. Fallback chart definitions are embedded in code.

**Risk**: chart drift and bespoke client forks.

### C. Hardcoded UK/ERM risk assumptions
1. UK region normalization map lives in code constants.
2. Risk monitoring directly imports client risk limits module from Python.

**Risk**: difficult to onboard non-UK assets/clients without code edits.

### D. Client constants duplicated across configs
1. LEI/report IDs exist in multiple client-related files.
2. Reporting date appears with conflicting values across files.

**Risk**: deterministic reporting drift and audit confusion.

### E. Regime/client mixing
Annex12 overlay holds both client-specific constants and regime-shaped values.

**Risk**: each new client+regime combination may clone near-identical overlay files.

### F. Theme duplication across layers
Theme fallback values appear in multiple Python modules plus YAML overrides.

**Risk**: inconsistent presentation and hidden override precedence.

---

## 6) Platform vs asset vs client vs regime diagnosis

### Orchestration layer
- **Intended**: platform-level reusable.
- **Current**: reusable core with ERM/client defaults embedded.
- **Diagnosis**: platform-capable but currently biased; medium drift risk.

### Field registry + alignment + canonical validation
- **Intended**: platform-level canonical truth.
- **Current**: strongly centralized and reused.
- **Diagnosis**: good ownership; keep strict no-override discipline.

### Asset defaults and static pools definitions
- **Intended**: asset-level.
- **Current**: mostly asset-level, but partially consumed through hardcoded filenames.
- **Diagnosis**: good direction, needs standardized loader/resolver.

### Client identity/branding and source mappings
- **Intended**: client-level.
- **Current**: mostly client-level, with some duplicated reporting constants.
- **Diagnosis**: partially correct, prune duplication.

### Regime/reporting rules
- **Intended**: regime-level.
- **Current**: mostly regime-level in dedicated files/modules.
- **Diagnosis**: good structure; avoid drifting client constants into regime logic files.

### Risk/controls
- **Intended**: asset+client composition.
- **Current**: mixed in Python code with UK assumptions.
- **Diagnosis**: functional, but non-standardized and too code-coupled.

---

## 7) Chart standardization diagnosis

### What is generic today
- `charts_plotly.py` provides reusable plotting/theme helpers.
- `static_pools_core.py` is product-agnostic compute engine.

### What is asset-specific today
- Static pools chart config in `config/asset/static_pools_config_erm.yaml`.
- ERM dashboard tab focus and vocabulary.

### What is client-specific today
- Branding values from `config_client_ERM_UK.yaml`.

### What should move to standard chart config
- Baseline MI sections/tabs.
- Default chart definitions, order, and common labels.
- Default enablement for generic charts.

### What should remain in asset chart config
- Asset-only chart extensions (e.g., ERM-specific stratifications, risk views).
- Asset-specific terminology defaults.

### What should be client overrides only
- Label wording tweaks.
- Minor order changes.
- Explicit chart enable/disable preferences.
- Branding/presentation-only variants.

### What is currently hardcoded/non-standardized
- Hardcoded static pools config filename and in-code fallback chart defs.
- Tab structure and numerous chart choices coded directly in Streamlit.
- Theme fallback duplication across modules.

---

## 8) Regime/config/reporting diagnosis

### Belongs in regime config (and largely already does)
- Annex constraints (format + ND permissions).
- Regime field behavior lists (choice/monetary/boolean/repeatable/sign).
- ESMA order/mapping behavior.

### Mixed into client/asset today
- Some Annex12 deterministic values in client annex overlay are regime-shaped but client-scoped.
- Dates and IDs in multiple files with weak precedence clarity.

### Deterministic client reporting values that should remain client-side
- Legal entity name.
- LEI / client legal IDs.
- Contact values and contractual static IDs if truly client-specific.

### ESMA/reporting logic that should remain regime-side
- Field constraints and allowed ND policy.
- XML path/multiplicity and ordering behavior.
- Generic rule transforms for regime outputs.

---

## 9) Priority cleanup recommendations

### Must fix before Client 1 onboarding
1. **Define and enforce merge precedence** (`platform -> asset -> regime -> client`) in one resolver path.
2. **Remove ERM/client hardcoded defaults from core runtime args** (or require explicit profile).
3. **Standardize chart layering** with explicit standard pack + asset pack + client overrides.
4. **Consolidate deterministic constants/dates** to one owner per concern; remove duplicates.
5. **Make risk limits and region maps declarative** (YAML/JSON), not embedded constants.

### Should fix soon after
1. Add schema validation for each config layer.
2. Add lints to prevent field semantic overrides outside field registry.
3. Extend run artifact to include full resolved config + provenance.
4. Normalize regime/client overlay pattern for all active regimes.

### Can defer
1. Legacy cleanup of older `config/system/config.py` consumers.
2. Broader UI modularization beyond chart pack ownership.
3. Optional tooling for config editing UX.

---

## 10) Proposed target hierarchy example

```text
config/
  platform/
    platform.yaml
    field_registry.yaml
    standard_chart_pack.yaml
  assets/
    equity_release/
      asset.yaml
      chart_pack.yaml
    sme/
      asset.yaml
      chart_pack.yaml
  regimes/
    esma_annex12/
      regime.yaml
    esma_annex2/
      regime.yaml
    esma_annex3/
      regime.yaml
    esma_annex4/
      regime.yaml
    esma_annex8/
      regime.yaml
    esma_annex9/
      regime.yaml
  clients/
    ere/
      client.yaml
      chart_overrides.yaml
```

### Runtime artifact
- `out/resolved_config.yaml` generated per run, with source paths + hashes and final effective values.

### Why this reduces drift
- One source of truth per concern.
- Deterministic ownership boundaries.
- Reusable standard/asset packs with minimal client overlays.
- Auditable resolved runtime behavior for onboarding and compliance.

