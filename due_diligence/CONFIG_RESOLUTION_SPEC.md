# Config Resolution Specification (Pass 1)

## Purpose
Define a deterministic, low-risk resolution model for Trakt configuration composition.

This spec is intended to enable Pass 2 adoption incrementally without forcing immediate repo-wide migration.

---

## Resolution model

### Context tuple
A runtime context is selected by:
- `platform`
- `asset`
- `regime`
- `client`

### Merge order
General config merge order:
1. platform config
2. asset config
3. regime config
4. client config

### Chart merge order
1. standard chart pack
2. asset chart pack
3. client chart overrides

### Field semantics
- `field_registry` is canonical.
- Non-registry layers should not redefine field meaning/format.
- Any field-level override attempt should be flagged.

---

## Merge semantics

### Dicts
- Recursive merge.
- Later layer wins on scalar key collisions.

### Lists
- Default policy for Pass 1 scaffold: replace-by-last-layer.
- Future extension: keyed list merge for chart IDs/rules where needed.

### Scalars
- Last writer wins by precedence.

### Null handling
- `null` in later layer overwrites prior value.
- Future option: explicit delete marker policy (out of scope Pass 1).

---

## Ownership expectations during resolution

- Platform provides global defaults only.
- Asset extends platform behavior.
- Regime applies reporting/output behavior.
- Client applies identity/branding/constants and constrained overrides.

Any resolved-key conflict that violates ownership policy should be captured as a warning in provenance diagnostics.

---

## Runtime override handling

### Inputs
Runtime overrides can come from:
- CLI arguments
- environment variables

### Rule
- Runtime overrides apply **after** layer merge.
- Runtime overrides should be tracked with source `runtime_override` in provenance.

---

## Provenance capture

Resolved artifact should include:
- effective config object
- per-path source map (which layer set final value)
- warnings (policy conflicts / missing expected files)
- context tuple used for resolution

Example structure:

```yaml
context:
  platform: default
  asset: equity_release
  regime: ESMA_Annex2
  client: ere
resolved:
  ...
provenance:
  pipeline.mi_enabled: client
  regime.code_order: regime
warnings:
  - "client attempted to override field_registry format for field X"
```

---

## Pass 1 scaffold scope

A Pass 1 scaffold may:
- load YAML files by provided paths
- merge dictionaries deterministically
- produce resolved config + provenance map

A Pass 1 scaffold should NOT:
- force runtime entrypoints to adopt new resolver immediately
- change existing orchestrator behavior by default
- attempt full ownership enforcement beyond lightweight warnings

---

## Adoption plan (non-disruptive)

### Pass 1 (this pass)
- Publish policy + duplication inventory + resolution spec.
- Add optional resolver helper module (not yet required by runtime entrypoints).

### Pass 2
- Integrate resolver into selected entrypoints with fallback to current behavior.
- Move chart loading to layered pack composition in a narrow, testable path.
- Add resolved config output artifact for key runs.

### Pass 3
- Extend ownership checks and linting.
- Migrate remaining embedded constants to declarative configs.
- Standardize regime/output config contracts further.

