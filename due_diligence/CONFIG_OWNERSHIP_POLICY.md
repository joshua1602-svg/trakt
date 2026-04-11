# Config Ownership Policy (Pass 1 Foundation)

## Purpose
This policy defines where configuration concerns belong in Trakt so future changes avoid drift, duplication, and hidden ERM/ERE coupling.

This is a **Pass 1 policy artifact**: it introduces boundaries and expectations without forcing immediate migration.

---

## Layer model and intended precedence

### General precedence
`platform -> asset -> regime -> client`

### Chart precedence
`standard_chart_pack -> asset_chart_pack -> client_chart_overrides`

### Field semantics precedence
`field_registry` remains canonical and should not be overridden casually in other layers.

---

## 1) Platform config

### Owns
- Global/static settings and conventions.
- Default validation behavior and stage ontology.
- Global feature flags that are truly cross-asset/client.
- Runtime defaults that are safe to share across all tenants.

### Must not own
- Client legal identity values (LEI/legal names).
- Asset-specific defaults (e.g., ERM risk assumptions).
- Regime filing rules.
- Client branding and chart label preferences.

---

## 2) Field registry

### Owns
- Canonical field definitions and meanings.
- Field format, required/optional semantics.
- Portfolio applicability metadata.
- Regime code mapping metadata at canonical-field level.

### Must not own
- Client branding/identity.
- Runtime chart ordering.
- Arbitrary client-specific value defaults.

---

## 3) Standard chart pack

### Owns
- Cross-asset standard MI chart set.
- Default dashboard sections/tabs ordering.
- Common labels/grouping/order used by default.

### Must not own
- Asset-only chart logic.
- Client-specific branding or bespoke labels.

---

## 4) Asset config

### Owns
- Asset-family default behavior.
- Asset-specific derivation/risk/forecast concepts.
- Asset applicability toggles and mappings.

### Must not own
- Client legal identity constants.
- Regime filing behavior that should be centralized.
- Full client chart forks.

---

## 5) Asset chart pack

### Owns
- Asset chart extensions and enablement relative to standard pack.
- Asset-specific analytical emphasis.

### Must not own
- Full duplicated standard chart pack.
- Client-specific presentation variants.

---

## 6) Regime config

### Owns
- Deterministic reporting behavior by regime.
- Regime-level constraints, mappings, output ordering.
- Filing and XML output rules by regime.

### Must not own
- Generic client branding.
- Asset defaults unrelated to filing behavior.

---

## 7) Client config

### Owns
- Tenant identity and legal constants.
- Client branding and deterministic source mappings.
- Client-level enable/disable controls where appropriate.

### Must not own
- Canonical field semantics.
- Broad regime logic replicated from regime layer.
- Asset-wide defaults that belong in asset config.

---

## 8) Client chart overrides

### Owns
- Minimal chart overlays only:
  - enable/disable selected charts
  - label/order/presentation tweaks
  - small additive changes

### Must not own
- Full forked chart pack.
- Duplicated standard/asset chart definitions.

---

## 9) Runtime/resolved config

### Owns
- Effective merged runtime config for selected context.
- Provenance metadata (which layer set each value).
- Validation/debug snapshot of final config.

### Must not own
- Business logic itself.
- Persistent ad hoc state unrelated to a run.

---

## Ownership guardrails

1. One source of truth per concern.
2. Field meaning belongs in field registry.
3. Asset config extends behavior; does not redefine canonical semantics.
4. Client config remains narrow and deterministic.
5. Regime config owns reporting behavior.
6. Charts inherit by overlay; no forked chart universe.

---

## Pass 1 enforcement scope

In Pass 1 this policy is **guidance + design contract**.

Immediate codebase-wide enforcement is out of scope for Pass 1.
Enforcement hooks (lint/validation checks) should be introduced in later passes.

