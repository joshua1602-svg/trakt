# RREL35 Amortisation Type — Enum Remediation

## Summary

The Delivery/XML readiness gate reported **1,526 `delivery_invalid` rows** for
**RREL35 amortisation_type**, sample value `Bullet`, failure type `enum_fail`.
This was a **code-list coverage / configuration issue — not an XML builder
issue**. The fix is entirely **config-driven**; no enum value is hard-coded in
Python.

## Why `Bullet` failed

The agentic pipeline validates Annex 2 enum values against the **regime contract**
`config/regime/annex2_delivery_rules.yaml::field_rules[RREL35].transform.enum_map`.
That map had been narrowed to ERM-style `OTHR` synonyms only:

```yaml
enum_map:
  Interest roll-up: OTHR
  interest roll-up: OTHR
  OTHR: OTHR
```

So when the source/asset value was `Bullet`, it was **not a known key and not a
known target code** → `enum_valid = false` → `delivery_invalid`. The repo's
system-level `config/system/enum_mapping.yaml` *did* already carry the full
`BULLET → BLLT` family, but the agentic delivery path does not read that file —
it reads the regime contract, which was the narrowed one.

## Authoritative workbook allowed values (RREL35)

| Code | Label |
| --- | --- |
| `FRXX` | French — total instalment (principal + interest) constant |
| `DEXX` | German — first instalment interest-only, then constant instalments |
| `FIXE` | Fixed amortisation schedule — constant principal per instalment |
| `BLLT` | Bullet — full principal repaid in the **last** instalment |
| `OTHR` | Other |

ND policy (workbook): **ND1–ND4 allowed, ND5 not allowed.** The prior config had
`nd_allowed: [ND5]`, which was both wrong (ND5 is forbidden here) and a symptom
of the same narrowing. It is corrected to `[ND1, ND2, ND3, ND4]`.

## The business distinction: generic `BLLT` vs ERM `OTHR`

- **Generic Annex 2 meaning:** `Bullet = BLLT` ("full principal repaid in the
  last instalment of a schedule").
- **Equity Release / lifetime mortgage:** the loan **rolls up interest** and
  repays at death/sale. The internal tape labels this `Bullet`, but it is **not**
  a scheduled bullet amortisation under the Annex definition. The correct
  regulatory reporting value for this product is **`OTHR`** unless product/legal
  review confirms a genuine scheduled bullet.

We therefore must **not** globally force `Bullet → OTHR`, and must **not** force
`Bullet → BLLT` for ERM. The decision is asset/client-specific.

## The fix (config-driven, two layers)

### 1. Regime contract — generic authoritative list

`config/regime/annex2_delivery_rules.yaml` (RREL35) now carries the full,
authoritative code list (generic ESMA meaning), with idempotent code
passthroughs and the generic internal synonyms:

```yaml
enum_map:
  French: FRXX
  German: DEXX
  Fixed amortisation schedule: FIXE
  Bullet: BLLT          # generic Annex meaning
  Other: OTHR
  FRXX: FRXX            # passthroughs
  DEXX: DEXX
  FIXE: FIXE
  BLLT: BLLT
  OTHR: OTHR
  Interest roll-up: OTHR
  interest roll-up: OTHR
nd_allowed: [ND1, ND2, ND3, ND4]
```

### 2. Asset/client policy — ERM override to `OTHR`

`config/asset/product_defaults_ERM.yaml` declares an explicit, documented
override under `reporting_policy.enum_overrides`:

```yaml
reporting_policy:
  enum_overrides:
    amortisation_type:
      Bullet: OTHR
      "Interest roll-up": OTHR
```

This override is **applied before** the generic regime `enum_map`, for this asset
class only. A non-ERM portfolio (no such override) still maps `Bullet → BLLT`.

### Where it is applied (no hard-coding)

The Projection Agent reads the override generically:

- `engine/projection_agent/gate4_adapter.py` — `load_asset_enum_overrides()`
  reads `reporting_policy.enum_overrides`; `apply_asset_enum_override()` applies
  it (case-insensitive) to a source label, returning `(value, applied)`.
- `engine/projection_agent/projection_agent.py` — `_project_cell()` and
  `_apply_static_default()` apply the override **ahead of** `apply_safe_transform`
  for both materialised values and configured asset defaults.

No `if value == "Bullet"` logic exists anywhere in Python; the `Bullet`/`OTHR`
references in the engine are docstring examples only. Removing the config
override reverts behaviour to the generic `BLLT` — proving it is config-driven.

## Audit trace

The projected target frame (`51_*`) and the delivery-normalised frame (`62_*`)
carry the mapping trace using existing columns:

| Column | RREL35 (ERM) | RREL35 (generic) |
| --- | --- | --- |
| `source_value_sample` | `Bullet` | `Bullet` |
| `projected_value` / `delivery_value` | `OTHR` | `BLLT` |
| `value_source` | `asset_policy` | `enum_map` |
| `projection_status` | `projected_from_transformed` (or `projected_asset_default` when sourced from the ERM default) | `projected_from_transformed` |

So `mapping_source` is auditable: `asset_policy` vs `enum_map` vs
`transformed_tape`.

## Re-run to clear `delivery_invalid`

The enum **mapping** happens at **projection** time, so the projected target
frame must be **regenerated** for the new mapping to flow through — the Delivery
Agent only *validates*, it does not re-map. Run, in order:

```bash
python -m engine.projection_agent.workflow \
  --validation-manifest onboarding_output/client_001/run_pre_xml_final_check_3/output/validation/40_validation_manifest.json

python -m engine.delivery_xml_agent.workflow \
  --projection-manifest onboarding_output/client_001/run_pre_xml_final_check_3/output/projection/50_projection_manifest.json

python scripts/inspect_delivery_xml_readiness.py --format-invalid \
  onboarding_output/client_001/run_pre_xml_final_check_3/output/delivery_xml
```

> If you re-run delivery **without** re-running projection, the literal `Bullet`
> already in the frame would now *pass* the membership check (because `Bullet` is
> a recognised enum_map key), but the delivered value would still be the label
> `Bullet`, not a code. Re-running projection is what makes the frame carry the
> correct code (`OTHR` for ERM), so re-projection is the correct path.

### Expected result

```text
RREL35 no longer appears in the format-invalid drill-down
delivery_format_invalid : 1 -> 0
delivery_invalid (rows) : 1526 -> 0
xml_generation_allowed  : still false   (other client/operator/config/source blockers remain)
xml_generated           : false
```

## Verification in this environment

The real `onboarding_output/...` run directory is **git-ignored and not present**
in this cloud workspace, so the live re-run must be performed in Codespaces with
the commands above. The end-to-end behaviour is proven at unit/integration level
(`tests/test_projection_agent_workflow.py::TestRrel35AmortisationEnum`,
`tests/test_delivery_xml_agent_workflow.py::TestRrel35DeliveryClears`):

- generic `Bullet → BLLT` (`value_source = enum_map`);
- ERM `Bullet → OTHR` (`value_source = asset_policy`), materialised and asset-default;
- removing the override reverts to `BLLT` (config-driven, not hard-coded);
- a delivery frame carrying RREL35 = `OTHR` is `deliverable`, not `delivery_invalid`;
- XML generation remains gated.
