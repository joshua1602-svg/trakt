# XML-Readiness Remediation Roadmap (Annex 2)

This roadmap converts the Delivery/XML Agent v1 delivery issues into a practical,
ordered remediation sequence to reach **XML preview** and then **production XML**.

```text
Onboarding → Transformation → Validation → Projection → Delivery/XML Agent v1 → (remediation) → XML
                                                          ^^^^^^^^^^^^^^^^^^^^^  current stage
```

> **Source of this roadmap.** It is derived from (a) the **confirmed pre-XML
> state** of run `run_pre_xml_final_check_3` as recorded in the task brief, and
> (b) the Delivery/XML Agent v1 blocker taxonomy
> (`engine/delivery_xml_agent/remediation.py`). The field/category lists below
> should be **reconciled against the actual `63_delivery_issues.csv`** once the
> agent has been run in Codespaces against the real projection manifest (see
> "How to regenerate the issue list" at the foot of this doc). No production XML
> is generated.

Delivery/XML Agent v1 verdict on the current run (expected):

```
delivery_xml_ran               = true
delivery_normalisation_complete = false
xml_generation_allowed         = false
xml_generated                  = false
ready_for_xml_delivery         = false
next_agent                     = operator_config_projection_remediation
```

---

## Top-level action plan

### A. What must be fixed before any XML preview

Every Delivery readiness gate must pass before `xml_generation_allowed` flips to
true (and `--allow-xml-preview` only writes a preview when it is). That means
clearing **all delivery-blocking rows and the required structural gates**:

1. **Client onboarding** — formal identifier policy for **RREL1 / RREL2**
   (these also satisfy the *required header/report metadata* gate).
2. **Operator review** — valuation / property / rate source ambiguity for
   **RREC9, RREC13, RREC17, RREL43**.
3. **Config mapping** — purpose enum mapping for **RREL27**.
4. **Source / projection mapping** — any remaining `source_mapping_unresolved`.
5. **ND / default policy** — any `nd_default_rule_missing` for a *mandatory* field
   that has no allowed/selected ND/default.
6. **Template / order** — add every required code missing from
   `esma_code_order.yaml::Record` so XML ordering is deterministic.

### B. What can be deferred until production XML

* **Delivery structure** — RREL↔RREC record hierarchy / collateral nesting /
  header-pool metadata shaping (v1 tags `record_group` only; nesting deferred to v2).
* Completing the **full 107-code order** (only *required* codes block preview).
* True **XSD validation** of a built tree, currency/`Ccy` amount source, and the
  header/pool/report metadata contract.

### C. What needs client / lender input

* **RREL1 (ScrtstnIdr) / RREL2 (Original Underlying Exposure Identifier)** —
  formal identifier policy. ND is **not** permitted for either, so a real
  identifier scheme must be agreed with the client/lender.

### D. What needs operator review

* **RREC9** property type source ambiguity.
* **RREC13** current valuation source ambiguity.
* **RREC17** original valuation source ambiguity.
* **RREL43** current interest rate source ambiguity.

These were deliberately **not** ND1-defaulted by Projection; an operator must
confirm the correct source before a value is projected.

### E. What needs config / rules work

* **RREL27 purpose** — enum mapping in config (`config_mapping_required`).
* Any `nd_default_rule_missing` items — define an *allowed* ND/default rule in
  `annex2_delivery_rules.yaml` / asset config, never a silent fill.

### F. What needs structural XML design

* RREL↔RREC hierarchy and **collateral cardinality** (nested vs separate section
  vs repeated related records).
* Header / pool / report-level metadata contract and currency source.
* Completing `esma_code_order.yaml` to all 107 workbook codes.

### Already deliverable (no action needed)

* **RREL24 maturity_date** → `ND5` (`projected_from_transformed`) — candidate deliverable.
* **RREL40 debt_to_income_ratio** → `ND5` (`projected_from_transformed`) — candidate deliverable.

These flowed through the asset-config ND5 policy and are **not** blockers; they
appear as `delivery_status = deliverable` in `62_delivery_normalised_frame.csv`.

---

## Remediation groups

### 1. Client onboarding decisions

| | |
| --- | --- |
| **Field codes** | RREL1, RREL2 |
| **Current blocker type** | `client_onboarding_dependency` (frame status `blocked_client_onboarding_dependency`) |
| **Business meaning** | Securitisation identifier (RREL1) and original underlying-exposure identifier (RREL2). Neither permits ND, so a formal client identifier policy is required to anonymise/assign IDs. |
| **Recommended owner** | Client onboarding (client/lender input) |
| **Recommended action** | Agree the formal identifier scheme during onboarding; feed it through Transformation so Projection can materialise real values. |
| **Needed before XML preview?** | **Yes** (also satisfies the required-header-metadata gate via RREL1) |
| **Needed before production XML?** | **Yes** |

### 2. Operator decisions

| | |
| --- | --- |
| **Field codes** | RREC9, RREC13, RREC17, RREL43 |
| **Current blocker type** | `operator_or_config_dependency` (frame status `blocked_operator_or_config_dependency`) |
| **Business meaning** | RREC9 property type; RREC13 current valuation; RREC17 original valuation; RREL43 current interest rate. Source ambiguity remains — multiple candidate source columns, none confirmed. |
| **Recommended owner** | Operator |
| **Recommended action** | Operator confirms the authoritative source field per code; never auto-ND/default an ambiguous valuation/rate. Once confirmed, re-run Projection. |
| **Needed before XML preview?** | **Yes** (mandatory fields; block `no_blocked_target_frame_rows`) |
| **Needed before production XML?** | **Yes** |

### 3. Config mapping decisions

| | |
| --- | --- |
| **Field codes** | RREL27 |
| **Current blocker type** | `config_dependency` (frame status `blocked_operator_or_config_dependency`, projection disposition `config_mapping_required`) |
| **Business meaning** | Loan **purpose** — needs a controlled enum mapping from the source value set to the ESMA code list. |
| **Recommended owner** | Config / rules |
| **Recommended action** | Add the purpose enum mapping (`transform.enum_map`) in `annex2_delivery_rules.yaml`; re-run Projection. |
| **Needed before XML preview?** | **Yes** |
| **Needed before production XML?** | **Yes** |

### 4. Source / projection mapping gaps

| | |
| --- | --- |
| **Field codes** | any remaining `source_mapping_unresolved` (reconcile from `63_*`; e.g. occupancy/related-field derivations) + any `delivery_format_invalid` |
| **Current blocker type** | `source_mapping_unresolved` (and `delivery_format_invalid` for malformed values) |
| **Business meaning** | Target field has related source data but no confirmed projection rule, or a projected value fails the regime regex/enum format check. |
| **Recommended owner** | Projection / Transformation |
| **Recommended action** | Add the explicit projection/source-mapping rule (never guess); fix values that fail `validators.regex` / `enum_map`. |
| **Needed before XML preview?** | **Yes** |
| **Needed before production XML?** | **Yes** |

### 5. ND / default policy gaps

| | |
| --- | --- |
| **Field codes** | any `nd_default_rule_missing` (mandatory field, blank, no allowed/selected ND/default). RREL24 / RREL40 are **already resolved** to ND5 and are *not* in this group. |
| **Current blocker type** | `nd_default_rule_missing` |
| **Business meaning** | A mandatory field is absent and there is no *allowed* ND or configured default to fall back to. |
| **Recommended owner** | Config / policy |
| **Recommended action** | Define an *allowed* ND/default rule in the regime/asset config where ESMA permits it; otherwise escalate to operator/source. Never silently fill. |
| **Needed before XML preview?** | **Yes** (for mandatory codes) |
| **Needed before production XML?** | **Yes** |

### 6. Delivery structure gaps

| | |
| --- | --- |
| **Field codes** | n/a (structural — applies to RREL exposure vs RREC collateral grouping) |
| **Current blocker type** | `delivery_structure_deferred` |
| **Business meaning** | The long target frame tags `record_group` (RREL = underlying exposure, RREC = collateral) but does not yet nest collateral under exposures or shape header/pool/report metadata. ESMA `auth.099` models collateral as repeatable related records. |
| **Recommended owner** | Delivery / XML design (v2) |
| **Recommended action** | Decide RREL↔RREC nesting & collateral cardinality; design header/pool/report metadata + currency source; add a runtime `annex2_xml_paths.yaml`. |
| **Needed before XML preview?** | **No** (v1 does not gate on it; record-group is preserved) |
| **Needed before production XML?** | **Yes** |

### 7. Template / order gaps

| | |
| --- | --- |
| **Field codes** | mandatory frame codes absent from `esma_code_order.yaml::Record` (reconcile from `60_delivery_manifest.json::missing_required_order_code_count`) |
| **Current blocker type** | `template_order_incomplete` |
| **Business meaning** | XSD requires strict `<xs:sequence>` order. `Record` lists 77 of the 107 universe codes; required codes missing from the order cannot be placed deterministically. |
| **Recommended owner** | Delivery / config |
| **Recommended action** | Add the missing **required** codes to `esma_code_order.yaml::Record` (full 107-code completion can follow before production). |
| **Needed before XML preview?** | **Yes** (gated by `template_code_order_complete`) |
| **Needed before production XML?** | **Yes** |

---

## Gate → group mapping

| Delivery readiness gate | Remediation group(s) that clear it |
| --- | --- |
| `projection_complete` / `ready_for_delivery_normalisation` | 1, 2, 3, 4, 5 (resolving upstream blockers re-runs Projection) |
| `no_delivery_blocking_projection_issues` | 1, 2, 3, 4, 5 |
| `no_blocked_target_frame_rows` | 1, 2, 3, 4, 5 |
| `no_mandatory_blank_without_nd` | 5 |
| `no_delivery_format_violations` | 4 |
| `required_header_metadata_present` | 1 (RREL1) |
| `record_grouping_determinable` | preserved today (RREL/RREC) — no action |
| `template_code_order_complete` | 7 |
| `ready_for_xml_delivery` | always false until all of the above pass |

---

## How to regenerate the issue list (Codespaces)

```bash
cd /workspaces/trakt
git checkout claude/delivery-xml-agent-v1-review-y5gb7u
git pull --ff-only origin claude/delivery-xml-agent-v1-review-y5gb7u

python -m engine.delivery_xml_agent.workflow \
  --projection-manifest onboarding_output/client_001/run_pre_xml_final_check_3/output/projection/50_projection_manifest.json

# then inspect:
python scripts/inspect_delivery_xml_readiness.py \
  onboarding_output/client_001/run_pre_xml_final_check_3/output/delivery_xml
```

Reconcile the **field codes** in groups 1–7 above against the printed
`Remediation groups` block and `63_delivery_issues.csv`. **No production XML is
generated** — XML preview remains hard-gated behind the readiness gates and the
`--allow-xml-preview` flag.
