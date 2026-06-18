# XML-Readiness Remediation Roadmap (Annex 2)

This roadmap converts the Delivery/XML Agent v1 delivery issues into a practical,
ordered remediation sequence to reach **XML preview** and then **production XML**.

```text
Onboarding → Transformation → Validation → Projection → Delivery/XML Agent v1 → (remediation) → XML
                                                          ^^^^^^^^^^^^^^^^^^^^^  current stage
```

> **Source of this roadmap.** It is derived from (a) the Delivery/XML Agent v1
> blocker taxonomy (`engine/delivery_xml_agent/remediation.py`) and (b) the
> **actual `63_delivery_issues.csv`** produced by running the agent in Codespaces
> against the real `run_pre_xml_final_check_3` projection manifest. The per-group
> field-code lists below are **reconciled to that real run** (see "Reconciled
> figures" below). No production XML is generated.

Delivery/XML Agent v1 verdict on the real run (`run_pre_xml_final_check_3`):

```
delivery_xml_ran               = true
delivery_normalisation_complete = false
xml_generation_allowed         = false
xml_generated                  = false
ready_for_xml_delivery         = false
next_agent                     = operator_config_projection_remediation
```

### Reconciled figures (run_pre_xml_final_check_3)

Delivery-normalised frame: **163,282 rows** (matches the projected target frame).

| Delivery status | Rows |
| --- | --- |
| deliverable | 83,930 |
| blocked | 47,306 |
| delivery_invalid (format/enum) | 1,526 |
| not_required_blank | 30,520 |
| **total** | **163,282** |

Delivery issues: **34** across 31 blocked ESMA codes (+ 1 format-invalid code,
+ structural + template-order). Issue mix by blocker type:

| Blocker type | Issues | Remediation group |
| --- | --- | --- |
| `client_onboarding_dependency` | 2 | 1 |
| `operator_or_config_dependency` | 7 | 2 |
| `config_dependency` | 11 | 3 |
| `source_mapping_unresolved` | 10 | 4 |
| `delivery_format_invalid` | 1 | 4 |
| `nd_default_rule_missing` | 1 | 5 |
| `delivery_structure_deferred` | 1 | 6 |
| `template_order_incomplete` | 1 | 7 |

> One ESMA code accounts for all **1,526** `delivery_invalid` rows (every record
> for that field fails the regime regex/enum check). Identify it from the
> `delivery_blocker_type = delivery_format_invalid` row in `63_delivery_issues.csv`
> and fix the projected value / format rule (group 4).

---

## Top-level action plan

### A. What must be fixed before any XML preview

Every Delivery readiness gate must pass before `xml_generation_allowed` flips to
true (and `--allow-xml-preview` only writes a preview when it is). That means
clearing **all delivery-blocking rows and the required structural gates**:

1. **Client onboarding** — formal identifier policy for **RREL1 / RREL2**
   (these also satisfy the *required header/report metadata* gate).
2. **Operator review** — valuation / property / rate source ambiguity
   (**7 codes** incl. RREC9, RREC13, RREC17, RREL43 — see group 2).
3. **Config mapping** — controlled enum/config mappings (**11 codes** incl.
   RREL27 purpose — see group 3).
4. **Source / projection mapping** — `source_mapping_unresolved` (**10 codes**)
   + the 1 `delivery_format_invalid` code (1,526 rows).
5. **ND / default policy** — `nd_default_rule_missing` (**RREL82**) for a
   *mandatory* field with no allowed/selected ND/default.
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
| **Field codes** | RREC1, RREC9, RREC13, RREC17, RREL9, RREL43, RREL69 (7 codes on the real run) |
| **Current blocker type** | `operator_or_config_dependency` (frame status `blocked_operator_or_config_dependency`) |
| **Business meaning** | RREC9 property type; RREC13 current valuation; RREC17 original valuation; RREL43 current interest rate; plus RREC1, RREL9, RREL69 surfaced on the real run with the same operator-review source ambiguity. Multiple candidate source columns, none confirmed. |
| **Recommended owner** | Operator |
| **Recommended action** | Operator confirms the authoritative source field per code; never auto-ND/default an ambiguous valuation/rate. Once confirmed, re-run Projection. |
| **Needed before XML preview?** | **Yes** (mandatory fields; block `no_blocked_target_frame_rows`) |
| **Needed before production XML?** | **Yes** |

### 3. Config mapping decisions

| | |
| --- | --- |
| **Field codes** | RREC7, RREC14, RREC16, RREL10, RREL11, RREL14, RREL26, RREL27, RREL44, RREL45, RREL75 (11 codes on the real run) |
| **Current blocker type** | `config_dependency` (frame status `blocked_operator_or_config_dependency`, projection disposition `config_mapping_required`) |
| **Business meaning** | Controlled enum / config mappings from source values to ESMA codes — e.g. RREL27 loan **purpose**. The real run surfaced 11 such config-mapping fields. |
| **Recommended owner** | Config / rules |
| **Recommended action** | Add the purpose enum mapping (`transform.enum_map`) in `annex2_delivery_rules.yaml`; re-run Projection. |
| **Needed before XML preview?** | **Yes** |
| **Needed before production XML?** | **Yes** |

### 4. Source / projection mapping gaps

| | |
| --- | --- |
| **Field codes** | `source_mapping_unresolved` (10): RREC2, RREC3, RREC4, RREC5, RREL3, RREL4, RREL5, RREL35, RREL67, RREL68, RREL84 — plus 1 `delivery_format_invalid` code (1,526 rows; identify from `63_*`) |
| **Current blocker type** | `source_mapping_unresolved` (and `delivery_format_invalid` for malformed values) |
| **Business meaning** | Target field has related source data but no confirmed projection rule (notably the RREL3/RREL4/RREL5 new/original identifier chain), or a projected value fails the regime regex/enum format check. |
| **Recommended owner** | Projection / Transformation |
| **Recommended action** | Add the explicit projection/source-mapping rule (never guess); fix values that fail `validators.regex` / `enum_map`. |
| **Needed before XML preview?** | **Yes** |
| **Needed before production XML?** | **Yes** |

### 5. ND / default policy gaps

| | |
| --- | --- |
| **Field codes** | RREL82 (1 on the real run). RREL24 / RREL40 are **already resolved** to ND5 (`projected_from_transformed`, deliverable) and are *not* in this group. |
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
| **Field codes** | 20 required codes absent from `esma_code_order.yaml::Record` on the real run: RREC5, RREC21, RREC23, RREL4, RREL6, RREL62, RREL63, RREL64, RREL65, RREL66, RREL67, RREL68, RREL70, RREL72, RREL76, RREL78, RREL79, RREL80, RREL81, RREL82 |
| **Current blocker type** | `template_order_incomplete` |
| **Business meaning** | XSD requires strict `<xs:sequence>` order. `Record` lists 77 of the 107 universe codes; the 20 required codes above are missing from the order and cannot be placed deterministically. (Some, e.g. RREL4/RREL67/RREL68/RREL82/RREC5, are *also* blocked upstream — fixing the order is independent of fixing their values.) |
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
