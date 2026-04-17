# ESMA Annex 2 XML Review — SYNTHETIC_012026_annex2.xml

**File:** `synthetic_demo/output/SYNTHETIC_012026_annex2.xml`  
**Schema:** `DRAFT1auth.099.001.04` v1.3.0 (ESMA Annex 2 — Non-ABCP Underlying Exposure Report)  
**Records:** 36 `UndrlygXpsrRcrd` elements  
**Size:** 627.2 KB  

---

## 1. Well-formedness

**Result: PASS**

The XML is well-formed per the XML 1.0 specification. The file declares UTF-8 encoding, has a single root element (`Document`), all tags are correctly nested and closed, and no illegal characters or malformed CDATA sections are present. ElementTree parses the file without error.

---

## 2. Namespace

**Result: PASS (namespace correctly declared)**

```xml
<Document
  xmlns="urn:esma:xsd:DRAFT1auth.099.001.04"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="urn:esma:xsd:DRAFT1auth.099.001.04 DRAFT1auth.099.001.04_1.3.0.xsd">
```

- The default namespace `urn:esma:xsd:DRAFT1auth.099.001.04` matches ESMA's official namespace for the v1.3.0 non-ABCP residential real estate exposure report.
- The XSI namespace is correctly used for `schemaLocation`.
- **Note:** `schemaLocation` uses a relative path (`DRAFT1auth.099.001.04_1.3.0.xsd`). For production delivery the XSD file must be co-located with the XML, or the path must be updated to an absolute URI.

---

## 3. Data Quality

**Result: PARTIAL — 4 enum fields contain raw lender values, not ESMA codes**

Numeric fields (balance, rate, LTV, age, valuation) map correctly:

| Field (RREL code) | Sample XML value | Expected type |
|---|---|---|
| `RREL1` (Loan ID) | `DEMO-0001` | String — correct |
| `RREL3` (Current balance) | `177334.06` | Decimal — correct |
| `RREL5` (Interest rate) | `7.108` | Decimal (%) — correct |
| `RREL8` (Origination date) | `2025-06-07` | ISO 8601 date — correct |
| `RREL11` (Current LTV) | `37.8919` | Decimal (%) — correct |
| `RREL14` (Property value) | `468000` | Decimal — correct |

Enum fields that failed normalisation:

| RREL / RREC code | Field | Raw value in XML | Required ESMA code | Impact |
|---|---|---|---|---|
| `RREC9` | Property type | `Detached House` | `RHOS` | XSD validation failure |
| `RREL42` | Interest rate type | `FIX` | `FXPR` | XSD validation failure |
| `RREL27` | Purpose | `Home improvements` | `IMRT` | XSD validation failure |

108 instances total (36 loans × 3 fields). The raw lender values were passed through without translation because `enum_mapping.yaml` does not contain mappings for these specific source values.

---

## 4. XSD Availability and Coverage

**Result: XSD present, validation fails due to enum values**

The XSD is available at `config/system/DRAFT1auth.099.001.04_1.3.0.xsd` (821 KB). The xml_builder_annex2.py script attempts XSD validation after construction and reports failure when enum constraints are violated.

The XSD enforces:
- Element ordering (code-order-yaml defines RREL/RREC sequence within each record)
- Cardinality (mandatory vs optional fields)
- Enum value sets (property type, rate type, purpose, amortisation type, etc.)
- Data type constraints (dates as `xs:date`, amounts as `xs:decimal`)

---

## 5. Element Ordering

**Result: PASS (no ordering violations detected)**

The XML builder uses `config/system/esma_code_order.yaml` to sequence RREL/RREC elements within each `UndrlygXpsrRcrd`. A spot-check of DEMO-0001 confirms the element ordering follows the prescribed sequence: `UndrlygXpsrId` → `UndrlygXpsrData` → `ResdtlRealEsttLn` → `PrfrmgLn` → activity date details → balance details → interest rate details → collateral details → obligor details.

---

## 6. Top 3 XSD Failure Causes

### Cause 1 — Unmapped property type enum (RREC9)
**Frequency:** 36/36 loans  
The lender tape contains free-text property type descriptions (`Detached House`, `detached`, `Semi Detached`, `Flat / Apartment`, `Bungalow`). The pipeline's `enum_mapping.yaml` does not include these specific source strings, so the raw values are projected to XML unchanged. The XSD restricts `RREC9` to ESMA codes (`RHOS`, `FLAS`, `BUNG`, `SEMI`, etc.).  
**Fix:** Add source → ESMA code mappings to `enum_mapping.yaml` for all observed property type variants.

### Cause 2 — Unmapped interest rate type enum (RREL42)
**Frequency:** 36/36 loans  
The lender tape uses inconsistent free-text for rate type (`Fixed`, `variable`, `FIXED `, `Variable rate`). Gate 2 normalises these to `FIX` or `VAR` but the XSD requires `FXPR` (fixed), `FLPR` (floating), or `MXPR` (mixed). Neither the intermediate normalisation target nor the enum map covers the final XSD code set.  
**Fix:** Update the canonical enum set for `interest_rate_type` to map `FIX` → `FXPR`, `VAR` → `FLPR`.

### Cause 3 — Unmapped purpose enum (RREL27)
**Frequency:** 36/36 loans  
Loan purpose is free-text in the source (`Home improvements`, `Refinance`, `Equity release`, `Debt Consolidation`, `Purchase Main Residence`). No mapping to ESMA codes (`IMRT`, `REFI`, `EQRL`, `DBTC`, `BUYR`) exists in the current `enum_mapping.yaml`.  
**Fix:** Add purpose → ESMA code mappings to `enum_mapping.yaml` for all observed purpose variants.

---

## Summary

| Check | Result |
|---|---|
| Well-formedness | PASS |
| Namespace declaration | PASS |
| Data quality — numerics | PASS |
| Data quality — enums | FAIL (3 fields, 108 instances) |
| XSD availability | PASS (v1.3.0, 821 KB) |
| Element ordering | PASS |
| XSD validation | FAIL (enum values) |

The XML is structurally sound and numerically correct. The only failures are enum code mapping gaps for three fields. Correcting `enum_mapping.yaml` for `property_type`, `interest_rate_type`, and `purpose` would bring the file to full XSD compliance with no structural changes needed.
