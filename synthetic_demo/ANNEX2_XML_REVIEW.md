# ESMA Annex 2 XML Review — Current Demo Pack Status

**Scope:** `synthetic_demo` demo pack in this repository revision.  
**Target schema family:** `DRAFT1auth.099.001.04` (Annex 2 / auth.099).  
**Expected namespace:** `urn:esma:xsd:DRAFT1auth.099.001.04`.

---

## Current truth source

- Demo orchestration points to the Annex 2 builder path (`engine/gate_5_delivery/xml_builder_annex2.py`) and Annex 2 workbook/sheet.  
- Delivery preflight report is present (`SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_delivery_report.json`) and indicates preflight `PASS`.  
- The final XML artifact file (`synthetic_demo/output/SYNTHETIC_012026_annex2.xml`) is **not present** in this pack revision.

Because the XML file is not bundled here, this markdown cannot assert a fresh in-pack XSD verdict from file inspection.

---

## Interpretation for demo readers

- This demo pack is **Annex 2**, not Annex 12.
- Namespace/version references in the HTML should align to Annex 2 (`auth.099`).
- Any older markdown text claiming a specific XSD FAIL/PASS from a missing XML file should be treated as stale legacy review output from a prior run context.

