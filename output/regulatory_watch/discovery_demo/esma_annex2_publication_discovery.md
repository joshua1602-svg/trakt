# ESMA Annex 2 Regulatory Watch — publication discovery

*Discovery date: 2026-02-16T00:00:00Z · annex2-discovery/1.0.0 · report contract v1.0.0 · Stage 1B, observational only.*

- **Discovery status:** `DISCOVERY_OK`
- Sources checked: 3
- Source failures: 0
- New ESMA publications: 7
- Previously seen (skipped): 0
- Potentially relevant to Annex 2: 5
- Human review required: 5

## Potentially relevant to Annex 2

1. **Securitisation reporting technical standards — XML schema and instructions (version 1.3.1)**
   Status: `TECHNICAL_SPEC_CHANGE` · type `TEMPLATE_SCHEMA_UPDATE` · relevance `RELEVANT` · confidence HIGH
   Reason: publishes an Annex 2 technical artefact (template/schema); Stage 1A must run to determine the exact field deltas
   Technical artefacts changed: Yes
   Stage 1A comparison: 1 regulatory delta(s); Trakt implementation impacts: 3
   Matches Stage 1A source: `esma_annex2_message_workbook, esma_annex2_reporting_instructions, esma_annex2_sample_message, esma_annex2_xml_schema`
   Human review: Yes
   URL: https://www.esma.europa.eu/document/securitisation-reporting-technical-standards-xml-schema-and-instructions

2. **Securitisation reporting validation rules package**
   Status: `TECHNICAL_SPEC_CHANGE` · type `VALIDATION_RULE_UPDATE` · relevance `RELEVANT` · confidence HIGH
   Reason: publishes updated validation rules; Stage 1A must run to determine the exact rule deltas
   Technical artefacts changed: Yes
   Stage 1A comparison: **not yet run** — technical-source verification required
   ⚠ `NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED`: https://www.esma.europa.eu/document/securitisation-validation-rules-package
   Human review: Yes
   URL: https://www.esma.europa.eu/document/securitisation-validation-rules-package

3. **Final Report — Draft RTS and ITS on securitisation reporting**
   Status: `FUTURE_CHANGE_EXPECTED` · type `FINAL_REPORT` · relevance `RELEVANT` · confidence HIGH
   Reason: a final report settles ESMA's position; technical artefacts typically follow
   Technical artefacts changed: No
   Human review: Yes
   URL: https://www.esma.europa.eu/document/final-report-draft-rts-and-its-securitisation-reporting

4. **Consultation Paper on amendments to the securitisation disclosure templates**
   Status: `POTENTIAL_FUTURE_CHANGE` · type `CONSULTATION` · relevance `RELEVANT` · confidence HIGH
   Reason: a consultation proposes change; nothing is in force yet
   Technical artefacts changed: No
   Consultation closes: 2026-05-15
   Human review: Yes
   URL: https://www.esma.europa.eu/document/consultation-paper-securitisation-disclosure-templates-review

5. **Questions and Answers on the Securitisation Regulation — disclosure templates**
   Status: `INTERPRETATION_REVIEW_REQUIRED` · type `Q_AND_A` · relevance `RELEVANT` · confidence HIGH
   Reason: a Q&A can change how a field must be populated without changing the machine-readable specification
   Technical artefacts changed: No
   Human review: Yes
   URL: https://www.esma.europa.eu/document/qa-securitisation-regulation-disclosure-templates

6. **ESMA report on the securitisation market in the European Union**
   Status: `INFORMATION_ONLY` · type `OTHER_REGULATORY_PUBLICATION` · relevance `NOT_RELEVANT` · confidence MEDIUM
   Reason: matched only broad securitisation terms with no reporting, disclosure or technical-standards signal
   Technical artefacts changed: No
   Human review: No
   URL: https://www.esma.europa.eu/document/esma-report-securitisation-market-european-union

## Not relevant

1 publication(s) rejected by the deterministic relevance filter. Each retains its reason in the JSON report:

- ESMA publishes latest edition of its Spotlight on Markets newsletter — no ESMA Annex 2 relevance term matched the title, summary, categories or URL

> 2 publication sighting(s) were the same publication reached through more than one allowlisted source, and were merged into a single development record.

> A technical-spec-change candidate was found. Stage 1B does not determine field deltas — run Stage 1A against the linked artefacts.

---

*Stage 1B detects and classifies ESMA publications. It does not determine field-level Annex 2 changes — only Stage 1A does — and it never modifies the active regime.*
