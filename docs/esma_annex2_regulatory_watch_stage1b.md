# ESMA Annex 2 Regulatory Watch — Stage 1B (publication discovery)

**Status: Stage 1B complete. Observational only.**

Stage 1A answers *"has the machine-readable Annex 2 specification changed?"* by
normalizing and diffing the technical artefacts. It cannot see anything coming:
a consultation, a final report or a Q&A moves nothing in the workbook or the
XSD, so Stage 1A correctly reports `CURRENT` right up until the day a new
technical package lands.

Stage 1B is that missing upstream layer.

> Detect new ESMA publications that may affect Annex 2, classify their status
> and relevance, preserve evidence, and create a reviewable
> regulatory-development record that can feed Stage 1A when a technical
> artefact actually changes.

---

## 1. The distinction Stage 1B exists to preserve

| | regulatory **development** | technical **regulatory delta** |
| --- | --- | --- |
| what it is | a publication that *may* affect Annex 2 | an exact, machine-readable difference in the Annex 2 field set |
| evidence | narrative | the workbook and the XSD |
| who decides | **Stage 1B** classifies it and asks a human to look | **Stage 1A**, and only Stage 1A |

Stage 1B therefore never converts narrative text into a regime change and never
computes a field delta. The strongest thing it can say about a technical
package is `TECHNICAL_SPEC_CHANGE`, which means *"Stage 1A should now run"* —
not *"these fields changed"*.

Two structural guards, both pinned by test:

* `regulatory_watch/discovery/` never imports Stage 1A's comparator,
  normalizer, impact engine or changelog parser. What cannot be reached cannot
  be reimplemented.
* `RegulatoryDevelopment` and `TechnicalSpecLinkage` have **no field** that
  could hold a field-level change — no `esma_code`, no `nd_allowed`, no
  `old_value`/`new_value`.

---

## 2. Architecture

```
config/regulatory_watch/esma_annex2_publication_sources.yaml   upstream allowlist
        │
        ▼
  sources.py ── retrieval.py (opt-in, isolated, off by default)
        │
        ▼
  parsers.py            bytes ─► RawPublication[]   (rss / atom / html listing)
        │                        structure-strict: a wrong shape FAILS,
        │                        it never degrades to an empty list
        ▼
  state.py              deterministic identity + per-source fingerprints
        │                        NEW / UNCHANGED / REVISED
        ▼
  triage.py             Stage A deterministic filter (broad, reason recorded)
        │               Stage B bounded classifier (deterministic; optional LLM)
        ▼
  linkage.py            technical URL ─► MATCHES_STAGE1A_SOURCE
        │                             │  NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED
        │                             └► quotes a Stage 1A report (never runs it)
        ▼
  pipeline.py           one weekly pass
        ▼
  report.py             JSON (UI-agnostic contract) + short Markdown
        ▼
  cli.py                discover → dedupe → classify → report
```

### Files

| path | role |
| --- | --- |
| `config/regulatory_watch/esma_annex2_publication_sources.yaml` | upstream ESMA publication allowlist |
| `regulatory_watch/discovery/contracts.py` | data contract + controlled vocabularies |
| `regulatory_watch/discovery/sources.py` | allowlist loading/validation |
| `regulatory_watch/discovery/retrieval.py` | optional, isolated network fetch |
| `regulatory_watch/discovery/parsers.py` | RSS / Atom / HTML listing parsers |
| `regulatory_watch/discovery/state.py` | identity, deduplication, revision |
| `regulatory_watch/discovery/triage.py` | two-stage relevance + classification |
| `regulatory_watch/discovery/linkage.py` | the Stage 1A boundary |
| `regulatory_watch/discovery/pipeline.py` | the weekly pass |
| `regulatory_watch/discovery/report.py` | JSON + Markdown emitters |
| `regulatory_watch/discovery/cli.py` | `discover` command |
| `scripts/run_regulatory_watch_discovery_demo.py` | end-to-end demonstration |

---

## 3. Official ESMA sources

| source | type | criticality | why |
| --- | --- | --- | --- |
| `https://www.esma.europa.eu/document/securitisation-reporting-technical-standards-xml-schema-and-instructions` | publication page | **gating** | The same URL Stage 1A's artefact allowlist cites for the auth.099 workbook and XSD. A change here is the strongest possible signal of an Annex 2 technical-spec change. |
| `https://www.esma.europa.eu/rss.xml` | RSS feed | **gating** | ESMA's site-wide publication feed. Broad by design — it is how a Q&A or consultation that never touches the technical-standards page gets caught. |
| `https://www.esma.europa.eu/esmas-activities/digital-finance-and-innovation/securitisation` | landing page | corroborating | The securitisation topic page. Overlaps the feed; catches items the feed categorises oddly. |

There is no crawling, no link-following and no search-engine use. Every URL must
be `https` on an official ESMA host, enforced by the loader.

### Reachability — read before the first live run

**No endpoint above has been fetched from this environment.** The build/CI
network policy denies `esma.europa.eu` (confirmed: the proxy returns 403 to
`CONNECT www.esma.europa.eu:443`). Their response shapes are therefore
*declared*, not confirmed, and every source carries `verified_reachable: false`.

That is safe by construction, not by hope:

* retrieval is off by default and all tests run from committed fixtures;
* the parsers validate structure and fail with `UNEXPECTED_SCHEMA` rather than
  returning an empty item list, so a wrong guess about the response shape
  surfaces as a failure and never as "no new publications";
* the first live run is a calibration run — confirm each URL, set
  `verified_reachable: true`, and adjust `parser` if ESMA serves a different
  format than declared.

---

## 4. Controlled vocabularies

**Publication type** — `CONSULTATION` · `FINAL_REPORT` · `RTS_ITS` · `Q_AND_A`
· `TECHNICAL_INSTRUCTION` · `TEMPLATE_SCHEMA_UPDATE` · `VALIDATION_RULE_UPDATE`
· `OTHER_REGULATORY_PUBLICATION`

**Development status** — `INFORMATION_ONLY` · `POTENTIAL_FUTURE_CHANGE` ·
`FUTURE_CHANGE_EXPECTED` · `INTERPRETATION_REVIEW_REQUIRED` ·
`TECHNICAL_SPEC_CHANGE` · `REVIEW_REQUIRED` · `NOT_RELEVANT`

**Discovery status** — `DISCOVERY_OK` · `DISCOVERY_PARTIAL` (a corroborating
source failed) · `DISCOVERY_FAILED` (a gating source failed). A discovery
failure is never readable as "nothing new".

**Technical linkage** — `NO_TECHNICAL_ARTEFACT` · `MATCHES_STAGE1A_SOURCE` ·
`NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED`

**Seen state** — `NEW` · `UNCHANGED` · `REVISED`

### Classification rules

Two ordering decisions carry the whole classifier, and both were wrong in the
first implementation until the demonstration exposed them:

1. **Document form beats subject matter.** "Consultation Paper on amendments to
   the disclosure templates" is a `CONSULTATION` about templates, not a
   template update. Checking subject-matter signals first marked every Q&A and
   consultation mentioning templates as a `TECHNICAL_SPEC_CHANGE` — precisely
   the confusion Stage 1B exists to prevent.
2. **…except when the publication *carries* the artefact.** The auth.099
   package bundles the XSD, the templates *and* the reporting instructions;
   typing it as an instructions document would hide the one publication that
   most needs Stage 1A to run. "Carrying" is tested on artefact-*format* terms
   in the title (`xml schema`, `xsd`, `schema and instructions`, `… package`)
   plus linked artefact files — never on subject terms like "disclosure
   template", which any commentary uses.

Dates (`effective_date`, `implementation_date`, `consultation_deadline`) are
captured only where the publication states them in an unambiguous ISO form.
"Early next year" stays unknown; a wrong effective date is worse than an absent
one.

---

## 5. Identity and deduplication

Identity is **source-independent** and keyed on the canonical URL — the one
thing every source agrees on for a web publication. ESMA publishes the auth.099
package in the feed *and* on the technical-standards page with a different
identifier in each; keying on the source, or on whichever identifier a source
supplied, would raise the same publication once per source every week.

Fingerprints, by contrast, are kept **per source**, because two sources
describe the same publication differently (the feed carries an editorial
summary, the listing page a link label). Comparing across sources made a
publication flip between `REVISED` and `UNCHANGED` depending on which source
was scanned first. A revision therefore means *"this source's description of
this publication changed"* — the signal that actually indicates ESMA re-issued
something.

An unreadable state store raises rather than looking empty: treating it as
empty would re-raise every historical publication as new and destroy the
dedupe guarantee.

---

## 6. The semantic classifier

Off by default. The deterministic classifier is the baseline and is what
survives if a backend fails, returns unparseable output, or answers outside the
controlled vocabulary. The backend is bounded by construction:

* it sees only title, source metadata and extracted official text — asserted by
  test, including that no Annex 2 field code or config ever reaches it;
* it must answer a strict schema, validated against the vocabulary;
* low confidence, out-of-vocabulary or a raised exception ⇒ `REVIEW_REQUIRED`;
* **escalate-only**: it may make a finding more urgent, never less. A model
  cannot quietly downgrade a `TECHNICAL_SPEC_CHANGE` to `INFORMATION_ONLY`.

It reuses `engine/onboarding_agent/llm_json.py` (the repo's hardened JSON
extractor) when importable, with a local fallback so the package stays
standalone. No new dependency is introduced.

---

## 7. Running it

```bash
# one weekly pass from committed fixtures (what CI and the demo do)
python -m regulatory_watch.discovery.cli discover \
    --out-dir output/regulatory_watch/discovery \
    --fixture-dir tests/fixtures/regulatory_watch/publications \
    --state output/regulatory_watch/discovery/seen_publications.json

# live (requires retrieval_enabled: true in the allowlist)
python -m regulatory_watch.discovery.cli discover \
    --out-dir output/regulatory_watch/discovery --live

# quote a Stage 1A report into a development record
... --stage1a-report <development_id>=<path to Stage 1A JSON>

# end-to-end demonstration (runs Stage 1A for real for the technical item)
python scripts/run_regulatory_watch_discovery_demo.py
```

Exit codes: `0` completed (`DISCOVERY_OK`/`DISCOVERY_PARTIAL`), `2` a gating
source could not be checked, `3` the allowlist or state store is unusable.

### Scheduling — deliberately not built here

The repository already has a scheduling abstraction: `function_app.py`
registers Azure Functions timer triggers (see `deliver_teams_notifications`,
`@app.timer_trigger(schedule=...)`). Stage 1B adds none of its own. Stage 2
deployment registers a weekly trigger against the same pattern — e.g.
`0 0 7 * * 1` (Mondays, 07:00) — calling
`regulatory_watch.discovery.cli.main(["discover", ...])`. Weekly is sufficient;
there is no polling loop anywhere in this package.

---

## 8. Limitations — what Stage 1B cannot establish

1. **Semantic relevance classification is not legal advice.** A development
   record is a prompt for a human, not a compliance conclusion.
2. **Narrative publications require human interpretation.** A Q&A can change
   how a field must be *populated* without changing anything machine-readable.
   Stage 1B can only say `INTERPRETATION_REVIEW_REQUIRED`.
3. **A `CURRENT` technical-spec status does not mean nothing is pending.**
   Stage 1A says the artefacts have not moved; Stage 1B's whole point is that a
   consultation or final report may be in flight at the same time. Read the two
   together.
4. **Publication monitoring cannot change the active regime.** No status here
   modifies configuration, promotes a version or generates XML.
5. **Endpoint shapes are unconfirmed.** See §3 — the first live run is a
   calibration run.
6. **No archived ESMA feed exists in the repository.** The two real historical
   publications used in the tests carry their real titles, URLs and dates, but
   the surrounding feed document is a structural fixture. No regulatory history
   has been invented.
7. **Coverage is bounded by the allowlist.** A relevant ESMA publication that
   appears on none of the three declared sources is not seen. That is the cost
   of not being a crawler, and it is why the feed is deliberately broad.
8. **The deterministic filter is tuned for recall, not precision.** It will pass
   through items a reviewer dismisses in seconds. That trade is deliberate: a
   false negative is invisible forever.

---

## 9. What Stage 1B deliberately does not do

No FCA. No other annex. No autonomous regulatory interpretation. No active
config changes. No OCC UI. No approval workflow. No config promotion. No
Teams/email alerts. No generic crawler. No scheduler. Stage 1A remains the sole
authority for deterministic technical-spec comparison.

## 10. Stage 2 (not implemented)

1. **OCC Regulatory Config view** — consumes both JSON contracts as-is
   (Stage 1A `contract_version` 1.1.0, Stage 1B 1.0.0).
2. **Human approval** — a reviewer accepts/dismisses each development record
   and each Stage 1A delta; today every finding is advisory.
3. **Versioned config promotion** — turning an approved Stage 1A delta into a
   new, explicitly-activated Annex 2 regime version.
4. **Scheduled execution** — a weekly Azure Functions timer trigger against the
   existing pattern, plus alerting on `DISCOVERY_FAILED`.
5. **Live calibration** — confirm the three endpoints, set
   `verified_reachable: true`, and enable retrieval.
