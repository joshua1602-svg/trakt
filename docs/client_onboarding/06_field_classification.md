# Field classification — and the questions removed

Every one of the 69 catalogue fields, classified into five buckets, with the
objective of asking as little as possible while still generating a complete
governed configuration.

## The result

| Bucket | Fields |
|---|---:|
| **1 — Client must provide** (cannot be inferred or defaulted) | 25 |
| **2 — Operator decides** (business decision) | 7 |
| **3 — Conditionally required** (only for certain products) | 4 |
| **4 — Automatically inferred** (sample pack, jurisdiction, previous answers) | 10 |
| **5 — System default or derived** (never asked unless overridden) | 23 |
| **Total** | **69** |

**Required fields a human must answer: 33 → 15.**

Most of the 25 in bucket 1 are optional (registered address, disclaimer, an
authorised approver). What actually gates approval is far smaller:

```
client_name · jurisdiction                          about the client
legal_name · roles · lei* · country*                per legal entity
reporting_contact_name · reporting_contact_email    who Trakt talks to
operational_contact_name · operational_contact_email
display_name · portfolio_type                       per portfolio
client_id · portfolio_id                            proposed by Trakt, confirmed
products                                            what the client buys
IVSS8                                               risk retention method, if
                                                    investor reporting applies
* conditional on the entity holding the originator role
```

Verified end to end: a client is onboarded to a complete, activated
configuration from **six answered steps** — name and country, one entity with
its roles and identifier, four contact fields, a sample pack, one portfolio name
and type, and the reporting products. Everything else follows.

---

## Bucket 1 — Client must provide

Facts only the client knows. These are the irreducible core.

| Field | Section | Required |
|---|---|---|
| Client name | client | **yes** |
| Jurisdiction | client | **yes** |
| Legal name | entities | **yes** |
| Roles | entities | **yes** |
| Company registration number | entities | no |
| Registered address | entities | no |
| Primary reporting contact | contacts | **yes** |
| Reporting email | contacts | **yes** |
| Operational contact | contacts | **yes** |
| Operational email | contacts | **yes** |
| Authorised approver | contacts | no |
| Portfolio name | portfolios | **yes** |
| How the book was acquired | portfolios | **yes** |
| Who sends the data | sources | no |
| Reporting products | reporting | **yes** |
| Risk retention method (IVSS8) | regime | conditional |
| Trigger and structural flags (IVSS11/12/13/20/30) | regime | no |
| Logo, disclaimer, brand colour | presentation | no |
| Reporting calendar note | presentation | no |

## Bucket 2 — Operator decides

Business decisions Trakt's operator owns. Two of them are **proposed** by Trakt
so the operator edits rather than invents.

| Field | Note |
|---|---|
| Client identifier | Proposed from the client's name, whole words only, collision-checked |
| Portfolio identifier | Proposed as `direct_001` / `acquired_002` — the convention already in the source registry and the storage layout |
| Book (funded / pipeline) | Which books this client sends |
| How the book is held | Warehouse / SPV / managed |
| How files arrive | SFTP, secure upload, email, system to system |
| Sample file provided · Mapping approved | Operational state of the source |

## Bucket 3 — Conditionally required

Required only when the answer that triggers them has been given. A client who
does not receive investor reporting is never asked an Annex 12 question.

| Field | Required when |
|---|---|
| Legal Entity Identifier | the entity holds the originator role |
| Country of establishment | the entity holds the originator role |
| Reporting contact telephone | investor reporting is selected |
| Investor report recipients | investor reporting is selected |

## Bucket 4 — Automatically inferred

Filled by Trakt, **shown with where it came from, always overridable.** These
were the largest source of unnecessary questions.

| Field | Inferred from |
|---|---|
| Reporting currency | The jurisdiction (`SE` → `SEK`) |
| Reporting time zone | The jurisdiction (`SE` → `Europe/Stockholm`) |
| Asset class | The sample pack's column headers, scored against the signal sets the existing bootstrap agent uses |
| Owning legal entity | The only entity, where there is one |
| Portfolio reporting currency | The client's reporting currency |
| Reporting period convention | The book's cadence (`monthly` → calendar month end) |
| File format | The sample pack's file extensions |
| Expected files | The file names in the sample pack |
| Report title | The client's name |
| Day-count convention · Payment frequency | The portfolio's asset class |
| Report UK geography as GBZZZ | A UK jurisdiction |

An inferred field is still **required and still validated**. Inference
discharges a requirement; it does not remove one. A jurisdiction with no rule,
or a tape whose columns are ambiguous, leaves the field as a question rather
than guessing — and a tape scoring equally for two asset classes is treated as
no answer, because that is exactly when a human should look.

## Bucket 5 — System default or derived, never asked

### Derived — a fixed relationship to another answer

These are not questions and are **not independently editable**. Two copies of
one fact that can disagree is the duplication onboarding exists to remove.

| Field | Derived from |
|---|---|
| Originator name / LEI / country (RREL82–84) | The entity holding the originator role |
| Original lender LEI / country (RREL80–81) | The entity holding the original lender role |
| Securitisation name, reporting entity name (IVSS3/IVSS4) | The reporting entity |
| Securitisation identifier (IVSS1) | The reporting entity's identifier |
| Reporting contact person / telephone / email (IVSS5–7) | The contacts step |
| Risk retention holder (IVSS9) | The role of the entity holding the retention |
| Underlying exposure type (IVSS10) | The portfolio's asset class |
| Still originating | Direct books originate; acquired books do not |
| Reporting cut-off | The book's cadence |
| Included in regulatory reporting | The reporting products and the book's dataset |
| Portfolio (on a source) | The portfolio the delivery belongs to |

**This was the single biggest reduction: 15 regime fields, every one of them a
fact already given, restated in regulator vocabulary.** The Annex 2 originator
block used to be three questions asked immediately after the operator had
entered the same company as an entity.

### System default

| Field | Default |
|---|---|
| Environment | `production` |
| Expected cadence | `monthly` |
| NUTS classification year | `2021` |
| Entity reference | Minted by Trakt |
| Onboarding reference | `ONB-2026-0001` |

A default is declared on the field, not written into a form, so one added to a
future regime's declaration takes effect with no code change. Two rules keep it
honest. Defaults are applied **last**, after every lookup and every derivation,
because a default is what applies when nothing is known — never a value that
displaces what is. And they only fill a part of the form the operator has
reached: an item exists because someone created it, a regime block exists
because someone bought the product, a plain section counts as reached once it
holds an answer. A case started and abandoned holds nothing.

There is no catalogue default for the reporting time zone, and there should not
be. A time zone is a fact about the client's country, so it comes from the
jurisdiction table and from that table's own fallback where the country is not
listed. A default here would be one country's clock quietly imposed on every
client Trakt ever onboards.

## Never collected at all

Declared under `not_collected` in the catalogue so the model is a complete
account, not only of the questions:

| Field | Why |
|---|---|
| Reporting period, static reporting date | Delivery-specific — supplied with each delivery |
| Expected schema fingerprint, file role signatures | Learned from the first real delivery; asking would be asking the operator to guess |
| Delivery location | Derived from the production storage layout |
| No-data policy | The governed regime rules carry defaults |

---

## What makes this safe

Three properties, each tested:

1. **Nothing is silent.** Every inferred or proposed value is recorded in the
   case's provenance with a plain-English origin, rendered under the field as
   "Taken from the currency used in SE" and listed again on the review screen.
2. **The operator always wins.** A field the operator has just edited is left
   alone for that save — including a deliberate blank. Without that rule,
   clearing an inferred currency to retype it produced `SEKEUR`.
3. **Inference discharges requirements; it does not remove them.** An inferred
   field is validated exactly as a typed one. A rule that fails to fire leaves a
   blocking problem, not a quietly incomplete configuration.

## What was deliberately not inferred

| Candidate | Why not |
|---|---|
| Legal name, country and address from the LEI via GLEIF | It is the obvious next win — one identifier would answer three fields — but it makes onboarding depend on a live external service. That is a decision to take deliberately, with a caching and outage story, not a rule to slip into a table. |
| Risk retention method (IVSS8) from the structure | The structure a book is held in does not determine the retention method. Guessing a regulatory declaration is worse than asking. |
| Reporting products from the asset class | Eligibility is derived; whether the client buys the product is commercial. |
| Operational contact from the reporting contact | Frequently different people. A wrong contact is discovered when a delivery fails. |
