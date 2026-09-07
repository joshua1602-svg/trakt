# The MI geography basis

*2026-09-07. Closes the region ownership conflict recorded in
`docs/mi_analytics_architecture_current_state_audit.md` and reversed in
`cdb84db`.*

## The question nobody had answered

A loan carries two geographies, and they are different facts:

| | |
|---|---|
| **borrower / obligor geography** | where the obligor is |
| **collateral / property geography** | where the security is |

For a lifetime mortgage they usually coincide. For an auto loan they do not
coincide at all. For a buy-to-let they routinely disagree. So

> "What is the balance by region?"

has no answer until somebody has decided **which geography this book reports
on**. Nobody had. Three separate layers each answered it by taking whichever
geography column happened to be populated first:

| layer | what it did |
|---|---|
| `funded_prep._coalesce_group_dimensions` | gap-filled the **borrower** column, row by row, from the **collateral** column |
| `engine.region_taxonomy.apply` | harmonised from the first populated of its source fields, obligor first |
| `llm_query_parser._preferred_region` | bound the head of a fixed preference order, chosen by column *presence* |

That is not a semantics. It is the absence of one, and it produced two
measurable defects on the live platform book (11,035 loans, 2026-06-30):

* **a borrower region column full of collateral geography.** A row whose obligor
  region was never collected reported the property's region as the borrower's,
  with nothing in the answer to say so. Where the obligor column held an ESMA
  no-data code the error was sharper: `ND1` means *not collected* — a
  declaration the lender makes — and overwriting it fabricates a fact.
* **a harmonised column that resolved for nobody.** The obligor column holds
  ITL3 codes (`TLI35`, `TLK30`, …), which no ITL1 taxonomy resolves, so
  `canonical_region_reporting` came back NULL on **11,035 of 11,035** rows,
  `region_mapping_method: unresolved`. It was still the head of the parser's
  preference order and still present on every serving frame, so a region
  question bound it anyway. The readable region names sat untouched in the
  column beside it.

## What decides now

The basis is a property of the **asset**. A lifetime mortgage's regional
concentration is a concentration of houses; an auto book's is a concentration of
people. It is read from the governed configuration hierarchy that already
records what kind of book this is:

| precedence | layer | owns |
|---|---|---|
| 1 | the question itself | a basis stated outright — `explicit_query` |
| 2 | portfolio registry | an exception for one book — `portfolio_registry` |
| 3 | client configuration | an exception for one client — `client_override` |
| 4 | asset configuration | the ordinary case — `asset_class_default` |
| 5 | — | nothing established one — `unconfigured`, and it says so |

```
OCC onboarding captures the asset class   (required, operator-confirmed)
  -> config/client/config_client_<id>.yaml    portfolio.asset_class
  -> config/asset/mi_geography.yaml           asset_class -> basis
  -> mi_agent.mi_geography.GeographyContract  the contract for a request
```

**The ordinary case needs no portfolio registry.** The asset layer says what
region means for a kind of book; the client layer says what kind of book this
client runs. Joining them is the whole mechanism, and OCC already writes the
client key (`writes_to: client_config:portfolio.asset_class`) that MI reads.

That join is what was missing at first. The basis came from the portfolio
registry alone, keyed by `source_portfolio_id`, so a deployment whose client
configuration declared `portfolio.asset_class: equity_release` — a complete
governed statement — still reported `basisSource: unconfigured` because a second
file listing its portfolios by id happened not to exist, and generic "region"
fell back to a field order no layer had chosen. The registry keeps what it
genuinely owns (per-portfolio metadata, and a per-portfolio exception) and loses
only the burden of being mandatory.

`config/asset/mi_geography.yaml`:

| asset class | primary basis |
|---|---|
| equity_release, lifetime_mortgage | collateral |
| residential_mortgage, residential_real_estate | collateral |
| commercial_real_estate | collateral |
| auto_finance, auto_loan | borrower |
| equipment_leasing, leasing | borrower |

Anything else has **no governed default**. It is not given one: MI would rather
say it does not know than assume, and onboarding raises it for operator review
like any other unestablished asset fact.

A portfolio may override the class default in the registry
(`mi_geography: {primary_basis: borrower}`). Onboarding *seeds* that key and
never overwrites it, so an operator's decision survives a re-run.

## The contract

| question says | measured on |
|---|---|
| "region", "by region", "geography", "area", "the Scottish balance" | the configured **primary basis** |
| "borrower region", "obligor region" | the **borrower** basis, always |
| "property region", "collateral region", "asset region" | the **collateral** basis, always |

Explicit language always wins over the configured default. And a stated basis
the book cannot support is **refused** — never the other geography quietly
wearing the label the reader asked for, because nothing in such an answer tells
them.

## What this module does not own

`mi_agent.mi_geography` names a **basis**. It resolves no geography *value*: no
place name, postcode, ITL or NUTS code appears in it or in its configuration.

* `mi_agent.region_resolution` remains the only owner of the ITL ladder that
  maps a reader's term onto a book's value. It is also what answers "does this
  column carry a *place*" — which is how a column of no-data declarations is
  recognised as carrying no geography without MI ever encoding one regulatory
  code.
* `engine.region_taxonomy` remains the only harmoniser of region vocabularies.
  It was not modified. What changed is that MI now **states the source columns**
  it harmonises from, taking them from the configured basis, via the
  `source_fields` argument `apply()` already accepted. On the live book that
  turns 0/11,035 resolved into **11,033/11,035, method `exact`, eleven governed
  regions**.

## Regulatory separation

Regulatory reporting has its own geography fields, its own sources and its own
contract, and this work does not touch them. `tests/test_mi_geography_leaves_the_regime_alone.py`
asserts the boundary in both directions:

* no module under `engine/gate_*`, `engine/regime_contract`,
  `engine/annex_delivery_agent`, `engine/projection_agent`,
  `engine/delivery_xml_agent` or `engine/region_taxonomy.py` imports
  `mi_agent.mi_geography` or mentions it;
* `mi_agent.mi_geography`'s executable code names no place and no no-data code;
* `ND1` survives MI preparation unchanged on both preparation paths while MI
  answers "Wales" on the same book — both true at once, because they are
  statements about different fields and neither is derived from the other.

The full-scale Annex 2 XML/XSD reproduction (11,035 records, 107 projected
fields, XSD PASSED, 0 errors) is byte-identical before and after this work.

## Which columns carry which basis

Fields are grouped by basis and, within a basis, by **granularity tier**:

| basis | tier | columns |
|---|---|---|
| collateral | reporting | `collateral_geography`, `property_region` |
| collateral | code | `geographic_region_collateral`, `geographic_region_collateral_itl3` |
| borrower | code | `geographic_region_obligor`, `geographic_region_obligor_itl3` |

Two columns in the **same tier** are one fact delivered under different names,
so they gap-fill each other freely. Two columns in **different tiers** do not:
filling a column of readable names from a column of ITL3 codes leaves one column
speaking two vocabularies — "London" beside "TLC31" — and splits a breakdown
into categories no reader can reconcile. The tier is chosen once per book, by
`field_for_basis`: the most readable tier the book actually populates.

`canonical_region_reporting` / `canonical_region_detail` belong to **no basis**.
They are derived across both, so they cannot answer a question that is about
one. They sit at the tail of the last-resort order, for a frame that carries
nothing else, and are reachable by name ("harmonised region", "canonical
region").

`*_itl3` columns carry a basis and gap-fill their tier head, but are never
offered as the region **axis**: a finer geography is not another spelling of
Region, and binding a breakdown to it turns eleven regions into a hundred and
seventy sub-regions nobody asked for.

## One contract per request

A dozen readers ask "which column does 'region' mean" between the request edge
and the receipt — the dimension binder, the categorical filter, the population
resolver, the facet detector, the coverage ledger, the dashboard's evolution
series. If any two answer differently, a correct answer is refused for having
lost a concept it applied, or a wrong one is published as right.

`mi_service.execute_governed_mi_query` opens one context for the request,
`_run_analysis` binds the contract into it as soon as the book is in hand, and
every reader takes it from there (`llm_query_parser._ACTIVE_GEOGRAPHY`). The
public entry points still accept the contract as an explicit argument: the
plumbing is implicit, the interface is not.

## Evidence

| what | where |
|---|---|
| the handoff, end to end | `tests/test_mi_geography_basis_lifecycle.py` |
| one basis never fills another | `mi_agent_api/tests/test_one_geography_basis_is_never_filled_from_another.py` |
| the query contract, end to end | `mi_agent_api/tests/test_region_means_the_configured_geography.py` |
| regulatory separation | `tests/test_mi_geography_leaves_the_regime_alone.py` |
| four surfaces agree cell for cell | `tests/test_regional_coherence_across_surfaces.py`, over `due_diligence/evidence/mi_geography/regional_coherence.py` |
