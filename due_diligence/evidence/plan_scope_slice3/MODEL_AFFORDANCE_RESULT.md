# Slice 3 portfolio affordance — the contract repaired, and measured once

The boundary run found the named source axis at 0/4 and the cause in our own
contract. That contract was repaired once. This is the one fresh measurement.

---

## 1. Run identity

| | |
|---|---|
| `LIVE_OPUS_CALLS` | **10** (authorised 10, spent 10, zero retries) |
| Model served | `claude-opus-5` on all ten, read back from the API |
| Bank | `slice3_affordance_v1`, sha256 `1acec678…11d18bc`, committed at `88fdf7f9` **before** the first call; the dispatch pinned the same hash |
| Run | `actions/runs/34605911511`, questions asked 13:44:40–13:47:22Z |
| Registry visible to the model | yes — three books, names and aliases only |
| Boundary | question → Opus → CandidateIntent → compiler → plan. No execution, no MI API, no serving path |

---

## 2. Result

```
cases 10   interpreted 10   PASS 10   CLARIFIED 0   FAIL 0
compile checked 9 (M1–M9)   PASS 9
```

| measure | result |
|---|---|
| `BACK_BOOK_TO_FUNDED` | **2/2** |
| `FRONT_BOOK_TO_PIPELINE` | **1/1** |
| `DIRECT_ROLE_PRESERVED` | **2/2** |
| `ACQUIRED_ROLE_PRESERVED` | **2/2** |
| `NAMED_SOURCE_ATOMIC` | **3/3** |
| `NAMED_SOURCE_REFERENCE_PRESERVED` | **3/3** |
| `SEASONING_INFERRED_FROM_BACK_BOOK` | **0** |
| `PHYSICAL_SOURCE_BINDINGS_AUTHORED_BY_MODEL` | **0** |
| `ATTRIBUTION_CAPABILITY_DROPS` | **0** |
| `SILENT_SCOPE_DROPS` / `WIDENINGS` / `NARROWINGS` | **0 / 0 / 0** |
| `TEMPORAL_DROPS` | **0** |

| | question | axis landed on | compiled scope |
|---|---|---|---|
| M1 | the back book balance | `base=funded` | *(none — the whole funded book)* |
| M2 | what is in the front book | `base=pipeline`, capability `pipeline` | *(none)* |
| M3 | acquired back book balance | `base=funded` + `lens=acquired` | `source_portfolio_type=acquired` |
| M4 | direct back book balance | `base=funded` + `lens=direct` | `source_portfolio_type=direct` |
| M5 | purchased loans | `lens=acquired` | `source_portfolio_type=acquired` |
| M6 | originated loans | `lens=direct` | `source_portfolio_type=direct` |
| M7 | ALP Acquired Back Book | `source_reference` only | `source_portfolio_id=alp_acquired` |
| M8 | the ALP back book *(alias)* | `source_reference` only | `source_portfolio_id=alp_acquired` |
| M9 | …each month | `source_reference` + `series` | `source_portfolio_id=alp_acquired` |
| M10 | how much of the increase came from the acquired back book | `base=funded` + `lens=acquired`, `period_movement`/`movement` | *(attribution — not compile-checked)* |

Not one case set `seasoning`. Not one authored an identifier. Not one dropped
the measure, the series or the attribution capability.

---

## 3. The model says why, in its own words

These are its non-blocking disclosures, and they name the two things that were
added to the contract:

**M1 — declined a book it could see.**
> *"The bare phrase 'back book' was read as the funded population, not as the
> governed book 'ALP Acquired Back Book' (alias 'ALP back book'), which the
> question does not name."*

**M2 — used the declared axes to rule out two wrong readings.**
> *"'Front book' read as the pipeline population … **per the governed scope
> axes**. It matches no governed source portfolio name, so it is not read as a
> named book, and it is not read as a seasoning restriction."*

**M8 — the atomic rule, applied exactly.**
> *"'ALP back book' matches the approved alias of the governed book 'ALP
> Acquired Back Book'; **taken as a name, so no acquired role and no back-book
> seasoning restriction were asserted separately**."*

**M10 — three axes readable in one phrase, and it still separated them.**
> *"'the acquired back book' read per governed scope axes as the funded
> population with the acquired origination role, not a seasoning restriction. It
> was not read as the governed source portfolio 'ALP Acquired Back Book', since
> the reader did not name that book."*

The registry was read and consciously **not** used on M1, M2 and M10, and used
on M7–M9. That is the axis distinction working, not a lookup succeeding.

---

## 4. Against the boundary run

| | boundary (`34b36c14`) | affordance (`88fdf7f9`) |
|---|---|---|
| named source usable | **0/4** | **3/3** |
| governed name kept atomic | 0/1 that would have served | **3/3** |
| seasoning inferred from a name or a lifecycle phrase | 3 cases | **0** |
| would have served a wrong population | **1** (M07, silently narrowed) | **0** |
| metadata retrievals, named cases | 7, 12, 12, 13 | **3, 3, 3** |

The retrieval counts are the clearest single number. The model was not searching
harder; it was searching for something that was not there. Told where to look,
it stops looking.

---

## 5. What did not move

* The 135-case signed-off corpus replays from disk with **135/135** outcomes and
  **135/135** plan identities identical. Zero model calls.
* `mi_agent_api`: 18 failed / 1863 passed before and after, the **same nine test
  ids** — no regression.
* Roles still bind one predicate; total funded is still unnarrowed; a governed
  name still resolves and an unknown one still refuses; `seasoning_segment`
  still binds when it is explicitly asked for; attribution is still its own
  capability.
* `outcomes.py` and the frozen banks are untouched. 23 of the prompt's 26
  paragraphs are present word for word, and every pre-existing metadata tool
  schema is byte-identical — both now asserted mechanically.

---

## 6. One boundary deliberately not crossed

`mi_agent/seasoning.py::_SEGMENT_PHRASES` still maps the raw phrase
`\bback book\b` → `seasoning_segment` on the **legacy serving path**. It is a
second owner, it carries its own governed lending-window ruling of 2026-08, and
changing it is a serving-semantics change rather than a model affordance. The
interpretation_v2 contract is now unambiguous about the axis; the legacy raw-text
path is not, and that is a decision for whoever owns that ruling.

---

```
SLICE_3_MODEL_AFFORDANCE = PASS
```
