# Slice 3 — legacy/fallback semantic owner audit

Audited at `85c64a38`. Offline, replayed intents only. No live Opus call, no
deployment, and **no product code changed**.

---

## 1. The serving order, from `mi_agent_api/mi_service.py`

```
1913  parsed = ParsedQuestion.parse(question, …)     ← LEGACY PARSE, ALWAYS, FIRST
        └─ parse_with_repair  →  resolve_seasoning_role(spec, question, cols)
              └─ seasoning.resolve_population_predicate
                    └─ lending_windows_named  →  _SEGMENT_PHRASES
        ⇒ spec.filters["seasoning_segment"] = "Back Book" | "Front Book"

2095  served = _governed_serving_attempt(routed)      ← GOVERNED ATTEMPT, SECOND
2222  served = _governed_serving_attempt(result)         (the point-in-time site)
        └─ plan_serving_canary.serve(...)  →  payload | None

      if served is not None:  return the GOVERNED envelope
      else:                   return the LEGACY envelope ALREADY BUILT ABOVE
```

The legacy envelope is not an alternative that is computed if needed. It is
computed **first, on every request**, and the governed attempt can only replace
it. `serve()` returns `None` — and therefore hands back that envelope — on
every one of: principal not in the canary, interpreter failure, CLARIFY,
REFUSE, ineligible plan, no snapshot catalogue, execution error, reconciliation
failure, empty measured population, render failure, or any unexpected error.

---

## 2. Controls

### A — `"What is the back book balance?"`

| | |
|---|---|
| **Governed** (replayed intent M1) | `population.base = funded`, no scope predicates |
| perimeter | `slice1`, **eligible = True**, reason `''` |
| plan population | `{base: funded, lens: all, seasoning: any, scope_predicates: []}` |
| **Legacy** (same question, production seam) | `filters = {"seasoning_segment": "Back Book"}` |

The governed reading is **the whole funded book**. The legacy reading is
**the seasoned cohort only**. Two different populations, same sentence.

### B — `"What is in the front book?"`

| | |
|---|---|
| **Governed** (replayed intent M2) | `population.base = pipeline`, capability `pipeline` |
| perimeter | `slice1`, **eligible = False**, reason `CAPABILITY_NOT_GENERIC` |
| **Legacy** | `filters = {"seasoning_segment": "Front Book"}` |

This one needs no forced condition. **The governed path declines it today**, so
a canary principal asking it receives the legacy answer — a subset of the
*funded* book — for a question the governed ontology says is about the
*pipeline*. Those are different datasets, not a wider and a narrower slice of
one.

### C / D — the same two questions, forced into a safe fallback

Forced by the commonest real decline: the interpreter unreachable (transport,
not semantics). `MI_AGENT_PLAN_SERVE=canary`, principal allow-listed.

```
handles(canary principal) = True

C  "What is the back book balance?"   serve() -> None
      legacy filters = {"seasoning_segment": "Back Book"}
D  "What is in the front book?"       serve() -> None
      legacy filters = {"seasoning_segment": "Front Book"}
```

### E — is `seasoning.py` still needed?

`_SEGMENT_PHRASES` holds ten phrases. Six are vintage vocabulary owned **nowhere
else** — the lending-window table does not cover them:

```
recent originations   newly originated   recently originated
legacy book           seasoned book      seasoned loans
```

Proven live: `"What is the balance of seasoned loans?"` →
`{"seasoning_segment": "Back Book"}`; `"…recent originations versus the back
book"` → `None` (two segments named, so a comparison is left alone, exactly as
the signed-off NL7 canonical requires).

Four are the contested lifecycle phrases: `front book`, `back book`,
`backbook`, `new originations`.

**A second consumer makes this more than a deletion.** `mask_segment_phrases`
uses the same table to blank these spans before the place-resolver runs, so that
"front book" is not read as a region called *Front* (defect P1J-1):

```
'How many acquired loans are in the front book?'
   masked -> 'How many acquired loans are in the           ?'
```

Narrowing the table without preserving the masking vocabulary would reintroduce
a geography defect this estate has already paid for.

---

## 3. A third finding, not asked for, found while tracing

`population.base` **does not reach the bound spec and is not reconciled.**

```
spec_for_plan(plan)     base=funded  →  filters {}   metric current_outstanding_balance
                        base=pipeline→  filters {}   metric current_outstanding_balance   ← identical
                        base=whole_book→filters {}   metric current_outstanding_balance   ← identical

check_eligibility       base=pipeline + generic_analysis  →  (True, '', '')   ← ADMITTED
_governed_plan_coverage reconciles requested.filters and requested.dimensions only
```

The slice-1 runtime executes against the frame the caller supplies, and
`mi_service` supplies the **funded** frame. So a governed plan stating
`population.base = pipeline` under `generic_analysis` would be declared
eligible, executed over funded rows, pass coverage (which never looks at
`base`), and be served as a governed answer.

Nothing shields this except the model's choice of capability. On M2 it happened
to choose `capability = pipeline`, which the perimeter refuses — that is luck,
not a control. And the lifecycle ontology landed in Slice 3 is precisely what
makes `base = pipeline` reachable from ordinary language for the first time.

`scope_predicates` are **not** affected: role and named-source predicates do
reach `spec.filters` via `plan_predicates` and are reconciled. Only `base` is
unguarded.

---

## 4. Verdict

The same words mean different populations depending on whether governed
execution happened to succeed. For "front book" the governed reading is not
reachable at all today. The conflict is not theoretical and it is not confined
to a fallback edge: it is the default for one of the two phrases.

```
SLICE_3_LEGACY_OWNER_AUDIT = REPAIR_REQUIRED
```

Not repaired. Nothing in this audit changed product code.
