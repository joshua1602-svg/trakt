### The sweep's own reach, and the case it cannot see

One 80-bank finding was reported as a classifier artefact. It is worth naming
precisely, because it is one instance of a general limitation rather than a
one-off.

**The case.** *"Show the funded balance bridge."* The sweep read
`spec.metric = None`, could not type the answer, and flagged a CURRENCY question
as returning an untyped result. The answer is in fact correct and is currency:
*"funded balance moved from £1.93bn in 2026-04 to £1.96bn at 2026-06 — a net
change of +£32.6m."*

**Why the sweep missed it.** It types an answer from the spec's metric slot. A
bridge decomposes a movement rather than reporting one measure, so the route
carries its measure in its own identity and leaves that slot empty. Every
specialist route does the same: bridge, cohort progression, forecast
extrapolation, risk limits, portfolio summary.

**How far that reaches, counted rather than estimated:**

| bank | in the sweep's reach | outside it |
|---|---|---|
| 252 calibration | 207 typed and executed | 21 `parse_only` (never executed), 21 declared `any`, 24 refuse/clarify |
| 30 simple-MI | 29 | 1 — *"What has changed versus the prior month?"* |
| 80 wide | 77 | 3 — the bridge, the same movement question, and *"What is the KFI to completion conversion rate?"* |
| 44-variation | all 44 | none — this one types from `analyticalFindings`, not from the spec |

**So it is a limitation of the instrument, and a fixable one.** The 44-bank
sweep already avoids it by typing from the structured findings a route emits
rather than from the spec it was built with. Applying the same to the other
three banks would bring the four specialist-route cases into reach; it was not
done here because the 30/80-bank capture records the spec and not the findings,
and widening the capture is a change to the measurement harness rather than to
the product.

**What this means for the result.** The type-conformance claim in this report
covers 313 of the 366 executed cases across the four banks. The remaining
cases are not asserted to be well-typed; they are asserted to be outside what
this instrument can check, and are listed above by name so the gap is countable
rather than implied.
