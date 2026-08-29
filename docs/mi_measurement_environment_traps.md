# Measurement environment traps — misconfigurations that do not fail

**One class of defect, three instances.** Each is an environment misconfiguration
that produces a **clean, quotable, entirely wrong measurement** instead of an
error. None announces itself. Each has cost a run.

> The shared property: the instrument reports a plausible table, every row
> internally consistent, and nothing in the output says the footing was wrong.
> A wrong number that looks wrong gets caught. These produce wrong numbers that
> look right.

---

## 1. The book fallback — a misconfigured path answers from the wrong book

**Recorded in** `question_interpretation/shipped_shapes.py::_apply_env`.

> *"misconfigured environment variables do NOT fail — the service falls back to
> a synthetic demo book and answers plausibly from the wrong data."*

`MI_AGENT_DATA_CSV` naming a path that does not resolve does not raise. The
service falls back to the bundled synthetic demo book and answers every question
from it.

**Detection:** `shipped_shapes --verify` prints the source actually loaded, and
prints `*** THIS IS THE BUNDLED SYNTHETIC DEMO BOOK, NOT YOUR FILE ***` when the
fallback fired.

---

## 2. `TRAKT_RUNTIME_MODE` defaults to production — every shape rates ABSENT

**Found during the Gate run.** `trakt_core.runtime.runtime_mode()` resolves
anything unset or unrecognised to `production`. Under production,
`trakt_core.policy` refuses both books as synthetic fixtures:

> *"The active data source is the synthetic demonstration dataset. Trakt does
> not answer production questions from synthetic data."*

**What it looks like if you miss it:** all 29 time-series phrasings refuse,
`route=None` on every one, and **every shape rates ABSENT with zero silent
drops**. That is a coherent, publishable table. It reads as a capability
finding. It is a configuration error.

It is doubly deceptive because ABSENT-with-honest-refusals is the *expected
shape* of a result on this surface — the failure mode mimics the finding.

**Correct setting:** `TRAKT_RUNTIME_MODE=development` (or `test`). This is the
sanctioned path for fixture data — `conftest.py` sets `test` for the whole suite
— and it cannot take effect in a deployed environment, because
`validate_runtime_mode` refuses a non-production mode when the Azure markers are
present.

---

## 3. A missing `demo_platform/workspace/` — one period, so no time axis exists

**Found during the Gate run.** The workspace is gitignored and regenerable, so a
fresh clone does not have it. Without it the book carries a **single reporting
period**.

**What it looks like if you miss it:** routes resolve, questions are answered,
nothing errors — but no artifact can carry more than one distinct period value,
so **every time-series shape rates ABSENT, including T1**, the simplest shape on
the surface and one that is PROVEN in a correctly built environment.

**Rebuild:**

```
python -m demo_platform.run_demo --generate --onboard --orchestrate
```

---

## What a correctly configured baseline looks like

Both books, deterministic arm, on the time-series surface:

```
  silent drops                    0
  honest refusals            21 of 29
  T1 PROVEN   T2 PARTIAL   T3-T6 ABSENT   T7 PARTIAL   T8 PARTIAL
```

**Reproduce the baseline before trusting any comparison built on top of it.**
Traps 2 and 3 both produce all-ABSENT tables, and the only way to tell a trap
from a finding is that the baseline does not match.

---

## The countermeasure

`question_interpretation/mi_capability_recontent.py` **refuses to measure** when
trap 2 or 3 is present, rather than producing a table:

```
REFUSING TO MEASURE — the environment would rate everything ABSENT:
  * TRAKT_RUNTIME_MODE resolves to 'production'; trakt_core.policy will refuse
    both books as synthetic fixtures and EVERY shape will rate ABSENT.
    Set TRAKT_RUNTIME_MODE=development (or test).
```

It exits non-zero. `tests/test_mi_capability_recontent.py` pins both refusals and
pins that a correctly configured environment is **not** blocked — a preflight
that always refuses is as useless as one that never does.

**Any new instrument on this surface should call `preflight()` before its first
question.** The cost of the check is one environment read; the cost of skipping
it is a published table that has to be withdrawn.
