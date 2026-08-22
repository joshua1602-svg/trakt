# Running your own questions, and what a lender's file must look like

Two audiences. §1–§4 are for whoever opens the Codespace. **§5 is written to be
sent to a lender as-is.**

Everything here was executed; no step is described from reading the code.

---

## 1. Does anything already take a file of questions? Now yes.

**It did not.** Every runner read a fixed bank; none accepted free text, a
questions file, or a book argument. The change is arguments on the runner that
already exists, not a second runner.

```bash
# your questions, the bundled demo book, deterministic parser
python3 -m question_interpretation.shipped_shapes --questions myquestions.txt

# your questions, YOUR book
python3 -m question_interpretation.shipped_shapes \
    --questions myquestions.txt \
    --csv /path/to/your_canonical_typed.csv

# with the model in the loop (needs ANTHROPIC_API_KEY in the environment)
python3 -m question_interpretation.shipped_shapes \
    --questions myquestions.txt --csv /path/to/book.csv --llm on

# print the book actually loaded, and stop
python3 -m question_interpretation.shipped_shapes --csv /path/to/book.csv --verify
```

`myquestions.txt` is one question per line; blank lines and `#` comments ignored.

**It reports, it does not grade.** A grade needs a declared expectation and your
questions have none — inventing a verdict is the defect this programme exists to
close. Each question prints the route, the measure and aggregation, the filters
and dimensions applied, the population covered, and the answer text.

The graded bank is untouched: running with no arguments still reports 15/15.

---

## 2. Can the transformation run from the command line? Yes — executed here.

```bash
python3 -m engine.gate_2_transform.canonical_transform \
    /path/to/your_tape.csv \
    --registry config/system/fields_registry.yaml \
    --output-dir out/
```

Produces `<stem>_canonical_typed.csv` and `<stem>_transform_report.json`.

**Measured on a deliberately minimal file: 9 columns in, 18 out, 200 rows in,
200 rows out.** The agent then loaded it and answered.

**The caveat that matters.** This is Gate 2. It expects columns already using
registry names. A lender's raw extract with its own column names needs Gate 1
(semantic alignment) first — that is the mapping step, and it is where a human
confirms what each of the lender's columns means. **If a lender can deliver a
file using the canonical names in §5, Gate 1 is not needed and this one command
is the whole pipeline.**

---

## 3. THE TRAP: a wrong path does not fail — it answers from the demo book

This cost a false result during this work and is the single most important thing
on the page.

Pointing the service at a book through the wrong environment variable **does not
error**. It silently falls back to a bundled synthetic demo book and answers
plausibly. In the run that caught it, the agent reported 36 rows and answered
four questions correctly-looking — **none of the 200 loans in the file under test
appeared in any of them.**

**So verify before trusting any number:**

```bash
python3 -m question_interpretation.shipped_shapes --csv /path/to/book.csv --verify
```

```
kind     explicit_csv                    <- must NOT say synthetic_demo
label    your_canonical_typed.csv        <- must be YOUR file
path     /path/to/your_canonical_typed.csv
rows     200                             <- must match your file
columns  23
```

`--verify` exits non-zero if `--csv` was given and the demo book loaded anyway.
Every `--questions` run prints the same block first, and shouts if the demo book
is in use.

---

## 4. What must be present in the Codespace

| requirement | check | if missing |
|---|---|---|
| Python 3.11 with `pandas`, `pyyaml`, `fastapi` | `python3 -c "import pandas, yaml, fastapi"` | `pip install -r requirements.txt` |
| The repo at the merged state | `git log --oneline -1` | `git pull` |
| **A book to query** | `--verify` above | supply `--csv`, or generate the demo book |
| `ANTHROPIC_API_KEY` — only for `--llm on` | `echo ${ANTHROPIC_API_KEY:+set}` | deterministic mode needs no key |
| `pyarrow` — optional | | only for parquet; CSV is unaffected |

**The demo book is gitignored (26 MB) and a fresh clone will not have it.** That
is what made earlier sessions skip 103 tests and nearly produce a false control.
`--verify` catches it in one second.

Trust nothing from a run whose `--verify` block you have not read.

---

## 5. FOR THE LENDER — what your file must contain

*Send from here down.*

We need one row per loan, as CSV, using the column names below. Extra columns are
harmless; missing ones narrow what can be asked rather than causing a failure.

### The minimum viable file — 6 columns

With these six, the agent answers whole-book totals and counts, averages and
weighted averages, thresholds and filters, and breakdowns by any band derived
from them.

| column | meaning |
|---|---|
| `loan_identifier` | unique per loan |
| `current_outstanding_balance` | current balance, currency units |
| `original_principal_balance` | advance at origination |
| `current_loan_to_value` | current LTV, percent |
| `origination_date` | ISO `YYYY-MM-DD` |
| `youngest_borrower_age` | years |

**Verified: a file with these plus interest rate and region produced a working
book and answered totals, counts, average LTV, LTV-band breakdowns, threshold
questions and a portfolio summary.**

### Strongly recommended — each unlocks a specific question type

| column | what it unlocks |
|---|---|
| `current_interest_rate` | rate questions and rate-band breakdowns |
| `geographic_region_obligor` **or a postcode column** | regional exposure and concentration |
| `property_valuation_amount` | valuation questions, LTV re-derivation |
| `reporting_date` | seasoning, vintage, and anything over time |
| `origination_channel` | channel breakdowns |
| `account_status` | status filters and breakdowns |
| `source_portfolio_type` | direct vs acquired comparisons |

**Measured limits without them:** on a file lacking usable geography,
*"balance by region"* is **refused** with the reason stated. Without a reporting
date, *"balance by vintage"* is **refused** — *"'Vintage' is not available in this
dataset"*. Nothing is guessed and no near-miss is substituted.

### What we derive — do not send these

Supplied 61 columns produce 76 the agent can query. **The 15 we derive:** LTV
band, original-LTV band, ticket-size band, age band, interest-rate band,
seasoning bucket, seasoning segment, months on book, time-on-book band, vintage
year, and four canonical region fields.

### Unrecognised columns

**Kept, not dropped — verified.** A column we do not recognise passes through
untouched and is simply not queryable. Nothing is discarded silently and nothing
is guessed at.

### Formats

- CSV, UTF-8, one header row.
- Dates ISO `YYYY-MM-DD`.
- Numbers unformatted — no thousands separators, no currency symbols.
- Percentages as numbers: `43.15` for 43.15%.
- Blanks empty, not `"N/A"` or `"-"`.
- One row per loan, at one reporting date.

### What happens if something is wrong

The agent **declines rather than approximates**. A field absent from your file
produces *"that is not available in this dataset"* naming the field. A question
spanning more periods than you supplied is told how many periods exist. A request
that cannot be honoured in full is withheld rather than answered over a wider
population with a footnote.

That behaviour is deliberate: a confident wrong number is worse than a refusal.
