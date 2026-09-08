# OCC Day-1 Operating Model

What the Operations Control Centre does for you when Client 1 goes live, what
you have to do, and what is deliberately not built yet.

Everything described here has been run end to end and proven. Nothing in this
document is a plan.

---

## 1. What OCC does

OCC is the control room for a client's reporting. In order, it:

- **onboards the client** — captures who they are, their portfolios, their legal
  entities and identifiers, and which reports they need, then creates the
  client's own governed configuration and switches them on;
- **knows what files to expect** and where they will arrive;
- **receives a delivery** the moment the file lands, and refuses anything from a
  client it has not been told about;
- **decides which reporting process is needed** — management information only,
  or management information plus the ESMA Annex 2 regulatory return — from the
  products chosen at onboarding, not from guesswork;
- **runs the data through three governed gates** that read the files, transform
  them, and check the figures;
- **asks you questions** wherever it cannot safely decide, and shows you what it
  found rather than asking you to trust it;
- **remembers your answers** so later months are quiet;
- **produces the governed management information** everything else is built on;
- **produces the Annex 2 return** where the client needs one, and validates it
  against the regulator's own schema before anyone sees it;
- **holds everything until you approve publication**;
- **records what happened** — every question, answer, approval, and the exact
  configuration and rule versions used.

---

## 2. A normal monthly reporting cycle

1. The lender's file arrives at its agreed location.
2. OCC picks it up automatically and recognises the client and the portfolio.
3. OCC decides whether this is an MI-only or MI + Annex 2 delivery.
4. OCC reads the file and matches its columns to the standard meanings.
5. **First month only:** OCC asks you about anything it cannot match with
   confidence. Later months normally ask nothing.
6. OCC transforms and checks the figures.
7. OCC assembles the governed management information.
8. **For a regulatory client**, OCC then projects the data, builds the XML and
   validates it against the regulator's schema.
9. OCC tells you everything is ready.
10. **You approve publication.** Nothing is published before you do.
11. The management information becomes available for reporting and queries, and
    the regulatory return is filed to its governed location.

**Where you are needed:** step 5 in the first month, and step 10 every month.

---

## 3. What is automated

- Picking up the file and identifying the client and portfolio.
- Refusing a delivery for a client who has not been onboarded and activated.
- Choosing MI or MI + Annex 2.
- Matching columns to meanings where it can do so safely.
- Applying every standing decision you have already approved.
- Running all three gates.
- Building the governed management information.
- Building the Annex 2 return and checking it against the regulator's schema.
- Refusing to keep an XML that fails that check.
- Preparing the publication and its version number.
- Recording the full audit trail.
- Restarting cleanly from the original file if the platform is interrupted.

**Proven twice, end to end:** an MI-only client for two consecutive months, and
a regulatory client for two consecutive months producing a schema-valid Annex 2
return. The second month asked nothing in both cases.

---

## 4. What you still have to do

### Every reporting cycle

- **Approve publication.** This is intentional and will not be automated. It is
  the only thing required every month once a client is settled in.

### Only when something needs review

- Answer OCC's questions on a new client's first delivery — typically naming the
  column each figure comes from, and confirming which column identifies the loan.
- Tell OCC what a file is, if its name is one OCC does not recognise.
- Answer the regulatory questions for a new regulatory client — mostly "the
  lender does not collect this, report it as not applicable" and "this lender's
  word for a loan status means this regulator's code". These are asked once and
  reused.
- Accept, with a written reason, a known data-quality issue in one month's file
  (see section 5).
- Confirm client or deal facts OCC has asked for.
- Note that a question you cannot answer yet can be **deferred** — OCC records
  who is waiting for what, keeps the item open and holds the report.

### Onboarding a new client

- Answer the onboarding questions, then approve and activate. Approval and
  activation are two separate steps on purpose.

---

## 5. When something goes wrong

| What happened | What OCC does | What you do |
|---|---|---|
| **A file OCC cannot recognise** | Holds the delivery and asks what the file is, offering the list of file types | Choose the type. OCC then processes it. It will ask again next month — see section 7 |
| **A column OCC cannot match** | Holds at the mapping stage and asks, showing what it found and what it guessed | Name the column, mark it not applicable, or defer it. Your answer is remembered for future months |
| **A structural data problem** — a figure the report cannot be produced without is missing entirely, or blank on every record | **Stops.** Tells you exactly which figure is missing and whether it is absent or empty. Offers no way to accept it | Get the file corrected, or correct the column mapping, then run it again. There is deliberately no override — not for you, not for an administrator |
| **A known data-quality issue** — a figure is there and correctly formatted but fails a business rule | Holds and shows you the exact rule, the field and how many records failed | Either fix it at source, or accept it for **this month only** with a written reason. The report is then produced, and the record permanently shows it was accepted rather than passed |
| **A regulatory problem** | Holds the regulatory return only. The management information is unaffected and still offered for publication | Answer the regulatory question. The return is then rebuilt and rechecked |
| **The platform is interrupted mid-run** | The run stops. Nothing partial is published | Restart the OCC service, and the run appears marked as interrupted with a "Run again" action. Press it. OCC fetches the original file back from its governed location and produces exactly the same result as an uninterrupted run |

---

## 6. What OCC remembers

**Remembered, and reused automatically in later months:**

- The client's configuration — identity, portfolios, legal entities, identifiers,
  which reports they need.
- Every column mapping you approve.
- Which column identifies the loan.
- Regulatory standing decisions — how a lender's wording translates into the
  regulator's codes, which figures the lender does not collect, and pool-level
  facts stated once.

**Deliberately NOT remembered:**

- **An accepted data-quality exception.** It applies to one file, for one
  reporting period, and nothing else. If the same problem appears next month,
  OCC asks again and holds the report until you decide again. This has been
  tested: the same defect in the following month is raised afresh and publication
  is refused until it is answered.
- **Publication approval.** Every period is approved separately.

**Asked again each month, and shouldn't be:**

- What an unrecognised file is. See section 7.

---

## 7. What OCC does not yet do

Stated plainly. None of these stops Client 1 going live.

- **You cannot change a standing decision through OCC.** If a column mapping is
  approved and later turns out to be wrong, there is no screen to correct it. The
  underlying capability exists and works, but it is not yet exposed. Until it is,
  a wrongly approved standing mapping needs an administrator or developer to
  correct. **This makes the first month's answers worth double-checking** — in
  particular which column identifies the loan.
- **OCC does not permanently learn an unrecognised file name.** You answer the
  same "what is this file?" question each month for a lender whose file name is
  outside the standard vocabulary. One click, no risk.
- **OCC does not recover from an interruption by itself.** It recovers correctly
  and completely, but the recovery happens when the OCC service restarts, not on
  its own schedule. Until the monitoring dashboard exists, a run that stays "in
  progress" far longer than usual is your signal.
- **There is no OCC monitoring or system-health dashboard.** You cannot see
  service health, run durations or a live queue in one place. Everything needed
  to build it is already recorded; the dashboard is the next piece of work.
- **There is no MI Query usage monitoring.** OCC does not record who asked what
  of the management information, or how it answered.
- **One item must be closed before a second commercial client.** If a client's
  own generated configuration ever became unreadable, OCC could fall back to the
  incumbent client's file. Every current path is protected by the check that
  refuses ungoverned clients, so it cannot happen for Client 1 — but it must be
  removed before a second client is onboarded.

---

## 8. Day 1 vs target state

| Capability | Day 1 | Target state |
|---|---|---|
| Normal reporting run | Fully automated end to end | Unchanged |
| Human review | First month asks; later months quiet | Fewer first-month questions as the library grows |
| Publication approval | You approve every period | Unchanged — an intentional control |
| Structural data failure | Stops, cannot be waived by anyone | Unchanged |
| Data-quality exception | Accept for one month with a written reason | Unchanged |
| Correcting an approved standing decision | Administrator or developer | Correct it yourself in OCC |
| File-role learning | Asked again each month | Learned from the first answer |
| Failure recovery | Restart the service, then "Run again" | OCC notices and offers it itself |
| Monitoring | No dashboard; watch for long-running deliveries | Full operations console |
| Multi-client isolation | Safe for Client 1; one item to close first | Closed before Client 2 |

---

## 9. Your reporting-cycle checklist

**When the file is due:**

1. Check the delivery arrived and a run started.

**While it runs:**

2. If OCC asks anything, answer it. If you cannot, defer it with a note — the
   report stays held, nothing is lost.
3. If OCC says a figure is missing entirely, do not look for a way round it.
   Get the file or the mapping corrected and run it again.
4. If OCC flags a data-quality issue you already know about, accept it with a
   written reason — that reason is the permanent record of why the report reads
   as it does.

**When it is ready:**

5. Review the summary. If it says checks were accepted rather than passed, that
   is a month you made a judgement — check it still reads correctly.
6. Approve publication.

**If a run has been "in progress" much longer than usual:**

7. Restart the OCC service, then press "Run again" on the run.

**First month with a new client, additionally:**

8. Check the mapping answers carefully — especially which column identifies the
   loan — because they will be reused every month and cannot yet be changed
   through OCC.

---

## 10. Bottom line

OCC can run Client 1's reporting: it takes the file from arrival to governed
management information and, where required, a schema-valid regulatory return,
with everything recorded. It needs you to answer the first month's questions and
to approve publication every period — and nothing else in a normal month. It
could still need a developer in one situation only: if a standing decision
approved in the first month turns out to be wrong, which is why those answers are
worth checking twice. Consciously deferred are correcting standing decisions
through the interface, permanently learning file names, automatic recovery from
interruptions, and the monitoring dashboard — none of which affects whether the
reporting is correct. Once the monitoring dashboard exists you will no longer
have to watch for a stalled run yourself, you will see service health and every
client's position in one place, and the manual service restart in step 7 becomes
something OCC offers you directly.
