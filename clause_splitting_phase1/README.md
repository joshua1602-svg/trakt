# Clause splitting — Phase 1 measurement instruments

Phase 1 only. Read-only. Nothing here is imported by product code, nothing is
wired in, and nothing behind a flag. This directory exists to answer one
question the existing banks cannot: **would a clause-splitting layer be worth
building?**

## Why these exist

The calibration bank and the generated harness sit at full marks. They can
detect regression and nothing else. Judged on them alone the best possible
result for a new layer is "nothing changed", which is not evidence of value.

So three instruments, authored and committed **before any splitter code
exists**, so that no expectation in them can be shaped by what a splitter turns
out to do:

| Instrument | Question it answers | Files |
|---|---|---|
| (a) Adversarial probe set | Does the known defect class become unexpressible? | `probes/adversarial_probes.yaml`, `run_probes.py` |
| (b) Vocabulary blast radius | Does adding a client's word stop being dangerous? | `probes/vocabulary_blast_radius.yaml`, `run_blast_radius.py` |
| (c) Time-series probe set | Does the capability with no home today appear? | `probes/time_series_probes.yaml`, `run_probes.py` |

## How a tree is scored

The instruments assert **classification, not outcome**. Two trees that both
refuse a question are not equivalent if only one refuses for the right reason,
and a regression bank cannot tell them apart.

Every probe declares the correct split in six spans — operation, subject,
grouping, filter, period, target — defined in `span_model.py`. Each span is
`filled`, `empty`, or `present-but-unresolvable`.

The release candidate has no spans. `rc_projection.py` reads its flat
`MIQuerySpec` into the same six-span shape so both trees are scored by the same
probes. That projection is deliberately **generous to the release candidate**:
`chart_type == "line"` is credited as a time axis, `aggregation == "count"` with
no metric is credited as a `loan_count` subject, and `ranking_mode` is credited
as a ranking operation. Without that generosity the release candidate would
fail every probe on a technicality rather than on capability, and the
comparison would prove nothing.

Two scoring rules keep the comparison honest:

* **A spurious binding fails a probe exactly as a missing one does.** A span the
  probe does not declare must be empty. Inventing a filter the question never
  stated is the silent-narrowing defect the layer exists to retire, so it
  cannot be scored as a near-miss.
* **A failure is tagged `structural` only when the tree has no channel for the
  expected answer AND put nothing in the span.** A tree that lacks the
  unresolvable state is not blamed for lacking it. A tree that bound the span
  to a confident wrong concept made a substantive wrong call and is not excused
  by the missing state.

Structural failures are reported separately from substantive ones, in both
directions, so neither tree is credited with a defect it merely cannot express.

## Running them

```bash
# (a) and (c), against the release candidate
python -m clause_splitting_phase1.run_probes --tree rc -v

# (b), both banks, in-memory registry perturbation only
python -m clause_splitting_phase1.run_blast_radius --tree rc

# (b) plus the disclosed probe-corpus sensitivity extension
python -m clause_splitting_phase1.run_blast_radius --tree rc --include-probes
```

Adding a second tree is a one-line registration in `trees.py`. No probe, no
expectation and no scoring rule changes when it appears — which is the point of
committing all of this first.

## What is deliberately NOT here

* No splitter. No rules engine. Those come after these are committed.
* No modification to any bank, fixture, registry file or product module. The
  blast-radius test perturbs a deep copy of the registry in memory; the file on
  disk is never written.
* No expectation derived from observed behaviour. Where a probe is marked
  `right_for_wrong_reason`, that marks a case whose end behaviour is defensible
  while its classification is not — the marking describes the *class of case*,
  and the expectation is still written from what the sentence says.
