# Stage 3 — pre-registration

Written **before** the merge module exists and **before** any bank was run with
the model on. Base commit `a171a20` (the Stage 2 head), tree clean.

## What is under test

The merge between the deterministic claim set and the model's bound concept
proposal. Three rules, as briefed:

1. the model may fill an **empty** slot;
2. the model may **not overwrite a filled one**;
3. a disagreement on a filled slot is a **finding**, not a resolution.

## The definition this whole stage turns on

**A slot filled by a governed default is FILLED, not empty.**

`chat_routing.py:1150–1166` is the guard that makes *"Show me the trend."*
refuse, and it fires on `subject.provenance == PROV_DEFAULT`. The schema is
explicit that such a claim is real: *"A `PROV_DEFAULT` subject is a real claim
carrying a real value — the series still plots the governed balance."*

If the merge treated a governed default as an empty slot, the model would fill
it, the provenance would stop being `default`, the guard would stop firing, and
*"Show me the trend."* would answer. That is precisely how the Opus run walked
through these guards: the model supplied the missing element itself, so no
default was ever recorded.

So defaults count as filled, and the model may not touch them.

## Prediction 1 — the brief's premise about the 21 is wrong, and I expect it to fail here

The brief states: *"On the 21, every relevant slot is empty — the concept never
arrived."*

Measured on the deterministic claim set, that is true for `row_predicates` and
`dimensions` and **false** for `source_scope`, `dataset` and `subject`, which
the deterministic side fills on **every** question with `provenance=default`:

```
What changed?          source_scope filled/default   dataset filled/default   subject filled
Show me the trend.     source_scope filled/default   dataset filled/default   subject filled/default
```

So I predict the merge, implemented exactly as briefed, **cannot reach** the
losses that live in those slots. Classified from Stage 1's own lost lists, the
20 type-(c) failures distribute as:

| slot the loss lives in | questions |
|---|---|
| `row_predicates` | 9 |
| `source_scope` **or** `row_predicates` (a scope word read as a lost narrowing) | 4 |
| `dimensions` | 3 |
| `subject` | 2 |
| `dataset` | 2 |
| `target` | 2 |
| nothing — no owner resolves it (Q15B) | 1 |

**Predicted reach: 12 of 20** — the 9 `row_predicates`, the 3 `dimensions` —
plus whichever of the 4 scope-or-predicate cases resolve as a row predicate.
**Predicted unreachable: 6** — 2 `subject`, 2 `dataset`, 2 `target` — because
those slots are never empty, plus Q15B, which no owner resolves at all.

If the measured reach is materially higher than 12, my reading of "empty" is
wrong and I will say so. If it is lower, the model is not proposing what I
expect it to.

## Prediction 2 — the three must-refuse questions still refuse

*"What changed?"*, *"Show me the trend."*, *"Compare us with the market."*

Predicted **because the defaulted slots are filled and untouchable**, not
because the model proposes nothing. I expect the model to propose *something*
for at least one of the three, and the merge to decline it. If the merge fills
any defaulted slot on these three, the stop condition has fired.

## Prediction 3 — the conflict count is low, and may be zero

A conflict needs the deterministic side to have filled a slot **from the
question** and the model to propose a different value for the same slot. On the
banks I expect this to be rare. **If it is zero I will report zero as a flat
result**, not as confirmation of anything: a conflict rule that never fires on
this corpus is untested by this corpus, and I will say that in those words.

## Prediction 4 — ambiguity is recorded, and distinguishable from absence

An ambiguous proposal (`direct` — two governed fields claim it) must produce a
record, not a silent non-fill. Predicted: it appears in the merge findings with
its own reason, and a merge with an ambiguous proposal is byte-different from a
merge with no proposal at all. This is Q20C's shape and the stop condition is
explicit.

## Prediction 5 — the prompt is unexercised and this run is measurement

Everything Stage 2 measured is a property of the binder. **No proposal prompt
has ever been sent to a model.** This run is measurement, not validation. A
poor proposal rate is a finding about the prompt. The prompt will **not** be
re-authored after seeing bank results; if it needs work that is a separate
change with its own before/after.

Specifically I do not know, and am not predicting:
- what fraction of proposals will bind rather than be rejected;
- whether the model will propose the wrong **kind** for the five cross-kind
  collision terms (`acquired`, `broker`, `loan size`, `owner occupied`,
  `ticket size`);
- whether it will propose anything at all for the two `target` losses, which
  have no proposable concept kind.

## Prediction 6 — Stage 1's two limits carry forward unchanged

Quoted verbatim from the Stage 1 report rather than rediscovered:

> **The check's recall is the owners' recall.** Q15B is missed because no owner
> resolves the hyphenated form, so no deterministic check can see it lost. That
> bound is the argument for the proposal step, not against the check.

> **The check compares presence, not correctness.** Q21C is silent because
> every stated concept is present and one was bound wrongly. Rule 3's
> disagreement reporting is what covers that case.

I predict Q15B stays missed on merged claims — the model may propose the
concept, but the check cannot see the loss either way — and Q21C stays silent.

## Registered line estimate

~350 production lines for the merge module. The stop condition is 3× that.

## What is NOT changed in this stage

The serving path. The merge is built and measured; it is not wired into
`/mi/query`. Any statement about the three must-refuse questions answering
today is therefore a statement about the estate as it ships, and the merge's
contribution is measured on claims, not on answers. I will say which is which
rather than letting the end-to-end result stand in for the merge's.
