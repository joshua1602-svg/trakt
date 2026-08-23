"""Compositional plan layer — scoping study instruments.

Read-only. Nothing here is imported by product code, nothing is wired in and
nothing sits behind a flag. Two instruments, both of which derive their numbers
from the source tree or from a live governed book rather than restating them:

* ``census.py``  — the shape cascades and the primitive implementations, counted
  from the source. Answers "how many places decide an answer's shape?" and "how
  many implementations does each primitive have?".
* ``compose.py`` — composes T1/T3/T4/T5/T6 from EXISTING, UNMODIFIED primitives
  against a live governed book, and reconciles each composition back to the
  shipped answer. Answers "do the routes factor?".

Neither builds anything. See docs/mi_compositional_plan_scoping.md.
"""
