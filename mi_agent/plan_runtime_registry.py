"""Which governed runtime executes which population. ONE PLACE TO READ IT.

WHAT THIS CLOSES. The question "which populations can this estate execute, and
by which owner" had no answer in the tree. Each runtime declared its own set, under
four different names (`POPULATION_BASES` three times, `SLICE_2_POPULATION_BASES`
once, `EXECUTABLE_POPULATIONS` twice), only one of them said which population it
actually LOADS, and the funded population gate in `plan_serving_canary` checked
against a default without naming whose declaration that was. So the estate's
executable set could not be enumerated, and widening it meant editing several
files and hoping none was missed.

WHAT THIS IS NOT. It computes nothing and decides nothing at request time. Every
runtime still owns its own declaration — `EXECUTABLE_POPULATIONS` and
`EXECUTION_POPULATION` on its own module — and this module only lists them, in
dispatch order, so the union can be read and tested. A runtime holding the same
value as another is a coincidence of scope, not a shared constant.

THE PARTITION, AND WHY IT IS LOAD-BEARING. The funded population gate speaks for
the funded runtimes BELOW it: material summary, attribution, metric delta, the
temporal runtime and the generic executor all execute over a funded frame. A
runtime that owns a DIFFERENT population must therefore be dispatched ABOVE that
gate, or the gate refuses its plans `POPULATION_NOT_EXECUTABLE` on behalf of
runtimes that were never going to execute them. That is not hypothetical: it is
all 26 `POPULATION_NOT_EXECUTABLE` fallbacks in the production question bank of
2026-09-28, every forecast question among them. So a runtime is listed in exactly
one of two tuples, and the tuple says where it is dispatched.

Adding a population is adding a runtime to `POPULATION_OWNING_RUNTIMES` and a
dispatch branch above the gate. The tests in
`tests/interpretation_v2/test_plan_runtime_registry.py` fail if either half is
missing, and fail if a base appears in the governed vocabulary with neither an
owner nor an explicit entry in `DELIBERATELY_UNEXECUTED`.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Tuple

from mi_agent import plan_attribution as attribution
from mi_agent import plan_material_summary as material_summary
from mi_agent import plan_metric_delta as metric_delta
from mi_agent import plan_pipeline_runtime as pipeline
from mi_agent import plan_runtime_adapter as generic
from mi_agent import plan_temporal_runtime as temporal

__all__ = ["POPULATION_OWNING_RUNTIMES", "FUNDED_RUNTIMES", "GOVERNED_RUNTIMES",
           "FUNDED_GATE_POPULATIONS", "DELIBERATELY_UNEXECUTED",
           "executable_populations"]

#: Runtimes owning a population OTHER than the funded book. Dispatched ABOVE the
#: funded population gate, each checked against its own declaration.
POPULATION_OWNING_RUNTIMES: Tuple = (pipeline,)

#: Runtimes executing over the funded book, in the order the canary tries them
#: BELOW the funded population gate. The order is the canary's and is
#: load-bearing — change-form owners before the temporal runtime (which claims
#: on period form alone, and so more broadly), the generic executor last.
FUNDED_RUNTIMES: Tuple = (material_summary, attribution, metric_delta,
                          temporal, generic)

#: Every governed runtime, in dispatch order.
GOVERNED_RUNTIMES: Tuple = POPULATION_OWNING_RUNTIMES + FUNDED_RUNTIMES


def executable_populations(runtimes: Iterable = GOVERNED_RUNTIMES) -> FrozenSet[str]:
    """The union of what the given runtimes DECLARE they execute."""
    out: FrozenSet[str] = frozenset()
    for runtime in runtimes:
        out = out | frozenset(runtime.EXECUTABLE_POPULATIONS)
    return out


#: What the funded population gate admits: exactly what the runtimes it guards
#: declare, and nothing another runtime declares. Derived rather than written
#: down, so it cannot drift from the runtimes it speaks for.
FUNDED_GATE_POPULATIONS: FrozenSet[str] = executable_populations(FUNDED_RUNTIMES)

#: Bases in the governed vocabulary that no runtime executes, ON PURPOSE. Each is
#: refused `POPULATION_NOT_EXECUTABLE` rather than approximated.
#:
#:   forecast    composes funded with pipeline at two cut-off dates. Its owner is
#:               the P0 forecast connectivity module, which removes it from here.
#:   whole_book  needs the limit schedule, and concentration limits are not yet
#:               established by the client. Stays refused, and says so.
DELIBERATELY_UNEXECUTED: FrozenSet[str] = frozenset({"forecast", "whole_book"})
