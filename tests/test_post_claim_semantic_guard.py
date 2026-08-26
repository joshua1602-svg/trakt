"""No route may recover analytical meaning from English after it is claimed.

THE RULE, and the line it draws. Recognition happens BEFORE a route owns a
question, and recognition reads wording — that is what it is for, and this
guard permits it. Once the registry has entered `handle`, the route owns the
question, and from that point what the question MEANS must come from the
contract, the plan, the execution or the receipt.

WHY AST AND NOT A GREP. `question` appears in strings, comments, envelope
payloads and audit fields throughout these modules; a substring scan cannot
tell a semantic read from an answer field, and would have to be tuned until it
agreed with whatever the code did on the day it was written. Only a call
argument, a subscript or a comparison counts here.

THE MUTATION TESTS ARE THE POINT. A guard that passes on the current tree
proves nothing on its own — the previous cut of this programme's substitution
detector passed against a defective tree because its signal only existed once
the defect was fixed. Each of the four below reintroduces a real defect into a
parsed copy of the source and requires the guard to fail on it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

CHAT = Path("mi_agent_api/chat_routing.py")
PCR = Path("mi_agent_api/period_change_route.py")

#: The seven migrated core routes: (conversion, handler, module).
HANDLERS = [
    ("C1", "_route_portfolio_summary", CHAT),
    ("C2", "_route_period_movement", CHAT),
    ("C3", "_route_geo", CHAT),
    ("C4", "_route_bridge", CHAT),
    ("C5", "_route_compare", CHAT),
    ("C6", "_route_evolution", CHAT),
    ("C7", "route_period_change", PCR),
]

#: Names that ARE the raw question inside a handler.
QUESTION_NAMES = {"question", "q", "raw_question"}

#: Callees that decide MEANING from text. Imported from the estate's own
#: inventory so this guard and the census cannot drift apart.
def _semantic_callees():
    from migration_phase0.semantic_owner_inventory import SEMANTIC_CALLEES
    return SEMANTIC_CALLEES


#: Callees that put the wording in front of a reader, or carry it for audit.
#: A handler may pass the question to these; none of them decides anything.
ALLOWED = {
    "_envelope", "_rank_refusal_envelope", "_failure_envelope",
    "_execution_failure_envelope", "_capability_unavailable_envelope",
    "_disclose_lens_scope", "_render", "PeriodChangeRequest",
    "_summary_population", "_table_artifact", "_chart_artifact",
    "_summary_kpi_artifact", "_stamp_analytical_intent", "dict", "str",
    "len", "bool", "list", "tuple", "_route_concentration_tests",
}


def _function(module: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {module}")


def post_claim_reads(node: ast.FunctionDef, *, vocabulary=frozenset()):
    """Every place this handler recovers meaning from the raw question.

    Four shapes, because the defect has taken four shapes in this estate:
    handing the text to a resolver, matching it against a regex, testing it for
    a keyword, and slicing or comparing it directly.
    """
    semantic = _semantic_callees()
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            args = list(sub.args) + [k.value for k in sub.keywords]
            carries = any(isinstance(n, ast.Name) and n.id in QUESTION_NAMES
                          for a in args for n in ast.walk(a))
            if not carries:
                continue
            func = sub.func
            if isinstance(func, ast.Attribute):
                base = getattr(func.value, "id", None)
                name = func.attr
                # A regex or a module-local vocabulary applied to the question
                # is interpretation taken HERE, not delegation.
                if base == "re" or base in vocabulary:
                    found.append((sub.lineno, f"{base}.{name}", "local vocabulary"))
                    continue
            else:
                name = getattr(func, "id", "…")
            if name in ALLOWED:
                continue
            decides = semantic.get(name)
            if decides:
                found.append((sub.lineno, name, decides))
        elif isinstance(sub, ast.Compare):
            # `"grew" in question`
            for node_ in [sub.left] + list(sub.comparators):
                if isinstance(node_, ast.Name) and node_.id in QUESTION_NAMES:
                    if any(isinstance(op, (ast.In, ast.NotIn)) for op in sub.ops):
                        found.append((sub.lineno, "in", "keyword test on the question"))
    return found


def module_vocabulary(module: Path) -> frozenset:
    """Module-level constants holding business language (word lists, regexes)."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            value = node.value
            if isinstance(value, (ast.Set, ast.Tuple, ast.List)):
                strings = [e for e in getattr(value, "elts", [])
                           if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                if len(strings) >= 3:
                    names.add(target.id)
            elif (isinstance(value, ast.Call)
                  and isinstance(value.func, ast.Attribute)
                  and getattr(value.func.value, "id", None) == "re"):
                names.add(target.id)
    return frozenset(names)


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("conversion,handler,module",
                         HANDLERS, ids=[c for c, *_ in HANDLERS])
def test_no_post_claim_raw_question_semantic_read(conversion, handler, module):
    reads = post_claim_reads(_function(module, handler),
                             vocabulary=module_vocabulary(module))
    assert reads == [], (
        f"{conversion} {handler} recovers meaning from the raw question after "
        f"claiming it: {reads}")


def test_the_census_and_this_guard_agree():
    """Two instruments, one answer. If they disagree, one of them is wrong."""
    from migration_phase0 import analytical_meaning_census as census
    result = census.run()
    assert sum(len(r["K1"]) + len(r["K2"]) for r in result["rows"]) == 0
    total = 0
    for _conversion, handler, module in HANDLERS:
        total += len(post_claim_reads(_function(module, handler),
                                      vocabulary=module_vocabulary(module)))
    assert total == 0


# --------------------------------------------------------------------------- #
# Mutation controls — the guard must FAIL on each reintroduced defect
# --------------------------------------------------------------------------- #
def _mutated(module: Path, handler: str, inject: str) -> ast.FunctionDef:
    """Parse the module with one statement injected into the handler's body."""
    source = module.read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == handler)
    target.body.insert(1, ast.parse(inject).body[0])
    return ast.fix_missing_locations(tree), target


@pytest.mark.parametrize("name,inject,module,handler", [
    ("C7 _resolve_lens reinstated",
     "resolved = chat_routing._resolve_lens(question, source_lens)",
     PCR, "route_period_change"),
    ("C7 requested_span reinstated",
     "span = _period_request.requested_span(question)",
     PCR, "route_period_change"),
    ("C7 post-claim re-recognition reinstated",
     "intent = recognise(question, spec=spec, view=view)",
     PCR, "route_period_change"),
    ("a measure resolver reinstated in a C1-C6 handler",
     "hits = detect_measure_set(question, semantics)",
     CHAT, "_route_geo"),
])
def test_the_guard_fails_when_a_read_is_reintroduced(name, inject, module,
                                                     handler):
    _tree, target = _mutated(module, handler, inject)
    reads = post_claim_reads(target, vocabulary=module_vocabulary(module))
    assert reads, f"the guard did not detect: {name}"


def test_the_guard_fails_on_a_new_route_local_vocabulary():
    """A three-word movement vocabulary declared inside the module, then used."""
    source = PCR.read_text(encoding="utf-8")
    tree = ast.parse(source
                     + '\n_NEW_MOVEMENT_WORDS = ("grew", "fell", "shrank")\n')
    vocabulary = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(
                        node.value, (ast.Set, ast.Tuple, ast.List)):
                    strings = [e for e in node.value.elts
                               if isinstance(e, ast.Constant)
                               and isinstance(e.value, str)]
                    if len(strings) >= 3:
                        vocabulary.add(target.id)
    assert "_NEW_MOVEMENT_WORDS" in vocabulary, \
        "the vocabulary detector missed a three-word list"
    handler = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "route_period_change")
    handler.body.insert(1, ast.parse(
        "moved = _NEW_MOVEMENT_WORDS.count(question)").body[0])
    reads = post_claim_reads(handler, vocabulary=frozenset(vocabulary))
    assert reads, "the guard did not detect a new route-local vocabulary"


def test_the_mutations_are_not_left_in_the_source():
    """Every mutation above is applied to a parsed COPY. Prove the file is clean."""
    for module in (CHAT, PCR):
        text = module.read_text(encoding="utf-8")
        assert "_NEW_MOVEMENT_WORDS" not in text
    reads = post_claim_reads(_function(PCR, "route_period_change"),
                             vocabulary=module_vocabulary(PCR))
    assert reads == []
