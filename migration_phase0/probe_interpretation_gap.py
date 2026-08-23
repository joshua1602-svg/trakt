"""What does the interpretation contract supply for portfolio_summary questions?

READ-ONLY. Objective 4 step 3 / deliverable 4: everything the shadow plan needs,
and whether the contract can supply it WITHOUT rereading the question.
"""
import os, sys, warnings, logging, json
warnings.simplefilter('ignore'); logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TRAKT_RUNTIME_MODE","development")
from demo_platform import config as cfg
os.environ.update(cfg.mi_env(period_role='current'))
os.environ['MI_AGENT_LLM_PARSER']='off'; os.environ['MI_AGENT_LLM_ENABLED']='0'
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent_api.data_source import semantics_path
from question_interpretation import projection
from mi_agent import portfolio_lens as lens_mod
from mi_agent_api import evolution as evo

sem = load_mi_semantics(semantics_path())
frames = evo.funded_frames(os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"],"alderbridge",None)
df = frames[-1]["df"]

QS = [
  "Please provide a portfolio summary",
  "Give me a summary of the portfolio",
  "Can you summarise the book for me?",
  "Summarise the acquired book",
  "What is the portfolio position for the direct book?",
]
for q in QS:
    qi = projection.project(q, semantics=sem, frame=df)
    lens = lens_mod.resolve_lens(q)
    print(f"\n{q!r}")
    print(f"  operation : type={qi.operation.type!r} state={qi.operation.state!r}")
    print(f"  subject   : {qi.subject.candidate_concept!r} state={qi.subject.state!r}")
    print(f"  dimensions: {[(d.raw_text, d.role, d.candidate_concept) for d in qi.dimensions]}")
    print(f"  filters   : {[(f.raw_text, f.provides, f.clause_id) for f in qi.filters]}")
    print(f"  population: {[(p.raw_text, p.concept, p.state) for p in qi.population]}")
    print(f"  time      : grain={qi.time.grain!r} window={qi.time.trend_window.state!r}")
    print(f"  residue   : {qi.residue}")
    print(f"  --- what the ROUTE derives from the raw question instead: ---")
    print(f"  portfolio_lens.resolve_lens(question) -> name={lens.name!r} "
          f"label={lens.label!r} filters={lens.filters}")

print("\n" + "="*70)
print("BOUNDARY: which population concepts DOES the contract carry?")
print("="*70)
CASES = [
  ("seasoning segment",  "What is the balance of the front book?"),
  ("seasoning segment",  "Summarise the front book"),
  ("portfolio lens",     "Summarise the acquired book"),
  ("portfolio lens",     "What is the balance of the acquired book?"),
  ("row filter",         "What is the balance for loans over £150k?"),
]
from mi_agent import seasoning as seasoning_mod
for kind, q in CASES:
    qi = projection.project(q, semantics=sem, frame=df)
    lens = lens_mod.resolve_lens(q)
    seg = seasoning_mod.resolve_population_predicate(q)
    print(f"\n[{kind}] {q!r}")
    print(f"   contract population : {[(p.raw_text, p.concept, p.state) for p in qi.population]}")
    print(f"   contract filters    : {[(f.raw_text, f.operator, f.value, f.provides) for f in qi.filters]}")
    print(f"   lens (raw question) : {lens.name!r} filters={lens.filters}")
    print(f"   seasoning (raw q)   : {seg}")
