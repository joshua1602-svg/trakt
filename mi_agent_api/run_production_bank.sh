#!/usr/bin/env bash
# Re-run the owner's production question bank on the App Service, over SSH.
#
# THE SAME 135 QUESTIONS as the 2026-09-28 run (qb_full.txt): the six categories
# below select exactly the ids, texts and order recorded in
# due_diligence/evidence/qb_plan_readback/qb_questions.json. The runner's own
# default adds the limits categories (160 questions), which is why they are
# named here.
#
#     for d in $(ls -td /tmp/*/ ); do [ -f "$d/mi_agent_api/app.py" ] && cd "$d" && break; done
#     bash mi_agent_api/run_production_bank.sh <principal object id> <id,id,...>
#     bash mi_agent_api/run_production_bank.sh <principal object id> all
#     bash mi_agent_api/run_production_bank.sh <principal object id> variants-recent
#     bash mi_agent_api/run_production_bank.sh <principal object id> variants
#     bash mi_agent_api/run_production_bank.sh <principal object id> twins
#     bash mi_agent_api/run_production_bank.sh <principal object id> signoff
#     bash mi_agent_api/run_production_bank.sh <principal object id> askback
#     bash mi_agent_api/run_production_bank.sh <principal object id> conversations
#
# `variants-recent` asks the held-out variants of the questions changed for
# since 2026-09-29, each after the bank question it varies (~42 questions), so
# one run holds both sides of every comparison. `variants` asks all 105
# variants, to compare with an `all` run on the same deploy (P0 design §25):
#     python due_diligence/evidence/qb_plan_readback/score_variants.py \
#         --variants <variants>.jsonl --bank <bank>.jsonl
# `twins` asks the conversation bank's 26 new stand-alone twins (§34) on their
# own, so each twin's outcome is known before any conversation is scored.
# `signoff` is the D13 sign-off run: the 135 bank questions, then the held-out
# variants no fix was made against (81), in ONE run — one log, one .jsonl,
# scored with score_variants.py --variants <it> --bank <it>.
# `askback` plays the conversation bank's ask-back conversations (§34 phase 1):
# each question that should ask back, its reply sent with the continuation the
# ask-back handed back, then the reply's stand-alone twin (~18 questions). The
# conversation is switched on for that run's process only — no user's service
# changes.
# `conversations` plays the WHOLE conversation bank (§34, §39) and its held-out
# set (§39.1): every turn of the 35 + 21 conversations the model reads
# (123 + 45), each message sent with the continuation the previous one handed
# back, then each turn's stand-alone twin (88 distinct; one asked earlier in
# the run is reused) — ~256 questions and ~115 conversation readings. Scored
# with
#     python due_diligence/evidence/qb_plan_readback/score_conversations.py <it>.jsonl
# The conversation is switched on for that run's process only.
#
# The second argument is REQUIRED: the named bank questions (a spot check of
# particular changes), or `all` for the whole bank. The whole bank is never the
# default — a command whose question list was lost (a line break in the paste)
# must stop here, not spend ~135 model interpretations.
#
# It asks each question once through the path POST /mi/query uses, so it costs
# ~135 model interpretations. It REFUSES before asking anything when the build
# is not the one being measured (vocabulary older than MINIMUM_VOCABULARY below),
# when the canary is not on for the principal, or when the evidence sink is not
# set — any of those would spend the run and record nothing to read back.
#
# The run is started under nohup, so closing the SSH tab does not stop it.
# Output goes to /home (persistent storage): <stamp>.log is what to hand back,
# with the START TIME it prints first; <stamp>.jsonl holds the full answers.
set -euo pipefail

PRINCIPAL="${1:-}"
SELECTION="${2:-}"
if [[ -z "${PRINCIPAL}" || -z "${SELECTION}" || $# -gt 2 ]]; then
  echo "usage: bash mi_agent_api/run_production_bank.sh <principal object id> <id,id,...|all|variants-recent|variants|twins|signoff|askback|conversations>" >&2
  echo "  <id,id,...>      only the named bank questions, comma-separated, no spaces" >&2
  echo "  all              the whole bank (~135 model interpretations)" >&2
  echo "  variants-recent  the recently changed questions and their held-out variants (~42)" >&2
  echo "  variants         all 105 held-out variants (compare with an 'all' run)" >&2
  echo "  twins            the conversation bank's new stand-alone twins (~27)" >&2
  echo "  signoff          the whole bank, then the unspent held-out variants (~216)" >&2
  echo "  askback          the ask-back conversations, replies and their twins (~22)" >&2
  echo "  conversations    the whole conversation bank, its held-out set and their twins (~256)" >&2
  echo "Nothing was asked." >&2
  exit 2
fi
IDS=""
HOLDOUT=""
TWINS=""
SIGNOFF=""
CONVERSATIONS=""
case "${SELECTION}" in
  all) ;;
  variants-recent) HOLDOUT="recent" ;;
  variants) HOLDOUT="all" ;;
  twins) TWINS="1" ;;
  signoff) SIGNOFF="1" ;;
  askback) CONVERSATIONS="C" ;;
  conversations) CONVERSATIONS="all" ;;
  *) IDS="${SELECTION}" ;;
esac
CATEGORIES="funded_kpi,funded_breakdown_1d,pipeline,pipeline_evolution,forecast,forecast_scale"
EXPECTED_QUESTIONS=135
# The oldest vocabulary the next measurement is for (2.10.0: conversion and
# pipeline change, §20; 2.9.0 and the catalogue batches before it): an older
# build is not the one being measured. 2.21.0: D23, amount or number; 2.22.0:
# ranking, the stage-rate wording, scenarios on the projection; 2.23.0: D27,
# the run-off model's completion rate and pull-through.
MINIMUM_VOCABULARY="2.23.0"

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${APP_ROOT}"
if [[ ! -f antenv/bin/activate ]]; then
  echo "no antenv under ${APP_ROOT}; run this from the deployed app directory" >&2
  exit 2
fi
# shellcheck disable=SC1091
. antenv/bin/activate

# The platform's instrumentation directory shadows the venv's libraries; see
# startup.sh, which drops it for the same reason before starting gunicorn.
if [[ -n "${PYTHONPATH:-}" ]]; then
  _kept=""
  IFS=':' read -ra _entries <<< "${PYTHONPATH}"
  for _entry in "${_entries[@]}"; do
    case "${_entry}" in
      /agents/python|/agents/python/*|"") continue ;;
    esac
    _kept="${_kept:+${_kept}:}${_entry}"
  done
  export PYTHONPATH="${_kept}"
fi

echo "app directory: ${APP_ROOT}"
python - "${PRINCIPAL}" "${CATEGORIES}" "${EXPECTED_QUESTIONS}" "${MINIMUM_VOCABULARY}" "${IDS}" "${HOLDOUT}" "${TWINS}" "${SIGNOFF}" "${CONVERSATIONS}" <<'PY'
import json, os, sys
from pathlib import Path

principal, categories, expected, vocabulary, ids, holdout, twins, signoff, conversations = sys.argv[1:10]
problems = []

build = Path("build_info.json")
commit = json.loads(build.read_text()).get("commit") if build.exists() else None
print(f"deployed commit: {commit or '(no build_info.json)'}")

from mi_agent.interpretation_v2.vocabulary import VOCABULARY_VERSION
print(f"vocabulary: {VOCABULARY_VERSION}")
# WHICH MODEL VIEW THIS RUN MEASURES, beside the signed-off baseline's: a run
# on a changed view is how that change is measured, so it is said, not refused.
try:
    from mi_agent.interpretation_v2.opus_interpreter import model_view_fingerprint
    view = model_view_fingerprint()
    recorded = json.loads(Path("config/mi/model_view_baseline.json").read_text())
    base = recorded.get("model_view_fingerprint")
    candidate = (recorded.get("candidate") or {}).get("model_view_fingerprint")
    print(f"model view: {view[:12]} "
          f"({'the signed-off baseline' if view == base else 'the CANDIDATE under measurement (design §40) — this run measures it against the baseline ' + str(base)[:12] if view == candidate else 'CHANGED since the signed-off baseline ' + str(base)[:12]})")
except Exception as exc:  # noqa: BLE001 - a preflight note, never a refusal
    print(f"model view: not read ({type(exc).__name__})")
if conversations:
    try:
        from mi_agent.interpretation_v2.conversation_reader import (
            reader_view_fingerprint)
        view = reader_view_fingerprint()
        base = (json.loads(Path("config/mi/model_view_baseline.json").read_text())
                .get("conversation_reader") or {}).get("fingerprint")
        print(f"conversation reader view: {view[:12]} "
              f"({'the recorded one' if view == base else 'CHANGED since the recorded ' + str(base)[:12]})")
    except Exception as exc:  # noqa: BLE001 - a preflight note, never a refusal
        print(f"conversation reader view: not read ({type(exc).__name__})")
def _v(text):
    return tuple(int(x) for x in str(text).split("."))
if _v(VOCABULARY_VERSION) < _v(vocabulary):
    problems.append(f"this build's vocabulary is {VOCABULARY_VERSION}, older "
                    f"than {vocabulary}: deploy claude/wizardly-faraday-o712ib first")

from mi_agent import plan_serving_canary as canary
from mi_agent import plan_shadow_evidence as evidence
mode = canary.serve_mode()
listed = principal.strip().lower() in canary.serve_principals()
sink = os.environ.get(evidence.SINK_ENV_VAR) or ""
print(f"{canary.SERVE_ENV_VAR}={mode}; principal "
      f"{'IS' if listed else 'is NOT'} on {canary.PRINCIPALS_ENV_VAR}")
print(f"{evidence.SINK_ENV_VAR}={sink or '(unset)'}")
if mode != canary.SERVE_CANARY:
    problems.append(f"{canary.SERVE_ENV_VAR} is {mode}: set it to canary")
if not listed:
    problems.append(f"the principal is not in {canary.PRINCIPALS_ENV_VAR}")
if not sink:
    problems.append(f"{evidence.SINK_ENV_VAR} is unset: nothing would be recorded")

from mi_agent_api.question_bank import (DEFAULT_BANKS, conversation_rows,
                                       holdout_rows, load_bank, twin_rows)
rows = load_bank(DEFAULT_BANKS)
if conversations:
    turns = conversation_rows(conversations)
    twins_asked = len({t["twin"] for t in turns if t.get("twin")})
    print(f"questions selected: {len(turns) + twins_asked} (conversations "
          f"{conversations}: {len(turns)} turns and {twins_asked} stand-alone twins)")
elif twins:
    print(f"questions selected: {len(twin_rows())} (conversation bank twins)")
elif signoff:
    wanted = set(categories.split(","))
    count = sum(1 for r in rows if r.get("category") in wanted)
    unspent = len(holdout_rows("unspent"))
    print(f"questions selected: {count + unspent} (sign-off: {count} bank "
          f"questions, then {unspent} unspent held-out variants)")
    if count != int(expected):
        problems.append(f"{count} bank questions selected, not {expected}")
elif holdout:
    print(f"questions selected: {len(holdout_rows(holdout))} "
          f"(held-out variants: {holdout})")
elif ids:
    wanted_ids = {i.strip() for i in ids.split(",") if i.strip()}
    count = sum(1 for r in rows if r.get("id") in wanted_ids)
    missing = sorted(wanted_ids - {r.get("id") for r in rows})
    print(f"questions selected: {count} (spot check)")
    if missing:
        problems.append(f"not in the bank: {', '.join(missing)}")
else:
    wanted = set(categories.split(","))
    count = sum(1 for r in rows if r.get("category") in wanted)
    print(f"questions selected: {count}")
    if count != int(expected):
        problems.append(f"{count} questions selected, not {expected}")

for p in problems:
    print(f"REFUSED: {p}")
sys.exit(1 if problems else 0)
PY

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
KIND="qb_plan_rerun"
[[ -n "${HOLDOUT}" ]] && KIND="qb_variants_${HOLDOUT}"
[[ -n "${TWINS}" ]] && KIND="qb_twins"
[[ -n "${SIGNOFF}" ]] && KIND="qb_signoff"
[[ -n "${CONVERSATIONS}" ]] && KIND="qb_conversations_${SELECTION}"
LOG="/home/${KIND}_${STAMP}.log"
OUT="/home/${KIND}_${STAMP}.jsonl"
START="$(date -u +%Y-%m-%dT%H:%M:%S)"
{
  echo "START TIME (UTC): ${START}"
  echo "deployed commit: $(python -c 'import json; print(json.load(open("build_info.json")).get("commit"))' 2>/dev/null || echo unknown)"
  echo
} > "${LOG}"
nohup python -m mi_agent_api.question_bank \
  --principal "${PRINCIPAL}" $(if [[ -n "${CONVERSATIONS}" ]]; then echo "--conversations ${CONVERSATIONS}"; elif [[ -n "${TWINS}" ]]; then echo "--twins"; elif [[ -n "${SIGNOFF}" ]]; then echo "--signoff --categories ${CATEGORIES}"; elif [[ -n "${HOLDOUT}" ]]; then echo "--holdout ${HOLDOUT}"; elif [[ -n "${IDS}" ]]; then echo "--ids ${IDS}"; else echo "--categories ${CATEGORIES}"; fi) --out "${OUT}" \
  >> "${LOG}" 2>&1 &

echo
echo "START TIME (UTC): ${START}"
echo "running as pid $! — log ${LOG}, answers ${OUT}"
echo "Ctrl-C stops watching, not the run. To watch again: tail -f ${LOG}"
echo
tail -f "${LOG}"
