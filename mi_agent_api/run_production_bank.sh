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
  echo "usage: bash mi_agent_api/run_production_bank.sh <principal object id> <id,id,...|all>" >&2
  echo "  <id,id,...>  only the named bank questions, comma-separated, no spaces" >&2
  echo "  all          the whole bank (~135 model interpretations)" >&2
  echo "Nothing was asked." >&2
  exit 2
fi
IDS=""
[[ "${SELECTION}" != "all" ]] && IDS="${SELECTION}"
CATEGORIES="funded_kpi,funded_breakdown_1d,pipeline,pipeline_evolution,forecast,forecast_scale"
EXPECTED_QUESTIONS=135
# The oldest vocabulary the next measurement is for (2.9.0: the weighting
# exclusions' population and the `lapsed` definition; catalogue batches 1 and
# 2 before it): an older build is not the one being measured.
MINIMUM_VOCABULARY="2.9.0"

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
python - "${PRINCIPAL}" "${CATEGORIES}" "${EXPECTED_QUESTIONS}" "${MINIMUM_VOCABULARY}" "${IDS}" <<'PY'
import json, os, sys
from pathlib import Path

principal, categories, expected, vocabulary, ids = sys.argv[1:6]
problems = []

build = Path("build_info.json")
commit = json.loads(build.read_text()).get("commit") if build.exists() else None
print(f"deployed commit: {commit or '(no build_info.json)'}")

from mi_agent.interpretation_v2.vocabulary import VOCABULARY_VERSION
print(f"vocabulary: {VOCABULARY_VERSION}")
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

from mi_agent_api.question_bank import DEFAULT_BANKS, load_bank
rows = load_bank(DEFAULT_BANKS)
if ids:
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
LOG="/home/qb_plan_rerun_${STAMP}.log"
OUT="/home/qb_plan_rerun_${STAMP}.jsonl"
START="$(date -u +%Y-%m-%dT%H:%M:%S)"
{
  echo "START TIME (UTC): ${START}"
  echo "deployed commit: $(python -c 'import json; print(json.load(open("build_info.json")).get("commit"))' 2>/dev/null || echo unknown)"
  echo
} > "${LOG}"
nohup python -m mi_agent_api.question_bank \
  --principal "${PRINCIPAL}" $([[ -n "${IDS}" ]] && echo "--ids ${IDS}" || echo "--categories ${CATEGORIES}") --out "${OUT}" \
  >> "${LOG}" 2>&1 &

echo
echo "START TIME (UTC): ${START}"
echo "running as pid $! — log ${LOG}, answers ${OUT}"
echo "Ctrl-C stops watching, not the run. To watch again: tail -f ${LOG}"
echo
tail -f "${LOG}"
