# Review B — Build-size and complexity review

Objective: determine whether the same working outcome could be achieved with
materially less code. Tools: `wc`/`git`, radon (cyclomatic complexity +
maintainability), vulture (dead code), jscpd (duplication), `tsc --noEmit`,
ts-prune, pytest-cov, AST-based import-cycle check. All runs recorded; no code
changed.

## 1. What the "13,478 lines / 64 files" actually is

| Category | Files | Lines | Share |
|---|---|---|---|
| Backend production Python (`operations_control/*.py`) | 8 | 2,672 | 20% |
| API production Python (`operations_control/api/`) | 4 | 785 | 6% |
| React/TS source (`frontend/…/src`, incl. 4 test files ≈300 and the 851-line mock) | 30 | 3,653 | 27% |
| UI config (`vite.config.ts`, tsconfigs, `index.html`) | 5 | 124 | 1% |
| **`package-lock.json` (generated)** | 1 | **4,905** | **36%** |
| Backend tests (`tests/operations_control/`) | 9 | 1,098 | 8% |
| Docs (Phase 1 design pack) | 10 | 1,059 | 8% (prior commit) |

**Hand-written production logic is ≈6,300 lines** (3,457 Python + ~2,800 TS
after excluding the mock and in-src tests). The 13.4k headline is inflated by
the npm lockfile (36%) and the design docs. There are no deployment files yet.

Largest 20 new files: headed by `MockOpsClient.ts` (851 — canned demo data),
`engine.py` (846), `adapters.py` (560), `api/app.py` (485), `stores.py` (439),
`NewWorkflow.tsx` (374), `WorkflowDetail.tsx` (321).

## 2. Mandatory quality checks — results

| Check | Result |
|---|---|
| Dead code (vulture, ≥80% confidence) | **1 finding**: unused `Field` import in `api/app.py:25` |
| Import cycles (AST walk of `operations_control`) | **None** at module level |
| Duplicate code (jscpd, 43 files, min 70 tokens) | **1 exact clone, 9 lines (0.12%)** |
| Python complexity (radon cc) | Hotspots: `adapters.translate_run_state` **F(48)**, `adapters.extract_mapping_decisions` **F(47)**, `engine.resolve_decision` **D(25)**; 10 further C-grade blocks. Maintainability: all files A except `engine.py` **C (8.55)** |
| TypeScript lint (`tsc --noEmit`) | Clean |
| TS unused exports (ts-prune) | Only barrel re-exports from `api/index.ts` (types); no dead components |
| Production dependency audit | UI: **5** runtime deps (react, react-dom, react-router-dom, clsx, lucide-react); backend: **0 new** Python deps (fastapi/uvicorn/pyyaml/pandas already in `requirements.txt`) |
| Test coverage (52 tests, all passing) | **80% overall**; weakest: `api/app.py` 61%, `adapters.py` 68%, `presenters.py` 69%; strongest: contracts 98%, rules 96%, language 97% |
| API endpoint inventory | 19 `/ops` + `/health` (listed in report C appendix); note: `POST /ops/rules/{id}/retire` was designed (doc 05) and implemented in the engine + tests but **has no API route** |
| Persistence-document inventory | 21 URI kinds in one `OpsLayout` authority (workflow, events, results+history, lease, deliveries, decisions, rules current+versions, audit+head, publications, 2 indexes) |

## 3. Reuse assessment against existing repository capabilities

| New component | Existing equivalent | Why reuse was insufficient | Duplication risk | Recommendation |
|---|---|---|---|---|
| `GovernedAgentResult` (contracts.py) | `trakt_core.envelope.GovernedResult` | Different object: per-stage operational doc (decisions, evidence, scopes, 6 operator statuses) vs frozen 4-status transport envelope with tenant/policy fields | **Medium — and a doc/code divergence**: design doc 05 says API responses ride inside `GovernedResult`; the implementation returns ad-hoc `{ok,…}` envelopes and never imports it | Either adopt the trakt_core envelope at the API boundary or amend doc 05; don't leave the claim false |
| `api/auth.py` (token + tenant binding) | `mi_agent_api/auth.py` (Easy Auth/SWA principal, roles); `mi_agent_operator` token check | Easy Auth requires Azure fronting absent in this slice; tenancy binding (per-client allowlist) exists nowhere | **Medium — third auth mechanism in the repo** | Phase 3: converge on the `mi_agent_api` dependency seam (Entra), keep only the tenancy-binding logic |
| `stores.py` storage use | `apps.blob_trigger_app.storage` | **Reused, not duplicated** (imports `Storage`/`open_storage`); adds atomic temp+rename JSON writes the existing layer lacks | Low | Keep; consider upstreaming atomic write later |
| Hash-chained audit (~60 lines) | `exception_db.py` remediation chain (SQLite, findings-specific); `trakt_core/audit.py` (log-only, deliberately non-persistent) | Neither persists arbitrary operational events to blob | Low (pattern reuse, small) | Keep; single-ledger convergence is a Phase 3 option |
| Decision store | `apps/blob_trigger_app/approvals.py` | Approvals are per-source promotion artefacts; decisions are per-item operator reviews. The promotion path **is reused** at publication (`write_pending/approve/promote`) | Low | Keep |
| Workflow state machine over orchestrator `RunState` | `RunState` itself; `run_records.py` statuses | Wrap-don't-modify makes a second, operator-level state doc inherent; orchestrator state stays the step-truth and `translate_run_state` is the single derivation point | **Medium** — two states must stay derived, never independently mutated | Keep, but document the invariant; never write stage status except via translation |
| Rule store + projector | `mapping_memory.py` client memory | Memory is client-scoped only — no file/portfolio/global scopes, no versioning, no supersession, no audit linkage (all required). Projector **writes through the existing store**, no format duplication | **Medium** — same fact in two places (rule store = record, memory = agent-readable projection) | Keep; keep projection idempotent and one-directional |
| React app | `frontend/mi-agent-ui` | Same stack/conventions reused; mi-agent-ui has no extractable component library (app-local, Plotly-heavy) | Low | Keep |
| `presenters.py` + `language.py` | `mi_agent_api/presenters.py` | MI presenters are MI-data-specific; the plain-English contract (forbidden-vocabulary tests) is the core UX requirement | Low | Merge the two new modules into one view layer (they overlap) |

**Wrappers with little material behaviour:** `GovernedAdapters` methods are
7 one-line delegations + observation — that observation is the whole point,
acceptable. `set_engine()` in `app.py` is a test seam (3 lines).
**Mock-mode risk:** `MockOpsClient` is statically imported, so it is bundled
into production builds; it activates only when `VITE_OPS_MODE=mock` at *build*
time, so it cannot be switched on at runtime — Low risk, but a dynamic import
would remove 851 lines from the bundle.
**Speculative generality:** enum/transformation rule projection plumbing has
no UI flow creating those kinds yet; `result_history_uri` per-result history
is written but nothing reads it; the engine's `retire()` has no route.

## 4. Scenarios

**A. Keep as implemented.** Defensible: ~6.3k production lines deliver 12
agreed capabilities (real agents, persistent state, GARs, approvals, scoped
rules, recovery, tenancy, publication gating, 6 screens, audit, classification,
idempotency) at 80% coverage, zero new backend deps, no cycles, 0.12%
duplication. The per-capability cost is modest; the headline number was mostly
lockfile.

**B. Moderate consolidation (recommended).** Low-risk reductions:
1. Slim `MockOpsClient.ts` to a small fixture module or load it via dynamic
   import (−500 to −850 lines, UI tests keep a trimmed fixture).
2. Merge `api/presenters.py` into `language.py` as one view module (−~80
   lines of overlap; presenters currently duplicate label logic).
3. Merge `classification.py` into `engine.py` or `rules.py` (−1 file, −~30).
4. Split/flatten the two F-grade translators (`translate_run_state`,
   `extract_mapping_decisions`) into per-stage functions — no line saving but
   removes the main maintainability risk; add tests to lift `app.py` 61% →
   ≥80%.
5. Delete unused `Field` import; either expose `retire` as a route or remove
   the method+test; drop `result_history_uri` writes until something reads
   them (−~40).
   Estimate: **−4 files, −900 to −1,100 lines**, test impact limited to import
   paths + trimmed mock fixtures, zero behaviour change.

**C. Minimum viable production slice.** One backend module set
(engine+stores+contracts collapsed, ~1,800 lines), API without dashboard/
rules-search/history niceties (~350), UI of 3 screens (Start, Workflow with
inline reviews, History; ~1,500 incl. config): **≈3,700 production lines,
~20 files**. Deferred: Rules Library UI (rules still persisted), dashboard
tiles, Review Centre as a separate screen, mock mode, backfill UI, evidence
rendering. Limitations: operators review inside the workflow screen only;
weaker at-a-glance ops visibility. All 9 non-negotiables (real execution,
persistence, GARs, approvals, scoped rules, recovery, tenancy, gating,
minimal UI) are retained. This is roughly **40% smaller**, at the cost of the
Review Centre/Rules/Dashboard surfaces that were explicitly requested in
Phase 2 — so it is a fallback shape, not the recommendation.

## 5. Answers to the specific challenges

- *Separate contract vs GovernedResult*: justified as an object, but the API
  should either genuinely use the trakt_core envelope or the design doc must
  be corrected (finding, Medium).
- *Another auth implementation*: pragmatic for the slice; third mechanism in
  the repo — converge in Phase 3 (Medium).
- *Another storage abstraction*: **not duplicated** — the existing one is
  imported; only atomic-write hardening was added.
- *Hash-chained audit*: ~60 lines, pattern borrowed from the remediation
  ledger; the existing audit framework is deliberately log-only, so
  persistence had to live somewhere.
- *Both workflow state and orchestrator state*: inherent to the no-modify
  constraint; the invariant (workflow stages derived only from `RunState`
  translation) must be preserved.
- *Both rule persistence and mapping memory*: memory lacks scopes/versions;
  the rule store is the record, memory is the agent-facing projection.
- *Extensive presenter/language layers*: the plain-English contract is a core
  requirement and is test-enforced; the two modules should merge (overlap).
- *Full mock mode*: build-time-gated and inert in production, but 851 lines /
  23% of UI source for demo data is the single largest trim target.
- *Six screens before proving one slice*: the vertical slice was proven with
  real agents; the six screens were explicitly in the approved Phase 2 scope.
- *13,400 lines to compose seams*: the true figure for hand-written
  production logic is ≈6,300; scenario B brings it to ≈5,300.
