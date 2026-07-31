import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FlaskConical, Plus } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type { CaseSummary, ScenarioSummary } from "@/api/agentTypes";
import { ErrorNote } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage, useLoad } from "@/lib/useLoad";
import { SyntheticBanner, stateTone } from "./shared";

/** Filters over the case list. Statuses come from the case, never computed here. */
const FILTERS = [
  { key: "all", label: copy.agent.filterAll },
  { key: "needs", label: copy.agent.filterNeedsYou },
  { key: "blocked", label: copy.agent.filterBlocked },
  { key: "ready", label: copy.agent.filterReady },
] as const;

type FilterKey = (typeof FILTERS)[number]["key"];

function matches(row: CaseSummary, filter: FilterKey): boolean {
  if (filter === "all") return true;
  if (filter === "blocked") return row.state === "BLOCKED";
  // Readiness is a WAYPOINT the run passes through, so "ready" is asked of the
  // readiness status the run recorded — not of where the run is now. A case
  // waiting for an activation decision has passed readiness and belongs here.
  if (filter === "ready") return row.readiness_status === "READY_FOR_EXECUTION";
  // "Needs you" is anything waiting on a human: an approval state on either
  // lifecycle, or an open decision. All three are properties the backend
  // already decided.
  return (
    row.open_decisions > 0 ||
    /REQUIRED|EXCEPTIONS/.test(row.state) ||
    ["draft", "in_review", "ready_for_approval", "changes_required"].includes(
      row.onboarding_status ?? "",
    )
  );
}

export function AgentCasesScreen() {
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();
  const [filter, setFilter] = useState<FilterKey>("all");
  const [instruction, setInstruction] = useState("");
  const [busy, setBusy] = useState(false);

  const meta = useLoad(() => client.getAgentMeta(), []);
  const cases = useLoad(() => client.listAgentCases(), []);

  const rows = useMemo(
    () => (cases.data ?? []).filter((row) => matches(row, filter)),
    [cases.data, filter],
  );

  async function create() {
    if (!instruction.trim() || busy) return;
    setBusy(true);
    try {
      const status = await client.createAgentCase(instruction.trim());
      navigate(`/agent/${status.case_ref}`);
    } catch (err) {
      toast.show(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  async function runScenario(scenario: ScenarioSummary) {
    if (busy) return;
    setBusy(true);
    try {
      const status = await client.runAgentScenario(scenario.fixture_id);
      navigate(`/agent/${status.case_ref}`);
    } catch (err) {
      toast.show(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  if (meta.error) {
    return (
      <Page title={copy.agent.title} subtitle={copy.agent.subtitle}>
        <ErrorNote message={meta.error} onRetry={() => void meta.reload()} />
      </Page>
    );
  }

  return (
    <Page title={copy.agent.title} subtitle={copy.agent.subtitle}>
      <SyntheticBanner />

      <section className="mt-6 rounded-2xl border border-stone-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-stone-900">{copy.agent.newCaseHeading}</h2>
        <p className="mt-1 text-sm text-stone-500">{copy.agent.newCasePrompt}</p>
        <textarea
          aria-label={copy.agent.newCaseHeading}
          className="mt-3 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm"
          rows={4}
          placeholder={copy.agent.newCasePlaceholder}
          value={instruction}
          onChange={(event) => setInstruction(event.target.value)}
        />
        <button
          type="button"
          disabled={busy || !instruction.trim()}
          onClick={() => void create()}
          className="mt-3 inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
        >
          <Plus className="h-4 w-4" aria-hidden />
          {busy ? copy.agent.sending : copy.agent.createButton}
        </button>
      </section>

      {(meta.data?.scenarios?.length ?? 0) > 0 && (
        <section className="mt-6">
          <h2 className="text-sm font-semibold text-stone-900">
            {copy.agent.scenariosHeading}
          </h2>
          <ul className="mt-3 grid gap-3 md:grid-cols-2">
            {meta.data!.scenarios.map((scenario) => (
              <li
                key={scenario.fixture_id}
                className="rounded-2xl border border-stone-200 bg-white p-4"
              >
                <p className="text-sm font-semibold text-stone-900">{scenario.label}</p>
                <p className="mt-1 text-sm text-stone-500">{scenario.description}</p>
                <p className="mt-2 text-xs text-stone-400">
                  {copy.agent.scenarioExpected}:{" "}
                  <span data-testid={`expected-${scenario.fixture_id}`}>
                    {scenario.expected_state}
                  </span>
                </p>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void runScenario(scenario)}
                  className="mt-3 rounded-xl border border-stone-300 px-3 py-1.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50"
                >
                  {copy.agent.scenarioRun}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="mt-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-stone-900">{copy.agent.casesHeading}</h2>
          <div className="flex gap-1" role="group" aria-label={copy.agent.casesHeading}>
            {FILTERS.map((option) => (
              <button
                key={option.key}
                type="button"
                aria-pressed={filter === option.key}
                onClick={() => setFilter(option.key)}
                className={clsx(
                  "rounded-lg px-3 py-1.5 text-sm font-medium",
                  filter === option.key
                    ? "bg-stone-900 text-white"
                    : "text-stone-600 hover:bg-stone-100",
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {cases.error && (
          <ErrorNote message={cases.error} onRetry={() => void cases.reload()} />
        )}
        {!cases.loading && rows.length === 0 && (
          <p className="mt-4 text-sm text-stone-500">{copy.agent.caseEmpty}</p>
        )}
        <ul className="mt-4 space-y-2">
          {rows.map((row) => (
            <li key={row.case_ref}>
              <Link
                to={`/agent/${row.case_ref}`}
                className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-stone-200 bg-white px-4 py-3 hover:border-stone-300"
              >
                <span className="min-w-0">
                  <span className="flex items-center gap-2">
                    <FlaskConical className="h-4 w-4 text-violet-500" aria-hidden />
                    <span className="truncate text-sm font-semibold text-stone-900">
                      {row.client_name || row.case_ref}
                    </span>
                    <span className="rounded-full border border-violet-200 bg-violet-50 px-2 py-0.5 text-xs font-medium text-violet-700">
                      {copy.agent.syntheticChip}
                    </span>
                  </span>
                  <span className="mt-1 block text-xs text-stone-500">
                    {[row.case_ref, row.onboarding_status_label].filter(Boolean).join(" · ")}
                  </span>
                </span>
                <span className="flex items-center gap-2">
                  {row.open_decisions > 0 && (
                    <span className="text-xs text-amber-700">
                      {row.open_decisions} {copy.common.decisionMany}
                    </span>
                  )}
                  <StatusChip status={stateTone(row.state)} label={row.state_label} />
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </Page>
  );
}
