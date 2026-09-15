import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Building2, FilePlus2, Import, PencilLine } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type { CaseSummary as AgentCaseSummary } from "@/api/agentTypes";
import type { ActiveClientRow, CaseRow } from "@/api/onboardingTypes";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Card, Note, PrimaryButton, SecondaryButton } from "@/components/onboarding/primitives";
import { Page } from "@/components/Page";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage, useLoad } from "@/lib/useLoad";

function StatusPill({ status, label }: { status: string; label: string }) {
  const tone =
    status === "approved" || status === "activated"
      ? "border-emerald-200 bg-emerald-50 text-emerald-800"
      : status === "awaiting_client" || status === "information_requested"
        ? "border-amber-200 bg-amber-50 text-amber-900"
        : status === "changes_required" || status === "withdrawn"
          ? "border-rose-200 bg-rose-50 text-rose-800"
          : "border-stone-200 bg-stone-100 text-stone-600";
  return (
    <span className={clsx("rounded-full border px-2.5 py-0.5 text-xs font-medium", tone)}>
      {label}
    </span>
  );
}

/**
 * One Agent case, in a governed queue, read-only.
 *
 * It links to the Agent tab rather than the case wizard: the wizard reads the
 * governed store, which has never heard of this case — that mismatch is what
 * made the Agent tab's own "Client onboarding" link 404. The row exists so an
 * operator looking at "Waiting on the client" sees the case that is, in fact,
 * waiting on a client.
 */
function AgentCaseRow({ row }: { row: AgentCaseSummary }) {
  return (
    <li className="py-3">
      <Link
        to={`/agent/${encodeURIComponent(row.case_ref)}`}
        data-testid="agent-case-row"
        className="flex flex-wrap items-start justify-between gap-3 hover:opacity-80"
      >
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium text-stone-900">
              {row.client_name || row.case_ref}
            </span>
            {row.onboarding_status && row.onboarding_status_label && (
              <StatusPill
                status={row.onboarding_status}
                label={row.onboarding_status_label}
              />
            )}
            <span className="rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-800">
              {copy.onboarding.agentChip}
            </span>
          </div>
          <p className="mt-1 break-words text-xs text-stone-500">
            {row.case_ref} · {copy.onboarding.agentRowHint}
          </p>
        </div>
      </Link>
    </li>
  );
}

function CaseList({
  heading,
  description,
  rows,
  agentRows = [],
  empty,
}: {
  heading: string;
  description?: string;
  rows: CaseRow[];
  /** Agent cases belonging in this queue. Read-only; see `AgentCaseRow`. */
  agentRows?: AgentCaseSummary[];
  empty: string;
}) {
  return (
    <Card title={heading} description={description}>
      {rows.length === 0 && agentRows.length === 0 ? (
        <p className="py-2 text-sm text-stone-500">{empty}</p>
      ) : (
        <ul className="divide-y divide-stone-100">
          {rows.map((row) => (
            <li key={row.case_id} className="py-3">
              <Link
                to={`/onboarding/cases/${encodeURIComponent(row.case_id)}`}
                className="flex flex-wrap items-start justify-between gap-3 hover:opacity-80"
              >
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium text-stone-900">{row.client_name}</span>
                    <StatusPill status={row.status} label={row.status_label} />
                    {row.kind !== "new_client" && (
                      <span className="rounded-full border border-stone-200 px-2 py-0.5 text-xs text-stone-500">
                        {row.kind_label}
                      </span>
                    )}
                  </div>
                  <p className="mt-1 break-words text-xs text-stone-500">
                    {row.case_id}
                    {row.portfolios > 0 &&
                      ` · ${row.portfolios} ${row.portfolios === 1 ? "portfolio" : "portfolios"}`}
                    {row.outstanding_requests > 0 &&
                      ` · ${row.outstanding_requests} awaiting the client`}
                    {row.updated_by && ` · last touched by ${row.updated_by}`}
                  </p>
                </div>
              </Link>
            </li>
          ))}
          {agentRows.map((row) => (
            <AgentCaseRow key={row.case_ref} row={row} />
          ))}
        </ul>
      )}
    </Card>
  );
}

function ActiveClients({
  rows,
  onAmend,
  busy,
}: {
  rows: ActiveClientRow[];
  onAmend: (clientId: string) => void;
  busy: boolean;
}) {
  return (
    <Card
      title={copy.onboarding.activeHeading}
      description={copy.onboarding.activeDescription}
    >
      {rows.length === 0 ? (
        <p className="py-2 text-sm text-stone-500">{copy.onboarding.noActive}</p>
      ) : (
        <ul className="divide-y divide-stone-100">
          {rows.map((row) => (
            <li key={row.client_id} className="flex flex-wrap items-start justify-between gap-3 py-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <Building2 className="h-4 w-4 shrink-0 text-stone-400" aria-hidden />
                  <Link
                    to={`/onboarding/clients/${encodeURIComponent(row.client_id)}`}
                    className="font-medium text-stone-900 hover:underline"
                  >
                    {row.display_name}
                  </Link>
                </div>
                <p className="mt-1 break-words text-xs text-stone-500">
                  {copy.common.version} {row.version} · {row.portfolios}{" "}
                  {row.portfolios === 1 ? "portfolio" : "portfolios"}
                  {row.products.length > 0 && ` · ${row.products.length} reporting products`}
                </p>
              </div>
              <SecondaryButton onClick={() => onAmend(row.client_id)} disabled={busy}>
                <span className="flex items-center gap-2">
                  <PencilLine className="h-4 w-4" aria-hidden />
                  {copy.onboarding.amend}
                </span>
              </SecondaryButton>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

/**
 * Client Onboarding home.
 *
 * The primary action starts a BLANK case: no client is selected, and nothing is
 * read from any existing configuration. Migration of a client Trakt already
 * serves is offered separately, at the foot of the page, because it is a
 * secondary path — not the way the product works.
 */
export function OnboardingHomeScreen() {
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const { data, error, loading, reload } = useLoad(() => client.getOnboardingHome(), []);
  // Agent cases, loaded SEPARATELY and allowed to fail.
  //
  // The tab is flag-gated and may not be mounted at all, in which case this
  // 404s — and Client Onboarding must still render. Its own failure is never
  // this page's failure, so the error is swallowed to an empty list rather
  // than surfaced: a missing Agent tab means there are no Agent cases, which
  // is exactly what an empty list says.
  const agent = useLoad(
    () => client.listAgentCases().catch(() => [] as AgentCaseSummary[]),
    [],
  );
  const agentCases = agent.data ?? [];
  const inAgentQueue = (...statuses: string[]) =>
    agentCases.filter((row) =>
      statuses.includes(String(row.onboarding_status ?? "")));

  async function start(kind: "new" | "migration" | "amendment", clientId?: string) {
    setBusy(true);
    try {
      const created =
        kind === "new"
          ? await client.startNewClientCase()
          : kind === "migration"
            ? await client.startMigrationCase(clientId ?? "")
            : await client.startAmendmentCase(clientId ?? "");
      navigate(`/onboarding/cases/${encodeURIComponent(created.case_id)}`);
    } catch (err) {
      toast.show(errorMessage(err), "error");
      setBusy(false);
    }
  }

  return (
    <Page
      title={copy.onboarding.title}
      subtitle={copy.onboarding.subtitle}
      actions={
        <PrimaryButton onClick={() => void start("new")} disabled={busy}>
          <span className="flex items-center gap-2">
            <FilePlus2 className="h-4 w-4" aria-hidden />
            {copy.onboarding.startNew}
          </span>
        </PrimaryButton>
      }
    >
      {loading && <Loading />}
      {error && !loading && !data && (
        <ErrorNote message={copy.onboarding.unavailable} onRetry={() => void reload()} />
      )}
      {data && !loading && (
        <div className="space-y-6">
          <Note>{copy.onboarding.intro}</Note>

          {/* Each queue is bucketed on the case's ONBOARDING status, which is
              the vocabulary these headings are named for — an Agent run has its
              own state machine and "PACK_SENT" is not an onboarding status. */}
          <CaseList
            heading={copy.onboarding.draftsHeading}
            description={copy.onboarding.draftsDescription}
            rows={data.drafts}
            agentRows={inAgentQueue("draft", "changes_required")}
            empty={copy.onboarding.noDrafts}
          />
          <CaseList
            heading={copy.onboarding.awaitingHeading}
            description={copy.onboarding.awaitingDescription}
            rows={data.awaiting_client}
            agentRows={inAgentQueue("information_requested", "awaiting_client")}
            empty={copy.onboarding.noAwaiting}
          />
          <CaseList
            heading={copy.onboarding.reviewHeading}
            description={copy.onboarding.reviewDescription}
            rows={[...data.in_review, ...data.approved]}
            agentRows={inAgentQueue("in_review", "ready_for_approval",
                                    "approved")}
            empty={copy.onboarding.noReview}
          />

          <ActiveClients
            rows={data.active_clients}
            busy={busy}
            onAmend={(id) => void start("amendment", id)}
          />

          {data.legacy_clients.length > 0 && (
            <Card
              title={copy.onboarding.migrateHeading}
              description={copy.onboarding.migrateDescription}
            >
              <ul className="divide-y divide-stone-100">
                {data.legacy_clients.map((row) => (
                  <li
                    key={row.client_id}
                    className="flex flex-wrap items-center justify-between gap-3 py-3"
                  >
                    <span className="text-sm text-stone-700">{row.display_name}</span>
                    <SecondaryButton
                      onClick={() => void start("migration", row.client_id)}
                      disabled={busy}
                    >
                      <span className="flex items-center gap-2">
                        <Import className="h-4 w-4" aria-hidden />
                        {copy.onboarding.migrate}
                      </span>
                    </SecondaryButton>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}
    </Page>
  );
}
