import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Building2, Plus } from "lucide-react";
import { useOpsClient } from "@/api/context";
import type { OnboardingClientRow } from "@/api/onboardingTypes";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Card, Note, PrimaryButton, SecondaryButton } from "@/components/onboarding/primitives";
import { Page } from "@/components/Page";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage, useLoad } from "@/lib/useLoad";

function StatusBadge({ status }: { status: string }) {
  const onboarded = status === "onboarded";
  return (
    <span
      className={
        onboarded
          ? "rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-800"
          : "rounded-full border border-stone-200 bg-stone-100 px-2.5 py-0.5 text-xs font-medium text-stone-600"
      }
    >
      {onboarded ? copy.onboarding.onboardedBadge : copy.onboarding.legacyBadge}
    </span>
  );
}

function ClientRow({
  row,
  onAdopt,
  busy,
}: {
  row: OnboardingClientRow;
  onAdopt: (clientId: string) => void;
  busy: boolean;
}) {
  const onboarded = row.status === "onboarded";
  return (
    <li className="flex flex-wrap items-start justify-between gap-4 border-b border-stone-100 px-1 py-4 last:border-0">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <Building2 className="h-4 w-4 text-stone-400" aria-hidden />
          <span className="font-medium text-stone-900">{row.display_name}</span>
          <StatusBadge status={row.status} />
        </div>
        {onboarded ? (
          <p className="mt-1 text-sm text-stone-500">
            {copy.common.version} {row.version} · {row.portfolios}{" "}
            {row.portfolios === 1 ? "portfolio" : "portfolios"}
            {row.updated_by ? ` · last changed by ${row.updated_by}` : ""}
          </p>
        ) : (
          <p className="mt-1 max-w-xl text-sm text-stone-500">{row.summary}</p>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {onboarded ? (
          <Link
            to={`/onboarding/clients/${encodeURIComponent(row.client_id)}`}
            className="rounded-xl border border-stone-300 px-4 py-2 text-sm font-medium text-stone-700 transition-colors hover:border-stone-500"
          >
            {copy.onboarding.open}
          </Link>
        ) : (
          <SecondaryButton onClick={() => onAdopt(row.client_id)} disabled={busy}>
            {copy.onboarding.adopt}
          </SecondaryButton>
        )}
      </div>
    </li>
  );
}

/**
 * Client Onboarding home. Lists every client the operator can see, whether or
 * not it has been through onboarding — a client configured before this existed
 * still works, and adopting it is an offer, not a demand.
 */
export function OnboardingHomeScreen() {
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const { data, error, loading, reload } = useLoad(() => client.getOnboardingClients(), []);

  async function startDraft(clientId?: string, adopt = false) {
    setBusy(true);
    try {
      const draft = await client.startOnboardingDraft({ client_id: clientId, adopt });
      navigate(`/onboarding/drafts/${draft.draft_id}`);
    } catch (err) {
      toast.show(errorMessage(err), "error");
      setBusy(false);
    }
  }

  const rows = data ?? [];
  const onboarded = rows.filter((r) => r.status === "onboarded");
  const legacy = rows.filter((r) => r.status !== "onboarded");

  return (
    <Page
      title={copy.onboarding.title}
      subtitle={copy.onboarding.subtitle}
      actions={
        <PrimaryButton onClick={() => void startDraft(undefined, false)} disabled={busy}>
          <span className="flex items-center gap-2">
            <Plus className="h-4 w-4" aria-hidden />
            {copy.onboarding.newClient}
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
          <Note>
            Onboarding sets a client's standing configuration — who they are, which books they
            have, which reports they receive. Deliveries are still started from Operations.
          </Note>

          <Card title={copy.onboarding.existingHeading}>
            {rows.length === 0 ? (
              <p className="py-4 text-sm text-stone-500">{copy.onboarding.empty}</p>
            ) : (
              <ul>
                {[...onboarded, ...legacy].map((row) => (
                  <ClientRow
                    key={row.client_id}
                    row={row}
                    busy={busy}
                    onAdopt={(id) => void startDraft(id, true)}
                  />
                ))}
              </ul>
            )}
          </Card>
        </div>
      )}
    </Page>
  );
}
