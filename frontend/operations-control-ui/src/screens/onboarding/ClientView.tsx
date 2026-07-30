import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { PencilLine } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type { OnboardingClientDetail } from "@/api/onboardingTypes";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import {
  ActionChip,
  Card,
  KeyValue,
  Note,
  PrimaryButton,
  SectionHeading,
} from "@/components/onboarding/primitives";
import { Page } from "@/components/Page";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage, useLoad } from "@/lib/useLoad";

type Tab = "general" | "entities" | "portfolios" | "reporting" | "sources" | "history" | "cases";

const TABS: { key: Tab; label: string }[] = [
  { key: "general", label: copy.onboarding.tabs.general },
  { key: "entities", label: copy.onboarding.tabs.entities },
  { key: "portfolios", label: copy.onboarding.tabs.portfolios },
  { key: "reporting", label: copy.onboarding.tabs.reporting },
  { key: "sources", label: copy.onboarding.tabs.sources },
  { key: "history", label: copy.onboarding.tabs.history },
  { key: "cases", label: copy.onboarding.tabs.cases },
];

const label = (v: string) => v.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

const ARTEFACT_LABELS: Record<string, string> = {
  client_config: "Client configuration",
  annex12_config: "Investor report configuration",
  portfolio_registry: "Portfolio metadata",
  source_registry: "Source registrations",
  tenancy: "Client index",
};

function block(detail: OnboardingClientDetail, key: string): Record<string, unknown> {
  return ((detail.answers ?? {})[key] ?? {}) as Record<string, unknown>;
}

function list(detail: OnboardingClientDetail, key: string): Record<string, unknown>[] {
  return ((detail.answers ?? {})[key] ?? []) as Record<string, unknown>[];
}

export function OnboardingClientScreen() {
  const { id = "" } = useParams();
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();
  const [tab, setTab] = useState<Tab>("general");
  const [busy, setBusy] = useState(false);
  const { data, error, loading, reload } = useLoad(() => client.getOnboardingClient(id), [id]);

  async function amend() {
    setBusy(true);
    try {
      const created = await client.startAmendmentCase(id);
      navigate(`/onboarding/cases/${encodeURIComponent(created.case_id)}`);
    } catch (err) {
      toast.show(errorMessage(err), "error");
      setBusy(false);
    }
  }

  if (loading) {
    return (
      <Page title={id}>
        <Loading />
      </Page>
    );
  }
  if (error || !data) {
    return (
      <Page title={id}>
        <ErrorNote message={copy.onboarding.unavailable} onRetry={() => void reload()} />
      </Page>
    );
  }
  if (data.status !== "active" || !data.answers) {
    return (
      <Page title={id} subtitle={copy.onboarding.clientSubtitle}>
        <Note>{data.summary}</Note>
      </Page>
    );
  }

  const clientBlock = block(data, "client");
  const contacts = block(data, "contacts");
  const presentation = block(data, "presentation");

  return (
    <Page
      title={String(clientBlock.client_name ?? id)}
      subtitle={copy.onboarding.clientSubtitle}
      actions={
        <div className="flex flex-wrap items-center gap-2">
          <Link
            to="/onboarding"
            className="rounded-xl border border-stone-300 px-4 py-2 text-sm font-medium text-stone-700 transition-colors hover:border-stone-500"
          >
            {copy.onboarding.allClients}
          </Link>
          <PrimaryButton onClick={() => void amend()} disabled={busy}>
            <span className="flex items-center gap-2">
              <PencilLine className="h-4 w-4" aria-hidden />
              {copy.onboarding.amend}
            </span>
          </PrimaryButton>
        </div>
      }
    >
      <div className="mb-6 flex flex-wrap items-center gap-3 text-sm text-stone-500">
        <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-800">
          {copy.common.version} {data.version} in force
        </span>
        {data.activated_by && (
          <span className="break-words">
            Activated by {data.activated_by}
            {data.activated_at ? ` on ${new Date(data.activated_at).toLocaleDateString()}` : ""}
          </span>
        )}
      </div>

      <nav className="mb-6 -mx-1 flex gap-1 overflow-x-auto border-b border-stone-200 px-1">
        {TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setTab(t.key)}
            className={clsx(
              "-mb-px shrink-0 whitespace-nowrap border-b-2 px-4 py-2 text-sm font-medium transition-colors",
              tab === t.key
                ? "border-stone-900 text-stone-900"
                : "border-transparent text-stone-500 hover:text-stone-800",
            )}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "general" && (
        <div className="space-y-6">
          <Card title={copy.onboarding.identityHeading}>
            <KeyValue label="Client name" value={String(clientBlock.client_name ?? "")} />
            <KeyValue label="Client identifier" value={String(clientBlock.client_id ?? "")} />
            <KeyValue label="Jurisdiction" value={String(clientBlock.jurisdiction ?? "")} />
            <KeyValue
              label="Reporting currency"
              value={String(clientBlock.reporting_currency ?? "")}
            />
            <KeyValue label="Reporting time zone" value={String(clientBlock.time_zone ?? "")} />
          </Card>
          <Card title={copy.onboarding.contactsHeading}>
            {Object.entries(contacts).map(([key, value]) => (
              <KeyValue key={key} label={label(key)} value={String(value ?? "")} />
            ))}
          </Card>
          {Object.keys(presentation).length > 0 && (
            <Card title={copy.onboarding.presentationHeading}>
              {Object.entries(presentation).map(([key, value]) => (
                <KeyValue key={key} label={label(key)} value={String(value ?? "")} />
              ))}
            </Card>
          )}
        </div>
      )}

      {tab === "entities" && (
        <div className="space-y-6">
          {list(data, "entities").map((entity, i) => (
            <Card key={String(entity.entity_id ?? i)} title={String(entity.legal_name ?? "")}>
              <KeyValue
                label="Roles"
                value={((entity.roles ?? []) as string[]).map(label).join(", ")}
              />
              <KeyValue label="Identifier" value={String(entity.lei ?? "")} />
              <KeyValue
                label="Country of establishment"
                value={String(entity.country_of_establishment ?? "")}
              />
              <KeyValue
                label="Registration number"
                value={String(entity.registration_number ?? "")}
              />
            </Card>
          ))}
        </div>
      )}

      {tab === "portfolios" && (
        <div className="space-y-6">
          {list(data, "portfolios").map((p, i) => (
            <Card key={String(p.portfolio_id ?? i)} title={String(p.display_name ?? "")}>
              <KeyValue label="Portfolio identifier" value={String(p.portfolio_id ?? "")} />
              <KeyValue label="Type" value={label(String(p.portfolio_type ?? ""))} />
              <KeyValue label="Asset class" value={label(String(p.asset_class ?? ""))} />
              <KeyValue label="How the book is held" value={label(String(p.structure ?? ""))} />
              <KeyValue
                label="Reporting period convention"
                value={label(String(p.period_convention ?? ""))}
              />
            </Card>
          ))}
        </div>
      )}

      {tab === "reporting" && (
        <Card title={copy.onboarding.reportingHeading}>
          <KeyValue
            label="Products"
            value={(
              ((data.answers.reporting ?? {}) as { products?: string[] }).products ?? []
            )
              .map(label)
              .join(", ")}
          />
          {Object.entries((data.answers.regime ?? {}) as Record<string, Record<string, unknown>>).map(
            ([product, answers]) => (
              <div key={product} className="mt-4">
                <SectionHeading>{label(product)}</SectionHeading>
                {Object.entries(answers).map(([key, value]) => (
                  <KeyValue key={key} label={key} value={String(value)} />
                ))}
              </div>
            ),
          )}
        </Card>
      )}

      {tab === "sources" && (
        <Card
          title={copy.onboarding.sourcesHeading}
          description={copy.onboarding.sourcesDescription}
        >
          {data.live_source_registrations.length === 0 ? (
            <p className="py-2 text-sm text-stone-500">{copy.onboarding.noSources}</p>
          ) : (
            <ul className="space-y-3">
              {data.live_source_registrations.map((row, i) => (
                <li key={i} className="border-b border-stone-100 pb-3 last:border-0 last:pb-0">
                  <div className="break-words text-sm font-medium text-stone-900">
                    {String(row.source_portfolio_id)} · {label(String(row.dataset))}
                  </div>
                  <p className="mt-0.5 break-words text-xs text-stone-500">
                    Expected {label(String(row.frequency))} from {String(row.source_system || "—")}{" "}
                    ·{" "}
                    {row.regime_required
                      ? "included in regulatory reporting"
                      : "management information"}{" "}
                    · {label(String(row.status))}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </Card>
      )}

      {tab === "history" && (
        <div className="space-y-6">
          <Note>{copy.onboarding.historyNote}</Note>
          {data.history.map((version) => (
            <Card
              key={version.version}
              title={`${copy.common.version} ${version.version}`}
              description={`${version.activated_by} · ${new Date(version.activated_at).toLocaleString()} · ${version.case_id}`}
              actions={
                <span
                  className={clsx(
                    "rounded-full border px-2.5 py-0.5 text-xs font-medium",
                    version.status === "active"
                      ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                      : "border-stone-200 bg-stone-100 text-stone-500",
                  )}
                >
                  {version.status === "active" ? "In force" : "Replaced"}
                </span>
              }
            >
              <p className="mb-4 break-words text-sm text-stone-600">{version.reason}</p>
              {version.changes.length > 0 && (
                <>
                  <SectionHeading>{copy.onboarding.whatChanged}</SectionHeading>
                  <ul className="mb-4 space-y-1">
                    {version.changes.slice(0, 12).map((change) => (
                      <li
                        key={change.path}
                        className="flex flex-wrap items-baseline justify-between gap-2 text-sm"
                      >
                        <span className="break-words text-stone-500">{change.label}</span>
                        <span className="break-words">
                          {change.before && (
                            <span className="text-stone-400 line-through">{change.before}</span>
                          )}{" "}
                          <span className="font-medium text-stone-900">{change.after || "—"}</span>
                        </span>
                      </li>
                    ))}
                    {version.changes.length > 12 && (
                      <li className="text-xs text-stone-400">
                        and {version.changes.length - 12} more
                      </li>
                    )}
                  </ul>
                </>
              )}
              {version.artefacts.length > 0 && (
                <>
                  <SectionHeading>{copy.onboarding.whatWasWritten}</SectionHeading>
                  <ul className="space-y-1">
                    {version.artefacts.map((artefact) => (
                      <li
                        key={artefact.target}
                        className="flex flex-wrap items-center justify-between gap-2 text-sm text-stone-600"
                      >
                        <span>{ARTEFACT_LABELS[artefact.kind] ?? label(artefact.kind)}</span>
                        <ActionChip action={artefact.action} tense="past" />
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </Card>
          ))}
        </div>
      )}

      {tab === "cases" && (
        <Card title={copy.onboarding.casesHeading}>
          {data.cases.length === 0 ? (
            <p className="py-2 text-sm text-stone-500">{copy.onboarding.noCases}</p>
          ) : (
            <ul className="divide-y divide-stone-100">
              {data.cases.map((row) => (
                <li key={row.case_id} className="py-3">
                  <Link
                    to={`/onboarding/cases/${encodeURIComponent(row.case_id)}`}
                    className="flex flex-wrap items-center justify-between gap-2 hover:opacity-80"
                  >
                    <span className="break-words text-sm text-stone-700">
                      {row.case_id} · {row.kind_label}
                    </span>
                    <span className="text-xs text-stone-500">{row.status_label}</span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Card>
      )}
    </Page>
  );
}
