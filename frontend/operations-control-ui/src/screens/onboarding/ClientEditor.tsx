import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
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

type Tab = "general" | "portfolios" | "reporting" | "regimes" | "sources" | "history";

const TABS: { key: Tab; label: string }[] = [
  { key: "general", label: copy.onboarding.tabs.general },
  { key: "portfolios", label: copy.onboarding.tabs.portfolios },
  { key: "reporting", label: copy.onboarding.tabs.reporting },
  { key: "regimes", label: copy.onboarding.tabs.regimes },
  { key: "sources", label: copy.onboarding.tabs.sources },
  { key: "history", label: copy.onboarding.tabs.history },
];

const label = (value: string) => value.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

/** Plain names for the governed artefacts — never a file path. */
const ARTEFACT_LABELS: Record<string, string> = {
  client_config: "Client configuration",
  annex12_config: "Investor report configuration",
  portfolio_registry: "Portfolio metadata",
  source_registry: "Source registrations",
};

function Tabs({ current, onSelect }: { current: Tab; onSelect: (tab: Tab) => void }) {
  return (
    <nav className="mb-6 flex flex-wrap gap-1 border-b border-stone-200">
      {TABS.map((tab) => (
        <button
          key={tab.key}
          type="button"
          onClick={() => onSelect(tab.key)}
          className={clsx(
            "-mb-px border-b-2 px-4 py-2 text-sm font-medium transition-colors",
            current === tab.key
              ? "border-stone-900 text-stone-900"
              : "border-transparent text-stone-500 hover:text-stone-800",
          )}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}

function GeneralTab({ detail }: { detail: OnboardingClientDetail }) {
  const c = detail.profile!.client;
  const sr = detail.profile!.static_reporting;
  return (
    <div className="space-y-6">
      <Card title="Identity">
        <KeyValue label="Client name" value={c.client_name} />
        <KeyValue label="Client identifier" value={c.client_id} />
        <KeyValue label="Legal entity name" value={c.legal_entity_name} />
        <KeyValue label="Legal entity identifier" value={c.lei} />
        <KeyValue label="Jurisdiction" value={c.jurisdiction} />
        <KeyValue label="Reporting currency" value={c.reporting_currency} />
        <KeyValue label="Reporting time zone" value={c.time_zone} />
      </Card>
      <Card title="Contacts">
        <KeyValue label="Primary reporting contact" value={c.primary_reporting_contact} />
        <KeyValue label="Reporting email" value={c.reporting_email} />
        <KeyValue label="Operational contact" value={c.operational_contact} />
        <KeyValue label="Operational email" value={c.operational_email} />
      </Card>
      <Card title="Report presentation">
        <KeyValue label="Report title" value={sr.app_title} />
        <KeyValue label="Brand colour" value={sr.primary_color} />
        <KeyValue label="Trustee" value={sr.trustee_name} />
        <KeyValue label="Warehouse lender" value={sr.warehouse_lender} />
        <KeyValue label="Payment convention" value={sr.payment_convention} />
        <KeyValue label="Day-count convention" value={sr.day_count_convention} />
      </Card>
    </div>
  );
}

function PortfoliosTab({ detail }: { detail: OnboardingClientDetail }) {
  return (
    <div className="space-y-6">
      {detail.profile!.portfolios.map((p) => (
        <Card key={p.portfolio_id} title={p.display_name}>
          <KeyValue label="Portfolio identifier" value={p.portfolio_id} />
          <KeyValue label="Type" value={label(p.portfolio_type)} />
          <KeyValue label="Asset class" value={label(p.asset_class)} />
          <KeyValue label="How the book is held" value={label(p.structure)} />
          <KeyValue label="Funded reporting cadence" value={label(p.funded_cadence)} />
          <KeyValue
            label="Pipeline reporting cadence"
            value={p.pipeline_cadence ? label(p.pipeline_cadence) : "No pipeline data"}
          />
          <KeyValue label="Who supplies the data" value={p.source_system} />
        </Card>
      ))}
    </div>
  );
}

function ReportingTab({ detail }: { detail: OnboardingClientDetail }) {
  const derived = detail.derived_reporting ?? {};
  return (
    <Card title="Reporting products">
      <ul className="space-y-3">
        {Object.entries(derived).map(([key, product]) => (
          <li
            key={key}
            className="flex flex-wrap items-start justify-between gap-3 border-b border-stone-100 py-3 last:border-0"
          >
            <div>
              <span className="text-sm font-medium text-stone-900">{label(key)}</span>
              <p className="mt-0.5 max-w-xl text-xs text-stone-500">{product.reason}</p>
            </div>
            <span
              className={clsx(
                "rounded-full border px-2.5 py-0.5 text-xs font-medium",
                product.applies
                  ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                  : "border-stone-200 bg-stone-100 text-stone-500",
              )}
            >
              {product.applies ? (product.derived ? "Always prepared" : "Prepared") : "Not prepared"}
            </span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function RegimesTab({ detail }: { detail: OnboardingClientDetail }) {
  const regime = detail.profile!.regime;
  const entries = Object.entries(regime);
  return (
    <div className="space-y-6">
      {entries.length === 0 && (
        <Note>No standing regime configuration is held for this client.</Note>
      )}
      {entries.map(([product, answers]) => (
        <Card key={product} title={label(product)}>
          {Object.entries(answers).map(([key, value]) => (
            <KeyValue key={key} label={key} value={String(value)} />
          ))}
        </Card>
      ))}
      {(detail.unrepresented ?? []).length > 0 && (
        <Card
          title="Recorded, but not written into configuration"
          description="Held with this client's history. Nothing today reads them."
        >
          <ul className="space-y-2">
            {(detail.unrepresented ?? []).map((entry) => (
              <li key={`${entry.product}-${entry.key}`} className="text-sm text-stone-500">
                <span className="font-medium text-stone-700">{entry.label}</span> — {entry.reason}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}

function SourcesTab({ detail }: { detail: OnboardingClientDetail }) {
  const rows = detail.live_source_registrations as Record<string, string | boolean>[];
  return (
    <Card
      title="What Trakt expects to receive"
      description="Created by onboarding, then kept up to date by real deliveries."
    >
      {rows.length === 0 ? (
        <p className="py-2 text-sm text-stone-500">Nothing is registered for this client yet.</p>
      ) : (
        <ul className="space-y-3">
          {rows.map((row, index) => (
            <li key={index} className="border-b border-stone-100 pb-3 last:border-0 last:pb-0">
              <div className="text-sm font-medium text-stone-900">
                {String(row.source_portfolio_id)} · {label(String(row.dataset))}
              </div>
              <p className="mt-0.5 text-xs text-stone-500">
                Expected {label(String(row.frequency))} from {String(row.source_system || "—")} ·{" "}
                {row.regime_required ? "included in regulatory reporting" : "management information"}{" "}
                · {label(String(row.status))}
              </p>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function HistoryTab({ detail }: { detail: OnboardingClientDetail }) {
  return (
    <div className="space-y-6">
      <Note>
        Every approved change is kept. Earlier versions are never overwritten, so what was in force
        on any date can still be read.
      </Note>
      {detail.history.map((version) => (
        <Card
          key={version.version}
          title={`${copy.common.version} ${version.version}`}
          description={`${version.approved_by} · ${new Date(version.approved_at).toLocaleString()}`}
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
          <p className="mb-4 text-sm text-stone-600">{version.reason}</p>
          {version.changes.length > 0 && (
            <>
              <SectionHeading>What changed</SectionHeading>
              <ul className="mb-4 space-y-1">
                {version.changes.slice(0, 12).map((change) => (
                  <li
                    key={change.path}
                    className="flex flex-wrap items-baseline justify-between gap-2 text-sm"
                  >
                    <span className="text-stone-500">{change.label}</span>
                    <span>
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
              <SectionHeading>What was written</SectionHeading>
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
  );
}

/** The existing-client editor: what is in force, and how it got there. */
export function OnboardingClientScreen() {
  const { id = "" } = useParams();
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();
  const [tab, setTab] = useState<Tab>("general");
  const [busy, setBusy] = useState(false);
  const { data, error, loading, reload } = useLoad(() => client.getOnboardingClient(id), [id]);

  async function edit() {
    setBusy(true);
    try {
      const draft = await client.startOnboardingDraft({
        client_id: id,
        adopt: data?.status !== "onboarded",
      });
      navigate(`/onboarding/drafts/${draft.draft_id}`);
    } catch (err) {
      toast.show(errorMessage(err), "error");
      setBusy(false);
    }
  }

  if (loading) return <Page title={id}><Loading /></Page>;
  if (error || !data) {
    return (
      <Page title={id}>
        <ErrorNote message={copy.onboarding.unavailable} onRetry={() => void reload()} />
      </Page>
    );
  }

  if (data.status !== "onboarded" || !data.profile) {
    return (
      <Page title={id} subtitle={copy.onboarding.editorSubtitle}>
        <div className="space-y-6">
          <Note>{data.summary}</Note>
          <PrimaryButton onClick={() => void edit()} disabled={busy}>
            {copy.onboarding.adopt}
          </PrimaryButton>
        </div>
      </Page>
    );
  }

  return (
    <Page
      title={data.profile.client.client_name || id}
      subtitle={copy.onboarding.editorSubtitle}
      actions={
        <div className="flex items-center gap-2">
          <Link
            to="/onboarding"
            className="rounded-xl border border-stone-300 px-4 py-2 text-sm font-medium text-stone-700 transition-colors hover:border-stone-500"
          >
            All clients
          </Link>
          <PrimaryButton onClick={() => void edit()} disabled={busy}>
            {copy.onboarding.edit}
          </PrimaryButton>
        </div>
      }
    >
      <div className="mb-6 flex flex-wrap items-center gap-3 text-sm text-stone-500">
        <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-800">
          {copy.common.version} {data.version} in force
        </span>
        {data.approved_by && (
          <span>
            Approved by {data.approved_by}
            {data.approved_at ? ` on ${new Date(data.approved_at).toLocaleDateString()}` : ""}
          </span>
        )}
      </div>

      <Tabs current={tab} onSelect={setTab} />

      {tab === "general" && <GeneralTab detail={data} />}
      {tab === "portfolios" && <PortfoliosTab detail={data} />}
      {tab === "reporting" && <ReportingTab detail={data} />}
      {tab === "regimes" && <RegimesTab detail={data} />}
      {tab === "sources" && <SourcesTab detail={data} />}
      {tab === "history" && <HistoryTab detail={data} />}
    </Page>
  );
}
