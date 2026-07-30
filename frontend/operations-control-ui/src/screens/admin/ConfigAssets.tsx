import { useState } from "react";
import { AlertTriangle, CheckCircle2, ChevronDown, Info, OctagonAlert } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type { AssetPackage } from "@/api/adminTypes";
import { AccessDenied } from "@/components/admin/AccessDenied";
import { CompatibilityTable } from "@/components/admin/CompatibilityTable";
import { EmptyNote, Field, SectionHeading } from "@/components/admin/primitives";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { copy } from "@/lib/copy";
import { formatDate } from "@/lib/format";
import { useAdminLoad } from "@/lib/useAdminLoad";
import { AdminTabs } from "./AdminLayout";
import { PackageWorkspace } from "./PackageWorkspace";

const READINESS_ICONS = {
  ready: CheckCircle2,
  attention: AlertTriangle,
  blocked: OctagonAlert,
} as const;

const READINESS_TONES = {
  ready: "border-emerald-200 bg-emerald-50/60 text-emerald-800",
  attention: "border-amber-200 bg-amber-50/60 text-amber-900",
  blocked: "border-rose-200 bg-rose-50/60 text-rose-900",
} as const;

/**
 * One asset package, operator first.
 *
 * The card answers the operational question — can this configuration safely
 * process a delivery? — in plain language. Everything an administrator needs is
 * still here, one disclosure away: identifiers, groupings, dependencies, parts
 * and activation metadata live under "Administration and technical details" so
 * they never crowd out the answer.
 */
function AssetCard({ asset }: { asset: AssetPackage }) {
  const [showTechnical, setShowTechnical] = useState(false);
  const readiness = asset.readiness;
  const state = readiness?.state ?? "attention";
  const Icon = READINESS_ICONS[state];

  return (
    <article data-asset={asset.id} className="rounded-2xl border border-stone-200 bg-white p-6">
      {/* Operational: what it is, whether it can be relied on. */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <h3 className="text-base font-semibold text-stone-900">{asset.label}</h3>
        <div className="flex flex-wrap items-center gap-2">
          <StatusChip status={asset.status} />
          <StatusChip status={asset.validated ? "valid" : "failed_checks"} />
        </div>
      </div>

      {readiness && (
        <p
          data-readiness={state}
          className={clsx(
            "mt-4 flex items-start gap-2 rounded-xl border px-4 py-3 text-sm leading-relaxed",
            READINESS_TONES[state],
          )}
        >
          <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <span>
            <span className="font-semibold">{readiness.label}. </span>
            {readiness.statement}
          </span>
        </p>
      )}

      <dl className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Field label={copy.admin.activeVersion}>v{asset.active_version}</Field>
        <Field label={copy.admin.activatedAt}>{formatDate(asset.last_activation.at) || "—"}</Field>
        <Field label={copy.admin.activatedBy}>{asset.last_activation.by || "—"}</Field>
        <Field label={copy.admin.assets.mappingDefaults}>
          {asset.mapping_defaults.setting_count} {copy.admin.assets.settings.toLowerCase()}
        </Field>
      </dl>

      <div className="mt-5 grid gap-5 sm:grid-cols-2">
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wide text-stone-500">
            {copy.admin.assets.supportedRegimes}
          </h4>
          <ul className="mt-2 flex flex-wrap gap-2">
            {asset.supported_regimes.length === 0 ? (
              <li className="text-sm text-stone-400">—</li>
            ) : (
              asset.supported_regimes.map((regime) => (
                <li
                  key={regime.id}
                  className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-700"
                >
                  {regime.label}
                </li>
              ))
            )}
          </ul>
        </div>
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wide text-stone-500">
            {copy.admin.assets.profiles}
          </h4>
          <p className="mt-2 text-sm text-stone-900">{asset.source_semantics.profile_count}</p>
        </div>
      </div>

      <div className="mt-5">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-stone-500">
          {copy.admin.readiness.warnings}
        </h4>
        {readiness && readiness.warnings.length > 0 ? (
          <ul className="mt-2 space-y-1.5">
            {readiness.warnings.map((warning, i) => (
              <li
                key={i}
                className="flex items-start gap-2 rounded-xl bg-amber-50 px-3 py-2 text-sm text-amber-900"
              >
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                {warning}
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-2 text-sm text-stone-500">{copy.admin.readiness.noWarnings}</p>
        )}
      </div>

      {/* Administrative: still complete, deliberately secondary. */}
      <div className="mt-6 border-t border-stone-100 pt-4">
        <button
          type="button"
          aria-expanded={showTechnical}
          onClick={() => setShowTechnical((open) => !open)}
          className="flex items-center gap-1.5 text-xs font-medium text-stone-500 hover:text-stone-700"
        >
          <ChevronDown
            className={clsx("h-3.5 w-3.5 transition-transform", showTechnical && "rotate-180")}
            aria-hidden
          />
          {showTechnical ? copy.admin.readiness.hide : copy.admin.readiness.show}
        </button>
        {showTechnical && (
          <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Field label={copy.admin.readiness.identifier}>
              <span className="font-mono text-xs text-stone-600">{asset.id}</span>
            </Field>
            <Field label={copy.admin.draftVersion}>
              {asset.draft_version ? `v${asset.draft_version}` : "—"}
            </Field>
            <Field label={copy.admin.assets.taxonomies}>
              {asset.taxonomies.length > 0 ? asset.taxonomies.join(", ") : "—"}
            </Field>
            <Field label={copy.admin.sections.dependencies}>
              {asset.dependencies.map((d) => d.name).join(", ") || "—"}
            </Field>
            <Field label={copy.admin.assets.sourceSemantics}>
              {asset.source_semantics.profile_version || "—"}
            </Field>
            <Field label={copy.admin.assets.issuePolicy}>{asset.issue_policy.rule_count}</Field>
          </dl>
        )}
      </div>
    </article>
  );
}

export function ConfigAssetsScreen() {
  const client = useOpsClient();
  const [showAdmin, setShowAdmin] = useState(false);
  const { data, error, forbidden, loading, reload } = useAdminLoad(
    () => client.getConfigCatalogue(),
    [],
  );

  if (forbidden) return <AccessDenied />;

  // The detailed lifecycle workspace below is about the selected asset, so it
  // is named for it rather than titled generically.
  const selected = data?.assets[0];
  const workspaceTitle = selected
    ? `${selected.label} ${copy.admin.readiness.configurationSuffix}`
    : copy.admin.assets.title;

  return (
    <Page title={copy.admin.title} subtitle={copy.admin.subtitle}>
      <AdminTabs />

      {loading && <Loading />}
      {error && !loading && !data && (
        <ErrorNote message={copy.admin.unavailable} onRetry={() => void reload()} />
      )}

      {data && !loading && (
        <div className="mb-10 space-y-6">
          <p className="flex items-start gap-2 rounded-xl border border-stone-200 bg-stone-50 px-4 py-3 text-sm text-stone-600">
            <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            {copy.admin.assets.clientNote}
          </p>

          <section>
            <SectionHeading>{copy.admin.readiness.heading}</SectionHeading>
            {data.assets.length === 0 ? (
              <EmptyNote>{copy.admin.assets.empty}</EmptyNote>
            ) : (
              <div className="space-y-3">
                {data.assets.map((asset) => (
                  <AssetCard key={asset.id} asset={asset} />
                ))}
              </div>
            )}
          </section>

          <CompatibilityTable matrix={data.compatibility} issues={data.issues} />
        </div>
      )}

      {/* Administrative area: version history, parts, impact, drafts and
          rollback. Nothing is removed — it is simply no longer the first thing
          an operator has to read. */}
      <section className="border-t border-stone-200 pt-6">
        <button
          type="button"
          aria-expanded={showAdmin}
          onClick={() => setShowAdmin((open) => !open)}
          className="flex items-center gap-2 text-sm font-semibold text-stone-700 hover:text-stone-900"
        >
          <ChevronDown
            className={clsx("h-4 w-4 transition-transform", showAdmin && "rotate-180")}
            aria-hidden
          />
          {copy.admin.readiness.technicalHeading}
        </button>
        {showAdmin && (
          <div className="mt-6">
            <PackageWorkspace
              layer="asset"
              title={workspaceTitle}
              subtitle={copy.admin.assets.subtitle}
            />
          </div>
        )}
      </section>
    </Page>
  );
}
