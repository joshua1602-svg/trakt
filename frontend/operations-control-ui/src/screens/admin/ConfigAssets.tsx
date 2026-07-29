import { Info } from "lucide-react";
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

function AssetCard({ asset }: { asset: AssetPackage }) {
  return (
    <article
      data-asset={asset.id}
      className="rounded-2xl border border-stone-200 bg-white p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-stone-900">{asset.label}</h3>
          <p className="mt-0.5 text-xs text-stone-500">{asset.id}</p>
        </div>
        <div className="flex items-center gap-2">
          <StatusChip status={asset.status} />
          <StatusChip status={asset.validated ? "valid" : "failed_checks"} />
        </div>
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Field label={copy.admin.activeVersion}>v{asset.active_version}</Field>
        <Field label={copy.admin.draftVersion}>
          {asset.draft_version ? `v${asset.draft_version}` : "—"}
        </Field>
        <Field label={copy.admin.activatedAt}>
          {formatDate(asset.last_activation.at) || "—"}
        </Field>
        <Field label={copy.admin.activatedBy}>{asset.last_activation.by || "—"}</Field>
      </dl>

      <div className="mt-5">
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

      <dl className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Field label={copy.admin.assets.profiles}>{asset.source_semantics.profile_count}</Field>
        <Field label={copy.admin.assets.mappingDefaults}>
          {asset.mapping_defaults.setting_count} {copy.admin.assets.settings.toLowerCase()}
        </Field>
        <Field label={copy.admin.assets.taxonomies}>
          {asset.taxonomies.length > 0 ? asset.taxonomies.join(", ") : "—"}
        </Field>
        <Field label={copy.admin.sections.dependencies}>
          {asset.dependencies.map((d) => d.name).join(", ") || "—"}
        </Field>
      </dl>
    </article>
  );
}

export function ConfigAssetsScreen() {
  const client = useOpsClient();
  const { data, error, forbidden, loading, reload } = useAdminLoad(
    () => client.getConfigCatalogue(),
    [],
  );

  if (forbidden) return <AccessDenied />;

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
            <SectionHeading>{copy.admin.assets.title}</SectionHeading>
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

      <PackageWorkspace
        layer="asset"
        title={copy.admin.assets.title}
        subtitle={copy.admin.assets.subtitle}
      />
    </Page>
  );
}
