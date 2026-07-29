import { AlertTriangle } from "lucide-react";
import { useOpsClient } from "@/api/context";
import type { RegimePackage } from "@/api/adminTypes";
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

function RegimeCard({ regime }: { regime: RegimePackage }) {
  return (
    <article
      data-regime={regime.id}
      className="rounded-2xl border border-stone-200 bg-white p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-stone-900">{regime.label}</h3>
          <p className="mt-0.5 text-sm text-stone-500">{regime.description}</p>
        </div>
        <div className="flex items-center gap-2">
          <StatusChip status={regime.status} />
          <StatusChip status={regime.validated ? "valid" : "failed_checks"} />
        </div>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Field label={copy.admin.regimes.regulator}>{regime.regulator}</Field>
        <Field label={copy.admin.regimes.annex}>{regime.annex}</Field>
        <Field label={copy.admin.activeVersion}>v{regime.active_version}</Field>
        <Field label={copy.admin.regimes.packageVersion}>{regime.package_version ?? "—"}</Field>
      </dl>

      <dl className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Field label={copy.admin.draftVersion}>
          {regime.draft_version ? `v${regime.draft_version}` : "—"}
        </Field>
        <Field label={copy.admin.activatedAt}>
          {formatDate(regime.last_activation.at) || "—"}
        </Field>
        <Field label={copy.admin.activatedBy}>{regime.last_activation.by || "—"}</Field>
        <Field label={copy.admin.regimes.fieldUniverse}>{regime.field_universe.count}</Field>
      </dl>

      <div className="mt-5">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-stone-500">
          {copy.admin.regimes.schema}
        </h4>
        <p className="mt-1 break-all text-sm text-stone-800">{regime.schema.file || "—"}</p>
        {regime.schema.is_draft && (
          <p className="mt-2 flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            {copy.admin.regimes.draftSchema}
          </p>
        )}
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-3">
        <Field label={copy.admin.regimes.codeOrder}>
          {regime.code_order.entry_count ?? 0}
          {regime.code_order.version ? ` · v${regime.code_order.version}` : ""}
        </Field>
        <Field label={copy.admin.regimes.unknownValues}>
          {regime.validation_policy.unknown_values || "—"}
        </Field>
        <Field label={copy.admin.regimes.deferred}>
          {regime.validation_policy.deferred_fields.length > 0
            ? regime.validation_policy.deferred_fields.join(", ")
            : "—"}
        </Field>
        <Field label={copy.admin.regimes.allowedCodes}>
          {regime.no_data_policy.codes_allowed.length > 0
            ? regime.no_data_policy.codes_allowed.join(", ")
            : "—"}
        </Field>
        <Field label={copy.admin.regimes.fieldsAllowing}>
          {regime.no_data_policy.fields_allowing_no_data}
        </Field>
        <Field label={copy.admin.sections.dependencies}>
          {regime.dependencies.map((d) => d.name).join(", ") || "—"}
        </Field>
      </dl>

      <div className="mt-5">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-stone-500">
          {copy.admin.regimes.compatibleAssets}
        </h4>
        <ul className="mt-2 flex flex-wrap gap-2">
          {regime.compatible_assets.length === 0 ? (
            <li className="text-sm text-stone-400">—</li>
          ) : (
            regime.compatible_assets.map((asset) => (
              <li
                key={asset.id}
                className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-700"
              >
                {asset.label}
              </li>
            ))
          )}
        </ul>
      </div>
    </article>
  );
}

export function ConfigRegimesScreen() {
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
          <section>
            <SectionHeading>{copy.admin.regimes.title}</SectionHeading>
            {data.regimes.length === 0 ? (
              <EmptyNote>{copy.admin.regimes.empty}</EmptyNote>
            ) : (
              <div className="space-y-3">
                {data.regimes.map((regime) => (
                  <RegimeCard key={regime.id} regime={regime} />
                ))}
              </div>
            )}
          </section>

          <CompatibilityTable matrix={data.compatibility} issues={data.issues} />
        </div>
      )}

      <PackageWorkspace
        layer="regime"
        title={copy.admin.regimes.title}
        subtitle={copy.admin.regimes.subtitle}
      />
    </Page>
  );
}
