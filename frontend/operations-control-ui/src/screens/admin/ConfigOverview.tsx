import { Link } from "react-router-dom";
import { AlertTriangle, OctagonAlert } from "lucide-react";
import { useOpsClient } from "@/api/context";
import type { AuditEntry, ConfigLayer } from "@/api/adminTypes";
import { AccessDenied } from "@/components/admin/AccessDenied";
import { AuditTimeline } from "@/components/admin/AuditTimeline";
import { CompatibilityTable } from "@/components/admin/CompatibilityTable";
import { ConfigurationPackageCard } from "@/components/admin/ConfigurationPackageCard";
import { EmptyNote, SectionHeading } from "@/components/admin/primitives";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { copy } from "@/lib/copy";
import { useAdminLoad } from "@/lib/useAdminLoad";
import { AdminTabs } from "./AdminLayout";

const LAYER_ROUTES: Record<ConfigLayer, string> = {
  system: "/admin/config/system",
  asset: "/admin/config/assets",
  regime: "/admin/config/regimes",
};

function Recent({ heading, entries }: { heading: string; entries: AuditEntry[] }) {
  return (
    <section>
      <SectionHeading>{heading}</SectionHeading>
      {entries.length === 0 ? (
        <EmptyNote>{copy.admin.overview.noRecent}</EmptyNote>
      ) : (
        <AuditTimeline entries={entries} />
      )}
    </section>
  );
}

export function ConfigOverviewScreen() {
  const client = useOpsClient();
  const { data, error, forbidden, loading, reload } = useAdminLoad(
    () => client.getConfigOverview(),
    [],
  );

  if (forbidden) return <AccessDenied />;

  const attention = data?.attention;
  const nothingNeedsAttention =
    attention &&
    attention.drafts_awaiting_action.length === 0 &&
    attention.validation_failures.length === 0 &&
    attention.compatibility_issues.length === 0;

  return (
    <Page title={copy.admin.title} subtitle={copy.admin.subtitle}>
      <AdminTabs />
      {loading && <Loading />}
      {error && !loading && !data && (
        <ErrorNote message={copy.admin.unavailable} onRetry={() => void reload()} />
      )}
      {data && !loading && (
        <div className="space-y-10">
          <section className="space-y-3">
            <ConfigurationPackageCard
              layer={data.layers.system}
              to={LAYER_ROUTES.system}
            />
            <ConfigurationPackageCard layer={data.layers.asset} to={LAYER_ROUTES.asset} />
            <ConfigurationPackageCard layer={data.layers.regime} to={LAYER_ROUTES.regime} />
          </section>

          <section>
            <SectionHeading>{copy.admin.overview.assetHeading}</SectionHeading>
            <div className="space-y-2">
              {data.assets.map((asset) => (
                <Link
                  key={asset.id}
                  to={`${LAYER_ROUTES.asset}?asset=${asset.id}`}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-stone-200 bg-white px-4 py-3 hover:border-stone-300 hover:bg-stone-50"
                >
                  <div>
                    <p className="text-sm font-medium text-stone-900">{asset.label}</p>
                    <p className="mt-0.5 text-xs text-stone-500">
                      {copy.admin.assets.supportedRegimes}:{" "}
                      {asset.supported_regimes.map((r) => r.label).join(", ") || "—"}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-stone-600">v{asset.active_version}</span>
                    <StatusChip status={asset.status} />
                  </div>
                </Link>
              ))}
            </div>
          </section>

          <section>
            <SectionHeading>{copy.admin.overview.regimeHeading}</SectionHeading>
            <div className="space-y-2">
              {data.regimes.map((regime) => (
                <Link
                  key={regime.id}
                  to={`${LAYER_ROUTES.regime}?regime=${regime.id}`}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-stone-200 bg-white px-4 py-3 hover:border-stone-300 hover:bg-stone-50"
                >
                  <div>
                    <p className="text-sm font-medium text-stone-900">{regime.label}</p>
                    <p className="mt-0.5 text-xs text-stone-500">
                      {regime.regulator} · {regime.annex}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-stone-600">v{regime.active_version}</span>
                    <StatusChip status={regime.status} />
                  </div>
                </Link>
              ))}
            </div>
          </section>

          <section>
            <SectionHeading>{copy.admin.overview.attentionHeading}</SectionHeading>
            {nothingNeedsAttention ? (
              <EmptyNote>{copy.admin.overview.noAttention}</EmptyNote>
            ) : (
              <div className="space-y-4">
                {attention!.validation_failures.length > 0 && (
                  <div>
                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-500">
                      {copy.admin.overview.validationFailures}
                    </h3>
                    <div className="space-y-2">
                      {attention!.validation_failures.map((failure) => (
                        <Link
                          key={`${failure.layer}-${failure.version}`}
                          to={LAYER_ROUTES[failure.layer]}
                          className="flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900 hover:bg-rose-100"
                        >
                          <OctagonAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                          {failure.label} v{failure.version} — {copy.admin.validation.fail}
                        </Link>
                      ))}
                    </div>
                  </div>
                )}

                {attention!.drafts_awaiting_action.length > 0 && (
                  <div>
                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-500">
                      {copy.admin.overview.draftsAwaiting}
                    </h3>
                    <div className="space-y-2">
                      {attention!.drafts_awaiting_action.map((draft) => (
                        <Link
                          key={`${draft.layer}-${draft.version}`}
                          to={LAYER_ROUTES[draft.layer]}
                          className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-dashed border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 hover:bg-amber-100"
                        >
                          <span>
                            {draft.label} v{draft.version} · {draft.created_by}
                          </span>
                          <StatusChip status={draft.validated ? "valid" : "not_checked"} />
                        </Link>
                      ))}
                    </div>
                  </div>
                )}

                {attention!.compatibility_issues.length > 0 && (
                  <div>
                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-500">
                      {copy.admin.overview.compatibilityIssues}
                    </h3>
                    <div className="space-y-2">
                      {attention!.compatibility_issues.map((issue) => (
                        <p
                          key={issue.message}
                          className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900"
                        >
                          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                          {issue.message}
                        </p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>

          <CompatibilityTable matrix={data.compatibility} />

          <Recent
            heading={copy.admin.overview.recentActivations}
            entries={data.recent_activations}
          />
          <Recent heading={copy.admin.overview.recentRollbacks} entries={data.recent_rollbacks} />
        </div>
      )}
    </Page>
  );
}
