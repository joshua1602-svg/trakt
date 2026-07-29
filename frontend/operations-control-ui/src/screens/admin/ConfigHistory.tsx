import { useOpsClient } from "@/api/context";
import { AccessDenied } from "@/components/admin/AccessDenied";
import { AuditTimeline } from "@/components/admin/AuditTimeline";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { copy } from "@/lib/copy";
import { useAdminLoad } from "@/lib/useAdminLoad";
import { AdminTabs } from "./AdminLayout";

/** Every configuration change: draft, check, activation, rollback. */
export function ConfigHistoryScreen() {
  const client = useOpsClient();
  const { data, error, forbidden, loading, reload } = useAdminLoad(
    () => client.getConfigAudit(),
    [],
  );

  if (forbidden) return <AccessDenied />;

  return (
    <Page title={copy.admin.audit.title} subtitle={copy.admin.audit.subtitle}>
      <AdminTabs />
      {loading && <Loading />}
      {error && !loading && !data && (
        <ErrorNote message={copy.admin.unavailable} onRetry={() => void reload()} />
      )}
      {data && !loading && (
        <div className="space-y-6">
          <p className="text-xs text-stone-500">
            {data.chain_intact ? copy.admin.audit.chainIntact : copy.admin.audit.chainBroken}
          </p>
          <AuditTimeline entries={data.entries} />
        </div>
      )}
    </Page>
  );
}
