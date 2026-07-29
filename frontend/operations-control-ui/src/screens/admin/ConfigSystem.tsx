import { Page } from "@/components/Page";
import { copy } from "@/lib/copy";
import { AdminTabs } from "./AdminLayout";
import { PackageWorkspace } from "./PackageWorkspace";

export function ConfigSystemScreen() {
  return (
    <Page title={copy.admin.title} subtitle={copy.admin.subtitle}>
      <AdminTabs />
      <PackageWorkspace
        layer="system"
        title={copy.admin.system.title}
        subtitle={copy.admin.system.subtitle}
      />
    </Page>
  );
}
