import type { ReactNode } from "react";
import { Page } from "@/components/Page";
import { copy } from "@/lib/copy";
import { occAgentEnabled } from "@/lib/features";

/**
 * Guard for the OCC Agent routes.
 *
 * The flag decides what is OFFERED. It is not the security boundary: the
 * backend does not mount these routes at all unless the same flag is set in its
 * own environment, so a hand-typed URL reaches a page that can call nothing.
 * This exists so that page says something useful instead of failing obscurely.
 */
export function AgentOnly({ children }: { children: ReactNode }) {
  if (!occAgentEnabled()) {
    return (
      <Page title={copy.agent.disabledTitle}>
        <p className="text-sm text-stone-600">{copy.agent.disabled}</p>
      </Page>
    );
  }
  return <>{children}</>;
}
